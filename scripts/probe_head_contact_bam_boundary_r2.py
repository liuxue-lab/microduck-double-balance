"""Independent entity/BAM memory probe on the compiled full model: zero dynamics, zero PPO.

One constructor CPU forward and one frozen-state GPU forward are budgeted.
Eight synthetic command ticks are not eight simulation steps. Scratch velocity
markers exist only for memory visibility tests and are restored without forward.
Sensor definitions remain compiled; their runtime/render lifecycle is excluded.
"""
from pathlib import Path
import argparse
from contextlib import contextmanager
import copy
import hashlib
import importlib.metadata
import json
import time
import traceback
import numpy as np
import mujoco

from head_contact_data_commit import commit_data, describe
import diagnose_head_contact_floor_cuda_events as events
import diagnose_head_contact_floor_cuda_schedule as bridge

ROOT = Path(__file__).resolve().parents[1]


def require(ok, message):
    if not bool(ok): raise RuntimeError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@contextmanager
def call_guards(mw, fw, wp, counts, progress):
    """Block native integration/capture/render setup; cap all allowed forwards."""
    originals = []
    def patch(obj, name, value):
        originals.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)
    def forbidden(*args, **kwargs):
        counts['blocked_step_or_capture_requests'] += 1
        save(progress, counts)
        raise RuntimeError('Zero-integration probe forbids step/integrator/capture')
    def render_forbidden(*args, **kwargs):
        counts['blocked_render_context_requests'] += 1
        save(progress, counts)
        raise RuntimeError('Entity memory probe excludes render context creation')
    def counted(fn, kind):
        def wrapper(*args, **kwargs):
            require(counts[kind+'_attempted'] < 1, kind+' forward budget exhausted')
            counts[kind+'_attempted'] += 1; save(progress, counts)
            value = fn(*args, **kwargs)
            counts[kind+'_completed'] += 1; save(progress, counts)
            return value
        return wrapper
    try:
        for name in ('mj_step','mj_step1','mj_step2','mj_Euler','mj_RungeKutta','mj_implicit'):
            if hasattr(mujoco, name): patch(mujoco, name, forbidden)
        for name in ('step','step1','step2','euler','rungekutta4','implicit'):
            if hasattr(fw, name): patch(fw, name, forbidden)
        for name in ('step','step1','step2','euler','rungekutta4','implicit'):
            if hasattr(mw, name): patch(mw, name, forbidden)
        for name in ('ScopedCapture','capture_launch'):
            patch(wp, name, forbidden)
        patch(mw, 'create_render_context', render_forbidden)
        patch(mujoco, 'mj_forward', counted(mujoco.mj_forward, 'CPU_forward'))
        native = fw.forward
        gpu = counted(native, 'GPU_forward')
        patch(fw, 'forward', gpu)
        patch(mw, 'forward', gpu)
        yield
    finally:
        for obj, name, value in reversed(originals): setattr(obj, name, value)


def initialize_memory_entities(scene, sim):
    """Initialize Entity/BAM readers without the Scene sensor/render lifecycle.

    Scene construction/compilation already included the original sensors. Only
    entities are initialized/reset here; no Scene initialize/reset/update or
    sensor lifecycle method is called. This is not a full Scene lifecycle test.
    """
    require(scene.sensor_context is None, 'Unexpected preexisting sensor context')
    sensors = dict(scene.sensors)
    entities = dict(scene.entities)
    model = sim.mj_model
    fields = ('nsensor', 'nsensordata', 'ncam', 'nbody', 'ngeom', 'nq', 'nv', 'nu')
    dimensions = {name: int(getattr(model, name)) for name in fields}
    for ent in entities.values():
        ent.initialize(model, sim.model, sim.data, scene.device)
    for ent in entities.values():
        ent.reset()
    require(scene.sensor_context is None, 'Entity initialization created sensor context')
    require(set(scene.sensors) == set(sensors) and
            all(scene.sensors[name] is sensor for name, sensor in sensors.items()),
            'Compiled sensor declarations/objects changed')
    require(set(scene.entities) == set(entities) and
            all(scene.entities[name] is ent for name, ent in entities.items()),
            'Entity ownership changed during initialization')
    require({name: int(getattr(model, name)) for name in fields} == dimensions,
            'Compiled model dimensions changed')
    return {'scope': 'entities_and_BAM_only',
            'compiled_sensor_definitions_retained': True,
            'sensor_lifecycle_initialized': False,
            'sensor_context_created': False,
            'full_scene_lifecycle_validated': False,
            'compiled_model_dimensions': dimensions,
            'entities_initialized_and_reset': list(entities),
            'excluded_sensor_runtime': [
                {'name': name, 'type': type(sensor).__module__ + '.' + type(sensor).__qualname__,
                 'requires_sensor_context': bool(sensor.requires_sensor_context)}
                for name, sensor in sensors.items()]}


def tensor_state(value, torch):
    if isinstance(value, torch.Tensor):
        a = value.detach().cpu().numpy()
        return {'shape':list(a.shape),'dtype':str(a.dtype),'sha256':hashlib.sha256(a.tobytes()).hexdigest()}
    if isinstance(value, torch.Generator): return tensor_state(value.get_state(), torch)
    if isinstance(value, (str,int,float,bool,type(None))): return value
    if isinstance(value, (list,tuple)): return [tensor_state(x,torch) for x in value]
    if isinstance(value, dict): return {k:tensor_state(v,torch) for k,v in value.items()}
    if type(value).__module__.startswith('mjlab.utils.buffers.'):
        return {'type':type(value).__qualname__, 'fields':tensor_state(vars(value),torch)}
    raise TypeError('Unsupported external state: '+type(value).__name__)


def external_state(act, robot, torch):
    fields = ['_prev_motor_torque','_dt','_base_kp','_friction_fields_checked',
              '_delay_buffer','vin_tensor','vin_drop_gain','kp_scale','kd_scale',
              'default_kp_scale','default_kd_scale','friction_scale','default_friction_scale',
              'diagnostics_enabled','last_diagnostics']
    result = {n:tensor_state(getattr(act,n),torch) for n in fields}
    result['firmware'] = {n:tensor_state(getattr(act._bam_model.actuator,n),torch) for n in ('kp','vin')}
    result['targets'] = {n:tensor_state(getattr(robot.data,n),torch) for n in ('joint_pos_target','joint_vel_target','joint_effort_target')}
    result['CPU_RNG'] = tensor_state(torch.get_rng_state(),torch)
    result['CUDA_RNG'] = tensor_state(torch.cuda.get_rng_state('cuda:0'),torch)
    return result


def run(out, plan_path):
    out.mkdir(parents=True, exist_ok=False)
    started=time.monotonic()
    counts={'attempted':0,'completed':0,'CPU_forward_attempted':0,'CPU_forward_completed':0,
            'GPU_forward_attempted':0,'GPU_forward_completed':0,'blocked_step_or_capture_requests':0,
            'blocked_render_context_requests':0,
            'bam_compute_calls':0,'delay_apply_calls':0,'logical_command_ticks':0,'commit_array_copy_calls':0,'PPO':0}
    progress=out/'progress.json';save(progress,counts)
    report={'status':'STARTED','PPO':0,'full_robot_steps':0,'cloud_hours':0,
            'formal_physics_adoption':False,'formal_task_migration':False,'stage11_complete':False,
            'zero_dynamics':True,'not_a_floor_or_full_robot_physics_acceptance':True,
            'revision':'r2_entity_initialization','full_scene_sensor_lifecycle_validated':False,
            'not_a_full_macro_rollback_implementation':True,'canonical_steps':33647}
    source_paths={};sim=act=baseline_data=None;restore_needed=False;wp=torch=None
    originals=[]
    try:
        plan=json.loads(plan_path.read_text())
        versions={name:importlib.metadata.version(name) for name in plan['versions']}
        require(versions==plan['versions'],'Pinned package version mismatch')
        for distname, files in plan['installed_source_hashes'].items():
            dist=importlib.metadata.distribution(distname)
            for name,h in files.items():source_paths[str(Path(dist.locate_file(name)))]=h
        for name,h in plan['project_source_hashes'].items():source_paths[str(ROOT/name)]=h
        require(all(Path(p).is_file() and sha(p)==h for p,h in source_paths.items()),'Source differs; preserved, stop before scene')
        save(out/'source-hashes.json',source_paths)
        require(mujoco.get_mjcb_control() is None,'Unexpected MuJoCo control callback')
        import torch
        import warp as wp
        import mujoco_warp as mw
        from mujoco_warp._src import forward as fw
        from mjlab.scene import Scene
        from mjlab.sim import Simulation
        from mjlab_microduck.robot.head_contact import physics_scene_pair
        from mjlab_microduck.actuator.friction_dr_bam import FrictionDRBamActuator
        require(torch.cuda.is_available(),'Local CUDA required for bridge memory semantics')
        device=wp.get_device('cuda:0');require(device.is_cuda and '5060' in device.name,'RTX5060 only')
        wp.set_device(device);torch.manual_seed(1105001);torch.cuda.manual_seed_all(1105001)

        class ZeroStepSimulation(Simulation):
            def _should_use_cuda_graph(self): return False
            def create_graph(self):
                self.step_graph=self.forward_graph=self.reset_graph=self.sense_graph=None
            def step(self): raise RuntimeError('Zero-step diagnostic subclass')

        with call_guards(mw,fw,wp,counts,progress):
            cfg,_,candidate=physics_scene_pair()
            require(cfg.sim.mujoco.timestep==.002 and cfg.decimation==10,'2ms/20ms contract changed')
            candidate=copy.deepcopy(candidate);candidate.num_envs=1
            scene=Scene(candidate,'cuda:0');model=scene.compile()
            sim_cfg=copy.deepcopy(cfg.sim);sim_cfg.nconmax=max(sim_cfg.nconmax or 0,256);sim_cfg.njmax=max(sim_cfg.njmax or 0,1024)
            require(model.nu==14 and model.neq==0 and model.nkey>=1,'Unexpected full-scene joints/equalities/keyframe')
            sim=ZeroStepSimulation(1,sim_cfg,model,'cuda:0')
            require(counts['CPU_forward_completed']==1 and not sim.use_cuda_graph,'Constructor forward/capture scope differs')
            sim.expand_model_fields(('dof_frictionloss','dof_damping'))
            initialization = initialize_memory_entities(scene, sim)
            save(out/'initialization-scope.json', initialization)
            report['initialization_scope'] = initialization
            robot=scene['robot']
            require(len(robot.actuators)==1,'Expected single canonical BAM actuator group')
            act=robot.actuators[0]
            require(type(act) is FrictionDRBamActuator,'Unexpected actuator type')
            require(len(act.target_names)==14 and all(not n.startswith('passive_') for n in act.target_names),'Actuator selector changed')
            require(act.cfg.delay_min_lag==3 and act.cfg.delay_max_lag==6 and act._dt==.002,'BAM cadence contract changed')
            require(not act.diagnostics_enabled,'Unexpected diagnostic side effects')
            # Initialize one frozen keyframe. This is not a rollout/reset validation.
            sim.data.qpos[:] = torch.as_tensor(model.key_qpos[0].copy(),device='cuda:0',dtype=torch.float32)[None]
            sim.data.qvel.zero_();sim.data.ctrl.zero_();sim.data.time.zero_()
            sim.forward();wp.synchronize();torch.cuda.synchronize()
            d=sim.wp_data;wm=sim.wp_model
            require(np.isfinite(d.qpos.numpy()).all() and np.isfinite(d.qvel.numpy()).all() and np.isfinite(d.qacc.numpy()).all(),'Nonfinite setup state')
            require(not sim.step_graph and not sim.forward_graph,'Unexpected graph')
            structure, arrays, owners, _=describe(d,wp.array)
            save(out/'data-layout-before.json',{'descriptors':structure,'owners':owners})
            capacity=sum(int(a.capacity) for a in arrays.values())
            require(capacity<256*1024*1024,'Unexpected full Data footprint')
            require(float(d.time.numpy()[0])==0,'Setup advanced time')
            before_model=bridge.fingerprint(wm)
            initial_qpos=d.qpos.numpy().copy();initial_qvel=d.qvel.numpy().copy()
            np.savez_compressed(out/'frozen-state.npz',qpos=initial_qpos,qvel=initial_qvel,time=d.time.numpy(),qacc=d.qacc.numpy())
            save(out/'scene-contract.json',{'device':device.name,'versions':versions,'nq':model.nq,'nv':model.nv,'nu':model.nu,'nbody':model.nbody,'ngeom':model.ngeom,'neq':model.neq,'nworld':1,'integrator':int(model.opt.integrator),'timestep':float(model.opt.timestep),'data_unique_arrays':len(arrays),'data_capacity_bytes':capacity,'actuator_names':list(act.target_names),'CPU_forward':1,'GPU_forward_budget':1,'graph_capture':False,'warmup_integrations':0})
            native_delay=act.apply_delay;native_compute=act.compute
            delayed=[];torques=[]
            def tracked_delay(command):
                require(counts['delay_apply_calls']<8,'Delay call budget exhausted')
                counts['delay_apply_calls']+=1
                return native_delay(command)
            def tracked_compute(command):
                require(counts['bam_compute_calls']<8,'BAM call budget exhausted')
                counts['bam_compute_calls']+=1
                delayed.append(command.position_target.detach().clone())
                torque=native_compute(command)
                require(bool(torch.isfinite(torque).all()),'Nonfinite BAM output')
                torques.append(torque.detach().cpu().numpy().copy())
                return torque
            originals=[('apply_delay',native_delay),('compute',native_compute)]
            act.apply_delay=tracked_delay;act.compute=tracked_compute
            base=robot.data.joint_pos.detach().clone();targets=[];tick_rows=[]
            for i in range(8):
                require(time.monotonic()-started<300,'Probe wall budget exceeded')
                target=base + (i+1)*2**-12
                targets.append(target[:,act.target_ids].detach().clone())
                robot.set_joint_position_target(target)
                scene.write_data_to_sim()
                wp.synchronize();torch.cuda.synchronize()
                lag=int(act._delay_buffer.current_lags[0].item())
                effective=min(lag,i)
                require(3<=lag<=6 and torch.equal(delayed[-1],targets[i-effective]),'Delayed target/history mismatch')
                require(int(act._delay_buffer._step_count[0])==i+1,'Delay advanced more than once per command tick')
                require(int(act._delay_buffer._buffer._num_pushes[0])==i+1,'Buffer append count mismatch')
                require(counts['bam_compute_calls']==counts['delay_apply_calls']==i+1,'BAM cadence mismatch')
                require(np.array_equal(d.qpos.numpy(),initial_qpos) and np.array_equal(d.qvel.numpy(),initial_qvel) and float(d.time.numpy()[0])==0,'Command-only tick changed physical state')
                counts['logical_command_ticks']=i+1;save(progress,counts)
                tick_rows.append({'logical_tick':i,'hypothetical_macro_time_s':i*.002,'physical_time_s':0.,'sampled_lag_ticks':lag,'effective_warmup_lag_ticks':effective,'target_source_tick':i-effective,'delay_counter':i+1})
            save(out/'command-ticks.json',tick_rows)
            np.savez_compressed(out/'command-data.npz',targets=np.stack([t.cpu().numpy() for t in targets]),delayed_targets=np.stack([t.cpu().numpy() for t in delayed]),motor_torques=np.stack(torques))
            after_model=bridge.fingerprint(wm)
            changed=[k for k in before_model if before_model[k]!=after_model[k]]
            require(set(changed)<={'dof_frictionloss','dof_damping'},'BAM changed unexpected Model fields')
            save(out/'model-before-bam.json',before_model);save(out/'model-after-bam.json',after_model)
            frozen_external=external_state(act,robot,torch);save(out/'external-state-before-memory.json',frozen_external)
            frozen_model=after_model;frozen_data=bridge.fingerprint(d)
            save(out/'data-before-memory-fingerprint.json',frozen_data)
            baseline_data=events.clone_data(d,wp)
            # Cache all readers before synthetic scratch mutation/commit.
            cached_qvel=sim.data.qvel._tensor
            require(robot.data.data is sim.data and act._data is sim.data,'Reader bridge ownership differs')
            require(sim.data.struct is d,'Simulation bridge points to different Data')
            ptr=cached_qvel.data_ptr();dof=int(act._dof_ids[0]);local_joint=int(act.target_ids[0])
            scratch=events.clone_data(d,wp)
            marker=initial_qvel.copy();marker[0,dof]=np.float32(2**-10)
            scratch.qvel.assign(marker);wp.synchronize()
            require(bridge.fingerprint(d)==frozen_data,'Rejected scratch changed original Data')
            require(not np.array_equal(scratch.qvel.numpy(),d.qvel.numpy()),'Synthetic marker was ineffective')
            rejected_hash=digest(bridge.fingerprint(scratch));del scratch
            require(external_state(act,robot,torch)==frozen_external and bridge.fingerprint(wm)==frozen_model,'Scratch rejection changed external state')
            candidate_data=events.clone_data(d,wp);candidate_data.qvel.assign(marker);wp.synchronize()
            # Real Data shape/scalar negative control: fail before any write.
            candidate_data.nworld=2
            rejected=False
            try: commit_data(d,candidate_data,wp)
            except ValueError: rejected=True
            finally: candidate_data.nworld=1
            require(rejected and bridge.fingerprint(d)==frozen_data,'Mismatched Data was not rejected before copying')
            restore_needed=True
            copied=commit_data(d,candidate_data,wp);counts['commit_array_copy_calls']+=copied;save(progress,counts)
            wp.synchronize();torch.cuda.synchronize()
            require(describe(d,wp.array)[2]==owners,'Existing Data owner identity changed')
            require(cached_qvel.data_ptr()==ptr and np.array_equal(cached_qvel.cpu().numpy(),marker),'Cached Torch reader stale after commit')
            require(float(robot.data.joint_vel[0,local_joint])==float(marker[0,dof]),'Entity did not see copied state')
            require(np.array_equal(act._data.qvel._tensor.cpu().numpy(),marker),'BAM did not see copied state')
            require(bridge.fingerprint(d)==bridge.fingerprint(candidate_data),'Complete Data copy mismatch')
            save(out/'data-committed-fingerprint.json',bridge.fingerprint(d))
            np.savez_compressed(out/'synthetic-memory-marker.npz',before_qvel=initial_qvel,marker_qvel=marker,committed_cached_reader=cached_qvel.cpu().numpy().copy(),entity_joint_vel=robot.data.joint_vel.cpu().numpy().copy(),entity_joint_v_adr=robot.data.indexing.joint_v_adr.cpu().numpy().copy())
            save(out/'reader-ownership.json',{'Data_qvel_pointer':int(d.qvel.ptr),'cached_Torch_pointer':int(ptr),'after_commit_Torch_pointer':int(cached_qvel.data_ptr()),'Simulation_Data_identity':id(d),'Simulation_bridge_Data_identity':id(sim.data.struct),'Entity_bridge_identity':id(robot.data.data),'Simulation_bridge_identity':id(sim.data),'BAM_bridge_identity':id(act._data),'marker_global_dof':dof,'marker_local_joint':local_joint,'forward_counts_at_marker':{k:v for k,v in counts.items() if 'forward' in k}})
            require(external_state(act,robot,torch)==frozen_external and bridge.fingerprint(wm)==frozen_model,'Commit changed frozen BAM/Model/RNG state')
            committed_hash=digest(bridge.fingerprint(d))
            copied_restore=commit_data(d,baseline_data,wp);counts['commit_array_copy_calls']+=copied_restore;save(progress,counts)
            wp.synchronize();torch.cuda.synchronize();restore_needed=False
            require(bridge.fingerprint(d)==frozen_data and describe(d,wp.array)[2]==owners,'Original Data not restored exactly')
            require(external_state(act,robot,torch)==frozen_external and bridge.fingerprint(wm)==frozen_model,'Restore changed external state')
            save(out/'external-state-after-memory.json',external_state(act,robot,torch))
            save(out/'data-restored-fingerprint.json',bridge.fingerprint(d))
            restored_structure,_,restored_owners,_=describe(d,wp.array)
            save(out/'data-layout-after.json',{'descriptors':restored_structure,'owners':restored_owners})
            report.update({'status':'BAM_MEMORY_BOUNDARY_PASS_REVIEW_REQUIRED','logical_command_ticks':8,
                           'not_physical_elapsed_s':.016,'physical_elapsed_s':0.,'changed_model_fields_at_bam_ticks':changed,
                           'memory_probe':{'synthetic_velocity_marker':True,'no_forward_with_marker':True,'rejected_scratch_sha256':rejected_hash,'committed_scratch_sha256':committed_hash,'restored_data_sha256':digest(frozen_data),'preflight_mismatch_rejected_before_copy':True,'owner_identity_preserved':True,'cached_Torch_Entity_BAM_visibility':True,'external_BAM_Model_RNG_unchanged':True,'exact_Data_restoration':True},
                           'configuration_unchanged':float(model.opt.timestep)==.002 and int(model.opt.integrator)==3})
            require(report['configuration_unchanged'],'Formal integrator/timestep changed')
            require(counts['CPU_forward_completed']==counts['GPU_forward_completed']==1 and counts['blocked_step_or_capture_requests']==counts['blocked_render_context_requests']==0,'Unexpected physics or render call count')
    except Exception as exc:
        report['status']='BAM_MEMORY_BOUNDARY_STOPPED_REVIEW_REQUIRED';report['error']=repr(exc)
        (out/'error.txt').write_text(traceback.format_exc())
    finally:
        if restore_needed and sim is not None and baseline_data is not None and wp is not None:
            try:
                counts['commit_array_copy_calls']+=commit_data(sim.wp_data,baseline_data,wp)
                report['failure_cleanup_Data_restored']=bridge.fingerprint(sim.wp_data)==bridge.fingerprint(baseline_data)
            except Exception as exc: report['failure_cleanup_error']=repr(exc)
        if act is not None:
            for name,value in originals: setattr(act,name,value)
        report['source_unchanged']=all(Path(p).is_file() and sha(p)==h for p,h in source_paths.items())
        report['counts']=counts;report['wall_seconds']=time.monotonic()-started
        if not report['source_unchanged']:report['status']='SOURCE_CHANGED_BLOCKED'
        save(progress,counts);save(out/'run-review.json',report)
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path);parser.add_argument('--plan',required=True,type=Path)
    args=parser.parse_args();result=run(args.out,args.plan)
    raise SystemExit(0 if result['status']=='BAM_MEMORY_BOUNDARY_PASS_REVIEW_REQUIRED' else 2)
