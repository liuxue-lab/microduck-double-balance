"""Independent cold/warm BAM command-boundary rollback: zero dynamics, zero PPO.

One constructor CPU forward and one frozen-state GPU forward are budgeted.
Four command attempts form two reject/retry pairs; no physical time elapses.
Full Data, mutable Model arrays, selected Torch/BAM state and RNG are restored.
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

from head_contact_data_commit_r3 import commit_data, describe
from head_contact_torch_state import TorchStateSnapshot
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
            'bam_compute_calls':0,'delay_apply_calls':0,'logical_command_ticks':0,'commit_array_copy_calls':0,'model_array_copy_calls':0,'torch_tensor_restore_calls':0,'rollback_calls':0,'accepted_logical_commands':0,'PPO':0}
    progress=out/'progress.json';save(progress,counts)
    report={'status':'STARTED','PPO':0,'full_robot_steps':0,'cloud_hours':0,
            'formal_physics_adoption':False,'formal_task_migration':False,'stage11_complete':False,
            'zero_dynamics':True,'not_a_floor_or_full_robot_physics_acceptance':True,
            'revision':'bam_macro_state_v1','full_scene_sensor_lifecycle_validated':False,
            'full_dynamic_macro_rollback_validated':False,'canonical_steps':33647}
    source_paths={};sim=act=baseline_data=None;restore_needed=False;wp=torch=None
    initial_snapshot=None;initial_model=None;model_fields=None;initial_data_fp=None
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
        require(torch.__version__.split('+')[0]=='2.9.1','Unexpected Torch core version')
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
            save(out/'array-storage-inventory.json', {
                path: {'size':int(a.size), 'shape':list(a.shape), 'strides':list(a.strides),
                       'dtype':str(a.dtype), 'device':str(a.device),
                       'pointer':owners[path][1], 'capacity':int(a.capacity)}
                for path,a in arrays.items()})
            save(out/'data-layout-before.json',{'descriptors':structure,'owners':owners})
            capacity=sum(int(a.capacity) for a in arrays.values())
            require(capacity<256*1024*1024,'Unexpected full Data footprint')
            require(float(d.time.numpy()[0])==0,'Setup advanced time')
            before_model=bridge.fingerprint(wm)
            initial_qpos=d.qpos.numpy().copy();initial_qvel=d.qvel.numpy().copy()
            np.savez_compressed(out/'frozen-state.npz',qpos=initial_qpos,qvel=initial_qvel,time=d.time.numpy(),qacc=d.qacc.numpy())
            save(out/'scene-contract.json',{'device':device.name,'versions':versions,'nq':model.nq,'nv':model.nv,'nu':model.nu,'nbody':model.nbody,'ngeom':model.ngeom,'neq':model.neq,'nworld':1,'integrator':int(model.opt.integrator),'timestep':float(model.opt.timestep),'data_unique_arrays':len(arrays),'data_capacity_bytes':capacity,'actuator_names':list(act.target_names),'CPU_forward':1,'GPU_forward_budget':1,'graph_capture':False,'warmup_integrations':0})
            from mjlab.utils.buffers import CircularBuffer, DelayBuffer
            fields = ('_prev_motor_torque','_dt','_base_kp','_friction_fields_checked',
                      '_delay_buffer','vin_tensor','vin_drop_gain','kp_scale','kd_scale',
                      'default_kp_scale','default_kd_scale','friction_scale','default_friction_scale',
                      'diagnostics_enabled','last_diagnostics')
            roots = [('actuator',act,fields),
                     ('firmware',act._bam_model.actuator,('kp','vin')),
                     ('targets',robot.data,('joint_pos_target','joint_vel_target','joint_effort_target'))]
            model_fields = {name:getattr(wm,name) for name in ('dof_frictionloss','dof_damping')}
            initial_model=events.clone_data(model_fields,wp)
            baseline_data=events.clone_data(d,wp)
            initial_snapshot=TorchStateSnapshot(torch,roots,(CircularBuffer,DelayBuffer),'cuda:0')
            initial_data_fp=bridge.fingerprint(d)
            initial_external=external_state(act,robot,torch)
            initial_model_fp=bridge.fingerprint(wm)
            require(act._delay_buffer._buffer._buffer is None,'Cold queue already allocated')
            require(int(act._delay_buffer._step_count[0])==0,'Cold queue already advanced')
            save(out/'initial-external.json',initial_external)
            save(out/'initial-model.json',initial_model_fp)
            save(out/'initial-data.json',initial_data_fp)
            save(out/'initial-torch-owners.json',initial_snapshot.owner_manifest())
            restore_needed=True
            native_delay=act.apply_delay;native_compute=act.compute
            command_arrays=[];observed={}
            def tracked_delay(command):
                require(counts['delay_apply_calls']<4,'Delay budget exhausted')
                counts['delay_apply_calls']+=1
                return native_delay(command)
            def tracked_compute(command):
                require(counts['bam_compute_calls']<4,'BAM budget exhausted')
                counts['bam_compute_calls']+=1
                observed['delayed']=command.position_target.detach().cpu().numpy().copy()
                torque=native_compute(command)
                require(bool(torch.isfinite(torque).all()),'Nonfinite BAM output')
                observed['torque']=torque.detach().cpu().numpy().copy()
                return torque
            originals=[('apply_delay',native_delay),('compute',native_compute)]
            act.apply_delay=tracked_delay;act.compute=tracked_compute

            def state():
                return {'Data':bridge.fingerprint(d),'Model':bridge.fingerprint(wm),
                        'external':external_state(act,robot,torch)}
            def restore(saved_data,saved_model,saved_torch):
                # Validate all selected storage before any rollback copy. Device
                # copy errors can still be partial: stop, record, never continue.
                saved_torch.preflight()
                require(describe(d,wp.array)[0]==describe(saved_data,wp.array)[0], 'Data restore layout mismatch')
                require(describe(model_fields,wp.array)[0]==describe(saved_model,wp.array)[0], 'Model restore layout mismatch')
                counts['commit_array_copy_calls']+=commit_data(d,saved_data,wp)
                counts['model_array_copy_calls']+=commit_data(model_fields,saved_model,wp)
                counts['torch_tensor_restore_calls']+=saved_torch.restore()
                wp.synchronize();torch.cuda.synchronize()
                counts['rollback_calls']+=1;save(progress,counts)
            base=robot.data.joint_pos.detach().clone()
            pair_rows=[]
            for logical_index in range(2):
                require(time.monotonic()-started<300,'Probe wall budget exceeded')
                if logical_index==0:
                    saved_data,saved_model,saved_torch=baseline_data,initial_model,initial_snapshot
                else:
                    require(act._delay_buffer._buffer._buffer is not None,'Warm queue not allocated')
                    saved_data=events.clone_data(d,wp)
                    saved_model=events.clone_data(model_fields,wp)
                    saved_torch=TorchStateSnapshot(torch,roots,(CircularBuffer,DelayBuffer),'cuda:0')
                saved_state=state();saved_owners=saved_torch.owner_manifest()
                save(out/f'pair-{logical_index}-before.json',saved_state)
                save(out/f'pair-{logical_index}-torch-owners-before.json',saved_owners)
                first=None
                for attempt in range(2):
                    require(time.monotonic()-started<300,'Probe wall budget exceeded')
                    # Deliberate tiny CPU/CUDA RNG witnesses make restoration of
                    # both global RNG streams observable, beyond BAM's own lag draw.
                    cpu_witness=torch.rand(3,device='cpu').numpy().copy()
                    cuda_witness=torch.rand(3,device='cuda:0').cpu().numpy().copy()
                    target=base+(logical_index+1)*2**-12
                    robot.set_joint_position_target(target)
                    scene.write_data_to_sim();wp.synchronize();torch.cuda.synchronize()
                    counts['logical_command_ticks']+=1;save(progress,counts)
                    require(int(act._delay_buffer._step_count[0])==logical_index+1,'Repeated command advanced delay twice')
                    require(int(act._delay_buffer._buffer._num_pushes[0])==logical_index+1,'History append count differs')
                    require(np.array_equal(d.qpos.numpy(),initial_qpos) and np.array_equal(d.qvel.numpy(),initial_qvel) and float(d.time.numpy()[0])==0,'Command trial changed physical state')
                    current=state()
                    require({k for k in saved_state['Model'] if saved_state['Model'][k]!=current['Model'][k]}<={'dof_frictionloss','dof_damping'},'Unexpected Model mutation')
                    item={'cpu_rng_witness':cpu_witness,'cuda_rng_witness':cuda_witness,
                          'target':target.detach().cpu().numpy().copy(),'delayed':observed['delayed'],
                          'torque':observed['torque'],'lag':act._delay_buffer.current_lags.detach().cpu().numpy().copy()}
                    command_arrays.append(item)
                    save(out/f'pair-{logical_index}-attempt-{attempt}.json',current)
                    if attempt==0:
                        first=current
                        restore(saved_data,saved_model,saved_torch)
                        restored=state()
                        save(out/f'pair-{logical_index}-restored.json',restored)
                        save(out/f'pair-{logical_index}-torch-owners-restored.json',saved_torch.owner_manifest())
                        require(restored==saved_state,'Rejected command did not restore full selected state')
                        require(saved_torch.owner_manifest()==saved_owners,'Torch owners not restored')
                        require(int(act._delay_buffer._step_count[0])==logical_index,'Rejected delay counter not restored')
                    else:
                        require(current==first,'Retry state differs despite rollback')
                        require(all(np.array_equal(item[k],command_arrays[-2][k]) for k in item),'Retry arrays/RNG witnesses differ')
                        counts['accepted_logical_commands']+=1
                pair_rows.append({'logical_index':logical_index,'cold_queue':logical_index==0,
                                  'attempts':2,'rollback_verified':True,'retry_bitwise_equal':True,
                                  'queue_count_after_retry':logical_index+1,
                                  'selected_torch_snapshot_bytes':saved_torch.tensor_bytes})
            save(out/'pair-review.json',pair_rows)
            np.savez_compressed(out/'command-retries.npz',**{k:np.stack([x[k] for x in command_arrays]) for k in command_arrays[0]})
            restore(baseline_data,initial_model,initial_snapshot)
            final=state();save(out/'final-restored-state.json',final)
            save(out/'final-torch-owners.json',initial_snapshot.owner_manifest())
            require(final=={'Data':initial_data_fp,'Model':initial_model_fp,'external':initial_external},'Final initial state not restored')
            require(describe(d,wp.array)[2]==owners,'Original Warp Data owners changed')
            save(out/'data-layout-after.json',{'descriptors':describe(d,wp.array)[0],'owners':describe(d,wp.array)[2]})
            require(act._delay_buffer._buffer._buffer is None,'Initial lazy queue allocation not restored')
            require(counts['CPU_forward_completed']==counts['GPU_forward_completed']==1,'Forward budget changed')
            require(counts['bam_compute_calls']==counts['delay_apply_calls']==counts['logical_command_ticks']==4,'Command budget differs')
            require(counts['rollback_calls']==3 and counts['accepted_logical_commands']==2,'Rollback/retry count differs')
            require(counts['blocked_step_or_capture_requests']==counts['blocked_render_context_requests']==0,'Forbidden call requested')
            require(float(model.opt.timestep)==.002 and int(model.opt.integrator)==3,'Formal integrator changed')
            restore_needed=False
            report.update(status='BAM_MACRO_STATE_PASS_REVIEW_REQUIRED',
                          cold_and_warm_rejection_retry_verified=True,
                          selected_Data_Model_BAM_Torch_RNG_restored=True,
                          initial_lazy_queue_restored=True,physical_elapsed_s=0.,
                          full_dynamic_macro_rollback_validated=False,torch_version=torch.__version__)
    except Exception as exc:
        report['status']='BAM_MACRO_STATE_STOPPED_REVIEW_REQUIRED';report['error']=repr(exc)
        (out/'error.txt').write_text(traceback.format_exc())
    finally:
        if restore_needed and sim is not None and baseline_data is not None and wp is not None:
            try:
                counts['commit_array_copy_calls']+=commit_data(sim.wp_data,baseline_data,wp)
                if initial_model is not None:
                    counts['model_array_copy_calls']+=commit_data(model_fields,initial_model,wp)
                if initial_snapshot is not None:
                    counts['torch_tensor_restore_calls']+=initial_snapshot.restore()
                wp.synchronize();torch.cuda.synchronize()
                report['failure_cleanup_Data_restored']=bridge.fingerprint(sim.wp_data)==initial_data_fp
                report['failure_cleanup_selected_state_restored']=state()=={'Data':initial_data_fp,'Model':initial_model_fp,'external':initial_external}
            except Exception as exc:report['failure_cleanup_error']=repr(exc)
        if act is not None:
            for name,value in originals:setattr(act,name,value)
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
    raise SystemExit(0 if result['status']=='BAM_MACRO_STATE_PASS_REVIEW_REQUIRED' else 2)
