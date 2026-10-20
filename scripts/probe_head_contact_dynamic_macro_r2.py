"""Bounded full-model diagnostic macro integration and rollback; zero PPO.

Two 2ms macros, each rejected once then retried: at most 256 RK4 microsteps.
BAM command/model friction is held within each macro. All original owners are
restored on rejection and at the end. Formal dt/integrator/material files stay
unchanged; temporary device RK4 options are diagnostic only. No Scene lifecycle.
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
import warnings
from head_contact_nominal_macro_state import nominal_state, geometry_row, validate_geometry
from head_contact_bounded_macro import BoundedRK4, diagnostic_rk_options, MICRO_DT, MICROSTEPS, MAX_STEPS
import numpy as np
import mujoco

from head_contact_data_commit_r3 import commit_data, describe
from head_contact_torch_state import TorchStateSnapshot
from head_contact_model_snapshot import snapshot_model_fields, best_effort_restore
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
    counts={'attempted':0,'completed':0,'CPU_kinematics_attempted':0,'CPU_kinematics_completed':0,'CPU_geom_distance_attempted':0,'CPU_geom_distance_completed':0,'CPU_forward_attempted':0,'CPU_forward_completed':0,
            'GPU_forward_attempted':0,'GPU_forward_completed':0,'setup_GPU_forward_completed':0,
            'endpoint_GPU_forward_attempted':0,'endpoint_GPU_forward_completed':0,'rk_stages_observed':0,
            'RK4_integrator_calls':0,'accepted_microsteps':0,'discarded_microsteps':0,'blocked_step_or_capture_requests':0,
            'blocked_render_context_requests':0,
            'bam_compute_calls':0,'delay_apply_calls':0,'logical_command_ticks':0,'commit_array_copy_calls':0,'model_array_copy_calls':0,'torch_tensor_restore_calls':0,'rollback_calls':0,'accepted_logical_commands':0,'PPO':0}
    progress=out/'progress.json';save(progress,counts)
    report={'status':'STARTED','PPO':0,'full_robot_steps':0,'cloud_hours':0,
            'formal_physics_adoption':False,'formal_task_migration':False,'stage11_complete':False,
            'zero_dynamics':False,'not_a_floor_or_full_robot_physics_acceptance':True,
            'revision':'bounded_dynamic_macro_r2_nominal_geometry','full_scene_sensor_lifecycle_validated':False,
            'full_dynamic_macro_rollback_validated':False,'canonical_completed':33647,'canonical_attempted':33648}
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

        with BoundedRK4(mujoco,mw,fw,wp,counts,lambda:save(progress,counts),started+600) as gate:
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
            # Reuse audited nominal reset geometry, without invoking Managers,
            # reset RNG/bookkeeping, sensors, or changing the compiled scene.
            from mjlab_microduck.robot import head_reference
            origins=scene.env_origins.detach().cpu().numpy()
            require(origins.shape==(1,3) and not origins.any(),'Expected one zero-origin diagnostic world')
            spawn=cfg.events['reset_double_balance'].params
            require(spawn['ball_radius']==.12 and spawn['root_height']==.125,'Existing lower-stack reset dimensions changed')
            nominal_qpos,nominal_ids,nominal_geometry=nominal_state(
                model,mujoco,robot.indexing.joint_q_adr.detach().cpu().numpy(),
                robot.data.default_joint_pos.detach().cpu().numpy(),head_reference,
                counts,lambda:save(progress,counts),lambda row:save(out/'nominal-CPU-initialization.json',row))
            sim.data.qpos[:] = torch.as_tensor(nominal_qpos,device='cuda:0',dtype=torch.float32)[None]
            sim.data.qvel.zero_();sim.data.ctrl.zero_();sim.data.time.zero_()
            sim.data.qacc.zero_();sim.data.qacc_warmstart.zero_()
            require(not sim.data.qfrc_applied.any() and not sim.data.xfrc_applied.any(),'Unexpected setup external wrench')
            torch.cuda.synchronize();wp.synchronize()
            require(all(getattr(sim.wp_model.callback,n) is None for n in ('passive','control','act_dyn','act_gain','act_bias','sensor','contactfilter')),'Unexpected native callback')
            sim.forward();wp.synchronize();torch.cuda.synchronize()
            d=sim.wp_data;wm=sim.wp_model
            setup_geometry=geometry_row(d.geom_xpos.numpy()[0],d.geom_xmat.numpy()[0],nominal_ids,model)
            setup_geometry.update(qpos=d.qpos.numpy()[0].tolist(),qvel=d.qvel.numpy()[0].tolist(),
                time_s=float(d.time.numpy()[0]),nacon=int(d.nacon.numpy()[0]),nefc=int(d.nefc.numpy()[0]))
            save(out/'nominal-CUDA-initialization.json',setup_geometry)
            validate_geometry(setup_geometry,nominal=True,expected_top=nominal_geometry['expected_top_position'])
            require(np.array_equal(d.qpos.numpy()[0],nominal_qpos.astype(np.float32))
                and not d.qvel.numpy().any() and float(d.time.numpy()[0])==0,'Nominal state upload differs')
            nc=int(d.nacon.numpy()[0]);nefc=int(d.nefc.numpy()[0])
            require(0<=nc<int(d.contact.dist.shape[0]) and 0<=nefc<int(d.njmax),'Setup contact/constraint capacity reached')
            contacts=d.contact.geom.numpy()[:nc];distances=d.contact.dist.numpy()[:nc]
            save(out/'nominal-CUDA-contacts.json',{'geom':contacts.tolist(),'dist':distances.tolist()})
            require(np.isfinite(distances).all() and np.isfinite(d.qfrc_constraint.numpy()).all(),'Nonfinite setup contact/force')
            require(not np.any(contacts==nominal_ids['geoms']['top']),'Nominal upper ball already in contact')
            report['nominal_initialization_verified']=True
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
            save(out/'scene-contract.json',{'device':device.name,'versions':versions,'nq':model.nq,'nv':model.nv,'nu':model.nu,'nbody':model.nbody,'ngeom':model.ngeom,'neq':model.neq,'nworld':1,'integrator':int(model.opt.integrator),'timestep':float(model.opt.timestep),'data_unique_arrays':len(arrays),'data_capacity_bytes':capacity,'actuator_names':list(act.target_names),'CPU_forward':1,'CPU_kinematics_budget':2,'CPU_geom_distance_budget':1,'GPU_forward_budget':1029,'microstep_budget':256,'microstep_dt_s':MICRO_DT,'macro_nominal_dt_s':.002,'macro_microsteps':64,'graph_capture':False,'warmup_integrations':0})
            from mjlab.utils.buffers import CircularBuffer, DelayBuffer
            fields = ('_prev_motor_torque','_dt','_base_kp','_friction_fields_checked',
                      '_delay_buffer','vin_tensor','vin_drop_gain','kp_scale','kd_scale',
                      'default_kp_scale','default_kd_scale','friction_scale','default_friction_scale',
                      'diagnostics_enabled','last_diagnostics')
            roots = [('actuator',act,fields),
                     ('firmware',act._bam_model.actuator,('kp','vin')),
                     ('targets',robot.data,('joint_pos_target','joint_vel_target','joint_effort_target'))]
            model_fields = {name:getattr(wm,name) for name in ('dof_frictionloss','dof_damping')}
            save(out/'model-layout-live-before.json',dict(zip(
                ('descriptors','arrays_omitted','owners','regions'),
                (describe(model_fields,wp.array)[0],None,describe(model_fields,wp.array)[2],describe(model_fields,wp.array)[3]))))
            initial_model=snapshot_model_fields(model_fields,wp)
            save(out/'model-layout-snapshot-cold.json',{'descriptors':describe(initial_model,wp.array)[0],
                                                       'owners':describe(initial_model,wp.array)[2]})
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
            # Reject any initial snapshot incompatibility before the first BAM call.
            initial_snapshot.preflight()
            require(describe(d,wp.array)[0]==describe(baseline_data,wp.array)[0],'Initial Data snapshot layout mismatch')
            require(describe(model_fields,wp.array)[0]==describe(initial_model,wp.array)[0],'Initial Model snapshot layout mismatch')
            restore_needed=True
            native_delay=act.apply_delay;native_compute=act.compute
            command_arrays=[];observed={}
            def tracked_delay(command):
                require(gate.phase is None and counts['delay_apply_calls']<4,'Delay budget exhausted')
                counts['delay_apply_calls']+=1
                return native_delay(command)
            def tracked_compute(command):
                require(gate.phase is None and counts['bam_compute_calls']<4,'BAM budget exhausted')
                counts['bam_compute_calls']+=1
                observed['delayed']=command.position_target.detach().cpu().numpy().copy()
                torque=native_compute(command)
                require(bool(torch.isfinite(torque).all()),'Nonfinite BAM output')
                observed['torque']=torque.detach().cpu().numpy().copy()
                return torque
            originals=[('apply_delay',native_delay),('compute',native_compute)]
            act.apply_delay=tracked_delay;act.compute=tracked_compute

            def state():
                require(all(getattr(wm,n) is a for n,a in model_fields.items()),'Live Model owner was replaced')
                return {'Data':bridge.fingerprint(d),'Model':bridge.fingerprint(wm),
                        'external':external_state(act,robot,torch)}
            def restore(saved_data,saved_model,saved_torch):
                # Validate all selected storage before any rollback copy. Device
                # copy errors can still be partial: stop, record, never continue.
                saved_torch.preflight()
                require(describe(d,wp.array)[0]==describe(saved_data,wp.array)[0], 'Data restore layout mismatch')
                live_layout=describe(model_fields,wp.array)
                saved_layout=describe(saved_model,wp.array)
                save(out/'latest-model-restore-preflight.json',{'live_descriptors':live_layout[0],
                    'snapshot_descriptors':saved_layout[0],'live_owners':live_layout[2],
                    'snapshot_owners':saved_layout[2]})
                require(live_layout[0]==saved_layout[0], 'Model restore layout mismatch')
                counts['commit_array_copy_calls']+=commit_data(d,saved_data,wp)
                counts['model_array_copy_calls']+=commit_data(model_fields,saved_model,wp)
                counts['torch_tensor_restore_calls']+=saved_torch.restore()
                wp.synchronize();torch.cuda.synchronize()
                counts['rollback_calls']+=1;save(progress,counts)
            # Resolve only actual names/types in this full model. Do not borrow
            # sphere-fixture geom IDs or change pair/material parameters.
            top_ids=[i for i in range(model.ngeom) if model.geom(i).name.split('/')[-1]=='top_ball_sphere']
            floor_ids=[i for i in range(model.ngeom) if int(model.geom_type[i])==int(mujoco.mjtGeom.mjGEOM_PLANE)]
            require(len(top_ids)==len(floor_ids)==1,'Unexpected top sphere/floor identity')
            top_id,floor_id=top_ids[0],floor_ids[0]
            require(int(model.geom_type[top_id])==int(mujoco.mjtGeom.mjGEOM_SPHERE),'Upper geom not a sphere')
            top_radius=float(model.geom_size[top_id,0])
            save(out/'diagnostic-integration-contract.json',{
                'CPU_model_integrator':int(model.opt.integrator),'CPU_model_timestep':float(model.opt.timestep),
                'temporary_device_integrator':'RK4','temporary_device_dt_s':MICRO_DT,
                'nominal_macro_dt_s':.002,'microsteps_per_macro':64,'max_microstep_calls':256,
                'max_RK_stage_forwards':1024,'max_endpoint_forwards':4,'setup_GPU_forwards':1,
                'BAM_dt_s':float(act._dt),'BAM_updates_per_attempted_macro':1,
                'top_geom':{'id':top_id,'name':model.geom(top_id).name,'radius':top_radius},
                'floor_geom':{'id':floor_id,'name':model.geom(floor_id).name},
                'event_subdivision':False,'material_changes':False,'formal_adoption':False,
                'native_clock_overwrite':False,'Scene_update_or_sensor_lifecycle':False})
            peaks={'all_contact_overlap_m':0.,'upper_ball_floor_overlap_m':0.,'max_abs_qvel':0.,
                   'max_abs_qacc':0.,'max_velocity_RK_residual':0.,'max_native_clock_minus_dt_sum_s':0.}
            report['diagnostic_peaks']=peaks
            report['diagnostic_peak_samples_observed']=0
            stage_log=out/'stage-log.jsonl'
            step_log=out/'microstep-log.jsonl'
            def primary():
                return {'qpos':d.qpos.numpy()[0].copy(),'qvel':d.qvel.numpy()[0].copy(),
                        'native_time_s':float(d.time.numpy()[0])}
            def held_inputs():
                return {n:getattr(d,n).numpy().tobytes() for n in ('ctrl','qfrc_applied','xfrc_applied')} | {
                    n:getattr(wm,n).numpy().tobytes() for n in ('dof_frictionloss','dof_damping')}
            def solved_row():
                p=primary();acc=d.qacc.numpy()[0].copy()
                require(np.isfinite(p['qpos']).all() and np.isfinite(p['qvel']).all() and np.isfinite(acc).all(),'Nonfinite dynamic state')
                require(np.isfinite(d.qfrc_constraint.numpy()).all(),'Nonfinite constraint force')
                nc=int(d.nacon.numpy()[0]);nefc=int(d.nefc.numpy()[0])
                require(0<=nc<int(d.contact.dist.shape[0]) and 0<=nefc<int(d.njmax),'Contact/constraint capacity reached')
                distances=d.contact.dist.numpy()[:nc].copy()
                require(np.isfinite(distances).all(),'Nonfinite contact distance')
                normal=d.geom_xmat.numpy()[0,floor_id].reshape(3,3)[:,2]
                require(np.max(np.abs(normal-[0,0,1]))<1e-6,'Unexpected floor orientation')
                geompos=d.geom_xpos.numpy()[0]
                separation=float(geompos[top_id,2]-geompos[floor_id,2])-top_radius
                peaks['all_contact_overlap_m']=max(peaks['all_contact_overlap_m'],float(max(0,-distances.min())) if nc else 0.)
                peaks['upper_ball_floor_overlap_m']=max(peaks['upper_ball_floor_overlap_m'],max(0.,-separation))
                peaks['max_abs_qvel']=max(peaks['max_abs_qvel'],float(np.max(np.abs(p['qvel']))))
                peaks['max_abs_qacc']=max(peaks['max_abs_qacc'],float(np.max(np.abs(acc))))
                row={**p,'qacc':acc,'nacon':nc,'nefc':nefc,
                        'contact_geom':d.contact.geom.numpy()[:nc].copy(),'contact_dist':distances,
                        'upper_ball_floor_separation_m':separation,
                        'top_geom_position':geompos[top_id].copy(),'floor_geom_position':geompos[floor_id].copy(),
                        'floor_normal':normal.copy()}
                report['diagnostic_peak_samples_observed']+=1
                if separation<-.001:
                    log_row(out/'guard-failure.jsonl',{'phase':gate.phase,'rk_stages_observed':counts['rk_stages_observed'],
                        'attempted':counts['attempted'],'completed':counts['completed'],**row})
                require(separation>=-.001,'Upper-ball floor penetration exceeded retained 1mm guard')
                return row
            def log_row(path,row):
                with path.open('a')as f:
                    f.write(json.dumps(row,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item(),allow_nan=False)+'\n')
            def integrate_macro(logical_index,attempt):
                held=held_inputs();external=external_state(act,robot,torch)
                calls=tuple(counts[k] for k in ('bam_compute_calls','delay_apply_calls','logical_command_ticks'))
                start=primary();local=[]
                require(float(model.opt.timestep)==.002 and int(model.opt.integrator)==3 and act._dt==.002,'Formal/BAM contract changed')
                with diagnostic_rk_options(wm,wp):
                    for index in range(MICROSTEPS):
                        before=primary();stages=[]
                        def observe(_m,_d,stage):
                            require(held_inputs()==held,'Held actuator inputs changed within an RK stage')
                            require(tuple(counts[k] for k in ('bam_compute_calls','delay_apply_calls','logical_command_ticks'))==calls,'BAM advanced within an RK stage')
                            row=solved_row();stages.append(row)
                            log_row(stage_log,{'logical_index':logical_index,'attempt':attempt,'micro_index':index,
                                'stage':stage,'nominal_stage_time_s':(logical_index*MICROSTEPS+index+[0,.5,.5,1][stage])*MICRO_DT,**row})
                        with warnings.catch_warnings(record=True)as caught:
                            warnings.simplefilter('always');gate.microstep(wm,d,observe)
                        require(not caught,'Warning during native RK4 integration')
                        after=primary()
                        require(np.isfinite(after['qpos']).all() and np.isfinite(after['qvel']).all(),'Nonfinite integrated state')
                        require(np.array_equal(stages[0]['qpos'],before['qpos']) and np.array_equal(stages[0]['qvel'],before['qvel']),'RK initial state mismatch')
                        residual=after['qvel'].astype(float)-before['qvel']-MICRO_DT*(np.array([1,2,2,1])/6 @ np.array([r['qacc'] for r in stages],float))
                        bound=8*np.finfo(np.float32).eps*(1+max(float(np.max(abs(before['qvel']))),float(np.max(abs(after['qvel']))),float(np.max(abs(MICRO_DT*np.array([r['qacc'] for r in stages]))))))
                        require(float(np.max(abs(residual)))<=bound,'RK weighted velocity increment mismatch')
                        expected=float(np.float32(np.float32(before['native_time_s'])+np.float32(MICRO_DT)))
                        require(after['native_time_s']==expected,'Native float32 clock update mismatch')
                        dt_sum=(logical_index*MICROSTEPS+index+1)*MICRO_DT
                        clock_error=after['native_time_s']-dt_sum
                        peaks['max_velocity_RK_residual']=max(peaks['max_velocity_RK_residual'],float(np.max(abs(residual))))
                        peaks['max_native_clock_minus_dt_sum_s']=max(peaks['max_native_clock_minus_dt_sum_s'],abs(clock_error))
                        require(abs(clock_error)<1e-7,'Native clock drift exceeded bounded-window 0.1us diagnostic limit')
                        row={'logical_index':logical_index,'attempt':attempt,'micro_index':index,
                            'actual_dt_sum_s':dt_sum,'native_clock_error_s':clock_error,'velocity_residual_max':float(np.max(abs(residual))),
                            'velocity_residual_bound':bound,**after}
                        local.append(row);log_row(step_log,row)
                    # Refresh derived kinematics for the exact integrated endpoint.
                    # BAM/Scene update is not called here or within microsteps.
                    gate.refresh(wm,d);endpoint=solved_row()
                require(held_inputs()==held and external_state(act,robot,torch)==external,'Held external state changed during dynamics')
                require(not np.array_equal(start['qpos'],endpoint['qpos']) or not np.array_equal(start['qvel'],endpoint['qvel']),'Dynamic window did not move')
                require(endpoint['native_time_s']>start['native_time_s'],'Physical clock did not advance')
                log_row(out/'macro-endpoints.jsonl',{'logical_index':logical_index,'attempt':attempt,**endpoint})
                return local
            base=robot.data.joint_pos.detach().clone()
            pair_rows=[]
            for logical_index in range(2):
                require(time.monotonic()-started<600,'Probe wall budget exceeded')
                if logical_index==0:
                    saved_data,saved_model,saved_torch=baseline_data,initial_model,initial_snapshot
                else:
                    require(act._delay_buffer._buffer._buffer is not None,'Warm queue not allocated')
                    saved_data=events.clone_data(d,wp)
                    saved_model=snapshot_model_fields(model_fields,wp)
                    save(out/'model-layout-snapshot-warm.json',{'descriptors':describe(saved_model,wp.array)[0],
                                                               'owners':describe(saved_model,wp.array)[2]})
                    saved_torch=TorchStateSnapshot(torch,roots,(CircularBuffer,DelayBuffer),'cuda:0')
                saved_state=state();saved_owners=saved_torch.owner_manifest()
                save(out/f'pair-{logical_index}-before.json',saved_state)
                save(out/f'pair-{logical_index}-torch-owners-before.json',saved_owners)
                first=None
                for attempt in range(2):
                    require(time.monotonic()-started<600,'Probe wall budget exceeded')
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
                    require(all(bridge.fingerprint(d)[n]==saved_state['Data'][n] for n in ('qpos','qvel','time')),'BAM command changed physical state before integration')
                    integrate_macro(logical_index,attempt)
                    current=state()
                    require({k for k in saved_state['Model'] if saved_state['Model'][k]!=current['Model'][k]}<={'dof_frictionloss','dof_damping'},'Unexpected Model mutation')
                    item={'cpu_rng_witness':cpu_witness,'cuda_rng_witness':cuda_witness,
                          'target':target.detach().cpu().numpy().copy(),'delayed':observed['delayed'],
                          'torque':observed['torque'],'lag':act._delay_buffer.current_lags.detach().cpu().numpy().copy(),
                          'qpos':d.qpos.numpy().copy(),'qvel':d.qvel.numpy().copy(),
                          'native_time':d.time.numpy().copy(),'qacc':d.qacc.numpy().copy()}
                    command_arrays.append(item)
                    save(out/f'pair-{logical_index}-attempt-{attempt}.json',current)
                    if attempt==0:
                        first=current
                        counts['discarded_microsteps']+=MICROSTEPS
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
                        counts['accepted_microsteps']+=MICROSTEPS
                        save(progress,counts)
                pair_rows.append({'logical_index':logical_index,'cold_queue':logical_index==0,
                                  'attempts':2,'rollback_verified':True,'retry_bitwise_equal':True,
                                  'queue_count_after_retry':logical_index+1,
                                  'selected_torch_snapshot_bytes':saved_torch.tensor_bytes,
                                  'accepted_microsteps':MICROSTEPS,'discarded_microsteps':MICROSTEPS,
                                  'actual_dt_sum_end_s':(logical_index+1)*MICROSTEPS*MICRO_DT,
                                  'native_device_time_end_s':float(d.time.numpy()[0])})
            save(out/'pair-review.json',pair_rows)
            np.savez_compressed(out/'command-retries.npz',**{k:np.stack([x[k] for x in command_arrays]) for k in command_arrays[0]})
            report['native_time_before_final_restore_s']=float(d.time.numpy()[0])
            report['accepted_physical_time_before_final_restore_s']=counts['accepted_microsteps']*MICRO_DT
            report['nominal_physical_time_before_final_restore_s']=.004
            restore(baseline_data,initial_model,initial_snapshot)
            final=state();save(out/'final-restored-state.json',final)
            save(out/'model-layout-live-after.json',{'descriptors':describe(model_fields,wp.array)[0],
                                                    'owners':describe(model_fields,wp.array)[2]})
            save(out/'final-torch-owners.json',initial_snapshot.owner_manifest())
            require(final=={'Data':initial_data_fp,'Model':initial_model_fp,'external':initial_external},'Final initial state not restored')
            require(describe(d,wp.array)[2]==owners,'Original Warp Data owners changed')
            save(out/'data-layout-after.json',{'descriptors':describe(d,wp.array)[0],'owners':describe(d,wp.array)[2]})
            require(act._delay_buffer._buffer._buffer is None,'Initial lazy queue allocation not restored')
            require(counts['CPU_kinematics_attempted']==counts['CPU_kinematics_completed']==2
                    and counts['CPU_geom_distance_attempted']==counts['CPU_geom_distance_completed']==1,'Nominal setup budget differs')
            require(counts['CPU_forward_completed']==1 and counts['setup_GPU_forward_completed']==1
                    and counts['endpoint_GPU_forward_completed']==4 and counts['GPU_forward_completed']==1029,'Forward count changed')
            require(counts['attempted']==counts['completed']==MAX_STEPS and counts['rk_stages_observed']==4*MAX_STEPS
                    and counts['RK4_integrator_calls']==MAX_STEPS,'Incomplete microstep/RK count')
            require(counts['accepted_microsteps']==counts['discarded_microsteps']==128,'Acceptance/discard accounting mismatch')
            require(counts['bam_compute_calls']==counts['delay_apply_calls']==counts['logical_command_ticks']==4,'Command budget differs')
            require(counts['rollback_calls']==3 and counts['accepted_logical_commands']==2,'Rollback/retry count differs')
            require(counts['blocked_step_or_capture_requests']==counts['blocked_render_context_requests']==0,'Forbidden call requested')
            require(float(model.opt.timestep)==.002 and int(model.opt.integrator)==3,'Formal integrator changed')
            restore_needed=False
            report.update(status='DYNAMIC_MACRO_PASS_REVIEW_REQUIRED',
                          cold_and_warm_rejection_retry_verified=True,bounded_dynamic_macro_rollback_verified=True,
                          fixed_microstep_diagnostic_only=True,adaptive_contact_algorithm_validated=False,
                          selected_Data_Model_BAM_Torch_RNG_restored=True,
                          initial_lazy_queue_restored=True,physical_elapsed_s=0.,
                          full_dynamic_macro_rollback_validated=False,torch_version=torch.__version__)
    except Exception as exc:
        report['status']='DYNAMIC_MACRO_STOPPED_REVIEW_REQUIRED';report['error']=repr(exc)
        (out/'error.txt').write_text(traceback.format_exc())
    finally:
        if restore_needed and sim is not None and baseline_data is not None and wp is not None:
            # A failed component must not prevent other independent state from
            # being restored. Never resume the diagnostic after any failure.
            def cleanup_data():
                n=commit_data(sim.wp_data,baseline_data,wp)
                counts['commit_array_copy_calls']+=n
                require(bridge.fingerprint(sim.wp_data)==initial_data_fp,'Data cleanup fingerprint mismatch')
                return {'array_copies':n,'fingerprint_matches':True}
            def cleanup_model():
                n=commit_data(model_fields,initial_model,wp)
                counts['model_array_copy_calls']+=n
                require(bridge.fingerprint(sim.wp_model)==initial_model_fp,'Model cleanup fingerprint mismatch')
                return {'array_copies':n,'full_Model_matches':True}
            def cleanup_torch():
                n=initial_snapshot.restore();counts['torch_tensor_restore_calls']+=n
                torch.cuda.synchronize()
                require(external_state(act,robot,torch)==initial_external,'Torch/BAM/RNG cleanup fingerprint mismatch')
                return {'tensor_copies':n,'external_state_matches':True}
            actions=[('Data',cleanup_data)]
            if initial_model is not None:actions.append(('Model',cleanup_model))
            if initial_snapshot is not None:actions.append(('Torch_BAM_RNG',cleanup_torch))
            report['failure_cleanup_components']=best_effort_restore(actions)
            try:
                wp.synchronize();torch.cuda.synchronize()
                actual={'Data':bridge.fingerprint(sim.wp_data),'Model':bridge.fingerprint(sim.wp_model),
                        'external':external_state(act,robot,torch)}
                save(out/'failure-cleanup-final-state.json',actual)
                report['failure_cleanup_selected_state_restored']=actual=={'Data':initial_data_fp,'Model':initial_model_fp,'external':initial_external}
            except Exception as exc:report['failure_cleanup_verification_error']=repr(exc)
        if act is not None:
            for name,value in originals:setattr(act,name,value)
        report['source_unchanged']=all(Path(p).is_file() and sha(p)==h for p,h in source_paths.items())
        report['full_robot_steps']=counts['completed']
        report['uncommitted_completed_microsteps']=counts['completed']-counts['accepted_microsteps']-counts['discarded_microsteps']
        report['canonical_completed_lower_bound']=33647+counts['completed']
        report['canonical_attempted_upper_bound']=33648+counts['attempted']
        report['counts']=counts;report['wall_seconds']=time.monotonic()-started
        if not report['source_unchanged']:report['status']='SOURCE_CHANGED_BLOCKED'
        save(progress,counts);save(out/'run-review.json',report)
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path);parser.add_argument('--plan',required=True,type=Path)
    args=parser.parse_args();result=run(args.out,args.plan)
    raise SystemExit(0 if result['status']=='DYNAMIC_MACRO_PASS_REVIEW_REQUIRED' else 2)
