"""G10 expanded native device blocks and post-accepted-macro recovery."""
from pathlib import Path
import argparse
import os,uuid
import copy
import hashlib
import importlib.metadata
import json
import sys
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parent))
from lifecycle_support import require
from expanded_segments import ExpandedGate as Gate,ExpandedSegments as Segments
from resource_lifetime import Resources
from portable_state import PortableState,write as write_portable,read as read_portable,exact_arrays
from joint_protocol import WorkerMailbox,verify_publication_receipt
from report_io import save
from sensor_state_snapshot import DataSnapshot,OwnerSnapshot,digest,sense_and_invalidate
from batch_reset_contract import check_reset,contact_rows
from batch_scene_contract import exact,owner_reset,append_contract,post_isolation
from manager_contract import first_observation_contract
from isolation_contract import immediate as immediate_contract,isolated,expiry
from isolation_policy import StreamBank,canonical_ids
from transaction_contract import compare_records
from scene_evidence import owner_state,model_state,array_meta,save_arrays,rng_raw,outputs


def verify_sources(repo,plan):
    paths={repo/n:h for n,h in plan['project_source_hashes'].items()}
    for distribution,entries in plan['installed_source_hashes'].items():
        dist=importlib.metadata.distribution(distribution)
        paths.update({Path(dist.locate_file(n)):h for n,h in entries.items()})
    for p,h in paths.items():require(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==h,'Pinned source differs: '+str(p))
    for n,v in plan['versions'].items():require(importlib.metadata.version(n)==v,'Version differs: '+n)
    for row in json.loads(Path(__file__).with_name('integration-source-pins.json').read_bytes())+[plan['new_source_pin']]:
        path=repo/row['relative_path'] if row['distribution']=='project' else Path(importlib.metadata.distribution(row['distribution']).locate_file(row['relative_path']))
        require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],'Additional source differs: '+str(path))
        paths[path]=row['sha256']
    require(len(paths)==219,'Source roster differs')
    return paths


def run(repo,out,plan,logical_world,parent_pid,role,restore_from):
    started=time.monotonic();caps=dict(plan['owner_limits'])
    macro_total=3 if role=='producer' else 1
    native_total=608 if role=='producer' else 128
    forward_total=2443 if role=='producer' else 516
    caps['integrations']=caps['RK4_integrator']=caps['position_write']=native_total
    caps['GPU_forward']=caps['GPU_solver']=caps['SolverContext_create']=forward_total
    identity=dict(logical_world=logical_world,pid=os.getpid(),owner_id=uuid.uuid4().hex,native_worlds=1)
    require(role in ('producer','consumer'),'Unknown process role')
    counts=dict(integrations_attempted=0,integrations_completed=0,PPO=0,cloud=0)
    report=dict(status='SINGLE_WORLD_NATIVE_OWNER_STARTED',counts=counts,canonical_completed_before=plan['canonical_completed_before'],
        canonical_attempted_before=plan['canonical_attempted_before'],G08='CLOSED_SCOPED',G09='CLOSED_SCOPED',stage11_complete=False,
        scope=plan['scope'],physical_rollout=False,nine_manager_batch_lifecycle_validated=False,
        opaque_native_handles_validated=False,CUDA_graphs_validated=False)
    gate=None;contact_order=None;reductions=None;branch_driver=None;segments=None;resources=None
    sim=None;ctx=None;completed=[]
    def persist():save(out/'progress.json',dict(elapsed_s=time.monotonic()-started,**counts))
    def phase_progress(completed,label):
        save(out/'phase-progress.json',dict(completed_units=completed,total_units=macro_total+2,label=label,integrations_at_boundary=counts.get('integrations_completed',0)))
    phase_progress(0,'初始化与编译')
    def milestone(label,detail):
        with (out/'milestones.jsonl').open('a') as f:f.write(json.dumps(dict(label=label,detail=detail,integrations=counts.get('integrations_completed',0)),ensure_ascii=False)+'\n')
    try:
        require(plan['protocol']=='g10-native-blocks-endpoint-recovery-v3' and logical_world in plan['execution_owners']==[0,1] and caps['integrations']==native_total and caps['PPO']==caps['cloud']==0,'Declared joint limits differ')
        from budget_contract import validate_runtime
        from contact_order_contract import ContactOrderHook
        from trajectory_candidate import TrajectoryContactGraph as LiveContactGraph
        report['budget_preflight']=validate_runtime(plan,ContactOrderHook,LiveContactGraph)
        save(out/'budget-preflight.json',report['budget_preflight'])
        pins=verify_sources(repo,plan);save(out/'source-hashes.json',{str(p):h for p,h in pins.items()})
        from g11_source_probe import collect as collect_g11
        report['G11_source_contracts']=collect_g11(pins,out)
        # Preserve additional fixed source bodies for independent Entity reset review.
        source_capture={}
        for p,h in pins.items():
            if str(p).endswith(('/mjlab/entity/data.py','/mjlab/actuator/actuator.py','/mjlab/sim/randomization.py')):
                require(p.stat().st_size<=131072,'Selected source capture cap')
                name=p.parent.name+'-'+p.name;target=out/'source'/name;target.parent.mkdir(exist_ok=True)
                target.write_bytes(p.read_bytes());source_capture[name]=dict(path=str(p),sha256=h)
        require(len(source_capture)==3,'Selected source capture roster differs')
        save(out/'selected-source-capture.json',source_capture)
        dist=importlib.metadata.distribution('mujoco-warp');native_sources={}
        for name in plan['source_capture']:
            path=Path(dist.locate_file(name));raw=path.read_bytes();target=out/'source'/Path(name).name;target.write_bytes(raw)
            native_sources[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),file=target.name)
        save(out/'native-source-capture.json',native_sources)
        bamdist=importlib.metadata.distribution('better-actuator-models')
        for name,h in plan['installed_source_hashes']['better-actuator-models'].items():
            raw=Path(bamdist.locate_file(name)).read_bytes()
            require(hashlib.sha256(raw).hexdigest()==h,'BAM source capture differs')
            (out/'source'/('bam-'+Path(name).name)).write_bytes(raw)
        seedpath=Path(__file__).with_name('pilot-seeds.json')
        require(hashlib.sha256(seedpath.read_bytes()).hexdigest()==plan['seed_sha256'],'Pilot seed bytes differ')
        seeds=json.loads(seedpath.read_text());require([v['tick'] for v in seeds['worlds']]==[7,14],'Pilot windows differ')
        seeds={**seeds,'worlds':[seeds['worlds'][logical_world]]}
        import numpy as np
        import warp as wp
        (out.parent/'kernel-cache').mkdir(exist_ok=True)
        wp.config.kernel_cache_dir=str(out.parent/'kernel-cache')
        import torch
        from pilot_managers import build as build_managers
        import mujoco
        import mujoco_warp as mw
        import warp as wp
        from mujoco_warp._src import forward as fw
        from mujoco_warp._src import io as mwio
        from mjlab.scene import Scene
        from mjlab.sim import Simulation
        from mjlab.sensor import raycast_sensor as raymod
        from mjlab.actuator.actuator import ActuatorCmd
        from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
        from rsl_rl.algorithms.ppo import PPO
        from mjlab_microduck.robot.head_contact import physics_scene_pair
        from mjlab_microduck.actuator.friction_dr_bam import FrictionDRBamActuator
        require(torch.__version__.split('+')[0]==plan['torch_core_version'],'Torch changed')
        require(torch.cuda.is_available(),'Local CUDA required; no fallback')
        device=wp.get_device('cuda:0');require(device.is_cuda and '5060' in device.name,'Only local RTX5060 allowed')
        require(mujoco.get_mjcb_control() is None,'CPU control callback present')
        wp.set_device(device);torch.manual_seed(1109002);torch.cuda.manual_seed_all(1109002)
        def sync():wp.synchronize();torch.cuda.synchronize()
        gate=Gate(counts,persist,float('inf'),sync);gate.caps=caps
        raw_step,raw_rk=fw.step,fw.rungekutta4
        graph_apis={n:getattr(wp,n) for n in ('capture_begin','capture_end','capture_launch')}
        for module,names in (
            (mujoco,('mj_step','mj_step1','mj_step2','mj_Euler','mj_RungeKutta','mj_implicit','mj_geomDistance')),
            (mw,('step','step1','step2','euler','implicit','rungekutta4','render')),
            (fw,('step','step1','step2','euler','implicit','rungekutta4')),
            (wp,('ScopedCapture','capture_begin','capture_end','capture_launch')),
            (ManagerBasedRlEnv,('__init__',)),(PPO,('__init__',)),(Simulation,('step',)),
            (ManagerBasedRlEnv,('step','reset'))):
            for name in names:
                if hasattr(module,name):gate.patch(module,name,gate.blocked(str(module)+'.'+name))
        def limit(obj,name,label):gate.patch(obj,name,gate.limited(label,caps[label],getattr(obj,name)))
        cpu_phase={'name':'simulation'}
        cpu_forward_total=gate.limited('CPU_forward',caps['CPU_forward'],mujoco.mj_forward)
        cpu_forward_by_phase={name:gate.limited('CPU_forward_'+name,1,cpu_forward_total)
                              for name in ('simulation','context')}
        def cpu_forward(*args,**kwargs):
            require(cpu_phase['name'] in cpu_forward_by_phase,'CPU forward outside construction phases')
            return cpu_forward_by_phase[cpu_phase['name']](*args,**kwargs)
        gate.patch(mujoco,'mj_forward',cpu_forward);limit(mujoco,'mj_kinematics','CPU_kinematics')
        forward=gate.limited('GPU_forward',caps['GPU_forward'],fw.forward)
        gate.patch(mw,'forward',forward);gate.patch(fw,'forward',forward)
        reset_native=gate.limited('GPU_reset_data',caps['GPU_reset_data'],mwio.reset_data)
        gate.patch(mw,'reset_data',reset_native);gate.patch(mwio,'reset_data',reset_native)
        limit(Simulation,'reset','Simulation_reset');limit(Scene,'initialize','Scene_initialize')
        limit(Scene,'write_data_to_sim','Scene_write')
        limit(Scene,'reset','Scene_reset')
        def zero_dt(args,kwargs):require((args[1] if len(args)>1 else kwargs.get('dt'))in (0.,.002),'Only declared Scene boundary updates permitted')
        gate.patch(Scene,'update',gate.limited('Scene_update',caps['Scene_update'],Scene.update,zero_dt))
        limit(Simulation,'expand_model_fields','Model_expansion');limit(Simulation,'set_sensor_context','Sensor_context_set')
        limit(Simulation,'sense','sense');limit(mw,'refit_bvh','BVH_refit')
        def context_check(args,kwargs):require(kwargs.get('nworld')==1 and kwargs.get('cam_active')==[False],'Only single-world rays with inactive camera')
        gate.patch(mw,'create_render_context',gate.limited('context_create',1,mw.create_render_context,context_check))
        def ray_check(args,kwargs):require(tuple(kwargs['pnt'].shape)==(1,6),'Ray dimensions differ')
        gate.patch(raymod,'rays',gate.limited('ray_batch',caps['ray_batch'],raymod.rays,ray_check))
        class EagerSimulation(Simulation):
            def _should_use_cuda_graph(self):return False
            def create_graph(self):
                counts['graph_setup_suppressed']=counts.get('graph_setup_suppressed',0)+1
                self.step_graph=self.forward_graph=self.reset_graph=self.sense_graph=None
        cfg,_,candidate=physics_scene_pair()
        require(cfg.sim.mujoco.timestep==.002 and cfg.decimation==10,'Formal timing changed')
        candidate=copy.deepcopy(candidate);candidate.num_envs=1
        scene=Scene(candidate,'cuda:0');model=scene.compile()
        dims={n:int(getattr(model,n)) for n in ('nbody','ngeom','nq','nv','nu','nsensor','nsensordata','ncam')}
        require(dims==dict(nbody=19,ngeom=85,nq=35,nv=32,nu=14,nsensor=11,nsensordata=28,ncam=1),'Dimensions differ')
        simcfg=copy.deepcopy(cfg.sim);simcfg.nconmax=256;simcfg.njmax=1024
        sim=EagerSimulation(1,simcfg,model,'cuda:0')
        require(counts.get('CPU_forward_simulation_completed')==1,'Simulation CPU initialization differs')
        cpu_phase['name']='context'
        require(all(getattr(sim.wp_model.callback,n) is None for n in ('passive','control','act_dyn','act_gain','act_bias','sensor','contactfilter')),'GPU callback present')
        sim.expand_model_fields(('dof_frictionloss','dof_damping'))
        scene.initialize(model,sim.model,sim.data)
        require(counts.get('CPU_forward_context_completed')==1,'Ray context CPU initialization differs')
        cpu_phase['name']='closed'
        require(len(scene.sensors)==14 and scene.sensor_context is not None,'Sensor roster differs')
        from trajectory_candidate import TrajectoryContactGraph as LiveContactGraph,TrajectoryReductions,FullContactArchive,collect_check
        from mujoco_warp._src import solver
        import rigid_metadata_kernel
        gate.limited('Rigid_module_load',1,lambda:wp.load_module(rigid_metadata_kernel,device=device))()
        reductions=TrajectoryReductions(sim.wp_model,sim.wp_data,wp,solver,gate,caps,out=out,world=logical_world,order_limits=plan['solver_order_evidence'])
        contact_archive=FullContactArchive(out,plan['contact_ordering']['layout'],plan['complete_contact_evidence'])
        contact_order=LiveContactGraph(sim.wp_model,sim.wp_data,wp,gate,counts,plan['contact_ordering']['layout'],plan['contact_ordering']['calls'],plan['graph_limits'],graph_apis,out)
        contact_order.configure(contact_archive,rigid_metadata_kernel.rigid_metadata)
        contact_order.prepare()
        segments=Segments(wp,out,gate,graph_apis,contact_order,sim.wp_model,sim.wp_data);gate.segments=segments
        segments.install(fw,solver);reductions.install()
        gate.patch(fw.collision_driver,'collision',contact_order.wrap(fw.collision_driver.collision))
        ctx=scene.sensor_context;require(not ctx.has_cameras and len(ctx.raycast_sensors)==1,'Camera outside scope')
        resources=Resources(wp,ctx,out,identity)
        require(int(ctx.render_context.nrender)==0 and not sim.use_cuda_graph,'Rendering or CUDA graph active')
        sim.set_sensor_context(ctx)
        robot=scene['robot'];require(len(robot.actuators)==1,'BAM roster differs')
        act=robot.actuators[0];require(type(act) is FrictionDRBamActuator,'BAM type differs')
        delay=act._delay_buffer
        require(delay is not None and delay.batch_size==1 and delay.min_lag==3 and delay.max_lag==6 and
                delay.update_period==0 and delay.hold_prob==0. and delay.per_env,'Pinned BAM delay configuration differs')
        require(act._dt==.002 and delay._buffer._buffer is None,'Expected cold BAM state')
        from startup_mapping import initialize_logical_world
        from joint_inputs import InputInvariant
        startup_proof=initialize_logical_world(act,logical_world,plan,torch,gate,out)
        report['startup_mapping']=startup_proof
        input_invariant=InputInvariant(logical_world,plan,out)
        apply_delay=gate.limited('BAM_apply_delay',caps['BAM_apply_delay'],act.apply_delay)
        gate.patch(act,'apply_delay',apply_delay)
        gate.patch(act,'compute',gate.limited('BAM_compute',caps['BAM_compute'],act.compute))
        roots=[('BAM',act,('_prev_motor_torque','_dt','_base_kp','_friction_fields_checked','_delay_buffer',
            'vin_tensor','vin_drop_gain','kp_scale','kd_scale','default_kp_scale','default_kd_scale',
            'friction_scale','default_friction_scale','diagnostics_enabled','last_diagnostics')),
            ('firmware',act._bam_model.actuator,('kp','vin'))]
        for name,ent in scene.entities.items():
            roots.append(('entity:'+name,ent.data,tuple(n for n,v in vars(ent.data).items() if isinstance(v,torch.Tensor))))
        roots += [('sensor:'+n,s,tuple(vars(s))) for n,s in scene.sensors.items()]
        roots += [('reset_mask',sim._reset_mask,('_tensor','_wp_array'))]
        # Model expansion is initialization only. Record every numeric Model array
        # and scalar both here and at exit; native context contents remain opaque.
        from nested_model_state import NestedModelTables
        nested_tables=NestedModelTables(sim.wp_model,wp,out)
        model_raw,model_meta=model_state(sim.wp_model,wp);save_arrays(out,'model-before',model_raw,**model_meta)
        model_fingerprint=dict(arrays=array_meta(model_raw),**model_meta)
        context_owners=dict(context=id(ctx),render_context=id(ctx.render_context),bvh=id(ctx.render_context.bvh))
        mask_owner=(id(sim._reset_mask),id(sim._reset_mask_wp),sim._reset_mask_wp.ptr,id(sim._reset_mask._torch_stream))
        for name in ('qpos','qvel'):
            getattr(sim.wp_data,name).assign(np.asarray([s[name] for s in seeds['worlds']],np.float32))
        sim.wp_data.time.assign(np.asarray([s['time_s'] for s in seeds['worlds']],np.float32))
        forward(sim.wp_model,sim.wp_data)
        def refresh():
            scene.update(0.)
            sense_and_invalidate(sim,scene.sensors)
            return outputs(scene,torch,counts)
        first_outputs=refresh()
        # Constructors finish before the default-RNG preservation boundary.
        from pilot_state import CompleteCheckpoint,SharedMacroBudget,GroundPairStateBridge
        from macro_state_bridge import MACRO_LIMITS
        from native_rk4_bridge import NativeRK4Bridge
        from ground_pair_driver import GroundPairDriver,macro_cost
        from ground_manager_adapter import ManagerSubstepAdapter
        from evidence_journal import EvidenceJournal
        from support_material_contract import candidate_solref,verify_effective
        bank=StreamBank(torch,gate,counts,out,caps,logical_world)
        managers=build_managers(scene,sim,cfg.actions['joint_pos'],gate,counts,out,torch,pins,caps,bank,act)
        env=managers.env;scene_roots=list(roots);roots+=managers.roots
        gate.patch(managers.noise,'apply',gate.limited('Observation_noise',caps['Observation_noise'],managers.noise.apply))
        obs_delay=env.observation_manager._group_obs_term_delay_buffer['diagnostic']['action_history']
        delay_class=type(obs_delay);native_delay_compute=delay_class.compute
        limited_obs_delay=gate.limited('Observation_delay',caps['Observation_delay'],lambda:native_delay_compute(obs_delay))
        def delay_dispatch(self):return limited_obs_delay() if self is obs_delay else native_delay_compute(self)
        gate.patch(delay_class,'compute',delay_dispatch)
        initial=CompleteCheckpoint(sim,scene,roots,bank,torch,wp,gate,caps)
        originals=dict(step=raw_step,rk=segments.rk(raw_rk))
        # Model options and non-whitelisted fields are never changed at reset.
        def record(label,output=None,delayed=None):
            sync();initial.data.preflight()
            require(initial.global_rng()==initial.rng,'Default RNG advanced after construction; preserve failure')
            data={n:a.numpy().copy() for n,a in initial.data.arrays.items()}
            raw,scalars,topology=owner_state(roots,torch,wp)
            save_arrays(out,label+'-data',data,owners=initial.data.owners,descriptors=initial.data.desc,last_contact_ordering=contact_order.last,ordering_freshness_not_asserted=True)
            save_arrays(out,label+'-owners',raw,scalars=scalars,topology=topology)
            save_arrays(out,label+'-rng',rng_raw(torch))
            arrays,metadata=model_state(sim.wp_model,wp)
            save_arrays(out,label+'-mutable-model',{n:arrays[n] for n in initial.allowed_model},all_model_hashes=array_meta(arrays),metadata=metadata)
            if output is not None:save_arrays(out,label+'-outputs',output)
            if delayed is not None:save_arrays(out,label+'-delay',{'position_target':delayed})
            return dict(data=data,owners=raw,scalars=scalars,topology=topology,outputs=output,delayed=delayed,rng=rng_raw(torch))
        initial_record=record('initial-cold',first_outputs)
        material0=sim.wp_model.geom_solref.numpy().copy()
        require(digest(material0)=='d3098dbc5e1ebceff9114b90ac905b28a9f86b48e0ca4efc6c04b2e9c9d29648','Pinned original material differs')
        material=candidate_solref(material0,.0025)
        gate.limited('Model_material_setup',1,lambda:sim.wp_model.geom_solref.assign(material))()
        require(sim.wp_model.geom_solref.numpy().tobytes()==material.tobytes(),'Material setup bytes differ')
        forward(sim.wp_model,sim.wp_data);refresh()
        save(out/'pre-macro-ordering.json',contact_order.last)
        # Fresh declared owners, not an invented replay of archived BAM history.
        # A single real observation push allocates the selected history buffers.
        action_input=torch.zeros((1,14),device='cuda:0',dtype=torch.float32)
        env.action_manager.process_action(action_input)
        env.obs_buf=env.observation_manager.compute(update_history=True)
        last_delay={'value':None}
        counted_delay=act.apply_delay
        def remember_delay(cmd):
            value=counted_delay(cmd);last_delay['value']=value.position_target.detach().cpu().numpy().copy();return value
        gate.patch(act,'apply_delay',remember_delay)
        macro=SharedMacroBudget({n:caps[n] for n in MACRO_LIMITS},counts=counts,persist=persist,sync=sync,deadline=float('inf'))
        bridge=GroundPairStateBridge(sim,scene,roots,bank,torch,wp,macro)
        engine=NativeRK4Bridge(fw,mw,sim.wp_model,sim.wp_data,wp,macro)
        engine.solver_order=reductions.order
        # All public step/RK entry points remain blocked outside owned hooks.
        engine.native_step=originals['step'];engine.native_rk=originals['rk']
        engine.contact_order=contact_order
        from adaptive_driver import AdaptiveContinuation
        from recovery_continuation import RecoveryContinuation,RecoveryAdapter
        from rebuild_boundary import restore_boundary
        require(restore_from is not None,'Immutable source checkpoint required')
        if role=='producer':
            portable=PortableState(sim,bridge.roots,torch,wp)
            source=restore_from/'logical-start'
        else:
            from endpoint_rebuild import EndpointPortableState
            portable=EndpointPortableState(sim,bridge.roots,torch,wp).configure(
                act,env,bank,logical_world,plan['endpoint_schema_sha256'][str(logical_world)],out)
            source=restore_from/'logical-endpoint'
        boundary=restore_boundary(portable,source,sim,scene.sensors,lambda:outputs(scene,torch,counts),out)
        report['portable_restore']=boundary['portable_restore'];report['rebuild_boundary_passed']=boundary['passed']
        report['rebuild_sense_exact']=boundary['after']['arrays'];portable=None
        nested_tables.check('loaded')
        start=CompleteCheckpoint(sim,scene,roots,bank,torch,wp,gate,caps)
        portable=PortableState(sim,start.roots,torch,wp);start_raw,start_schema=portable.snapshot();portable=None
        write_portable(out/'logical-start',start_raw,start_schema)
        phase_progress(1,'完整状态与缓存已加载；本进程开始新的图覆盖/续算范围')
        endpoints={};schemas={};checks=[]
        def execute_trajectory(mode,fixed):
            nonlocal bridge,engine
            segments.enabled=mode!='fixed-eager'
            bridge=GroundPairStateBridge(sim,scene,roots,bank,torch,wp,macro)
            engine=NativeRK4Bridge(fw,mw,sim.wp_model,sim.wp_data,wp,macro)
            engine.native_step=originals['step'];engine.native_rk=originals['rk']
            engine.solver_order=reductions.order;engine.contact_order=contact_order
            ticks=(seeds['worlds'][0]['tick']+int(fixed),)
            driver=RecoveryContinuation(bridge,engine,ticks) if fixed else AdaptiveContinuation(bridge,engine,ticks)
            journal=EvidenceJournal(out/(mode+'-raw'),bridge,byte_limit=64*1024*1024,file_limit=600,record_limit=450)
            driver.emit=journal
            adapter=RecoveryAdapter(env,driver) if fixed else ManagerSubstepAdapter(env,driver)
            result=adapter.advance();journal.finalize()
            if fixed:
                require(result['integrations']==128 and not result['adaptive_acceptance'] and driver.commits==1,'Fixed continuation schedule differs')
            else:
                require(result['accepted_resolution']==128 and result['replayed'] and driver.commits==1,'Original adaptive rejection/replay missing')
                GroundPairDriver.check_required_witness(driver,driver.accepted_trial)
            sensor_output=refresh();record(mode,sensor_output,last_delay['value'])
            portable=PortableState(sim,start.roots,torch,wp);raw,schema=portable.snapshot()
            write_portable(out/(mode+'-endpoint'),raw,schema)
            endpoints[mode]=raw;schemas[mode]=schema
            save(out/(mode+'-result.json'),result);completed.append(mode);segments.enabled=False
            nested_tables.check(mode)
            return raw,schema
        if role=='producer':
            raw,schema=execute_trajectory('adaptive-graph',False)
            for reference_mode in ('eager','graph'):
                old,old_schema=read_portable(restore_from/(reference_mode+'-endpoint'))
                check=dict(name='archived-'+reference_mode+'-vs-expanded-adaptive-graph',**exact_arrays(old,raw),schema_exact=json.loads(json.dumps(schema))==old_schema)
                checks.append(check)
            save(out/'adaptive-endpoint-comparisons.json',checks)
            require(all(r['passed'] and r['schema_exact'] for r in checks),'Expanded adaptive endpoint differs; preserve evidence and stop this dependent branch')
            write_portable(out/'logical-endpoint',raw,schema)
            endpoint_raw,endpoint_schema=raw,schema
            endpoint=CompleteCheckpoint(sim,scene,roots,bank,torch,wp,gate,caps)
            phase_progress(2,'扩展图路径完成；与两条历史端点严格比较通过')
            modes=('fixed-eager','fixed-graph')
        else:
            endpoint=start;endpoint_raw,endpoint_schema=start_raw,start_schema
            write_portable(out/'logical-endpoint',endpoint_raw,endpoint_schema)
            modes=('restored-fixed-graph',)
        for index,mode in enumerate(modes):
            endpoint.restore()
            execute_trajectory(mode,True)
            phase_progress((3 if role=='producer' else 2)+index,'已完成 '+mode+'；完整原始轨迹已保留')
        if role=='producer':
            checks.append(dict(name='fixed-eager-vs-expanded-fixed-graph',**exact_arrays(endpoints['fixed-eager'],endpoints['fixed-graph']),schema_exact=schemas['fixed-eager']==schemas['fixed-graph']))
        else:
            for reference_mode in ('fixed-eager','fixed-graph'):
                old,old_schema=read_portable(restore_from/(reference_mode+'-endpoint'))
                checks.append(dict(name='producer-'+reference_mode+'-vs-restored-fixed-graph',**exact_arrays(old,endpoints['restored-fixed-graph']),schema_exact=json.loads(json.dumps(schemas['restored-fixed-graph']))==old_schema))
        save(out/'endpoint-comparisons.json',checks)
        require(all(r['passed'] and r['schema_exact'] for r in checks),'Complete continuation endpoint differs; raw evidence preserved')
        endpoint.restore()
        restored=PortableState(sim,start.roots,torch,wp);raw,schema=restored.snapshot();restored=None
        write_portable(out/'post-continuation-restored',raw,schema)
        restore_check=exact_arrays(endpoint_raw,raw)
        save(out/'post-continuation-restore.json',dict(**restore_check,schema_exact=schema==endpoint_schema))
        require(restore_check['passed'] and schema==endpoint_schema,'Whole post-macro checkpoint restoration differs')
        report['segment_lifecycle']=segments.close();gate.segments=None
        # Restore the original cold numerical state before retiring native owners.
        report['whole_run_restore']=initial.restore();record('final-original-restored')
        report['nested_Model_final']=nested_tables.check('final')
        report['contact_graph']=contact_order.summary();contact_order.close()
        report['resource_lifecycle']=resources.close(ctx,sim)
        require(counts['integrations_attempted']==counts['integrations_completed']==native_total,'Exact microstep schedule differs')
        require(counts['GPU_forward_attempted']==counts['GPU_forward_completed']==forward_total,'Exact physical forward schedule differs')
        require(verify_sources(repo,plan)==pins,'Pinned sources changed')
        report.update(status='G10_BLOCKS_RECOVERY_MEASURED_REVIEW_REQUIRED',role=role,identity=identity,
            completed_macros=list(completed),checks=checks,all_endpoint_comparisons_passed=all(r['passed'] and r['schema_exact'] for r in checks),
            scope=plan['scope'],
            full_native_RK4_graph=False,G10='OPEN',G11='OPEN',PPO=0,cloud_training=0,
            portable_scope='complete post-one-accepted-macro state and selected owners; fixed128 continuation; no interrupted private-stage recovery')
        phase_progress(macro_total+2,'本进程矩阵、完整恢复与所有已登记资源释放完成')
    except BaseException as exc:
        report.update(status='G10_BLOCKS_RECOVERY_STOPPED_REVIEW_REQUIRED',error=repr(exc),traceback=traceback.format_exc(),role=role,identity=identity,completed_macros=list(completed))
        # No physical retry after native or protocol failure. Cleanup is best effort.
        if segments is not None:
            try:report['segment_cleanup']=segments.close()
            except BaseException as error:report['segment_cleanup_error']=repr(error)
        if contact_order is not None:
            try:contact_order.close()
            except BaseException as error:report['contact_cleanup_error']=repr(error)
        if resources is not None and sim is not None:
            try:report['resource_cleanup']=resources.close(ctx,sim)
            except BaseException as error:report['resource_cleanup_error']=repr(error)
    finally:
        if resources is not None:resources.unpatch()
        if gate is not None:gate.close()
        report['counts']=dict(counts);report['elapsed_s']=time.monotonic()-started
        persist();save(out/'run-review.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return 0 if report['status']=='G10_BLOCKS_RECOVERY_MEASURED_REVIEW_REQUIRED' else 2

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--world',type=int,choices=(0,1),required=True);p.add_argument('--role',choices=('producer','consumer'),required=True);p.add_argument('--restore-from',type=Path)
    p.add_argument('--parent-pid',type=int,required=True);p.add_argument('--execute-declared-plan',action='store_true',required=True);a=p.parse_args()
    require(sys.flags.isolated and sys.dont_write_bytecode and not sys.flags.optimize,'Use venv Python -I -B')
    return run(a.repo,a.out,json.loads(a.plan.read_bytes()),a.world,a.parent_pid,a.role,a.restore_from)
if __name__=='__main__':raise SystemExit(main())
