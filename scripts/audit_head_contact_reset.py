"""Native Entity reset IO audit, not a Manager rollout; CUDA forward only."""
from __future__ import annotations
import argparse, hashlib, importlib.util, inspect, json, time, traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
t=load('frozen_tensor_audit',ROOT/'scripts/audit_head_contact_tensor.py')
a=t.a;require,save=a.require,a.save
support=load('frozen_mesh_distance',ROOT/'scripts/analyze_head_contact_support.py')
CONTRACT=ROOT/'docs/audits/stage-11-head-reset-contract.json'
PRIOR=Path('/home/lx/microduck-double-balance/artifacts/double-balance-stage11/head-tensor/20261004T071751Z.zip')
RESET_BATCHES=((0,1,2),(2,),(1,0))
NUMERICAL_ATOL=2e-5


def validate_sensor_context(scene):
    """Retain the inherited foot ray sensor; require zero camera sensors.

    A raycast BVH resource is expected during native Scene initialization.
    No sensor execution or camera rendering is requested by this audit.
    """
    context=scene.sensor_context
    require(context is not None,'Inherited foot raycast context missing')
    require(not context.has_cameras and not context.camera_sensors,'Camera sensors outside reset IO scope')
    sensors={name:sensor for name,sensor in scene.sensors.items() if sensor.requires_sensor_context}
    require(set(sensors)=={'foot_height_scan'},'Unexpected context-dependent sensor set')
    sensor=sensors['foot_height_scan']
    require(type(sensor).__name__=='TerrainHeightSensor' and type(sensor).__module__=='mjlab.sensor.terrain_height_sensor','Unexpected foot sensor implementation')
    require(context.has_raycasts and len(context.raycast_sensors)==1 and context.raycast_sensors[0] is sensor,'Raycast context identity mismatch')
    return {'context_present':True,'camera_sensors':0,'raycast_sensor_names':['foot_height_scan'],
            'raycast_class':'mjlab.sensor.terrain_height_sensor.TerrainHeightSensor',
            'raycast_BVH_resource_initialized':True,'camera_render_or_sense_requested':False}


def validate_reset_arrays(before, after, selected, expected_robot, expected_lower, expected_top, robot_qadr, default_joints):
    """Independent qpos/qvel/clock comparison, using compiled named addresses."""
    selected=np.asarray(selected,int);other=np.setdiff1d(np.arange(len(before['qpos'])),selected)
    for k in ('qpos','qvel','ctrl','time','xfrc_applied'):
        require(np.array_equal(before[k][other],after[k][other]),'Nonselected state changed: '+k)
    for k in ('ctrl','time','xfrc_applied'):
        require(np.array_equal(before[k],after[k]),'Reset IO changed '+k)
    require(np.isfinite(after['qpos']).all() and np.isfinite(after['qvel']).all(),'Nonfinite reset state')
    require(not np.any(after['qvel'][selected]),'Selected residual velocity')
    for key,expected in [('robot',expected_robot),('ball',expected_lower),('top_ball',expected_top)]:
        np.testing.assert_allclose(after[key+'_position'][selected],expected,atol=NUMERICAL_ATOL,rtol=0)
        np.testing.assert_allclose(after[key+'_quaternion'][selected],np.tile([1.,0,0,0],(len(selected),1)),atol=NUMERICAL_ATOL,rtol=0)
    np.testing.assert_array_equal(after['qpos'][selected][:,robot_qadr],default_joints[selected])
    return {'selected':selected.tolist(),'nonselected':other.tolist(),'nonselected_qpos_qvel_ctrl_time_wrench_exact':True,'selected_velocities_exact_zero':True,'root_and_ball_pose_residual_max':float(max(np.max(np.abs(after[k+'_position'][selected]-v)) for k,v in [('robot',expected_robot),('ball',expected_lower),('top_ball',expected_top)]))}


def extra_sources(out):
    result={};contract=json.loads(CONTRACT.read_text())
    for name,expected in contract['additional_mjlab_sources'].items():
        root=Path(importlib.util.find_spec('mjlab').origin).parent;blob=(root/name).read_bytes();actual=hashlib.sha256(blob).hexdigest()
        result[name]={'sha256':actual,'reviewed_sha256':expected,'matches':actual==expected}
        p=out/'installed-source/mjlab'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(blob)
    save(out/'reset-dependency-sources.json',result)
    require(all(r['matches'] for r in result.values()),'Native entity/bridge source mismatch; no model construction')
    return result


def read_state(scene,wd):
    result={k:getattr(wd,k).numpy().copy() for k in ('qpos','qvel','ctrl','time','xfrc_applied')}
    for name in ('robot','ball','top_ball'):
        data=scene[name].data
        result[name+'_position']=data.root_link_pos_w.detach().cpu().numpy().copy()
        result[name+'_quaternion']=data.root_link_quat_w.detach().cpu().numpy().copy()
    return result


def run(out,prior):
    import mujoco,mujoco_warp as mw,warp as wp,torch
    from mjlab.scene import Scene
    from mjlab.sim.sim_data import WarpBridge
    from mjlab_microduck.tasks import mdp
    from mjlab_microduck.tasks.microduck_head_contact_env_cfg import make_microduck_head_contact_env_cfg
    require(torch.cuda.is_available() and '5060' in torch.cuda.get_device_name(0),'Authorized local RTX5060 required')
    cfg=make_microduck_head_contact_env_cfg(static_audit_only=True);cfg.scene.num_envs=3
    require(cfg.events['stage11_release_block'].func is mdp.stage11_head_release_block,'Startup guard changed')
    require(cfg.terminations['top_ball_lost'].func is mdp.stage11_head_release_block,'Termination guard changed')
    require(tuple(cfg.actions['ball_hold'].levels)==(0.,) and not cfg.curriculum,'Nominal unassisted config changed')
    scene=Scene(cfg.scene,device='cuda:0');model=scene.compile();cfg.sim.mujoco.apply(model)
    fields=a.load_module('reset_physical_fields',ROOT/'scripts/audit_head_task_prototype.py').PHYSICS_FIELDS
    physical={k:hashlib.sha256(np.asarray(getattr(model,k)).tobytes()).hexdigest() for k in fields}
    require(physical==prior,'Compiled physical arrays differ from tensor return')
    ids=a.resolve_ids(model);save(out/'compiled-identities.json',ids)
    counters={'forward_calls':0,'reset_calls':0};samples=[]
    def progress(phase):
        save(out/'execution-progress.json',{'phase':phase,**counters,'new_integration_steps':0})
    progress('before_native_initialization')
    with wp.ScopedDevice('cuda:0'):
        wm=mw.put_model(model);wd=mw.put_data(model,mujoco.MjData(model),nworld=3,nconmax=512,njmax=4096)
        mb,db=WarpBridge(wm,nworld=3),WarpBridge(wd)
        device_fields={k:v for k,v in vars(wm).items() if isinstance(v,wp.array)}
        device_fields.update({'opt.'+k:v for k,v in vars(wm.opt).items() if isinstance(v,wp.array)})
        device_before={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in device_fields.items()}
        save(out/'device-model-signatures-before.json',device_before)
        # Use one actual Warp stream for native Entity Torch writes and forwards.
        # This audit does not claim asynchronous stream-performance coverage.
        stream=torch.cuda.ExternalStream(wp.get_stream().cuda_stream,device='cuda:0')
        with torch.cuda.stream(stream):
            scene.initialize(model,mb,db)
            sensor_metadata=validate_sensor_context(scene)
            save(out/'sensor-context-review.json',sensor_metadata)
            progress('native_initialization_and_sensor_identity_checked')
            # Record exactly which native actuator implementations were initialized.
            actuator_sources={}
            for act in scene['robot'].actuators:
                for cls in type(act).__mro__:
                    try:path=inspect.getsourcefile(cls)
                    except TypeError:continue
                    if not path:continue
                    blob=Path(path).read_bytes();key=cls.__module__
                    actuator_sources[key]={'sha256':hashlib.sha256(blob).hexdigest(),'bytes':len(blob)}
                    p=out/'actuator-source'/Path(*key.split('.')).with_suffix('.py');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(blob)
            save(out/'actuator-sources.json',actuator_sources)
            def forward():
                require(counters['forward_calls']<9,'Forward budget exceeded')
                before={k:getattr(wd,k).numpy().copy() for k in ('qpos','qvel','time')}
                mw.forward(wm,wd);wp.synchronize()
                counters['forward_calls']+=1
                progress('forward_completed')
                require(all(np.array_equal(getattr(wd,k).numpy(),v) for k,v in before.items()),'Forward advanced primary state')
                require(int(wd.nacon.numpy()[0])<wd.naconmax and np.max(wd.nefc.numpy())<wd.njmax,'Solver capacity reached')
            sim=SimpleNamespace(mj_model=model,wp_model=wm,wp_data=wd,model=mb,data=db,num_envs=3,device='cuda:0',forward=forward)
            env=SimpleNamespace(scene=scene,sim=sim,num_envs=3,device='cuda:0')
            robot=scene['robot'];qadr=robot.indexing.joint_q_adr.detach().cpu().numpy()
            defaults=robot.data.default_joint_pos.detach().cpu().numpy().copy()
            require(defaults.shape==(3,len(qadr)),'Native default joint width differs from indexing')
            yaw_ids,yaw_names=robot.find_joints('^head_yaw$');require(len(yaw_ids)==1,'Head yaw identity not unique')
            allids=torch.arange(3,device='cuda:0',dtype=torch.long)
            origins=scene.env_origins.detach().cpu().numpy().copy()
            save(out/'native-indexing.json',{'joint_names':robot.joint_names,'robot_joint_qadr':qadr,'default_joint_pos':defaults,'origins':origins,'yaw_name':yaw_names,'startup_events_executed':False,'action_manager_constructed':False})
            for revision,batch in enumerate(RESET_BATCHES):
                # Distinct artificial prior states make stale reads and cross-world writes observable.
                for entity_index,name in enumerate(('robot','ball','top_ball')):
                    poses=torch.zeros(3,7,device='cuda:0');poses[:,3]=1.
                    poses[:,:3]=torch.tensor(origins,device='cuda:0')+torch.tensor([[.3,.2,.6],[.4,-.3,.7],[-.2,.4,.8]],device='cuda:0')
                    poses[:,0]+=.4*entity_index
                    scene[name].write_root_link_pose_to_sim(poses,env_ids=allids)
                    scene[name].write_root_link_velocity_to_sim(torch.arange(1,19,device='cuda:0',dtype=torch.float32).reshape(3,6)*.01,env_ids=allids)
                jp=robot.data.default_joint_pos.clone();jp[:,yaw_ids[0]]+=torch.tensor([.1,-.1,.2],device='cuda:0')
                robot.write_joint_state_to_sim(jp,torch.full_like(jp,.03),env_ids=allids)
                forward();before=read_state(scene,wd)
                if hasattr(env,'_basketball_state'):anchor_before=env._basketball_state.anchor.detach().cpu().numpy().copy()
                else:anchor_before=None
                selected=torch.tensor(batch,device='cuda:0',dtype=torch.long)
                start_count=counters['forward_calls']
                mdp.reset_stage11_head_reference(env,selected,**cfg.events['reset_stage11_head_reference'].params)
                counters['reset_calls']+=1
                progress('reset_function_returned')
                require(counters['forward_calls']==start_count+1,'Reset intermediate forward count changed')
                # Native reset deliberately leaves the final forward to its caller.
                forward();after=read_state(scene,wd)
                sids,_=robot.find_sites('^stage11_head_material_frame$');require(len(sids)==1,'Reference site missing')
                site=wd.site_xpos.numpy()[:,ids['site']];rotation=wd.site_xmat.numpy()[:,ids['site']].reshape(3,3,3)
                expected_robot=origins[list(batch)]+np.array([0.,0.,.368]);expected_lower=origins[list(batch)]+np.array([0.,0.,.121])
                expected_top=site[list(batch)]+rotation[list(batch),: ,2]*.0205
                review=validate_reset_arrays(before,after,batch,expected_robot,expected_lower,expected_top,qadr,defaults)
                anchor=env._basketball_state.anchor.detach().cpu().numpy().copy()
                np.testing.assert_allclose(anchor[list(batch)],expected_lower,atol=NUMERICAL_ATOL,rtol=0)
                if anchor_before is not None:
                    other=np.setdiff1d(np.arange(3),batch);np.testing.assert_array_equal(anchor[other],anchor_before[other])
                capture=mdp.stage11_head_contact_tensor_snapshot(sim,phase='static_forward',revision=revision,static_audit_only=True)
                tensors={k:t.tensor_cpu(v) for k,v in capture['tensors'].items()};np.savez_compressed(out/f'tensor-reset-{revision}.npz',**tensors)
                require(tensors['valid_environment'].all() and tensors['contact_capacity_ok'] and tensors['invalid_world_contacts']==0,'Invalid reset snapshot')
                for name,fn in [('actor',mdp.stage11_head_actor_state),('critic',mdp.stage11_head_critic_state)]:
                    actual=fn(env).detach().cpu().numpy();p=tensors['position_material'].copy();p[:,2]-=.02
                    expected=np.concatenate([p*np.array([20.,25.,50.]),tensors['rate_material']*2.],axis=-1)
                    if name=='critic':
                        spin=np.einsum('nji,nj->ni',tensors['rotation'],tensors['ball_omega']-tensors['head_omega']);expected=np.concatenate([expected,spin*.2],axis=-1)
                    require(np.isfinite(actual).all(),'Nonfinite raw feature result')
                    np.testing.assert_allclose(actual,expected,atol=NUMERICAL_ATOL,rtol=0);np.save(out/f'{name}-features-{revision}.npy',actual)
                geom=ids['head_geom'];mesh=int(model.geom_dataid[geom]);va=int(model.mesh_vertadr[mesh]);vn=int(model.mesh_vertnum[mesh]);fa=int(model.mesh_faceadr[mesh]);fn=int(model.mesh_facenum[mesh])
                vertices=np.array(model.mesh_vert[va:va+vn],float);faces=np.array(model.mesh_face[fa:fa+fn],int)
                gp=wd.geom_xpos.numpy();gr=wd.geom_xmat.numpy().reshape(3,model.ngeom,3,3);geometry=[]
                for world in batch:
                    center=after['top_ball_position'][world];local=gr[world,geom].T@(center-gp[world,geom])
                    gap=support.nearest_distance(local,vertices[faces])-.02
                    # Prove the center is outside the closed shell along the reference normal.
                    verts_world=vertices@gr[world,geom].T+gp[world,geom]
                    normal=rotation[world,:,2];outside_margin=float((center-site[world])@normal-((verts_world-site[world])@normal).max())
                    require(outside_margin>0 and abs(gap-.0005)<=NUMERICAL_ATOL,'Nominal reset is not the audited clear head placement')
                    require(tensors['head_count'][world]==0 and tensors['floor_count'][world]==0,'Unexpected upper-ball contact at clear reset')
                    geometry.append({'world':world,'unsigned_mesh_distance_minus_radius_m':gap,'outside_shell_projection_margin_m':outside_margin})
                ncon=int(wd.nacon.numpy()[0]);raw={k:getattr(wd.contact,k).numpy()[:ncon].copy() for k in ('worldid','geom','pos','frame','dist','efc_address')}
                save(out/f'contacts-reset-{revision}.json',raw)
                np.savez_compressed(out/f'state-reset-{revision}.npz',**{'before_'+k:v for k,v in before.items()},**{'after_'+k:v for k,v in after.items()},reference_site=site,reference_rotation=rotation,anchor=anchor)
                review.update(revision=revision,mesh_clearance=geometry,head_count=tensors['head_count'],floor_count=tensors['floor_count'],basketball_hold_bookkeeping=env._basketball_state.hold.detach().cpu().numpy(),assistance_applied=False,full_manager_execution=False)
                samples.append(review);save(out/'reset-cases.json',samples)
            require(counters=={'forward_calls':9,'reset_calls':3},'Unexpected audit count')
            require(not np.any(wd.xfrc_applied.numpy()),'Unexpected external wrench')
            device_after={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in device_fields.items()}
            save(out/'device-model-signatures-after.json',device_after)
            require(device_before==device_after,'Native entity initialization/reset changed device model arrays')
    require(physical=={k:hashlib.sha256(np.asarray(getattr(model,k)).tobytes()).hexdigest() for k in fields},'Physical model mutated')
    save(out/'physical-array-signatures.json',physical)
    progress('all_reset_IO_checks_completed')
    return {'device':torch.cuda.get_device_name(0),'worlds':3,**counters,'selected_reset_samples':6,'native_scene_entity_IO':True,'sensor_context':sensor_metadata,'manager_startup_actions_or_step_executed':False,'zero_external_wrench':True,'device_model_array_count_unchanged':len(device_before),'startup_and_termination_release_guards_preserved':True,'new_integration_steps':0,'limit':'Direct native reset-function IO only; no action manager, BAM startup event or full Manager lifecycle. hold bookkeeping may be one; no assistance was applied and this is not unassisted strategy evidence.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reset-forward-only',required=True,action='store_true');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False);start=time.monotonic();attempts=[]
    report=dict(status='NATIVE_RESET_EXECUTION_BLOCKED',stage11_status='MATERIAL_BLOCKED_REVIEW_REQUIRED',stage11_complete=False,new_integration_steps=0,new_ppo_updates=0,cloud_hours=0,canonical_total_completed_steps=7895,formal_task_migration=False,policy_success=None)
    try:
        report['prior_source_count']=len(a.source_audit(out));report['stream_source_count']=len(t.additional_sources(out));report['reset_source_count']=len(extra_sources(out))
        contract=json.loads(CONTRACT.read_text());z=support.PinnedZip(PRIOR,contract['tensor_zip_sha256']);prior=z.json('tensor/physical-array-signatures.json');r=z.json('run-summary.json')
        require(r['execution_invariants_unchanged'] and r['canonical_total_known_completed_steps']==7895 and r['physics_integration_steps']==0,'Tensor provenance changed');save(out/'input-provenance.json',{'zip_sha256':z.sha,'consumed':z.used})
        with a.zero_integration_guard() as attempts:
            report['reset']=run(out,prior)
            require(not attempts and time.monotonic()-start<600,'Zero-step or wall budget failed')
        report['status']='NATIVE_RESET_IO_REVIEW_REQUIRED'
    except Exception as exc:report['error']=str(exc);(out/'error.txt').write_text(traceback.format_exc())
    if (out/'execution-progress.json').exists():report['execution_progress']=json.loads((out/'execution-progress.json').read_text())
    report['forbidden_integration_call_attempts']=attempts;report['wall_seconds']=time.monotonic()-start
    save(out/'reset-review.json',report);print('Stage11ResetResult='+report['status'],flush=True)
    return 0 if report['status']=='NATIVE_RESET_IO_REVIEW_REQUIRED' else 2


if __name__=='__main__':raise SystemExit(main())
