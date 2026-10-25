"""Only the missing G6 autonomous GPU half-step sensitivity counterpart.

Reuses immutable G5 clone/event helpers. No CPU or G5 rollout is repeated.
Native RK4 dispatch and arithmetic/physical gates remain identical.
"""
from pathlib import Path
import argparse,gzip,importlib.metadata,json,time,traceback,warnings
import numpy as np
import mujoco
import diagnose_head_contact_floor_rk_reference as rk
import diagnose_head_contact_floor_cuda_schedule as bridge
import diagnose_head_contact_floor_cuda_events as events

TOTAL=4000
PRIOR=30357
BASE_DT=1/64000
N=3200
WEIGHTS=bridge.WEIGHTS
primary=bridge.primary
guard_state=bridge.guard_state
clone_data=events.clone_data
digest_data=events.digest_data
array_leaves=events.array_leaves

def native_trial(wm,d,fw,wp,rec,b,out,item):
    rk.require(b['attempted']<TOTAL,'4000-trial budget exhausted')
    rk.require(time.monotonic()<rec.deadline,'Wall budget reached before integration')
    h=item['actual_float32_dt_s'];wm.opt.timestep.fill_(h);wp.synchronize()
    rk.require(float(wm.opt.timestep.numpy()[0])==h,'Device timestep assignment failed')
    before=primary(d);guard_state(before);rec.begin(item)
    b['attempted']+=1;rk.save(out/'progress.json',b)
    with warnings.catch_warnings(record=True)as caught:
        warnings.simplefilter('always');fw.step(wm,d);wp.synchronize()
    b['completed']+=1;rk.save(out/'progress.json',b)
    after=primary(d);guard_state(after);rk.require(not caught,'Warning in CUDA integration')
    rk.require(len(rec.rows)==4,'Native RK4 did not produce four forward stages')
    rk.require(np.array_equal(before['qpos'],rec.rows[0]['state']['qpos'])and np.array_equal(before['qvel'],rec.rows[0]['state']['qvel']),'First RK stage state mismatch')
    acc=np.array([s['qacc']for s in rec.rows],np.float64);vel=np.array([s['state']['qvel']for s in rec.rows],np.float64)
    rv=after['qvel'].astype(float)-before['qvel']-h*(WEIGHTS@acc)
    rx=after['qpos'][:3].astype(float)-before['qpos'][:3]-h*(WEIGHTS@vel[:,:3])
    eps=np.finfo(np.float32).eps
    vt=8*eps*(1+max(float(np.max(abs(before['qvel']))),float(np.max(abs(after['qvel']))),float(np.max(abs(h*acc)))))
    xt=8*eps*(1+max(float(np.max(abs(before['qpos'][:3]))),float(np.max(abs(after['qpos'][:3])))))
    rk.require(np.max(abs(rv))<=vt and np.max(abs(rx))<=xt,'GPU weighted-increment identity inconsistent with float32 bound')
    expected_clock=float(np.float32(np.float32(before['native_device_time_s'])+np.float32(h)))
    rk.require(after['native_device_time_s']==expected_clock,'Native float32 clock update differs from source')
    rec.max_overlap=max(rec.max_overlap,.02-float(after['qpos'][2]))
    return {'schedule':item,'before':before,'after':after,'stages':rec.rows,'identities':{'velocity_residual_max':float(np.max(abs(rv))),'position_residual_max':float(np.max(abs(rx))),'float32_velocity_bound':vt,'float32_position_bound':xt}}


class RefinedStepper(events.EventStepper):
    def trial(self,d,start,end,macro,depth,path):
        rk.require(self.b['attempted']<TOTAL,'4000 CUDA trial budget exhausted')
        rk.require(time.monotonic()<self.deadline,'Wall budget reached before copy')
        h=float(np.float32(end-start));rk.require(0<h<=float(np.float32(BASE_DT))+1e-13,'Invalid local timestep')
        original=digest_data(d);td=clone_data(d,self.wp);self.last_trial=td
        item={'index':self.b['attempted'],'macro_index':macro,'start_s':start,'end_s':end,'requested_dt_s':end-start,'actual_float32_dt_s':h}
        try:row=native_trial(self.m,td,self.fw,self.wp,self.rec,self.b,self.out,item)
        finally:
            rk.require(digest_data(d)==original,'Trial mutated original full Warp Data')
        row.update({'trial_id':self.b['attempted'],'before_full_data_sha256':original,'after_full_data_sha256':digest_data(td),'source_full_data_unchanged':True,'depth':depth,'tree_path':path})
        return td,row


def compare(reference,gpu):
    grid=.39+np.arange(3201)*BASE_DT
    rk.require(gpu['qpos'].shape==(3201,7)and gpu['qvel'].shape==(3201,6)and np.array_equal(gpu['time'],grid),'Complete G6 original grid required')
    n=len(reference['time']);rk.require(n in [1601,3201],'Unsupported archived grid')
    rs=1 if n==1601 else 2
    rk.require(reference['qpos'].shape==(n,7)and reference['qvel'].shape==(n,6),'Archived shape mismatch')
    rk.require(np.max(abs(reference['time'][::rs]-grid[::2]))<1e-12,'Shifted original common grid')
    for key in ['qpos','qvel']:
        rk.require(np.isfinite(reference[key]).all()and np.isfinite(gpu[key]).all(),'Nonfinite comparison arrays')
        rk.require(np.array_equal(gpu[key][0],reference[key][0].astype(np.float32)),'Archived initial primary state differs')
    rp,rv=reference['qpos'],reference['qvel'];gp,gv=gpu['qpos'],gpu['qvel']
    dx=np.linalg.norm(rp[::rs,:3]-gp[::2,:3],axis=1);dv=np.linalg.norm(rv[::rs,:3]-gv[::2,:3],axis=1)
    metrics={'peak_overlap':abs(max(0,.02-float(rp[:,2].min()))-max(0,.02-float(gp[:,2].min()))),'position':float(dx.max()),'velocity':float(dv.max()),'peak_upward_velocity':abs(max(0,float(rv[:,2].max()))-max(0,float(gv[:,2].max())))}
    checks={k:{'value':v,'limit':rk.LIMITS[k],'pass':v<=rk.LIMITS[k]}for k,v in metrics.items()}
    result={'checks':checks,'all_pass':all(c['pass']for c in checks.values()),'common_samples':1601,'reference_peak_samples':n,'G6_peak_samples':3201,'time_shift_s':0,'full_window_s':[.39,.44]}
    if n==3201:
        rk.require(np.max(abs(reference['time']-grid))<1e-12,'Shifted fine reference grid')
        dense={'max_position_difference_m':float(np.linalg.norm(rp[:,:3]-gp[:,:3],axis=1).max()),'max_velocity_difference_m_s':float(np.linalg.norm(rv[:,:3]-gv[:,:3],axis=1).max())}
        result['dense_3201_checks']=dense;result['all_pass']&=dense['max_position_difference_m']<=rk.LIMITS['position']and dense['max_velocity_difference_m_s']<=rk.LIMITS['velocity']
    return result

def run(out,model_path,e6,g5,fixed,input_hashes):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();deadline=started+600
    b={'attempted':0,'completed':0,'rk_stages_observed':0,'accepted_substeps':0,'discarded_trials':0,'localized_switch_intervals':0,'setup_forward_attempts':0,'setup_forward_completed':0,'PPO':0,'prior_canonical_steps':PRIOR};rk.save(out/'progress.json',b)
    report={'status':'STARTED','formal_physics_adoption':False,'formal_task_migration':False,'stage11_complete':False,'PPO':0,'cloud_hours':0,'full_robot_steps':0,'GPU_chooses_events':True,'CPU_schedule_replayed':False,'original_2ms_floor_block_retained':True}
    original=wm=fw=d=engine=None;sig=device_before=native_forward=old_dt=None;source_paths={};poses=[];vels=[];times=[];native_times=[];actual_times=[]
    try:
        versions={n:importlib.metadata.version(n)for n in ['mujoco','mujoco-warp','warp-lang','mjlab']}
        rk.require(versions=={'mujoco':'3.10.0','mujoco-warp':'3.8.1','warp-lang':'1.12.0','mjlab':'1.3.0'},'Pinned runtime mismatch')
        for p,h in input_hashes.items():rk.require(Path(p).is_file()and not Path(p).is_symlink()and rk.sha(p)==h,'Input mismatch: '+p)
        init=json.loads((Path(e6)/'initialization.json').read_text());original=mujoco.MjModel.from_binary_path(str(model_path));sig=rk.signature(original)
        rk.require((original.nq,original.nv,original.nu,original.na,original.nsensor,original.nplugin)==(7,6,0,0,0,0),'Autonomous fixture only')
        rk.require(original.opt.timestep==.002 and original.opt.integrator==3 and original.opt.disableflags==0,'Original baseline changed')
        m=rk.configure(original,BASE_DT);configured=rk.signature(m);initial=mujoco.MjData(m)
        initial.qpos[:]=init['state']['qpos'];initial.qvel[:]=init['state']['qvel'];initial.time=init['state']['time_s'];initial.qacc_warmstart[:]=init['qacc_warmstart']
        q0=initial.qpos.copy();v0=initial.qvel.copy();t0=initial.time;rk.require(t0==.39 and mujoco.get_mjcb_control()is None,'Unexpected initialization/callback')
        b['setup_forward_attempts']+=1;rk.save(out/'progress.json',b);mujoco.mj_forward(m,initial);b['setup_forward_completed']+=1;rk.save(out/'progress.json',b)
        rk.require(np.array_equal(initial.qpos,q0)and np.array_equal(initial.qvel,v0)and initial.time==t0 and not np.any(initial.warning.number),'Upload forward mutated/warned')
        initial.qacc_warmstart[:]=init['qacc_warmstart']
        import warp as wp,mujoco_warp as mw
        from mujoco_warp._src import forward as fw
        native_forward=fw.forward
        plan=json.loads((Path(__file__).resolve().parents[1]/'docs/audits/stage-11-floor-cuda-refinement-plan.json').read_text())
        source_paths={str(Path(mw.__file__).parent/n):h for n,h in plan['dependency_sources'].items()}
        rk.require(all(rk.sha(p)==h for p,h in source_paths.items()),'Dependency source mismatch');rk.require(rk.sha(fw.__file__)==bridge.FORWARD_SHA,'RK4 source mismatch')
        device=wp.get_device('cuda:0');rk.require(device.is_cuda and '5060'in device.name,'RTX5060 only');wp.set_device(device)
        wm=mw.put_model(m);d=mw.put_data(m,initial,nworld=1,nconmax=256,njmax=1024);wp.synchronize()
        old_dt=float(wm.opt.timestep.numpy()[0]);device_before=bridge.fingerprint(wm);rk.save(out/'device-model-before.json',device_before)
        first=bridge.primary(d);rk.require(np.array_equal(first['qpos'],q0.astype(np.float32))and np.array_equal(first['qvel'],v0.astype(np.float32))and first['native_device_time_s']==float(np.float32(t0)),'Upload primary mismatch')
        rk.require(np.array_equal(d.qacc_warmstart.numpy()[0],np.array(init['qacc_warmstart'],np.float32)),'Upload warmstart mismatch')
        arrays=array_leaves(d,wp);allocated=sum(int(a.capacity)for a in {id(a):a for a in arrays}.values());rk.require(allocated<32*1024*1024,'Unexpected Data allocation footprint')
        rk.save(out/'initialization.json',{'CPU_E6':init,'CUDA':first,'device':device.name,'versions':versions,'data_arrays':len(arrays),'data_array_capacity_bytes':allocated,'warmup_integrations':0,'capture':False,'clock_policy':'Native float32 time cloned/restored with all Data. Physical time is sum of accepted float32 dt only. No native time overwrite. Autonomous fixture only.'})
        poses.append(first['qpos']);vels.append(first['qvel']);times.append(.39);native_times.append(first['native_device_time_s']);actual_times.append(.39)
        with gzip.open(out/'trial-log.jsonl.gz','wt')as log,(out/'states.partial.jsonl').open('x')as partial:
            engine=RefinedStepper(wm,wp,fw,b,out,deadline,log);engine.last_accepted=d
            with bridge.observe_native_forward(fw,engine.rec):
                for i in range(N):
                    d=engine.advance(d,.39+i*BASE_DT,.39+(i+1)*BASE_DT,i);p=bridge.primary(d)
                    poses.append(p['qpos']);vels.append(p['qvel']);times.append(.39+(i+1)*BASE_DT);native_times.append(p['native_device_time_s']);actual_times.append(engine.integrated_time)
                    partial.write(json.dumps({'nominal_time_s':times[-1],'actual_dt_sum_time_s':actual_times[-1],**p},default=rk.serial)+'\n');partial.flush()
        rk.require(b['completed']==b['attempted']==b['accepted_substeps']+b['discarded_trials']and b['completed']==N+2*b['discarded_trials']and b['rk_stages_observed']==4*b['completed'],'Trial accounting mismatch')
        gpu={'qpos':np.array(poses),'qvel':np.array(vels),'time':np.array(times)};report['comparisons']={}
        # These trajectories are deliberately loaded after event decisions end.
        for name,p in [('CUDA_self_G5',Path(g5)/'states.npz'),('CPU_E6',Path(e6)/'states.npz'),('CUDA_fixed_E6',Path(fixed)/'states.npz')]:
            with np.load(p,allow_pickle=False)as z:reference={k:z[k].copy()for k in ['qpos','qvel','time']}
            report['comparisons'][name]=compare(reference,gpu)
        report['all_original_comparisons_pass']=all(c['all_pass']for c in report['comparisons'].values())
        report['status']='CUDA_SELF_SELECTED_G6_GATES_PASS_REVIEW_REQUIRED'if report['all_original_comparisons_pass']else'CUDA_SELF_SELECTED_G6_COMPARISON_BLOCKED_REVIEW_REQUIRED'
        rk.require(rk.signature(m)==configured,'CPU diagnostic model changed')
    except BaseException as e:
        report['status']='CUDA_SELF_SELECTED_G6_STOPPED_REVIEW_REQUIRED';report['error']=repr(e);(out/'error.txt').write_text(traceback.format_exc())
    finally:
        if fw is not None and native_forward is not None:fw.forward=native_forward
        if wm is not None and old_dt is not None:
            wm.opt.timestep.fill_(old_dt);wp.synchronize();after=bridge.fingerprint(wm);rk.save(out/'device-model-after.json',after);report['device_model_restored']=after==device_before
        if engine is not None:
            report.update({'peak_all_trial_internal_endpoint_overlap_m':engine.rec.max_overlap,'peak_accepted_internal_endpoint_overlap_m':engine.max_accepted_overlap,'max_depth':engine.max_depth,'min_accepted_dt_s':engine.min_h if np.isfinite(engine.min_h)else None})
            rk.save(out/'localized-switches.json',engine.localized)
            if engine.rec.rows:rk.save(out/'last-observed-stages.json',engine.rec.rows,allow_nonfinite=True)
            for label,state in [('last-trial',engine.last_trial),('last-accepted',engine.last_accepted)]:
                if state is not None:
                    p=bridge.primary(state);np.savez_compressed(out/(label+'-state.npz'),qpos=p['qpos'],qvel=p['qvel'],native_device_time=p['native_device_time_s'],qacc_warmstart=state.qacc_warmstart.numpy());rk.save(out/(label+'-full-data-fingerprint.json'),bridge.fingerprint(state))
        np.savez_compressed(out/'states.npz',qpos=np.array(poses),qvel=np.array(vels),time=np.array(times),native_device_time=np.array(native_times),actual_dt_sum_time=np.array(actual_times))
        if times:report['max_native_clock_minus_actual_dt_sum_s']=float(np.max(abs(np.array(native_times)-actual_times)));report['max_actual_dt_sum_minus_nominal_s']=float(np.max(abs(np.array(actual_times)-times)))
        report.update(b);report['canonical_completed_lower_bound']=PRIOR+b['completed'];report['canonical_attempted_upper_bound']=PRIOR+b['attempted'];report['wall_seconds']=time.monotonic()-started
        report['inputs_unchanged']=all(Path(p).is_file()and rk.sha(p)==h for p,h in input_hashes.items());report['dependency_sources_unchanged']=all(rk.sha(p)==h for p,h in source_paths.items());report['original_model_unchanged']=original is not None and rk.signature(original)==sig;report['native_forward_restored']=fw is None or fw.forward is native_forward
        if not all(report[k]for k in ['inputs_unchanged','dependency_sources_unchanged','original_model_unchanged','native_forward_restored'])or report.get('device_model_restored')is False:report['status']='SOURCE_OR_HOOK_CHANGED_BLOCKED'
        rk.save(out/'run-review.json',report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['output','model','e6','g5','fixed','input-hashes']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();r=run(a.output,a.model,a.e6,a.g5,a.fixed,json.loads(a.input_hashes.read_text()))
    print('Stage11CUDARefinement='+r['status']);print('Stage11CompletedTrials='+str(r['completed']));print('Stage11AttemptedTrials='+str(r['attempted']))
    raise SystemExit(0 if r['status']in ['CUDA_SELF_SELECTED_G6_GATES_PASS_REVIEW_REQUIRED','CUDA_SELF_SELECTED_G6_COMPARISON_BLOCKED_REVIEW_REQUIRED']else 2)
