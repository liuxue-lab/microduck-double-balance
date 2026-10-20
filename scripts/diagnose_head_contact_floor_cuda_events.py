"""One autonomous CUDA contact-switch subdivision reference, no training.

GPU contacts choose the tree; archived trajectories are read only after rollout.
Every rejected trial is counted, and every Warp Data field is copied/checked.
"""
from pathlib import Path
import argparse,copy,dataclasses,gzip,hashlib,importlib.metadata,json,time,traceback
import numpy as np
import mujoco
import diagnose_head_contact_floor_rk_reference as rk
import diagnose_head_contact_floor_cuda_schedule as bridge

TOTAL=2400
PRIOR=28657
BASE_DT=1/32000
N=1600
EVENT_WIDTH=1e-8
MAX_DEPTH=24
RADIUS32=float(np.float32(.02))

def digest_data(d):
    return hashlib.sha256(json.dumps(bridge.fingerprint(d),sort_keys=True,allow_nan=False).encode()).hexdigest()

def array_leaves(v,wp):
    if isinstance(v,wp.array):return [v]
    if dataclasses.is_dataclass(v):return [a for f in dataclasses.fields(v)for a in array_leaves(getattr(v,f.name),wp)]
    if isinstance(v,(list,tuple)):return [a for x in v for a in array_leaves(x,wp)]
    if isinstance(v,dict):return [a for x in v.values()for a in array_leaves(x,wp)]
    rk.require(v is None or isinstance(v,(int,float,bool,str)),'Unsupported mutable Warp Data field')
    return []

def clone_data(d,wp):
    memo={}
    def clone(v):
        if id(v)in memo:return memo[id(v)]
        if isinstance(v,wp.array):result=wp.clone(v);memo[id(v)]=result;return result
        if dataclasses.is_dataclass(v):
            names={f.name for f in dataclasses.fields(v)}
            rk.require(set(vars(v))==names,'Undeclared dynamic Data attributes')
            result=copy.copy(v);memo[id(v)]=result
            for n in names:setattr(result,n,clone(getattr(v,n)))
            return result
        if isinstance(v,list):result=[clone(x)for x in v]
        elif isinstance(v,tuple):result=tuple(clone(x)for x in v)
        elif isinstance(v,dict):result={k:clone(x)for k,x in v.items()}
        else:
            rk.require(v is None or isinstance(v,(int,float,bool,str)),'Unsupported mutable Warp Data field');return v
        memo[id(v)]=result;return result
    original=array_leaves(d,wp)
    # Distinct overlapping views would need view-aware copying. This fixture's
    # pinned make_data/put_data allocates independent arrays; otherwise stop.
    regions=[]
    for a in {id(a):a for a in original}.values():
        if a.size:
            rk.require(a.is_contiguous,'Noncontiguous Data array unsupported')
            start=int(a.ptr);end=start+int(a.capacity)
            rk.require(all(end<=lo or start>=hi for lo,hi in regions),'Aliased Data views unsupported');regions.append((start,end))
    result=clone(d);wp.synchronize()
    new=array_leaves(result,wp)
    rk.require(len(original)==len(new),'Data structure copy mismatch')
    rk.require(all(not a.size or a.ptr!=b.ptr for a,b in zip(original,new)),'Cloned Data shares mutable storage')
    rk.require(digest_data(result)==digest_data(d),'Full Data clone content mismatch')
    return result

def switches(row):
    # Pinned CUDA write_contact uses dist < margin (strict), unlike CPU's
    # zero-gap inclusion. Preserve the native float32 rule without epsilon.
    contacts=[bool(s['contacts'])for s in row['stages']]+[row['after']['qpos'][2]<RADIUS32]
    forces=[s['qfrc_constraint'][2]>0 for s in row['stages']]
    return {'geometry_switch':any(contacts)and not all(contacts),'force_switch':any(forces)and not all(forces),
            'contact_mask_including_endpoint':contacts,'stage_positive_normal_force':forces}

def decide(row):
    flags=switches(row);changing=flags['geometry_switch']or flags['force_switch']
    h=row['schedule']['actual_float32_dt_s'];indicator=float(h*max(np.linalg.norm(s['qacc'][:3])for s in row['stages']))
    return (not changing)or(h<=EVENT_WIDTH and indicator<=1e-4),flags,indicator

class Recorder(bridge.Recorder):
    def __call__(self,m,d):
        super().__call__(m,d)
        self.rows[-1]['solver_niter']=d.solver_niter.numpy()[0].item()
        self.rows[-1]['qacc_warmstart']=d.qacc_warmstart.numpy()[0].copy()
        st=self.rows[-1]
        rk.require(bool(st['contacts'])==(st['state']['qpos'][2]<RADIUS32),'Native float32 plane/sphere classification differs')

class EventStepper:
    def __init__(self,wm,wp,fw,b,out,deadline,log):
        self.m=wm;self.wp=wp;self.fw=fw;self.b=b;self.out=out;self.deadline=deadline;self.log=log
        self.rec=Recorder(wp,b,out,deadline);self.last_trial=None;self.last_accepted=None
        self.integrated_time=.39;self.max_accepted_overlap=0.;self.max_depth=0;self.min_h=float('inf');self.localized=[]

    def trial(self,d,start,end,macro,depth,path):
        rk.require(self.b['attempted']<TOTAL,'2400 CUDA trial budget exhausted')
        rk.require(time.monotonic()<self.deadline,'Wall budget reached before copy')
        h=float(np.float32(end-start));rk.require(0<h<=float(np.float32(BASE_DT))+1e-13,'Invalid local timestep')
        original=digest_data(d);td=clone_data(d,self.wp);self.last_trial=td
        item={'index':self.b['attempted'],'macro_index':macro,'start_s':start,'end_s':end,'requested_dt_s':end-start,'actual_float32_dt_s':h}
        try:row=bridge.step_once(self.m,td,self.fw,self.wp,self.rec,self.b,self.out,item)
        finally:
            rk.require(digest_data(d)==original,'Trial mutated original full Warp Data')
        row.update({'trial_id':self.b['attempted'],'before_full_data_sha256':original,'after_full_data_sha256':digest_data(td),'source_full_data_unchanged':True,'depth':depth,'tree_path':path})
        return td,row

    def advance(self,d,start,end,macro,depth=0,path=''):
        rk.require(depth<=MAX_DEPTH,'Subdivision depth budget exhausted')
        td,row=self.trial(d,start,end,macro,depth,path);accepted,flags,indicator=decide(row);h=row['schedule']['actual_float32_dt_s']
        row.update(flags);row.update({'accepted':accepted,'sampled_acceleration_times_width_m_s':indicator})
        self.max_depth=max(self.max_depth,depth)
        self.log.write(json.dumps(row,default=rk.serial,allow_nan=False)+'\n');self.log.flush()
        if accepted:
            self.b['accepted_substeps']+=1;self.last_accepted=td;self.integrated_time+=h;self.min_h=min(self.min_h,h)
            self.max_accepted_overlap=max(self.max_accepted_overlap,.02-row['after']['qpos'][2],max(.02-s['state']['qpos'][2]for s in row['stages']),max([-c['dist']for s in row['stages']for c in s['contacts']]+[0.]))
            rk.require(abs(self.integrated_time-end)<1e-8,'Accepted dt sum diverged from original physical time')
            if flags['geometry_switch']or flags['force_switch']:
                self.b['localized_switch_intervals']+=1
                self.localized.append({'trial_id':row['trial_id'],'start_s':start,'end_s':end,'dt_s':h,**flags,'sampled_acceleration_times_width_m_s':indicator})
            rk.save(self.out/'progress.json',self.b);return td
        self.b['discarded_trials']+=1;rk.save(self.out/'progress.json',self.b)
        rk.require(depth<MAX_DEPTH,'Unresolved contact switch at depth limit')
        mid=start+(end-start)/2;rk.require(start<mid<end,'Host subdivision clock stagnated')
        left=self.advance(d,start,mid,macro,depth+1,path+'L')
        return self.advance(left,mid,end,macro,depth+1,path+'R')

def compare(reference,gpu):
    rg=reference['time'];gg=gpu['time'];grid=.39+np.arange(N+1)*BASE_DT
    rk.require(gpu['qpos'].shape==(N+1,7)and gpu['qvel'].shape==(N+1,6)and np.array_equal(gg,grid),'Incomplete GPU 50ms grid')
    rk.require(reference['qpos'].shape[0]in [1601,3201],'Unsupported archived grid')
    stride=(len(rg)-1)//N
    rk.require(np.max(abs(rg[::stride]-grid))<1e-12,'Shifted archived reference grid')
    rp=reference['qpos'];rv=reference['qvel'];gp=gpu['qpos'];gv=gpu['qvel']
    metrics={'peak_overlap':abs(max(0,.02-float(rp[:,2].min()))-max(0,.02-float(gp[:,2].min()))),
      'position':float(np.linalg.norm(rp[::stride,:3]-gp[:,:3],axis=1).max()),'velocity':float(np.linalg.norm(rv[::stride,:3]-gv[:,:3],axis=1).max()),
      'peak_upward_velocity':abs(max(0,float(rv[:,2].max()))-max(0,float(gv[:,2].max())))}
    return {'checks':{k:{'value':v,'limit':rk.LIMITS[k],'pass':v<=rk.LIMITS[k]}for k,v in metrics.items()},'all_pass':all(v<=rk.LIMITS[k]for k,v in metrics.items()),'common_samples':1601,'reference_peak_samples':len(rg),'GPU_peak_samples':1601,'time_shift_s':0,'full_window_s':[.39,.44]}

def run(out,model_path,e6,e5,fixed,input_hashes):
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
        plan=json.loads((Path(__file__).resolve().parents[1]/'docs/audits/stage-11-floor-cuda-events-plan.json').read_text())
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
            engine=EventStepper(wm,wp,fw,b,out,deadline,log);engine.last_accepted=d
            with bridge.observe_native_forward(fw,engine.rec):
                for i in range(N):
                    d=engine.advance(d,.39+i*BASE_DT,.39+(i+1)*BASE_DT,i);p=bridge.primary(d)
                    poses.append(p['qpos']);vels.append(p['qvel']);times.append(.39+(i+1)*BASE_DT);native_times.append(p['native_device_time_s']);actual_times.append(engine.integrated_time)
                    partial.write(json.dumps({'nominal_time_s':times[-1],'actual_dt_sum_time_s':actual_times[-1],**p},default=rk.serial)+'\n');partial.flush()
        rk.require(b['completed']==b['attempted']==b['accepted_substeps']+b['discarded_trials']and b['completed']==N+2*b['discarded_trials']and b['rk_stages_observed']==4*b['completed'],'Trial accounting mismatch')
        gpu={'qpos':np.array(poses),'qvel':np.array(vels),'time':np.array(times)};report['comparisons']={}
        # These trajectories are deliberately loaded after event decisions end.
        for name,p in [('CPU_E5',Path(e5)/'states.npz'),('CPU_E6',Path(e6)/'states.npz'),('CUDA_fixed_E6',Path(fixed)/'states.npz')]:
            with np.load(p,allow_pickle=False)as z:reference={k:z[k].copy()for k in ['qpos','qvel','time']}
            report['comparisons'][name]=compare(reference,gpu)
        report['all_original_comparisons_pass']=all(c['all_pass']for c in report['comparisons'].values())
        report['status']='CUDA_SELF_SELECTED_EVENTS_GATES_PASS_REVIEW_REQUIRED'if report['all_original_comparisons_pass']else'CUDA_SELF_SELECTED_EVENTS_COMPARISON_BLOCKED_REVIEW_REQUIRED'
        rk.require(rk.signature(m)==configured,'CPU diagnostic model changed')
    except BaseException as e:
        report['status']='CUDA_SELF_SELECTED_EVENTS_STOPPED_REVIEW_REQUIRED';report['error']=repr(e);(out/'error.txt').write_text(traceback.format_exc())
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
    for n in ['output','model','e6','e5','fixed','input-hashes']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();r=run(a.output,a.model,a.e6,a.e5,a.fixed,json.loads(a.input_hashes.read_text()))
    print('Stage11CUDAEvents='+r['status']);print('Stage11CompletedTrials='+str(r['completed']));print('Stage11AttemptedTrials='+str(r['attempted']))
    raise SystemExit(0 if r['status']in ['CUDA_SELF_SELECTED_EVENTS_GATES_PASS_REVIEW_REQUIRED','CUDA_SELF_SELECTED_EVENTS_COMPARISON_BLOCKED_REVIEW_REQUIRED']else 2)
