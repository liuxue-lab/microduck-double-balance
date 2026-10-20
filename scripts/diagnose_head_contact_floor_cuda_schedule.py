"""One CUDA RK4 rollout on the reviewed E6 accepted-step schedule; no training.

The CPU schedule is frozen input, not a GPU adaptive collision algorithm.
Native device clock and actual float32 timestep sums are recorded separately.
"""
from pathlib import Path
from contextlib import contextmanager
import argparse,dataclasses,gzip,hashlib,importlib.metadata,json,time,traceback,warnings
import numpy as np
import mujoco
import diagnose_head_contact_floor_rk_reference as rk

TOTAL=3244
PRIOR=25413
FORWARD_SHA='c764b6da0b55c05f97b9368f7c77d4826cbafafe93a15f682a878eef7f9e3de3'
WEIGHTS=np.array([1,2,2,1])/6

def schedule_from_rows(rows):
    accepted=[r for r in rows if r['accepted']]
    rk.require(len(rows)==3288 and len(accepted)==TOTAL,'Expected reviewed E6 trial/leaf counts')
    schedule=[];last=.39;counts={i:0 for i in range(3200)}
    for i,r in enumerate(accepted):
        start=r['before']['time_s'];end=r['after']['time_s'];h=r['dt_s'];macro=r['macro_index']
        rk.require(start==last and 0<h<=1/64000+1e-13 and abs(end-start-h)<1e-14,'Non-contiguous or invalid CPU schedule')
        rk.require(macro in counts,'Bad macro index');counts[macro]+=1
        last_of_macro=i==TOTAL-1 or accepted[i+1]['macro_index']!=macro
        if last_of_macro:rk.require(abs(end-(.39+(macro+1)/64000))<1e-12,'CPU output grid changed')
        schedule.append({'index':i,'macro_index':macro,'cpu_trial_id':r['trial_id'],'start_s':start,'end_s':end,'requested_dt_s':h,'actual_float32_dt_s':float(np.float32(h)),'last_of_macro':last_of_macro})
        last=end
    rk.require(abs(last-.44)<1e-12 and all(counts.values())and sum(s['last_of_macro']for s in schedule)==3200,'Incomplete original 50ms schedule')
    rk.require(max(abs(.39+sum(s['actual_float32_dt_s']for s in schedule[:i+1])-s['end_s'])for i,s in enumerate(schedule))<1e-8,'Float32 timestep sum changes comparison time')
    return schedule

def fingerprint(value):
    if hasattr(value,'numpy'):
        a=value.numpy();return {'shape':list(a.shape),'dtype':str(a.dtype),'sha256':hashlib.sha256(a.tobytes()).hexdigest()}
    if dataclasses.is_dataclass(value):return {f.name:fingerprint(getattr(value,f.name))for f in dataclasses.fields(value)}
    if isinstance(value,dict):return {str(k):fingerprint(v)for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [fingerprint(v)for v in value]
    if isinstance(value,(str,int,float,bool,type(None))):return value
    return repr(value)

def primary(d):
    return {'qpos':d.qpos.numpy()[0].copy(),'qvel':d.qvel.numpy()[0].copy(),'native_device_time_s':float(d.time.numpy()[0])}

def guard_state(state):
    rk.require(np.isfinite(state['qpos']).all()and np.isfinite(state['qvel']).all(),'Nonfinite GPU primary state')
    rk.require(float(state['qpos'][2])-.02>=-.001,'Original 1mm physical penetration gate exceeded')

def check_contacts(rows):
    for c in rows:
        rk.require(c['geom'].tolist()==[0,2]and c['dim']==4,'Unexpected head/pair/dimension in floor-only bridge')
        rk.require(np.isfinite(c['dist'])and c['dist']>=-.001,'Nonfinite/excessive contact penetration')
        rk.require(np.array_equal(c['solref'],np.array([.001,1],np.float32)),'Contact solref changed')
        rk.require(np.array_equal(c['solimp'],np.array([.95,.99,.001,.5,2],np.float32)),'Contact solimp changed')
        rk.require(np.array_equal(c['friction'],np.array([.6,.6,.005,.0001,.0001],np.float32)),'Contact friction changed')
        rk.require(np.max(abs(c['frame'][0]-[0,0,1]))<=1e-6,'Floor normal changed')

@contextmanager
def observe_native_forward(fw,observer):
    native=fw.forward
    def wrapped(m,d):
        result=native(m,d);observer(m,d);return result
    fw.forward=wrapped
    try:yield
    finally:fw.forward=native

class Recorder:
    def __init__(self,wp,b,out,deadline):self.wp=wp;self.b=b;self.out=out;self.deadline=deadline;self.rows=[];self.item=None;self.max_overlap=0.
    def begin(self,item):self.rows=[];self.item=item
    def __call__(self,m,d):
        rk.require(time.monotonic()<self.deadline,'600-second wall budget reached')
        self.wp.synchronize();i=len(self.rows);rk.require(i<4 and self.item is not None,'Unexpected native forward phase')
        self.b['rk_stages_observed']+=1;rk.save(self.out/'progress.json',self.b)
        p=primary(d);count=int(d.nacon.numpy()[0]);rk.require(0<=count<=1 and int(d.nefc.numpy()[0])<1024,'Unexpected contact count/capacity')
        fields={k:getattr(d.contact,k).numpy()[:count].copy()for k in ['geom','dist','pos','frame','dim','friction','solref','solimp']}
        contacts=[{k:v[j]for k,v in fields.items()}for j in range(count)]
        row={'stage':i,'state':p,'nominal_stage_time_s':self.item['start_s']+[0,.5,.5,1][i]*self.item['actual_float32_dt_s'],
             'contacts':contacts,'qacc':d.qacc.numpy()[0].copy(),'qfrc_constraint':d.qfrc_constraint.numpy()[0].copy(),'qfrc_smooth':d.qfrc_smooth.numpy()[0].copy()}
        self.rows.append(row);guard_state(p);check_contacts(contacts)
        rk.require(np.isfinite(row['qacc']).all()and np.isfinite(row['qfrc_constraint']).all(),'Nonfinite GPU solve')
        rk.require(not np.any(d.qfrc_applied.numpy())and not np.any(d.xfrc_applied.numpy()),'External force introduced')
        self.max_overlap=max(self.max_overlap,.02-float(p['qpos'][2]),max([-float(c['dist'])for c in contacts]+[0.]))

def step_once(wm,d,fw,wp,rec,b,out,item):
    rk.require(b['attempted']<TOTAL,'3244-call budget exhausted')
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

def comparison(cpu,gpu):
    rk.require(cpu['qpos'].shape==gpu['qpos'].shape==(3201,7)and cpu['qvel'].shape==gpu['qvel'].shape==(3201,6),'Complete original output grids required')
    grid=.39+np.arange(3201)/64000
    rk.require(np.max(abs(cpu['time']-grid))<1e-12 and np.array_equal(gpu['time'],grid),'Shifted or incomplete comparison times')
    dx=np.linalg.norm(cpu['qpos'][:,:3]-gpu['qpos'][:,:3],axis=1);dv=np.linalg.norm(cpu['qvel'][:,:3]-gpu['qvel'][:,:3],axis=1)
    metrics={'peak_overlap':abs(max(0,.02-float(cpu['qpos'][:,2].min()))-max(0,.02-float(gpu['qpos'][:,2].min()))),
      'position':float(dx[::2].max()),'velocity':float(dv[::2].max()),'peak_upward_velocity':abs(max(0,float(cpu['qvel'][:,2].max()))-max(0,float(gpu['qvel'][:,2].max())))}
    checks={k:{'value':v,'limit':rk.LIMITS[k],'pass':v<=rk.LIMITS[k]}for k,v in metrics.items()}
    dense={'max_position_difference_m':float(dx.max()),'max_velocity_difference_m_s':float(dv.max())}
    dense_pass=dense['max_position_difference_m']<=rk.LIMITS['position']and dense['max_velocity_difference_m_s']<=rk.LIMITS['velocity']
    return {'checks':checks,'original_common_samples':1601,'dense_samples':3201,'dense_checks':dense,'all_pass':all(v['pass']for v in checks.values())and dense_pass,'time_shift_s':0,'start_s':.39,'end_s':.44,'formal_acceptance':False}

def run(out,model_path,e6,input_hashes):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();deadline=started+600
    b={'attempted':0,'completed':0,'rk_stages_observed':0,'setup_forward_attempts':0,'setup_forward_completed':0,'PPO':0,'prior_canonical_steps':PRIOR};rk.save(out/'progress.json',b)
    report={'status':'STARTED','formal_physics_adoption':False,'formal_task_migration':False,'stage11_complete':False,'PPO':0,'cloud_hours':0,'full_robot_steps':0,'not_GPU_adaptive_event_algorithm':True,'original_2ms_floor_block_retained':True}
    original=wm=rec=fw=d=None;sig=device_before=None;native_forward=None;poses=[];vels=[];clocks=[];native_times=[];actual_times=[];source_paths={};versions={};old_dt=None
    try:
        versions={n:importlib.metadata.version(n)for n in ['mujoco','mujoco-warp','warp-lang','mjlab']}
        rk.require(versions=={'mujoco':'3.10.0','mujoco-warp':'3.8.1','warp-lang':'1.12.0','mjlab':'1.3.0'},'Pinned runtime versions differ')
        for p,h in input_hashes.items():rk.require(Path(p).is_file()and not Path(p).is_symlink()and rk.sha(p)==h,'Input mismatch: '+p)
        with gzip.open(Path(e6)/'trial-log.jsonl.gz','rt')as f:schedule=schedule_from_rows([json.loads(line)for line in f])
        rk.save(out/'frozen-schedule.json',schedule)
        with np.load(Path(e6)/'states.npz',allow_pickle=False)as z:cpu={k:z[k].copy()for k in ['qpos','qvel','time']}
        init=json.loads((Path(e6)/'initialization.json').read_text())
        original=mujoco.MjModel.from_binary_path(str(model_path));sig=rk.signature(original)
        rk.require((original.nq,original.nv,original.nu,original.na,original.nsensor,original.nplugin)==(7,6,0,0,0,0),'Only autonomous fixed sphere fixture supported')
        rk.require(original.opt.timestep==.002 and original.opt.integrator==3 and original.opt.disableflags==0,'Original physics baseline changed')
        model=rk.configure(original,1/64000);model_sig=rk.signature(model);initial=mujoco.MjData(model)
        initial.qpos[:]=init['state']['qpos'];initial.qvel[:]=init['state']['qvel'];initial.time=init['state']['time_s'];initial.qacc_warmstart[:]=init['qacc_warmstart']
        q0=initial.qpos.copy();v0=initial.qvel.copy();t0=initial.time
        rk.require(mujoco.get_mjcb_control()is None,'Unexpected CPU control callback')
        b['setup_forward_attempts']+=1;rk.save(out/'progress.json',b);mujoco.mj_forward(model,initial);b['setup_forward_completed']+=1;rk.save(out/'progress.json',b)
        rk.require(np.array_equal(initial.qpos,q0)and np.array_equal(initial.qvel,v0)and initial.time==t0 and not np.any(initial.warning.number),'CPU upload forward mutated state or warned')
        initial.qacc_warmstart[:]=init['qacc_warmstart']
        import warp as wp,mujoco_warp as mw
        from mujoco_warp._src import forward as fw
        native_forward=fw.forward
        source_paths={str(Path(mw.__file__).parent/n):h for n,h in json.loads((Path(__file__).resolve().parents[1]/'docs/audits/stage-11-floor-cuda-schedule-plan.json').read_text())['dependency_sources'].items()}
        rk.require(all(rk.sha(p)==h for p,h in source_paths.items()),'Installed dependency source differs from reviewed runtime')
        rk.require(rk.sha(fw.__file__)==FORWARD_SHA,'Unexpected native RK4 implementation')
        device=wp.get_device('cuda:0');rk.require(device.is_cuda and '5060'in device.name,'Use canonical laptop RTX5060 only')
        wp.set_device(device);wm=mw.put_model(model);d=mw.put_data(model,initial,nworld=1,nconmax=256,njmax=1024);wp.synchronize()
        old_dt=float(wm.opt.timestep.numpy()[0]);device_before=fingerprint(wm);rk.save(out/'device-model-before.json',device_before)
        first=primary(d);rk.require(np.array_equal(first['qpos'],q0.astype(np.float32))and np.array_equal(first['qvel'],v0.astype(np.float32))and first['native_device_time_s']==float(np.float32(t0)),'Unexpected upload state')
        rk.require(np.array_equal(d.qacc_warmstart.numpy()[0],np.array(init['qacc_warmstart'],np.float32)),'Warmstart upload changed')
        rk.save(out/'initialization.json',{'CPU_E6':init,'CUDA':first,'qacc_warmstart':d.qacc_warmstart.numpy(),'device':device.name,'versions':versions,'capture':False,'warmup_steps':0,'native_stage_clock_policy':'Pinned RK4 source does not perturb native time between stages; this fixture is autonomous. Save native clock, compare using cumulative actual integration dt, no native clock overwrite.'})
        poses.append(first['qpos']);vels.append(first['qvel']);clocks.append(.39);native_times.append(first['native_device_time_s']);actual_times.append(.39)
        rec=Recorder(wp,b,out,deadline);integrated_time=.39
        with gzip.open(out/'stage-log.jsonl.gz','wt')as log,(out/'states.partial.jsonl').open('x')as partial,observe_native_forward(fw,rec):
            for item in schedule:
                row=step_once(wm,d,fw,wp,rec,b,out,item);integrated_time+=item['actual_float32_dt_s']
                row['actual_cumulative_dt_time_s']=integrated_time;log.write(json.dumps(row,default=rk.serial,allow_nan=False)+'\n');log.flush()
                rk.require(abs(integrated_time-item['end_s'])<1e-8,'Actual dt sum diverged from original time grid')
                if item['last_of_macro']:
                    p=row['after'];poses.append(p['qpos']);vels.append(p['qvel']);clocks.append(.39+(item['macro_index']+1)/64000);native_times.append(p['native_device_time_s']);actual_times.append(integrated_time)
                    partial.write(json.dumps({'nominal_time_s':clocks[-1],'actual_cumulative_dt_time_s':integrated_time,**p},default=rk.serial)+'\n');partial.flush()
        rk.require(b['completed']==b['attempted']==TOTAL and b['rk_stages_observed']==4*TOTAL,'Incomplete schedule count')
        gpu={'qpos':np.array(poses),'qvel':np.array(vels),'time':np.array(clocks)};report['comparison']=comparison(cpu,gpu)
        report['peak_any_internal_or_endpoint_overlap_m']=rec.max_overlap
        report['max_native_clock_minus_integrated_time_s']=float(np.max(abs(np.array(native_times)-actual_times)))
        report['max_actual_dt_sum_minus_nominal_time_s']=float(np.max(abs(np.array(actual_times)-clocks)))
        report['status']='CUDA_SCHEDULE_REFERENCE_GATES_PASS_REVIEW_REQUIRED'if report['comparison']['all_pass']else'CUDA_SCHEDULE_REFERENCE_BLOCKED_REVIEW_REQUIRED'
        rk.require(rk.signature(model)==model_sig,'CPU diagnostic model changed')
    except BaseException as e:
        report['status']='CUDA_SCHEDULE_STOPPED_REVIEW_REQUIRED';report['error']=repr(e);(out/'error.txt').write_text(traceback.format_exc())
    finally:
        if fw is not None and native_forward is not None:fw.forward=native_forward
        if wm is not None and old_dt is not None:
            wm.opt.timestep.fill_(old_dt);wp.synchronize();after=fingerprint(wm);rk.save(out/'device-model-after.json',after);report['device_model_restored']=after==device_before
        if rec is not None and rec.rows:rk.save(out/'last-observed-stages.json',rec.rows,allow_nonfinite=True)
        if d is not None:
            last=primary(d);np.savez_compressed(out/'last-primary-state.npz',qpos=last['qpos'],qvel=last['qvel'],native_device_time=last['native_device_time_s'],qacc_warmstart=d.qacc_warmstart.numpy())
        np.savez_compressed(out/'states.npz',qpos=np.array(poses),qvel=np.array(vels),time=np.array(clocks),native_device_time=np.array(native_times),actual_dt_sum_time=np.array(actual_times))
        report.update(b);report['canonical_completed_lower_bound']=PRIOR+b['completed'];report['canonical_attempted_upper_bound']=PRIOR+b['attempted'];report['wall_seconds']=time.monotonic()-started
        report['inputs_unchanged']=all(Path(p).is_file()and rk.sha(p)==h for p,h in input_hashes.items());report['dependency_sources_unchanged']=all(rk.sha(p)==h for p,h in source_paths.items())
        report['original_model_unchanged']=original is not None and rk.signature(original)==sig;report['native_forward_restored']=fw is None or fw.forward is native_forward
        if not report['inputs_unchanged']or not report['dependency_sources_unchanged']or not report['native_forward_restored']or not report['original_model_unchanged']or report.get('device_model_restored')is False:report['status']='SOURCE_OR_HOOK_CHANGED_BLOCKED'
        rk.save(out/'run-review.json',report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['output','model','e6','input-hashes']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();r=run(a.output,a.model,a.e6,json.loads(a.input_hashes.read_text()))
    print('Stage11CUDASchedule='+r['status']);print('Stage11CompletedCalls='+str(r['completed']));print('Stage11AttemptedCalls='+str(r['attempted']))
    raise SystemExit(0 if r['status']in ['CUDA_SCHEDULE_REFERENCE_GATES_PASS_REVIEW_REQUIRED','CUDA_SCHEDULE_REFERENCE_BLOCKED_REVIEW_REQUIRED']else 2)
