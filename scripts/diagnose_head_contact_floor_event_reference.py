"""Native RK4 reference with local subdivision at observed contact switches.

Only an isolated CPU fixture is used. All mj_step trials, including discarded
parents, are counted. No geometric projection, altered margin or new force law.
"""
from pathlib import Path
import argparse, copy, gzip, hashlib, json, time, traceback
import numpy as np
import mujoco
import diagnose_head_contact_floor_rk_reference as rk

CASES=(('E5',1/32000,1600),('E6',1/64000,3200))
TOTAL=6400
EVENT_WIDTH=1e-8
MAX_DEPTH=24
PRIOR=20429
STATE_SPEC=mujoco.mjtState.mjSTATE_INTEGRATION

def snapshot(m,d):
    x=np.empty(mujoco.mj_stateSize(m,STATE_SPEC));mujoco.mj_getState(m,d,x,STATE_SPEC)
    return x

def clone_data(m,d):
    trial=mujoco.MjData(m);mujoco.mj_copyData(trial,m,d)
    rk.require(np.array_equal(snapshot(m,trial),snapshot(m,d)),'Full integration state copy mismatch')
    return trial

def switches(row):
    # Native plane/sphere geometry includes equality at zero margin. The
    # endpoint has no current forward result, so only its primary gap is used.
    contact=[bool(s['contacts'])for s in row['stages']]+[row['post_gap_m']<=0]
    force=[s['solver_result']['qfrc_constraint'][2]>0.0 for s in row['stages']]
    return {'geometry_switch':any(contact)and not all(contact),
            'force_switch':any(force)and not all(force),
            'contact_mask_including_endpoint':contact,'stage_positive_normal_force':force}

class EventStepper:
    def __init__(self,m,budget,out,deadline,log):
        self.m=m;self.b=budget;self.out=out;self.deadline=deadline;self.log=log
        self.trace=rk.StageTrace(budget,out,deadline)
        self.last_accepted=None;self.last_trial=None
        self.max_all_trial_overlap=0.;self.max_accepted_overlap=0.
        self.min_accepted_dt=float('inf');self.max_depth=0;self.localized=[]

    def trial(self,d,end):
        rk.require(time.monotonic()<self.deadline,'Internal wall budget reached')
        rk.require(self.b['attempted']<TOTAL,'6400 mj_step trial budget exhausted')
        original=snapshot(self.m,d);td=clone_data(self.m,d);self.last_trial=td
        h=end-float(d.time);rk.require(0<h<=1/32000+1e-14,'Invalid trial interval')
        self.m.opt.timestep=h;before=rk.state(td);warnings=td.warning.number.copy()
        rk.state_guard(td);self.trace.begin(td.time,h)
        self.b['attempted']+=1;rk.save(self.out/'progress.json',self.b)
        try:
            mujoco.mj_step(self.m,td)
            self.b['completed']+=1;rk.save(self.out/'progress.json',self.b)
        finally:
            rk.require(np.array_equal(snapshot(self.m,d),original),'Trial mutated its original full integration state')
        rk.state_guard(td);rk.require(np.array_equal(td.warning.number,warnings),'New MuJoCo warning')
        rk.require(abs(td.time-end)<1e-13,'Trial did not reach requested endpoint')
        identities=self.trace.finish(td,before)
        for s in self.trace.rows:
            rk.require(s['solver_result']['qfrc_constraint'][2]>=-1e-12,'Unexpected tensile floor force')
            rk.require(bool(s['contacts'])==(s['gap_m']<=0),'Plane/sphere event classification differs from native geometry')
        row={'trial_id':self.b['attempted'],'dt_s':h,'before':before,'after':rk.state(td),
             'before_full_integration_state_sha256':hashlib.sha256(original.tobytes()).hexdigest(),
             'original_full_integration_state_unchanged':True,'stages':self.trace.rows,
             'post_gap_m':rk.gap(td),'identities':identities}
        return td,row

    def advance(self,d,end,macro,depth=0,path=''):
        rk.require(depth<=MAX_DEPTH,'Event subdivision depth budget reached')
        td,row=self.trial(d,end);h=row['dt_s'];flags=switches(row)
        changing=flags['geometry_switch']or flags['force_switch']
        acceleration=max(np.linalg.norm(s['solver_result']['qacc'][:3])for s in row['stages'])
        indicator=float(h*acceleration)
        # This sampled acceleration indicator is an extra local check, not a
        # certified global error bound. The original full-window pair gates stay.
        leaf=(not changing)or(h<=EVENT_WIDTH and indicator<=1e-4)
        row.update(flags);row.update({'macro_index':macro,'depth':depth,'tree_path':path,
                     'accepted':leaf,'sampled_acceleration_times_width_m_s':indicator})
        overlap=max(0.,-row['post_gap_m'],max(-s['gap_m']for s in row['stages']))
        self.max_all_trial_overlap=max(self.max_all_trial_overlap,overlap);self.max_depth=max(self.max_depth,depth)
        self.log.write(json.dumps(row,default=rk.serial,allow_nan=False)+'\n');self.log.flush()
        if leaf:
            self.b['accepted_substeps']+=1;self.last_accepted=td
            self.max_accepted_overlap=max(self.max_accepted_overlap,overlap);self.min_accepted_dt=min(self.min_accepted_dt,h)
            if changing:
                self.b['localized_switch_intervals']+=1
                self.localized.append({'trial_id':row['trial_id'],'start_s':row['before']['time_s'],
                    'end_s':row['after']['time_s'],'width_s':h,'geometry_switch':flags['geometry_switch'],
                    'force_switch':flags['force_switch'],'sampled_acceleration_times_width_m_s':indicator})
            rk.save(self.out/'progress.json',self.b)
            return td
        self.b['discarded_trials']+=1;rk.save(self.out/'progress.json',self.b)
        rk.require(depth<MAX_DEPTH,'Cannot resolve switch inside depth budget')
        mid=float(d.time)+h/2
        rk.require(float(d.time)<mid<end,'Subdivision cannot advance floating-point clock')
        left=self.advance(d,mid,macro,depth+1,path+'L')
        return self.advance(left,end,macro,depth+1,path+'R')

def run(out,model_path,cases,rk_cases,input_hashes):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();deadline=started+300
    b={'attempted':0,'completed':0,'rk_stages_observed':0,'accepted_substeps':0,'discarded_trials':0,
       'localized_switch_intervals':0,'PPO':0,'prior_canonical_steps':PRIOR}
    rk.save(out/'progress.json',b)
    report={'status':'STARTED','cases':[],'PPO':0,'full_robot_steps':0,'cloud_hours':0,
            'stage11_complete':False,'formal_physics_adoption':False,'original_2ms_floor_block_retained':True,
            'historical_C5_C6_and_R5_R6_failure_retained':True}
    previous=mujoco.get_mjcb_control();original=None;sig=None;engine=None;series=[]
    try:
        rk.require(previous is None,'Existing global control callback; preserve and stop')
        rk.require(mujoco.__version__=='3.10.0','Requires archived MuJoCo 3.10.0')
        for p,h in input_hashes.items():rk.require(Path(p).is_file()and not Path(p).is_symlink()and rk.sha(p)==h,'Input mismatch: '+p)
        original=mujoco.MjModel.from_binary_path(str(model_path));sig=rk.signature(original);rk.save(out/'original-model-signature.json',sig)
        rk.require((original.nq,original.nv,original.nu,original.na,original.nplugin)==(7,6,0,0,0),'Wrong fixture')
        rk.require(original.opt.integrator==3 and original.opt.solver==2 and original.opt.timestep==.002 and original.opt.disableflags==0,'Original numerical settings changed')
        rk.require(original.opt.iterations==10 and original.opt.tolerance==1e-8 and original.opt.ls_iterations==20 and original.opt.ls_tolerance==.01,'Solver changed')
        rk.require(np.array_equal(original.opt.gravity,[0,0,-9.81])and original.body_mass[2]==.01,'Gravity/mass changed')
        rk.require(np.all(original.geom_margin==0)and np.all(original.geom_gap==0),'Nonzero margin/gap unsupported')
        init=json.loads((Path(cases)/'C5/initialization.json').read_text())
        rk.require(init==json.loads((Path(cases)/'C6/initialization.json').read_text()),'Old initializations differ')
        for name,base_dt,n in CASES:
            dest=out/name;dest.mkdir();m=rk.configure(original,base_dt);configured=rk.signature(m);d=mujoco.MjData(m)
            d.qpos[:]=init['qpos'];d.qvel[:]=init['qvel'];d.qacc_warmstart[:]=init['qacc_warmstart'];d.time=init['time_s']
            rk.require(d.time==.39,'Start time changed')
            rk.save(dest/'initialization.json',{'state':rk.state(d),'qacc_warmstart':d.qacc_warmstart.copy(),
                'complete_historical_state_restore':False,'trial_rollback':'mj_copyData entire MjData, checked with mjSTATE_INTEGRATION'})
            poses=[d.qpos.copy()];vels=[d.qvel.copy()];times=[d.time];base_counts=b.copy()
            summary={'name':name,'base_dt':base_dt,'planned_output_intervals':n,'horizon_completed':False,
                     'peak_endpoint_overlap_m':0.,'peak_endpoint_upward_velocity_m_s':0.}
            report['cases'].append(summary);engine=None
            try:
                with gzip.open(dest/'trial-log.jsonl.gz','wt')as log,(dest/'states.partial.jsonl').open('x')as partial:
                    engine=EventStepper(m,b,out,deadline,log);engine.last_accepted=d
                    mujoco.set_mjcb_control(engine.trace)
                    partial.write(json.dumps(rk.state(d),default=rk.serial)+'\n');partial.flush()
                    for i in range(n):
                        rk.require(abs(d.time-(.39+i*base_dt))<1e-12,'Original output grid changed')
                        d=engine.advance(d,.39+(i+1)*base_dt,i)
                        poses.append(d.qpos.copy());vels.append(d.qvel.copy());times.append(d.time)
                        partial.write(json.dumps(rk.state(d),default=rk.serial)+'\n');partial.flush()
                        summary['peak_endpoint_overlap_m']=max(summary['peak_endpoint_overlap_m'],-rk.gap(d))
                        summary['peak_endpoint_upward_velocity_m_s']=max(summary['peak_endpoint_upward_velocity_m_s'],float(d.qvel[2]))
                    summary['horizon_completed']=True
            finally:
                mujoco.set_mjcb_control(previous);m.opt.timestep=base_dt
                summary['model_only_declared_numerical_changes']=rk.signature(m)==configured
                for k in ['attempted','completed','accepted_substeps','discarded_trials','rk_stages_observed','localized_switch_intervals']:summary[k]=b[k]-base_counts[k]
                summary['output_states_saved']=len(poses)
                if engine is not None:
                    summary.update({'peak_all_trial_overlap_m':engine.max_all_trial_overlap,'peak_accepted_internal_overlap_m':engine.max_accepted_overlap,
                       'min_accepted_dt_s':engine.min_accepted_dt if np.isfinite(engine.min_accepted_dt)else None,'max_depth':engine.max_depth})
                    rk.save(dest/'localized-switches.json',engine.localized)
                    if engine.last_accepted is not None:
                        ld=engine.last_accepted;np.savez_compressed(dest/'last-accepted-state.npz',full_integration_state=snapshot(m,ld),qpos=ld.qpos,qvel=ld.qvel,time=ld.time)
                    if engine.last_trial is not None:
                        td=engine.last_trial;np.savez_compressed(dest/'last-trial-state.npz',full_integration_state=snapshot(m,td),qpos=td.qpos,qvel=td.qvel,time=td.time)
                    if engine.trace.rows:rk.save(dest/'last-observed-stages.json',engine.trace.rows,allow_nonfinite=True)
                np.savez_compressed(dest/'states.npz',qpos=np.array(poses),qvel=np.array(vels),time=np.array(times),dt=base_dt)
                rk.save(dest/'summary.json',summary)
            rk.require(summary['model_only_declared_numerical_changes'],'Undeclared model change')
            rk.require(summary['accepted_substeps']+summary['discarded_trials']==summary['completed'],'Trial accounting mismatch')
            series.append({'qpos':np.array(poses),'qvel':np.array(vels),'time':np.array(times)})
            rk.save(out/'run-review.json',report)
        report['comparison']=rk.compare(*series,*report['cases'])
        old_series=[];old_summaries=[];differences=[]
        for ix,old_name in enumerate(['R5','R6']):
            with np.load(Path(rk_cases)/old_name/'states.npz',allow_pickle=False)as z:a={k:z[k].copy()for k in ['qpos','qvel','time']}
            old_series.append(a);old_summaries.append(json.loads((Path(rk_cases)/old_name/'summary.json').read_text()))
            rk.require(np.max(abs(a['time']-series[ix]['time']))<1e-10,'Old/new physical grids differ')
            differences.append({'old_case':old_name,'new_case':CASES[ix][0],
                'max_position_difference_m':float(np.linalg.norm(a['qpos'][:,:3]-series[ix]['qpos'][:,:3],axis=1).max()),
                'max_linear_velocity_difference_m_s':float(np.linalg.norm(a['qvel'][:,:3]-series[ix]['qvel'][:,:3],axis=1).max())})
        report['old_R5_R6_recomputed_without_integration']=rk.compare(*old_series,*old_summaries)
        rk.require(not report['old_R5_R6_recomputed_without_integration']['all_four_pass'],'Historical failure unexpectedly disappeared')
        report['same_grid_old_new_differences']=differences
        report['status']='EVENT_REFERENCE_PAIR_GATES_PASS_REVIEW_REQUIRED'if report['comparison']['all_four_pass']else'EVENT_REFERENCE_SENSITIVITY_BLOCKED_REVIEW_REQUIRED'
    except BaseException as e:
        report['status']='EVENT_REFERENCE_STOPPED_REVIEW_REQUIRED';report['error']=repr(e);(out/'error.txt').write_text(traceback.format_exc())
    finally:
        mujoco.set_mjcb_control(previous)
        report.update(b);report['canonical_completed_lower_bound']=PRIOR+b['completed'];report['canonical_attempted_upper_bound']=PRIOR+b['attempted']
        report['step_count_exact']=b['completed']==b['attempted'];report['wall_seconds']=time.monotonic()-started
        report['inputs_unchanged']=all(Path(p).is_file()and rk.sha(p)==h for p,h in input_hashes.items())
        report['original_model_unchanged']=original is not None and rk.signature(original)==sig
        report['control_callback_restored']=mujoco.get_mjcb_control()is previous
        if not report['inputs_unchanged']or not report['original_model_unchanged']or not report['control_callback_restored']:report['status']='SOURCE_OR_CALLBACK_CHANGED_BLOCKED'
        rk.save(out/'run-review.json',report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['output','model','cases','rk-cases','input-hashes']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();r=run(a.output,a.model,a.cases,a.rk_cases,json.loads(a.input_hashes.read_text()))
    print('Stage11EventReference='+r['status']);print('Stage11CompletedTrialCalls='+str(r['completed']));print('Stage11AttemptedTrialCalls='+str(r['attempted']))
    print('Stage11AcceptedSubsteps='+str(r['accepted_substeps']));print('Stage11DiscardedTrials='+str(r['discarded_trials']))
    raise SystemExit(0 if r['status']in ['EVENT_REFERENCE_PAIR_GATES_PASS_REVIEW_REQUIRED','EVENT_REFERENCE_SENSITIVITY_BLOCKED_REVIEW_REQUIRED']else 2)
