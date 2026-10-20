"""Bounded RK4 reference at the existing C5/C6 time grids; production stays frozen."""
from pathlib import Path
import argparse, copy, gzip, hashlib, json, time, traceback
import numpy as np
import mujoco

CASES=(('R5',1/32000,1600),('R6',1/64000,3200))
TOTAL=4800
START=.39
LIMITS={'peak_overlap':.0001,'position':.0002,'velocity':.05,'peak_upward_velocity':.02}

def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)
def require(ok,msg):
    if not ok:raise RuntimeError(msg)
def save(p,x,allow_nonfinite=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,default=serial,ensure_ascii=False,indent=2,allow_nan=allow_nonfinite)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def signature(m):
    return {'arrays':{k:hashlib.sha256(getattr(m,k).tobytes()).hexdigest() for k in dir(m) if isinstance(getattr(m,k),np.ndarray)},'options':{k:(getattr(m.opt,k).tolist() if isinstance(getattr(m.opt,k),np.ndarray) else getattr(m.opt,k)) for k in dir(m.opt) if not k.startswith('_') and not callable(getattr(m.opt,k))}}
def gap(d):return float(d.qpos[2]-.02)
def state(d):return {'qpos':d.qpos.copy(),'qvel':d.qvel.copy(),'time_s':float(d.time)}
def forces(d):return {'qacc':d.qacc.copy(),'qfrc_smooth':d.qfrc_smooth.copy(),'qfrc_constraint':d.qfrc_constraint.copy(),'solver_niter':d.solver_niter.copy()}
def state_guard(d):
    require(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all(),'Nonfinite state')
    require(gap(d)>=-.001,'Original 1mm floor penetration gate exceeded')

class StageTrace:
    """Control callback reads geometry, never sets controls or state.

    At callback i>0, qacc/qfrc still hold stage i-1 results. These are assigned
    explicitly to that preceding stage. A weighted velocity identity checks this
    source-derived observation convention; current-stage contact force is never
    read before its solve.
    """
    def __init__(self,budget,out,deadline):self.budget=budget;self.out=out;self.deadline=deadline;self.rows=[];self.start=None;self.dt=None
    def begin(self,t,dt):self.rows=[];self.start=t;self.dt=dt
    def __call__(self,m,d):
        require(time.monotonic()<self.deadline,'Internal wall budget reached')
        require(self.start is not None and len(self.rows)<4,'Unexpected callback outside native four RK stages')
        i=len(self.rows)
        if i:self.rows[-1]['solver_result']=forces(d)
        self.budget['rk_stages_observed']+=1;save(self.out/'progress.json',self.budget)
        row={'stage':i,'state':state(d),'gap_m':gap(d),'contacts':[],'phase':'geometry_before_current_stage_constraint_solve'}
        self.rows.append(row)
        for ci in range(d.ncon):
            c=d.contact[ci];row['contacts'].append({'geom':c.geom.copy(),'dist':float(c.dist),'pos':c.pos.copy(),'frame':c.frame.copy(),'dim':int(c.dim),'solref':c.solref.copy(),'solimp':c.solimp.copy(),'friction':c.friction.copy()})
        state_guard(d)
        require(abs(d.time-(self.start+[0,.5,.5,1][i]*self.dt))<1e-11,'Unexpected native RK stage clock')
        for c in row['contacts']:
            require(sorted(c['geom'].tolist())==[0,2] and c['dim']==4,'Unexpected stage contact')
            require(c['dist']>=-.001 and np.allclose(c['solref'],[.001,1],rtol=0,atol=0),'Stage penetration/material gate failed')
    def finish(self,d,before):
        require(len(self.rows)==4,'RK callback count differs from four')
        self.rows[-1]['solver_result']=forces(d)
        weights=np.array([1,2,2,1])/6
        acc=np.array([r['solver_result']['qacc']for r in self.rows]);vel=np.array([r['state']['qvel']for r in self.rows])
        require(np.isfinite(acc).all(),'Nonfinite stage acceleration')
        dv=d.qvel-before['qvel'];dx=d.qpos[:3]-before['qpos'][:3]
        rv=dv-self.dt*(weights@acc);rx=dx-self.dt*(weights@vel[:,:3])
        require(np.max(abs(rv))<=1e-10 and np.max(abs(rx))<=1e-12,'Native RK stage-observation identity failed; no force attribution')
        return {'velocity_identity_residual_max':float(np.max(abs(rv))),'position_identity_residual_max':float(np.max(abs(rx)))}

def configure(original,dt):
    m=copy.deepcopy(original);before=signature(original);expected=copy.deepcopy(before)
    m.opt.timestep=dt;m.opt.integrator=mujoco.mjtIntegrator.mjINT_RK4;m.geom_solref[m.geom('sphere').id,0]=.001
    expected['options']['timestep']=dt;expected['options']['integrator']=1
    solref=original.geom_solref.copy();solref[m.geom('sphere').id,0]=.001;expected['arrays']['geom_solref']=hashlib.sha256(solref.tobytes()).hexdigest()
    require(signature(m)==expected and signature(original)==before,'Undeclared model change')
    return m

def one_step(m,d,budget,out,trace):
    require(time.monotonic()<trace.deadline,'Internal wall budget reached');require(budget['attempted']<TOTAL,'4800-attempt budget exhausted')
    state_guard(d);before=state(d);warnings=d.warning.number.copy();trace.begin(d.time,m.opt.timestep)
    budget['attempted']+=1;save(out/'progress.json',budget)
    mujoco.mj_step(m,d)
    budget['completed']+=1;save(out/'progress.json',budget)
    state_guard(d);require(np.array_equal(d.warning.number,warnings),'New MuJoCo warning');require(abs(d.time-before['time_s']-m.opt.timestep)<1e-11,'Unexpected step clock')
    ids=trace.finish(d,before)
    return {'before':before,'stages':trace.rows,'after':state(d),'post_gap_m':gap(d),'identities':ids}

def compare(a,b,sa,sb):
    require(len(a['time'])==1601 and len(b['time'])==3201,'Comparison requires full unchanged 50ms windows')
    require(np.max(abs(a['time']-(START+np.arange(1601)/32000)))<1e-10 and np.max(abs(b['time']-(START+np.arange(3201)/64000)))<1e-10,'Wrong explicit grids')
    dx=np.linalg.norm(a['qpos'][:,:3]-b['qpos'][::2,:3],axis=1);dv=np.linalg.norm(a['qvel'][:,:3]-b['qvel'][::2,:3],axis=1)
    # Same endpoint-sampled metric as historical C5/C6; RK internal stages have
    # a separate stricter physical penetration guard and are not substituted.
    metrics={'peak_overlap':abs(sa['peak_endpoint_overlap_m']-sb['peak_endpoint_overlap_m']),'position':float(dx.max()),'velocity':float(dv.max()),'peak_upward_velocity':abs(sa['peak_endpoint_upward_velocity_m_s']-sb['peak_endpoint_upward_velocity_m_s'])}
    checks={k:{'value':v,'limit':LIMITS[k],'pass':v<=LIMITS[k]}for k,v in metrics.items()}
    return {'common_samples':1601,'first_time_s':START,'last_time_s':.44,'time_shift_s':0,'checks':checks,'all_four_pass':all(v['pass']for v in checks.values()),'max_position_time_s':START+int(dx.argmax())/32000,'max_velocity_time_s':START+int(dv.argmax())/32000,'formal_material_acceptance':False,'historical_C5_C6_failure_retained':True}

def run(out,model_path,cases,input_hashes):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();deadline=start+300
    b={'attempted':0,'completed':0,'rk_stages_observed':0,'PPO':0,'prior_canonical_steps':15629};save(out/'progress.json',b)
    report={'status':'STARTED','cases':[],'PPO':0,'cloud_hours':0,'full_robot_steps':0,'stage11_complete':False,'formal_physics_adoption':False,'original_2ms_floor_block_retained':True,'historical_C5_C6_failure_retained':True}
    original=None;original_sig=None;previous_cb=mujoco.get_mjcb_control();trace=None;series=[]
    try:
        require(previous_cb is None,'Existing global control callback; preserve it and stop')
        require(mujoco.__version__=='3.10.0','Requires archived MuJoCo 3.10.0')
        for n,h in input_hashes.items():require(Path(n).is_file() and not Path(n).is_symlink() and sha(n)==h,'Input SHA mismatch: '+n)
        original=mujoco.MjModel.from_binary_path(str(model_path));original_sig=signature(original);save(out/'original-model-signature.json',original_sig)
        require((original.nq,original.nv,original.nu,original.na)==(7,6,0,0),'Wrong fixture')
        require(original.opt.integrator==3 and original.opt.solver==2 and original.opt.timestep==.002 and original.opt.disableflags==0,'Original solver/integrator changed')
        require(original.opt.iterations==10 and original.opt.tolerance==1e-8 and original.opt.ls_iterations==20 and original.opt.ls_tolerance==.01,'Solver numerical settings changed')
        require(np.array_equal(original.opt.gravity,[0,0,-9.81]) and original.body_mass[2]==.01,'Gravity/mass changed')
        init=json.loads((Path(cases)/'C5/initialization.json').read_text());require(init==json.loads((Path(cases)/'C6/initialization.json').read_text()),'Old initializations differ')
        for name,dt,n in CASES:
            dest=out/name;dest.mkdir();m=configure(original,dt);d=mujoco.MjData(m)
            d.qpos[:]=init['qpos'];d.qvel[:]=init['qvel'];d.qacc_warmstart[:]=init['qacc_warmstart'];d.time=init['time_s'];require(d.time==START,'Start clock moved')
            save(dest/'initialization.json',{'state':state(d),'qacc_warmstart':d.qacc_warmstart.copy(),'complete_historical_state_restore':False,'method':'Same archived A0/C5/C6 free-flight initialization; no additional forward or integration'})
            poses=[d.qpos.copy()];vels=[d.qvel.copy()];times=[d.time]
            summary={'name':name,'dt':dt,'planned_steps':n,'completed_steps':0,'attempted_steps':0,'peak_endpoint_overlap_m':0.,'peak_internal_stage_overlap_m':0.,'peak_endpoint_upward_velocity_m_s':0.,'horizon_completed':False}
            case_start_attempts=b['attempted'];case_start_completed=b['completed']
            report['cases'].append(summary);trace=StageTrace(b,out,deadline);mujoco.set_mjcb_control(trace)
            try:
                with gzip.open(dest/'stage-log.jsonl.gz','wt')as log,(dest/'states.partial.jsonl').open('x')as partial:
                    partial.write(json.dumps(state(d),default=serial)+'\n');partial.flush()
                    for i in range(n):
                        require(abs(d.time-(START+i*dt))<1e-11,'Explicit pre-step grid mismatch')
                        summary['attempted_steps']+=1
                        row=one_step(m,d,b,out,trace)
                        summary['completed_steps']+=1;poses.append(d.qpos.copy());vels.append(d.qvel.copy());times.append(d.time)
                        partial.write(json.dumps(state(d),default=serial)+'\n');partial.flush();log.write(json.dumps(row,default=serial,allow_nan=False)+'\n');log.flush()
                        summary['peak_endpoint_overlap_m']=max(summary['peak_endpoint_overlap_m'],-gap(d));summary['peak_internal_stage_overlap_m']=max(summary['peak_internal_stage_overlap_m'],max(-x['gap_m']for x in trace.rows))
                        summary['peak_endpoint_upward_velocity_m_s']=max(summary['peak_endpoint_upward_velocity_m_s'],float(d.qvel[2]))
                    summary['horizon_completed']=True
            finally:
                mujoco.set_mjcb_control(previous_cb)
                summary['attempted_steps']=b['attempted']-case_start_attempts;summary['completed_steps']=b['completed']-case_start_completed
                summary['verified_saved_states']=len(poses)
                np.savez_compressed(dest/'states.npz',qpos=np.array(poses),qvel=np.array(vels),time=np.array(times),dt=dt)
                np.savez_compressed(dest/'last-primary-state.npz',qpos=d.qpos,qvel=d.qvel,qacc=d.qacc,qacc_warmstart=d.qacc_warmstart,time=d.time)
                save(dest/'summary.json',summary)
                if trace and trace.rows:
                    # A partial/in-flight stage dump is evidence, not a completed step.
                    save(dest/'last-observed-stages.json',trace.rows,allow_nonfinite=True)
            series.append({'qpos':np.array(poses),'qvel':np.array(vels),'time':np.array(times)})
            save(out/'run-review.json',report)
        report['comparison']=compare(*series,*report['cases'])
        historical=[];historical_summaries=[];same_dt=[]
        for ix,name in enumerate(['C5','C6']):
            with np.load(Path(cases)/name/'states.npz',allow_pickle=False)as data:old={k:data[k].copy()for k in ['qpos','qvel','time']}
            historical.append(old)
            historical_summaries.append({'peak_endpoint_overlap_m':float(max(0,.02-old['qpos'][:,2].min())),'peak_endpoint_upward_velocity_m_s':float(max(0,old['qvel'][:,2].max()))})
            require(np.max(abs(old['time']-series[ix]['time']))<1e-10,'Old/new same-dt grids differ')
            same_dt.append({'old_case':name,'new_case':CASES[ix][0],'max_position_difference_m':float(np.linalg.norm(old['qpos'][:,:3]-series[ix]['qpos'][:,:3],axis=1).max()),'max_linear_velocity_difference_m_s':float(np.linalg.norm(old['qvel'][:,:3]-series[ix]['qvel'][:,:3],axis=1).max()),'time_shift_s':0,'physical_parameter_change':False,'changed_integrator_only_relative_to_old_diagnostic':True})
        report['historical_implicit_pair_recomputed_without_integration']=compare(*historical,*historical_summaries)
        require(not report['historical_implicit_pair_recomputed_without_integration']['all_four_pass'],'Historical failure unexpectedly disappeared')
        report['same_dt_old_new_comparisons']=same_dt
        report['status']='RK_REFERENCE_PAIR_GATES_PASS_REVIEW_REQUIRED'if report['comparison']['all_four_pass']else 'RK_REFERENCE_SENSITIVITY_BLOCKED_REVIEW_REQUIRED'
    except BaseException as e:
        report['status']='RK_REFERENCE_STOPPED_REVIEW_REQUIRED';report['error']=repr(e);(out/'error.txt').write_text(traceback.format_exc())
    finally:
        mujoco.set_mjcb_control(previous_cb)
        report.update({'attempted_steps':b['attempted'],'completed_steps':b['completed'],'rk_stages_observed':b['rk_stages_observed'],'step_count_exact':b['attempted']==b['completed'],'canonical_completed_lower_bound':15629+b['completed'],'canonical_attempted_upper_bound':15629+b['attempted'],'wall_seconds':time.monotonic()-start})
        report['inputs_unchanged']=all(Path(p).is_file()and sha(p)==h for p,h in input_hashes.items());report['original_model_unchanged']=original is not None and signature(original)==original_sig;report['control_callback_restored']=mujoco.get_mjcb_control()is previous_cb
        if not report['inputs_unchanged']or not report['original_model_unchanged']or not report['control_callback_restored']:report['status']='SOURCE_OR_CALLBACK_CHANGED_BLOCKED'
        save(out/'run-review.json',report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--cases',type=Path,required=True);p.add_argument('--input-hashes',type=Path,required=True)
    a=p.parse_args();r=run(a.output,a.model,a.cases,json.loads(a.input_hashes.read_text()));print('Stage11RKReference='+r['status']);print('Stage11NewCompletedSteps='+str(r['completed_steps']));print('Stage11NewAttemptedSteps='+str(r['attempted_steps']));print('Stage11RKStagesObserved='+str(r['rk_stages_observed']));raise SystemExit(0 if r['status']in ['RK_REFERENCE_PAIR_GATES_PASS_REVIEW_REQUIRED','RK_REFERENCE_SENSITIVITY_BLOCKED_REVIEW_REQUIRED']else 2)
