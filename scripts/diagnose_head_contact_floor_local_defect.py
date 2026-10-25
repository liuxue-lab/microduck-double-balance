"""Stage 11: four same-state local integration defects, at most twelve calls.

Independent CPU material fixture. No training, production physics or old replay.
"""
from pathlib import Path
import argparse, copy, gzip, hashlib, json, time, traceback
import numpy as np
import mujoco

INDICES=(13,44,143,1400)
H=1/32000
MAX_STEPS=12

def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)

def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(x,default=serial,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def require(ok,message):
    if not ok:raise RuntimeError(message)

def signature(m):
    return {'arrays':{k:hashlib.sha256(getattr(m,k).tobytes()).hexdigest() for k in dir(m) if isinstance(getattr(m,k),np.ndarray)},
            'options':{k:(getattr(m.opt,k).tolist() if isinstance(getattr(m.opt,k),np.ndarray) else getattr(m.opt,k)) for k in dir(m.opt) if not k.startswith('_') and not callable(getattr(m.opt,k))}}

def geometric_guard(m,d):
    require(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all(),'Nonfinite primary state')
    gap=float(d.qpos[2]-m.geom_size[m.geom('sphere').id,0]-m.geom_pos[m.geom('floor').id,2])
    require(gap>=-.001,'Original 1 mm floor penetration limit exceeded')
    return gap

def one_step(m,d,budget,out,deadline):
    require(time.monotonic()<deadline,'120 second internal wall budget exhausted')
    require(budget['attempted']<MAX_STEPS,'12-attempt limit reached')
    geometric_guard(m,d)
    before={'qpos':d.qpos.copy(),'qvel':d.qvel.copy(),'time':float(d.time)}
    warnings=d.warning.number.copy()
    budget['attempted']+=1;save(out/'progress.json',budget)
    mujoco.mj_step(m,d)
    budget['completed']+=1;save(out/'progress.json',budget)
    # Save before checking so a failed call's returned state is retained.
    raw={'before':before,'after':{'qpos':d.qpos.copy(),'qvel':d.qvel.copy(),'time':float(d.time)},'qacc':d.qacc.copy(),'qacc_warmstart_after':d.qacc_warmstart.copy(),'qfrc_smooth':d.qfrc_smooth.copy(),'qfrc_constraint':d.qfrc_constraint.copy(),'solver_niter':d.solver_niter.copy(),'contacts':[]}
    for i in range(d.ncon):
        c=d.contact[i];f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f)
        raw['contacts'].append({'geom':c.geom.copy(),'dist':float(c.dist),'pos':c.pos.copy(),'frame':c.frame.copy(),'force':f,'solref':c.solref.copy(),'solimp':c.solimp.copy(),'friction':c.friction.copy(),'dim':int(c.dim)})
    # NPZ preserves NaN/Inf if encountered before the JSON guard.
    np.savez_compressed(out/f"attempt-{budget['attempted']:02d}-state.npz",qpos=d.qpos,qvel=d.qvel,qacc=d.qacc,qacc_warmstart=d.qacc_warmstart,time=d.time)
    require(np.isfinite(d.qacc).all(),'Nonfinite acceleration')
    raw['post_floor_gap_m']=geometric_guard(m,d)
    save(out/f"attempt-{budget['attempted']:02d}.json",raw)
    require(np.array_equal(d.warning.number,warnings),'New MuJoCo warning')
    require(abs(d.time-before['time']-m.opt.timestep)<1e-12,'Unexpected clock advance')
    for c in raw['contacts']:
        require(sorted(c['geom'].tolist())==[0,2],'Unexpected contact pair')
        require(c['dist']>=-.001 and np.isfinite(c['force']).all(),'Contact penetration or force invalid')
        require(c['dim']==4 and np.allclose(c['solref'],[.001,1],rtol=0,atol=0),'Diagnostic material changed')
    return raw

def run(out,model_path,case_root,input_hashes):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();deadline=started+120
    budget={'attempted':0,'completed':0,'PPO':0,'prior_canonical_steps':15617}
    save(out/'progress.json',budget)
    report={'status':'STARTED','cases':[],'stage11_complete':False,'formal_physics_adoption':False,'full_robot_steps':0,'cloud_hours':0,'PPO':0,'scope':'same C5 saved primary state; one 31.25 us step versus two 15.625 us steps, fixed diagnostic solref=0.001 s'}
    original=None;frozen=None
    try:
        require(mujoco.__version__=='3.10.0','MuJoCo version differs from C5/C6')
        for name,digest in input_hashes.items():require(Path(name).is_file() and not Path(name).is_symlink() and sha(name)==digest,'Input SHA mismatch: '+name)
        original=mujoco.MjModel.from_binary_path(str(model_path));frozen=signature(original);save(out/'original-model-signature.json',frozen)
        require((original.nq,original.nv,original.nu)==(7,6,0),'Unexpected fixture dimensions')
        require(int(original.opt.integrator)==3 and int(original.opt.solver)==2 and original.opt.disableflags==0,'Integrator/solver changed')
        require(original.opt.iterations==10 and original.opt.tolerance==1e-8 and original.opt.ls_iterations==20 and original.opt.ls_tolerance==.01,'Solver settings changed')
        require(np.array_equal(original.opt.gravity,[0,0,-9.81]) and original.body_mass[2]==.01,'Gravity/mass changed')
        with np.load(Path(case_root)/'C5/states.npz',allow_pickle=False) as data:a={k:data[k].copy() for k in data.files}
        with gzip.open(Path(case_root)/'C5/contacts.jsonl.gz','rt') as f:logs=[json.loads(s) for s in f]
        require(float(a['dt'])==H and len(a['time'])==1601,'Incomplete archived C5')
        for k in INDICES:
            require(np.array_equal(a['qpos'][k],logs[k]['qpos_before']) and np.array_equal(a['qvel'][k],logs[k]['qvel_before']),'Saved phases differ')
            case={'saved_C5_index':k,'initial_time_s':.39+k*H,'branches':{},'warmstart_inferred_not_complete_saved_state':True}
            report['cases'].append(case)
            warm=(a['qvel'][k]-a['qvel'][k-1])/H
            for branch,dt,n in [('coarse',H,1),('fine',H/2,2)]:
                m=copy.deepcopy(original);m.opt.timestep=dt;m.geom_solref[m.geom('sphere').id,0]=.001
                modified=signature(m);expected=copy.deepcopy(frozen);expected['options']['timestep']=dt
                solref=original.geom_solref.copy();solref[m.geom('sphere').id,0]=.001;expected['arrays']['geom_solref']=hashlib.sha256(solref.tobytes()).hexdigest()
                require(modified==expected,'Undeclared in-memory model change')
                d=mujoco.MjData(m);d.qpos[:]=a['qpos'][k];d.qvel[:]=a['qvel'][k];d.time=case['initial_time_s'];d.qacc_warmstart[:]=warm
                case['branches'][branch]={'dt':dt,'initial_qpos':d.qpos.copy(),'initial_qvel':d.qvel.copy(),'initial_warmstart':warm.copy(),'steps':[]}
                for _ in range(n):
                    row=one_step(m,d,budget,out,deadline);case['branches'][branch]['steps'].append(row);save(out/'run-review.json',report)
                require(abs(d.time-(case['initial_time_s']+H))<1e-12,'Common terminal time mismatch')
                if branch=='coarse':
                    qerr=float(np.max(abs(d.qpos-a['qpos'][k+1])));verr=float(np.max(abs(d.qvel-a['qvel'][k+1])))
                    case['coarse_archive_reproduction']={'max_qpos_error':qerr,'max_qvel_error':verr,'limits':[1e-12,1e-9]}
                    require(qerr<=1e-12 and verr<=1e-9,'Coarse archived single-step reproduction failed')
            c=case['branches']['coarse']['steps'][0];f0,f1=case['branches']['fine']['steps']
            dv=f1['after']['qvel'][:3]-c['after']['qvel'][:3];dx=f1['after']['qpos'][:3]-c['after']['qpos'][:3]
            acc_c=c['qacc'][:3];acc_0=f0['qacc'][:3];acc_1=f1['qacc'][:3]
            case['local_defect']={'fine_minus_coarse_position_m':dx,'fine_minus_coarse_linear_velocity_m_s':dv,'first_acceleration_difference_m_s2':acc_0-acc_c,'velocity_defect_from_logged_accelerations_m_s':H/2*(acc_0+acc_1)-H*acc_c,'position_defect_from_logged_accelerations_m':H*H*(.5*acc_0+.25*acc_1-acc_c),'velocity_identity_residual_m_s':dv-(H/2*(acc_0+acc_1)-H*acc_c),'position_identity_residual_m':dx-H*H*(.5*acc_0+.25*acc_1-acc_c)}
            save(out/f'case-{k}.json',case)
        report['status']='LOCAL_DEFECT_12_STEPS_COMPLETE_REVIEW_REQUIRED'
    except BaseException as exc:
        report['status']='LOCAL_DEFECT_STOPPED_REVIEW_REQUIRED';report['error']=repr(exc);(out/'error.txt').write_text(traceback.format_exc())
    finally:
        if 'd' in locals():
            np.savez_compressed(out/'last-primary-state.npz',qpos=d.qpos,qvel=d.qvel,qacc=d.qacc,qacc_warmstart=d.qacc_warmstart,time=d.time,attempted=budget['attempted'],completed=budget['completed'])
        report['attempted_steps']=budget['attempted'];report['completed_steps']=budget['completed'];report['step_count_exact']=budget['attempted']==budget['completed']
        report['canonical_completed_lower_bound']=15617+budget['completed'];report['canonical_attempted_upper_bound']=15617+budget['attempted']
        report['input_hashes_unchanged']=all(Path(p).is_file() and sha(p)==h for p,h in input_hashes.items())
        report['original_model_unchanged']=original is not None and signature(original)==frozen
        if not report['input_hashes_unchanged'] or not report['original_model_unchanged']:report['status']='INPUT_OR_MODEL_CHANGED_BLOCKED'
        report['not_started_indices']=[k for k in INDICES if not any(c['saved_C5_index']==k for c in report['cases'])]
        report['wall_seconds']=time.monotonic()-started;save(out/'run-review.json',report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--cases',type=Path,required=True);p.add_argument('--input-hashes',type=Path,required=True)
    a=p.parse_args();r=run(a.output,a.model,a.cases,json.loads(a.input_hashes.read_text()));print('Stage11LocalDefect='+r['status']);print('Stage11NewCompletedSteps='+str(r['completed_steps']));print('Stage11NewAttemptedSteps='+str(r['attempted_steps']));raise SystemExit(0 if r['status']=='LOCAL_DEFECT_12_STEPS_COMPLETE_REVIEW_REQUIRED' else 2)
