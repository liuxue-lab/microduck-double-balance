import hashlib
import math
import statistics
import sys
from common import require as need

GATES = ('top_center', 'top_height', 'top_speed', 'lower_offset', 'lower_height_min',
         'lower_height_max', 'robot_tilt', 'lower_ball_speed', 'unassisted')

VALUES = ('top_center_m', 'top_height_error_m', 'top_speed_m_s', 'lower_offset_m',
          'lower_height_m', 'robot_tilt_rad', 'lower_ball_speed_m_s', 'hold')

THRESHOLDS = dict(top_center_m=.012, top_height_error_m=.006, top_speed_m_s=.08,
                  lower_offset_m=.06, lower_height_min_m=.20, lower_height_max_m=.29,
                  robot_tilt_deg=20., lower_ball_speed_m_s=.15, hold=1e-6,
                  consecutive_steps=250, control_dt=.02, horizon_seconds=10.)

def summary_audit(saved,recomputed):
    """Compare derived display statistics, never physical success thresholds."""
    differences=[];invalid=[]
    def compare(a,b,path):
        if isinstance(b,dict):
            if not isinstance(a,dict) or set(a)!=set(b):
                invalid.append(dict(path=path,kind='schema'));return
            for key in sorted(b):compare(a[key],b[key],path+'.'+key)
        elif isinstance(b,list):
            if not isinstance(a,list) or len(a)!=len(b):
                invalid.append(dict(path=path,kind='schema'));return
            for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'[%d]'%i)
        elif type(b) is float:
            if type(a) is not float or not math.isfinite(a) or not math.isfinite(b):
                invalid.append(dict(path=path,kind='type_or_nonfinite',saved=repr(a),recomputed=repr(b)));return
            if a!=b:
                delta=abs(a-b);limit=max(1e-12,2e-12*max(abs(a),abs(b)))
                item=dict(path=path,saved=a,recomputed=b,absolute_difference=delta,allowed_difference=limit,
                          within_roundoff_tolerance=delta<=limit)
                differences.append(item)
                if delta>limit:invalid.append(item)
        elif type(a) is not type(b) or a!=b:
            invalid.append(dict(path=path,kind='exact_value',saved=repr(a),recomputed=repr(b)))
    compare(saved,recomputed,'summary')
    return dict(valid=not invalid,exact_equal=not differences and not invalid,
        float_abs_tolerance=1e-12,float_rel_tolerance=2e-12,
        scope='Derived summary float values only; all episode decisions/counts and physical thresholds remain exact',
        differences=differences,invalid=invalid,python=sys.version,executable=sys.executable)

def wilson(successes,count):
    need(0<=successes<=count and count>0,'Invalid episode count')
    p=successes/count;z=1.959963984540054;den=1+z*z/count
    center=(p+z*z/(2*count))/den
    radius=z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/den
    return [max(0.,center-radius),min(1.,center+radius)]

def summarize(rows):
    need(bool(rows),'No evaluation episodes')
    count=len(rows);success=sum(r['success'] for r in rows)
    steps=sum(r['steps'] for r in rows)
    lengths=sorted(r['longest_stable_steps']*.02 for r in rows)
    return dict(episodes=count,successes=success,success_fraction=success/count,
        success_wilson95=wilson(success,count),ever_5s_count=sum(r['ever_5s'] for r in rows),
        physically_terminated_count=sum(r['terminated'] for r in rows),
        mean_survival_seconds=statistics.mean(r['steps']*.02 for r in rows),
        longest_stable_seconds=dict(median=statistics.median(lengths),min=min(lengths),max=max(lengths)),
        stable_fraction_of_first_episode_steps=sum(r['stable_steps'] for r in rows)/steps,
        gate_violation_fraction={g:sum(r['gate_failure_counts'][i] for r in rows)/steps for i,g in enumerate(GATES)},
        run_break_counts={g:sum(r['run_break_counts'][i] for r in rows) for i,g in enumerate(GATES)},
        values_mean_over_first_episode_steps={v:sum(r['value_sums'][i] for r in rows)/steps for i,v in enumerate(VALUES)},
        termination_counts={t:sum(t in r['terminations'] for r in rows) for t in sorted({t for r in rows for t in r['terminations']})})


def fingerprint(tensors):
    h=hashlib.sha256()
    for name,tensor in sorted(tensors.items()):
        value=tensor.detach().cpu().contiguous()
        h.update(name.encode());h.update(str((str(value.dtype),tuple(value.shape))).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()

def initial_fingerprint(raw,obs):
    values={}
    for name in ('robot','ball','top_ball'):
        e=raw.scene[name]
        values.update({name+'/'+field:getattr(e.data,field) for field in
            ('root_link_pose_w','root_link_vel_w','joint_pos','joint_vel')})
    values.update({'obs/'+name:value for name,value in obs.items()})
    for field in sorted(raw.event_manager.domain_randomization_fields):
        values['randomized_model/'+field]=getattr(raw.sim.model,field)
    return dict(sha256=fingerprint(values),fields=sorted(values),
                field_diagnostics={name:dict(sha256=fingerprint({name:value}),shape=list(value.shape),dtype=str(value.dtype),
                    finite=bool(value.isfinite().all())) for name,value in sorted(values.items())},
                scope='initial physical states, initial observations, declared randomized model fields')

class Observer:
    """Sample after the original metric computation, before any reset/forward."""
    def __init__(self,raw,torch,mdp):
        self.raw,self.torch,self.mdp=raw,torch,mdp
        self.original=raw.metrics_manager.compute
        names=raw.metrics_manager.active_terms
        self.stable_col=names.index('native_joint_geometric_stable_fraction')
        self.success_col=names.index('native_joint_geometric_success_5s')
        n,device=raw.num_envs,raw.device
        self.active=torch.ones(n,dtype=torch.bool,device=device)
        self.steps=torch.zeros(n,dtype=torch.long,device=device)
        self.stable_steps=torch.zeros_like(self.steps)
        self.current=torch.zeros_like(self.steps);self.longest=torch.zeros_like(self.steps)
        self.ever=torch.zeros(n,dtype=torch.bool,device=device)
        self.failures=torch.zeros((n,len(GATES)),dtype=torch.long,device=device)
        self.breaks=torch.zeros_like(self.failures)
        self.values=torch.zeros((n,len(VALUES)),dtype=torch.float64,device=device)

    def compute(self):
        self.original()
        raw,t,mdp=self.raw,self.torch,self.mdp
        p,v,_=mdp._stage11_head_kinematics(raw)
        r,b=raw.scene['robot'],raw.scene['ball']
        center=t.linalg.vector_norm(p[:,:2],dim=-1);height_error=(p[:,2]-.02).abs()
        speed=t.linalg.vector_norm(v,dim=-1)
        offset=t.linalg.vector_norm(r.data.root_link_pos_w[:,:2]-b.data.root_link_pos_w[:,:2],dim=-1)
        height=r.data.root_link_pos_w[:,2]-b.data.root_link_pos_w[:,2]
        tilt=mdp.wrestle_tilt(raw,'robot');ball_speed=t.linalg.vector_norm(b.data.root_link_lin_vel_w[:,:2],dim=-1)
        hold=mdp._bb_state(raw).hold
        values=t.stack((center,height_error,speed,offset,height,tilt,ball_speed,hold),dim=1)
        gates=t.stack((center<=.012,height_error<=.006,speed<=.08,offset<=.06,height>=.20,
                      height<=.29,tilt<=math.radians(20),ball_speed<=.15,hold<=1e-6),dim=1)
        cached=raw.metrics_manager._step_values
        stable=cached[:,self.stable_col].bool();ever=cached[:,self.success_col].bool()
        a=self.active
        valid=t.isfinite(values[a]).all() & (gates.all(dim=1)[a]==stable[a]).all()
        valid &= (hold[a]<=1e-6).all() & (raw.command_manager.get_command('twist')[a]==0).all()
        valid &= ~raw.termination_manager.get_term('nan_state')[a].any()
        need(bool(valid),'Nonfinite state, changed metric gates, nonzero command/assistance, or nan_state')
        self.steps+=a.long();self.stable_steps+=(stable&a).long()
        self.failures+=(~gates&a[:,None]).long()
        self.breaks+=(~gates&(a&(self.current>0))[:,None]).long()
        self.current=t.where(a,t.where(stable,self.current+1,t.zeros_like(self.current)),self.current)
        self.longest=t.maximum(self.longest,self.current)
        self.ever|=a&ever
        self.values+=t.where(a[:,None],values.double(),t.zeros_like(values,dtype=t.float64))
        need(bool(((self.longest[a]>=250)==self.ever[a]).all()),'Cached five-second latch differs from independent consecutive-step count')

    def rows(self,seed,terminated,terms):
        arrays={name:getattr(self,name).cpu().tolist() for name in
                ('steps','stable_steps','longest','ever','failures','breaks','values')}
        return [dict(seed=seed,env_id=i,steps=arrays['steps'][i],stable_steps=arrays['stable_steps'][i],
            longest_stable_steps=arrays['longest'][i],ever_5s=arrays['ever'][i],
            terminated=terminated[i],success=arrays['ever'][i] and not terminated[i],
            terminations=terms[i],gate_failure_counts=arrays['failures'][i],run_break_counts=arrays['breaks'][i],
            value_sums=arrays['values'][i]) for i in range(self.raw.num_envs)]
