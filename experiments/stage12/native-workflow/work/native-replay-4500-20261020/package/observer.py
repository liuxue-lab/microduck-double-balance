"""Observer copied unchanged from the completed native evaluation."""

import hashlib
import math
from replay_common import need

GATES = ('top_center', 'top_height', 'top_speed', 'lower_offset', 'lower_height_min',
         'lower_height_max', 'robot_tilt', 'lower_ball_speed', 'unassisted')

VALUES = ('top_center_m', 'top_height_error_m', 'top_speed_m_s', 'lower_offset_m',
          'lower_height_m', 'robot_tilt_rad', 'lower_ball_speed_m_s', 'hold')

THRESHOLDS = dict(top_center_m=.012, top_height_error_m=.006, top_speed_m_s=.08,
                  lower_offset_m=.06, lower_height_min_m=.20, lower_height_max_m=.29,
                  robot_tilt_deg=20., lower_ball_speed_m_s=.15, hold=1e-6,
                  consecutive_steps=250, control_dt=.02, horizon_seconds=10.)

def fingerprint(tensors):
    h=hashlib.sha256()
    for name,tensor in sorted(tensors.items()):
        value=tensor.detach().cpu().contiguous()
        h.update(name.encode());h.update(str((str(value.dtype),tuple(value.shape))).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()

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
