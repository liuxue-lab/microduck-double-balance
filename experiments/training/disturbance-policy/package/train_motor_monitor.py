"""Check original BAM electrical bounds on all training worlds, every substep.

Joint q excursions remain trainable violations, priced by the reward; this
monitor never certifies a training sample as an accepted policy evaluation.
"""
from common import require
from model_limits_runtime import Guard
from joint_statistics import JointStatistics


class TrainingMotorMonitor:
    def __init__(self, raw, torch, folder):
        # Reuse the proven source/transmission/parameter mapping checks.
        self.guard=Guard(raw,torch,folder)
        self.raw,self.torch=raw,torch
        self.original=raw.sim.step
        self.active=False
        self.steps=0
        self.last=[None]*len(self.guard.groups)
        self.parameters=[]
        for group in self.guard.groups:
            bam=group._bam_model
            self.parameters.append((float(bam.kt.value),float(bam.R.value),
                None if bam.actuator.max_current is None else float(bam.actuator.max_current),
                float(bam.actuator.max_pwm)))
        joints = self.guard.metadata['joints']
        self.qids = torch.tensor([int(raw.sim.mj_model.jnt_qposadr[j['global_joint_id']])
                                  for j in joints], dtype=torch.long, device=raw.device)
        bounds = torch.tensor([[j['lower_rad'], j['upper_rad']] for j in joints], device=raw.device)
        q = raw.sim.data.qpos[:, self.qids]
        require(tuple(q.shape) == (raw.num_envs, 14), 'Direct qpos joint mapping differs')
        require(torch.allclose(q, raw.scene['robot'].data.joint_pos, rtol=0., atol=1e-7),
                'Direct qpos and named robot positions disagree before training')
        self.joint_stats = JointStatistics(torch, q, bounds, [j['name'] for j in joints], raw.physics_dt)
        raw.sim.step=self.step

    def step(self):
        if not self.active:return self.original()
        t=self.torch
        checks=[(self.raw.sim.data.xfrc_applied==0).all()]
        for index,(group,(kt,resistance,cap,pwm)) in enumerate(zip(self.guard.groups,self.parameters)):
            d=group.last_diagnostics
            require(d is not None and d is not self.last[index], 'Missing fresh training BAM diagnostics')
            self.last[index]=d
            current=d['motor_torque']/kt
            duty=d['duty_applied']
            omega=d['joint_velocity'];voltage=d['effective_vin'];kd=group.kd_scale
            expected=kt/resistance*(voltage*duty-kt*omega*kd)
            torque=d['motor_torque'];ctrl=self.raw.sim.data.ctrl[:,group.global_ctrl_ids]
            tolerance=(32*2**-23)*t.maximum(t.maximum(torque.abs(),expected.abs()),t.ones_like(torque))
            checks.extend([t.isfinite(current).all(),t.isfinite(duty).all(),
                t.isfinite(d['joint_velocity']).all(),t.isfinite(d['effective_vin']).all(),
                t.isfinite(kd).all(),(voltage>0).all(),(kd>0).all(),t.isfinite(ctrl).all(),
                ((torque-expected).abs()<=tolerance).all(),
                ((ctrl-torque).abs()<=(32*2**-23)*t.maximum(torque.abs(),t.ones_like(torque))).all(),
                (duty.abs()<=pwm+(32*2**-23)*t.maximum(duty.abs(),t.ones_like(duty)*max(1.,pwm))).all()])
            if cap is not None:
                tol=(32*2**-23)*t.maximum(current.abs(),t.ones_like(current)*max(1.,cap))
                checks.append((current.abs()<=cap+tol).all())
        require(bool(t.stack(checks).all()), 'Training BAM current/PWM/nonfinite/external-wrench fault; preserve')
        result=self.original();self.steps+=1
        checks=[]
        for group in self.guard.groups:
            force=self.raw.sim.data.actuator_force[:,group.global_ctrl_ids]
            limits=self.raw.sim.model.actuator_forcerange[0,group.global_ctrl_ids]
            lo,hi=limits[:,0],limits[:,1]
            scale=t.maximum(t.maximum(lo.abs(),hi.abs()),t.ones_like(lo))
            tol=(32*2**-23)*t.maximum(force.abs(),scale)
            checks.extend([t.isfinite(force).all(),(force>=lo-tol).all(),(force<=hi+tol).all()])
        robot=self.raw.scene['robot']
        q=self.raw.sim.data.qpos[:,self.qids]
        checks.extend([t.isfinite(q).all(),t.isfinite(robot.data.joint_vel).all()])
        require(bool(t.stack(checks).all()), 'Training force-range/nonfinite-state fault; preserve')
        self.joint_stats.observe(q)
        return result

    def close(self):
        self.raw.sim.step=self.original
        self.guard.close()
