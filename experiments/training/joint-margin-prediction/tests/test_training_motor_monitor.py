"""CPU interface doubles for all-world electrical checks and direct qpos sampling."""
from pathlib import Path
from types import SimpleNamespace as NS
import math
import unittest
from unittest.mock import patch
import numpy as np
from test_constraint_v3 import Tensor, Backend as ArrayBackend
import train_motor_monitor as module

class Backend(ArrayBackend):
    tensor=staticmethod(lambda x,dtype=None,device=None:Tensor(x,dtype))
    allclose=staticmethod(np.allclose)
    isfinite=staticmethod(np.isfinite)

Tensor.abs=lambda self:Tensor(np.abs(self))

def fixture():
    n,j=3,14;kt=.366;resistance=2.811
    omega=Tensor(np.zeros((n,j)));duty=Tensor(np.full((n,j),.2));voltage=Tensor(np.full((n,j),7.2))
    tau=kt/resistance*(voltage*duty)
    group=NS(_bam_model=NS(kt=NS(value=kt),R=NS(value=resistance),
        actuator=NS(max_current=1.75,max_pwm=1.)),kd_scale=Tensor(np.ones((n,j))),
        global_ctrl_ids=list(range(j)),last_diagnostics=dict(joint_velocity=omega,
        duty_applied=duty,effective_vin=voltage,motor_torque=tau))
    # Deliberately reverse named servo addresses and include seven non-servo qpos.
    names=['joint-'+str(i) for i in range(j-1)]+['head_roll'];addresses=list(range(7,21))[::-1]
    data=NS(xfrc_applied=Tensor(np.zeros((n,1,6))),ctrl=Tensor(tau.copy()),
        actuator_force=Tensor(tau.copy()),qpos=Tensor(np.zeros((n,21))))
    robot=NS(data=NS(joint_pos=Tensor(np.zeros((n,j))),joint_vel=omega))
    raw=NS(device='cpu',num_envs=n,physics_dt=.002,scene={'robot':robot},sim=NS(data=data,
        mj_model=NS(jnt_qposadr=addresses),model=NS(actuator_forcerange=Tensor(np.tile([-1,1],(1,j,1)))),
        step=lambda:'integrated'))
    guard=NS(groups=[group],close=lambda:None,metadata=dict(joints=[dict(name=name,
        global_joint_id=i,lower_rad=math.radians(-25),upper_rad=math.radians(25)) for i,name in enumerate(names)]))
    with patch.object(module,'Guard',return_value=guard):
        monitor=module.TrainingMotorMonitor(raw,Backend(),Path('.'))
    monitor.active=True
    return monitor,raw,group

class TrainingMotorTests(unittest.TestCase):
    def test_valid_multiple_worlds_integrate_and_restore(self):
        m,raw,g=fixture();self.assertEqual(m.step(),'integrated');self.assertEqual(m.steps,1)
        self.assertEqual(m.joint_stats.take_rollout(1)['world_physics_samples'],3)
        m.close();self.assertEqual(raw.sim.step(),'integrated')

    def test_direct_qpos_violation_recorded_despite_stale_robot_cache_and_reset(self):
        m,raw,g=fixture()
        def integrate():raw.sim.data.qpos[2,7]=math.radians(27)
        m.original=integrate;m.step()
        self.assertEqual(raw.scene['robot'].data.joint_pos[2,13],0)
        raw.sim.data.qpos[:]=0
        m.joint_stats.after_control_step(Tensor([False,False,True],bool),Tensor([False,False,True],bool))
        row=m.joint_stats.take_rollout(1)
        self.assertEqual(row['head_roll_violation_fraction'],1/3)
        self.assertAlmostEqual(row['head_roll_peak_abs_degrees'],27)
        self.assertEqual(row['ended_episodes_with_joint_violation'],1)

    def test_last_world_current_violation_is_not_hidden_by_first(self):
        m,raw,g=fixture();g.last_diagnostics['motor_torque'][2,13]=1.
        with self.assertRaises(RuntimeError):m.step()
        self.assertEqual(m.steps,0)

    def test_control_mapping_mismatch_stops_before_integration(self):
        m,raw,g=fixture();raw.sim.data.ctrl[2,13]+=.1
        with self.assertRaises(RuntimeError):m.step()
        self.assertEqual(m.steps,0)

    def test_force_range_checked_after_integration(self):
        m,raw,g=fixture();raw.sim.data.actuator_force[2,13]=1.1
        with self.assertRaises(RuntimeError):m.step()
        self.assertEqual(m.steps,1)

    def test_nonfinite_direct_qpos_stops_even_if_cached_position_finite(self):
        m,raw,g=fixture();raw.sim.data.qpos[2,7]=float('nan')
        with self.assertRaises(RuntimeError):m.step()
        self.assertEqual(m.joint_stats.steps,0)

if __name__=='__main__':unittest.main(verbosity=2)
