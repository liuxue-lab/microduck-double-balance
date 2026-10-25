"""Array-backend checks of the all-world BAM guard; actual Torch/GPU still pending."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'package'))
import train_motor_monitor as module


class Tensor(np.ndarray):
    def __new__(cls,values):return np.asarray(values,dtype=float).view(cls)
    def abs(self):return np.abs(self)


class Backend:
    @staticmethod
    def isfinite(x):return np.isfinite(x)
    @staticmethod
    def stack(x):return Tensor(np.stack(x))
    @staticmethod
    def maximum(x,y):return np.maximum(x,y)
    @staticmethod
    def ones_like(x):return np.ones_like(x)


def fixture():
    kt=.366;resistance=2.811
    omega=Tensor(np.zeros((3,2)));duty=Tensor(np.full((3,2),.2));voltage=Tensor(np.full((3,2),7.2))
    tau=kt/resistance*(voltage*duty)
    group=SimpleNamespace(_bam_model=SimpleNamespace(kt=SimpleNamespace(value=kt),R=SimpleNamespace(value=resistance),
        actuator=SimpleNamespace(max_current=1.75,max_pwm=1.)),kd_scale=Tensor(np.ones((3,2))),
        global_ctrl_ids=[0,1],last_diagnostics=dict(joint_velocity=omega,duty_applied=duty,effective_vin=voltage,motor_torque=tau))
    data=SimpleNamespace(xfrc_applied=Tensor(np.zeros((3,1,6))),ctrl=Tensor(tau.copy()),actuator_force=Tensor(tau.copy()))
    robot=SimpleNamespace(data=SimpleNamespace(joint_pos=Tensor(np.zeros((3,2))),joint_vel=omega))
    raw=SimpleNamespace(sim=SimpleNamespace(data=data,model=SimpleNamespace(
        actuator_forcerange=Tensor([[[-1,1],[-1,1]]])),step=lambda:'integrated'),scene={'robot':robot})
    guard=SimpleNamespace(groups=[group],close=lambda:None)
    with patch.object(module,'Guard',return_value=guard):monitor=module.TrainingMotorMonitor(raw,Backend(),Path('.'))
    monitor.active=True
    return monitor,raw,group


class TrainingMotorTests(unittest.TestCase):
    def test_valid_multiple_worlds_integrate_and_restore(self):
        m,raw,g=fixture();self.assertEqual(m.step(),'integrated');self.assertEqual(m.steps,1)
        m.close();self.assertEqual(raw.sim.step(),'integrated')

    def test_last_world_current_violation_is_not_hidden_by_first(self):
        m,raw,g=fixture();g.last_diagnostics['motor_torque'][2,1]=1.
        with self.assertRaises(RuntimeError):m.step()
        self.assertEqual(m.steps,0)

    def test_bam_control_mapping_mismatch_stops_before_integration(self):
        m,raw,g=fixture();raw.sim.data.ctrl[2,1]+=0.1
        with self.assertRaises(RuntimeError):m.step()
        self.assertEqual(m.steps,0)

    def test_applied_torque_violation_checked_after_integration(self):
        m,raw,g=fixture();raw.sim.data.actuator_force[2,1]=1.1
        with self.assertRaises(RuntimeError):m.step()
        self.assertEqual(m.steps,1)


if __name__=='__main__':unittest.main(verbosity=2)
