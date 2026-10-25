"""Verify no delay stretching and no retroactive commands at control boundaries."""
from types import SimpleNamespace as NS
import unittest
import torch
from mjlab.utils.buffers import DelayBuffer
from native_delay import OriginalDelayClock


class DelayClock(unittest.TestCase):
    def test_virtual_2ms_samples_match_the_original_buffer(self):
        def buffer():
            b=DelayBuffer(min_lag=3,max_lag=6,batch_size=1,device='cpu',
                hold_prob=0.,update_period=0,per_env_phase=True)
            b.append(torch.zeros(1,14));b.compute()
            return b
        torch.manual_seed(27);reference=buffer();initial=torch.get_rng_state().clone()
        expected=[]
        for tick in range(30):
            reference.append(torch.full((1,14),float(tick//10+1)))
            expected.append(reference.compute().clone())
        final_rng=torch.get_rng_state().clone()
        torch.manual_seed(27);actual=buffer();self.assertTrue(torch.equal(initial,torch.get_rng_state()))
        group=NS(command_field='position',_delay_buffer=actual,
            cfg=NS(delay_min_lag=3,delay_max_lag=6,delay_update_period=0,delay_hold_prob=0.),
            apply_delay=lambda cmd:cmd)
        sim=NS(step_calls=0,mj_data=NS(time=0.),after_step=lambda:None)
        raw=NS(sim=sim,scene={'robot':NS(actuators=[group])})
        clock=OriginalDelayClock(raw)
        from mjlab.actuator.actuator import ActuatorCmd
        for control in range(3):
            for substep,index in enumerate((0,2,5,7)):
                target=torch.full((1,14),float(control+1));zero=torch.zeros_like(target)
                cmd=ActuatorCmd(position_target=target,velocity_target=zero,effort_target=zero,pos=zero,vel=zero)
                result=group.apply_delay(cmd)
                self.assertTrue(torch.equal(result.position_target,expected[control*10+index]))
                sim.step_calls+=1;sim.mj_data.time=sim.step_calls*.005;sim.after_step()
        self.assertTrue(torch.equal(final_rng,torch.get_rng_state()))
        self.assertEqual(clock.snapshot()['original_delay_samples'],[30])
        self.assertEqual(clock.snapshot()['native_delay_calls'],[12])
        clock.close()


if __name__=='__main__':unittest.main()
