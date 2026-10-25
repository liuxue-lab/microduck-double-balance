"""Native physics evidence and fixed-deadline success remain separate."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from support import c, case_row, loop_evidence, loop_ops

class Acceptance(unittest.TestCase):
    def reports(self,realtime=False):
        shared=dict(completed_requests=6000,transcript_sha256='a'*64)
        sim=dict(status='LOOP_SIM_FUNCTIONAL_PASSED',original_records_unchanged=True,
            completed_control_steps=6000,cases=[case_row(x,realtime) for x in c.CASES],
            transport=shared,full_loop_realtime_acceptance=realtime)
        return sim,dict(status='LOOP_BRIDGE_COMPLETE',**shared),dict(status='LOOP_SERVER_COMPLETE',
            model_unchanged=True,recurrent_reset_sequences=[0,3000],**shared)

    def test_all_deadlines_required_even_with_fast_average(self):
        for realtime in (False,True):
            sim,bridge,server=self.reports(realtime)
            for mean in (11.,1.):
                sim['cases'][0]['timing']['all_steps']['environment_ms']['mean']=mean
                result=loop_evidence.verify_reports(sim,bridge,server)
                self.assertTrue(result['functional_closed_loop'])
                self.assertIs(result['full_loop_realtime_acceptance'],realtime)
                self.assertFalse(result['physical_robot_acceptance'])

    def test_incomplete_physics_forged_flags_and_transcript_mismatch_fail(self):
        sim,bridge,server=self.reports()
        mutations=[lambda r:r['cases'][0]['runtime_adapters']['backend'].update(native_step_calls=11999),
            lambda r:r['cases'][0]['runtime_adapters']['backend'].update(reset_calls=3),
            lambda r:r['cases'][1]['model_limits'].update(motor_checks_before_integration=11999),
            lambda r:r['cases'][1]['disturbance']['pulses'][0].update(applied_physics_steps=19),
            lambda r:r['cases'][0]['live_action_parity'].update(local_fallback=True),
            lambda r:r['cases'][0]['timing'].update(realtime_acceptance=True),
            lambda r:r['cases'][0]['timing'].update(schedule_rebases=1),
            lambda r:r['cases'][0]['timing'].update(first_tick_included=False),
            lambda r:r['cases'][0]['timing'].update(incomplete_tick=True),
            lambda r:r['cases'].reverse(),
            lambda r:r['transport'].update(transcript_sha256='b'*64)]
        for mutate in mutations:
            candidate=copy.deepcopy(sim);mutate(candidate)
            with self.assertRaises(RuntimeError):loop_evidence.verify_reports(candidate,bridge,server)
        with self.assertRaises(RuntimeError):
            loop_evidence.verify_reports(sim,bridge,dict(server,recurrent_reset_sequences=[0]))

    def test_status_success_and_watch_after_completion(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)
            with patch.object(loop_ops,'ROOT',path),patch('builtins.print'):
                for status in ('NATIVE_MUJOCO_REALTIME_PASSED','NATIVE_MUJOCO_FUNCTIONAL_PASSED_REALTIME_PENDING'):
                    (path/'report.json').write_text(json.dumps(dict(status=status,directory=d,phase='finished')))
                    self.assertEqual(loop_ops.show(),0)
                with patch.object(loop_ops.time,'sleep',side_effect=KeyboardInterrupt):
                    with self.assertRaises(KeyboardInterrupt):loop_ops.show(watch=True)
