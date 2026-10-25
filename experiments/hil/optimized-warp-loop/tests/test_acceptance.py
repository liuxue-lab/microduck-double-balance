import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest

from support import c, case_row, loop_evidence, loop_ops


class Acceptance(unittest.TestCase):
    def reports(self):
        shared = dict(completed_requests=6000, transcript_sha256='a'*64)
        sim = dict(status='LOOP_SIM_FUNCTIONAL_PASSED', original_records_unchanged=True,
                   completed_control_steps=6000, cases=[case_row(case) for case in c.CASES], transport=shared)
        return sim, dict(status='LOOP_BRIDGE_COMPLETE', **shared), dict(status='LOOP_SERVER_COMPLETE',
            model_unchanged=True, recurrent_reset_sequences=[0,3000], **shared)

    def test_functional_success_never_becomes_realtime_success(self):
        sim, bridge, server = self.reports()
        for mean in (23., 1.):
            sim['cases'][0]['timing']['all_steps']['environment_ms']['mean'] = mean
            result = loop_evidence.verify_reports(sim, bridge, server)
            self.assertTrue(result['functional_closed_loop'])
            self.assertFalse(result['full_loop_realtime_acceptance'])
            self.assertEqual(result['real_time_HIL'], 'PENDING')
            self.assertTrue(result['original_realtime_requirement_retained'])

    def test_missing_audits_forged_check_flags_or_transcripts_fail(self):
        sim, bridge, server = self.reports()
        mutations = [lambda r:r['cases'][0]['runtime_adapters']['packed_checks'].update(packed_reads_after_integration=29999),
            lambda r:r['cases'][1]['runtime_adapters']['action_graph'].update(staged_controls=2999),
            lambda r:r['cases'][0]['live_action_parity'].update(local_fallback=True),
            lambda r:r['cases'][0]['timing'].update(incomplete_tick=True),
            lambda r:r['cases'].reverse(),
            lambda r:r['transport'].update(transcript_sha256='b'*64)]
        for mutate in mutations:
            candidate = copy.deepcopy(sim)
            mutate(candidate)
            with self.assertRaises(RuntimeError):
                loop_evidence.verify_reports(candidate, bridge, server)
        with self.assertRaises(RuntimeError):
            loop_evidence.verify_reports(sim, bridge, dict(server, recurrent_reset_sequences=[0]))

    def test_watch_remains_open_after_completion(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            (path/'report.json').write_text(json.dumps(dict(
                status='CUMULATIVE_LOOP_FUNCTIONAL_PASSED_REALTIME_PENDING', directory=d, phase='finished')))
            with patch.object(loop_ops, 'ROOT', path), patch.object(loop_ops.time, 'sleep', side_effect=KeyboardInterrupt), patch('builtins.print'):
                with self.assertRaises(KeyboardInterrupt):
                    loop_ops.show(watch=True)
