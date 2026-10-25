import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'package'))
import bootstrap
bootstrap.load_base(Path(os.environ.get('MICRODUCK_REFERENCE_PACKAGE', str(bootstrap.BASE_PACKAGE))))
import action_contract as contract
import auxiliary

REPORT = Path(os.environ.get('MICRODUCK_REVIEWED_REPORT', str(
    bootstrap.STAGE / 'native-pc-timeline-fixed-v2/evidence/reviewed-repair-report.json')))


def row(mode, mean, p95):
    value = json.loads(REPORT.read_bytes())['arms']['inference-only']
    value.update(status='ACTIONGRAPH_ARM_COMPLETE', mode=mode)
    value['episode_result']['stop_reason'] = 'ACTIONGRAPH_SAMPLE_COMPLETE'
    value['full_environment_step_ms'].update(mean=mean, p95=p95, p99=p95 + 1., max=p95 + 2.)
    return value


class Selection(unittest.TestCase):
    def arms(self):
        return {'baseline': row('baseline', 25., 26.), 'action-graph': row('action-graph', 22., 23.),
                'cumulative': row('cumulative', 21., 22.)}

    def test_fastest_qualifying_candidate_selected(self):
        self.assertEqual(contract.preliminary(self.arms()), 'cumulative')

    def test_mean_only_gain_is_insufficient(self):
        arms = self.arms()
        for mode in ('action-graph', 'cumulative'):
            arms[mode]['full_environment_step_ms'].update(mean=15., p95=25.9)
        self.assertIsNone(contract.preliminary(arms))
        decision = contract.recommendation(arms)
        self.assertEqual(decision['choice'], 'KEEP_INFERENCE_ONLY')
        self.assertFalse(decision['full_loop_realtime_acceptance'])

    def test_repeat_must_beat_faster_baseline(self):
        arms = self.arms()
        arms['candidate-confirm'] = row('cumulative', 22., 23.)
        arms['baseline-confirm'] = row('baseline', 23., 24.)
        decision = contract.recommendation(arms, 'cumulative')
        self.assertEqual(decision['choice'], 'KEEP_INFERENCE_ONLY')
        self.assertFalse(decision['confirmation_passed'])

    def test_two_successful_measurements_still_do_not_claim_hil_pass(self):
        arms = self.arms()
        arms['candidate-confirm'] = row('cumulative', 21.1, 22.1)
        arms['baseline-confirm'] = row('baseline', 24.9, 25.9)
        decision = contract.recommendation(arms, 'cumulative')
        self.assertEqual(decision['choice'], 'VERIFIED_CUMULATIVE_CANDIDATE')
        self.assertTrue(decision['confirmation_passed'])
        self.assertFalse(decision['production_activated'])
        self.assertFalse(decision['full_loop_realtime_acceptance'])

    def test_incomplete_or_nonfinite_arm_cannot_be_selected(self):
        for field, value in (('p95', float('nan')), ('mean', 0.)):
            arm = row('action-graph', 22., 23.)
            arm['full_environment_step_ms'][field] = value
            self.assertFalse(contract.arm_valid(arm))
        arm = row('action-graph', 22., 23.)
        arm['episode_result']['model_limits']['motor_checks_before_integration'] -= 1
        self.assertFalse(contract.arm_valid(arm))


class AuxiliaryTrace(unittest.TestCase):
    def test_all_control_boundaries_required_and_rng_difference_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            a, b = Path(td) / 'a.jsonl', Path(td) / 'b.jsonl'
            rows = [dict(step=i, groups=[{'history': 'identical'}], cpu_rng='abc', cuda_rng=['def']) for i in range(161)]
            def save(path): path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            save(a); save(b)
            self.assertEqual(auxiliary.compare(a, b)['snapshots'], 161)
            rows[158]['cuda_rng'] = ['different']
            save(b)
            with self.assertRaisesRegex(RuntimeError, 'step 158'): auxiliary.compare(a, b)
            rows.pop()
            save(b)
            with self.assertRaises(RuntimeError): auxiliary.compare(a, b)


if __name__ == '__main__':
    unittest.main(verbosity=2)
