"""Exact numerical preservation and conservative repeated performance gates."""
import json
from pathlib import Path
import math

from bootstrap import require
from readgraph_contract import STEPS, WARMUP, compare_traces, compare_audits
from readgraph_contract import valid_partial as bounded
import auxiliary

MODES = ('baseline', 'action-graph', 'cumulative')
MINIMUM_GAIN = .05


def valid_partial(row):
    return bounded(row, 'ACTIONGRAPH_SAMPLE_COMPLETE')


def arm_valid(row):
    if not (row.get('status') == 'ACTIONGRAPH_ARM_COMPLETE' and row.get('control_steps') == STEPS and
            row.get('physics_steps') == STEPS * 10 and row.get('runtime_bindings_restored') is True and
            valid_partial(row.get('episode_result', {}))):
        return False
    timing = row.get('full_environment_step_ms', {})
    return (timing.get('count') == STEPS - WARMUP and
            all(type(timing.get(k)) in (int, float) and math.isfinite(timing[k]) and timing[k] > 0
                for k in ('mean', 'p95', 'p99', 'max')))


def compare_folders(first, second):
    a = json.loads((first / 'worker-report.json').read_bytes())
    b = json.loads((second / 'worker-report.json').read_bytes())
    require(arm_valid(a) and arm_valid(b), 'Cannot compare incomplete arms')
    parity = compare_traces(first / 'state-trace.jsonl', second / 'state-trace.jsonl')
    parity.update(compare_audits(a, b))
    parity['all_numeric_state_differences_zero'] = all(v == 0 for v in parity['max_absolute_error_by_field'].values())
    parity['delay_and_rng'] = auxiliary.compare(first / 'aux-state.jsonl', second / 'aux-state.jsonl')
    parity['passed'] = (parity['passed'] and parity['scalar_comparisons'] == 73440 and
        parity['all_numeric_state_differences_zero'] and all(parity[k] for k in
        ('full_audit_equal', 'trial_equal', 'disturbance_equal')) and parity['delay_and_rng']['passed']
        and parity['delay_and_rng']['exact_action_history_and_targets'])
    require(parity['passed'], 'State, audit, force schedule or RNG differed')
    return parity


def gains(baseline, candidate):
    return {k: 1. - candidate[k] / baseline[k] for k in ('mean', 'p95')}


def preliminary(arms):
    base = arms['baseline']['full_environment_step_ms']
    choices = []
    for mode in MODES[1:]:
        row = arms[mode]
        require(arm_valid(row), 'Candidate arm is incomplete')
        gain = gains(base, row['full_environment_step_ms'])
        if min(gain.values()) >= MINIMUM_GAIN:
            choices.append((row['full_environment_step_ms']['mean'], mode))
    return min(choices)[1] if choices else None


def recommendation(arms, selected=None):
    result = dict(choice='KEEP_INFERENCE_ONLY', production_activated=False, full_loop_realtime_acceptance=False,
                  minimum_mean_and_p95_gain=MINIMUM_GAIN, selection_retested=False,
                  normal_environment_steps_only=True, profiler_enabled=False)
    if selected is None:
        result['reason'] = 'No exact-parity candidate improved both mean and p95 by at least 5%'
        return result
    for name in ('baseline', 'baseline-confirm', selected, 'candidate-confirm'):
        require(arm_valid(arms[name]), 'Incomplete confirmation arm: ' + name)
    require(arms['candidate-confirm']['mode'] == selected, 'Confirmation used a different candidate')
    reference = {k: min(arms['baseline']['full_environment_step_ms'][k],
                        arms['baseline-confirm']['full_environment_step_ms'][k]) for k in ('mean', 'p95')}
    checks = {name: gains(reference, arms[name]['full_environment_step_ms'])
              for name in (selected, 'candidate-confirm')}
    passed = all(min(row.values()) >= MINIMUM_GAIN for row in checks.values())
    result.update(selection_retested=True, proposed_mode=selected, conservative_baseline_ms=reference,
                  gains_against_faster_baseline=checks, confirmation_passed=passed,
                  reason='Both candidate measurements must beat the faster baseline on mean and p95')
    if passed:
        result['choice'] = 'VERIFIED_' + selected.upper().replace('-', '_') + '_CANDIDATE'
    return result
