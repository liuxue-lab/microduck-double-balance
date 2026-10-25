import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'package'))
import bootstrap
BASE = Path(os.environ.get('MICRODUCK_REFERENCE_PACKAGE', str(bootstrap.BASE_PACKAGE)))
old = bootstrap.load_base(BASE)
with patch.object(bootstrap, 'load_base', return_value=old):
    import loop_sim
    import loop_ops
import loop_contract as c
import cumulative_runtime
import loop_evidence
from telemetry import Timing, States


def adapter_stats(steps=3000):
    return dict(runtime_bindings_restored=True, phase_cache=dict(cache_hits=1),
        bam_graphs=[dict(replays=steps*10, same_input_checked_calls=40)],
        action_graph=dict(closed=True, bindings_restored=True, replays=steps*10, controls=steps,
                          staged_controls=steps, same_input_checked_calls=40, comparison_failure=None),
        static_quaternion=dict(closed=True, checked_calls=40, calls=steps),
        packed_checks={k:steps*10 for k in ('zero_wrench_checks_before_write',
            'packed_reads_before_integration', 'packed_reads_after_integration')},
        scalar_audit=dict(calls={k:steps*10 for k in ('before_integration', 'after_integration')},
            same_input_checked_calls={k:40 for k in ('before_integration', 'after_integration')},
            last_comparison_failure=None),
        inference_only=dict(omitted_reward_calls=steps, actor_group_calls=steps, unexpected_critic_calls=0),
        state_export=dict(samples=steps+1), close_errors=[])


def case_row(case):
    adapters = adapter_stats()
    return dict(**case, loop_passed=True, acceptance=dict(passed=True), completed_control_steps=3000,
        observed_physics_steps=30000, runtime_adapters=adapters,
        adapter_checks=cumulative_runtime.adapter_checks(adapters),
        live_action_parity=dict(compared_actions=3000, every_applied_action_came_from_Orin=True, local_fallback=False),
        timing=dict(completed_ticks=3000, incomplete_tick=False,
                    all_steps=dict(environment_ms=dict(count=3000, mean=23., p95=24., max=30.))))
