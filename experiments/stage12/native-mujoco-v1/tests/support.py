import copy
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'package'))
import bootstrap
BASE=Path(os.environ.get('MICRODUCK_REFERENCE_PACKAGE',str(bootstrap.BASE_PACKAGE)))
bootstrap.BASE_PACKAGE=BASE
old=bootstrap.load_base(BASE)
with patch.object(bootstrap,'load_base',return_value=old):
    import loop_sim
    import loop_ops
import loop_contract as c
import native_runtime
import native_preflight
import loop_evidence
from telemetry import Timing, States

ROWS=json.loads((ROOT/'tests/fixtures/physical-cases.json').read_text())['cases']
def adapter_stats(steps=3000):
    assert steps==3000
    return copy.deepcopy(ROWS[0]['runtime_adapters'])

def case_row(case,realtime=False):
    row=copy.deepcopy(next(r for r in ROWS if r['case_id']==case['case_id']))
    row.update(loop_passed=True,adapter_checks=native_runtime.adapter_checks(row['runtime_adapters']),
        live_action_parity=dict(compared_actions=3000,every_applied_action_came_from_Orin=True,local_fallback=False),
        timing=dict(completed_ticks=3000,incomplete_tick=False,release_schedule_anchored=True,
            dropped_ticks=0,schedule_rebases=0,first_tick_included=True,
            deadline_misses=0 if realtime else 1,realtime_acceptance=realtime,
            all_steps=dict(environment_ms=dict(count=3000,mean=11.,max=30.),
                release_to_finish_ms=dict(count=3000,max=18. if realtime else 25.,over_20ms=0 if realtime else 1))))
    return row
