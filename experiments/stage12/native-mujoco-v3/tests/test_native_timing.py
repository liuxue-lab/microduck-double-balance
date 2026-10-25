"""A deadline miss remains a miss; no rebasing, dropped frames, or hidden warmup."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import telemetry
from telemetry import Timing,FIELDS
from native_plan import make_schedule,pulse_index


class Clock:
    def __init__(self):self.now=1_000_000_000;self.sleeps=[]
    def clock(self):return self.now
    def sleep(self,s):self.sleeps.append(s);self.now+=round(s*1e9)


class TimingTests(unittest.TestCase):
    def run_ticks(self,durations):
        clock=Clock();timing=Timing(Path('/unused'),clock.clock,clock.sleep,spin_ns=0)
        with patch.object(telemetry,'STEPS',len(durations)):
            for duration in durations:
                timing.begin();timing.policy_returned()
                timing.add(**{k:0. for k in FIELDS if k not in ('start_to_start_ms','release_lateness_ms')})
                clock.now+=duration*1_000_000;timing.end()
            return timing,timing.summary(),clock

    def test_all_ticks_count_and_50hz_releases_remain_anchored(self):
        timing,result,clock=self.run_ticks([12,15,17])
        self.assertTrue(result['realtime_acceptance'])
        self.assertEqual(result['deadline_misses'],0)
        self.assertEqual([r['release_ns'] for r in timing.rows],[1_000_000_000,1_020_000_000,1_040_000_000])
        self.assertEqual(len(clock.sleeps),2)

    def test_first_tick_overrun_is_not_discarded_and_next_release_is_not_rebased(self):
        timing,result,clock=self.run_ticks([28,15,5])
        self.assertFalse(result['realtime_acceptance'])
        self.assertEqual(result['deadline_misses'],2)
        self.assertEqual(timing.rows[1]['release_lateness_ms'],8.)
        self.assertEqual(result['dropped_ticks'],0)
        self.assertEqual(result['schedule_rebases'],0)

    def test_force_duration_impulse_and_onsets_survive_step_change(self):
        plan=make_schedule(2026102500)
        self.assertEqual([p['start_seconds'] for p in plan],list(range(2,60,2)))
        self.assertTrue(all(p['expected_physics_steps']==20 for p in plan))
        self.assertEqual(sum(pulse_index(i) is not None for i in range(12000)),580)
        self.assertTrue(all(abs(p['end_seconds']-p['start_seconds']-.1)<1e-12 for p in plan))


if __name__=='__main__':unittest.main()
