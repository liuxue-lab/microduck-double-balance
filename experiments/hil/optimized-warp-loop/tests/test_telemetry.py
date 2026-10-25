import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

from support import Timing, States
from telemetry import FIELDS


class Telemetry(unittest.TestCase):
    def test_clock_domains_and_last_interval_are_not_fabricated(self):
        now = [0]
        with tempfile.TemporaryDirectory() as d:
            timing = Timing(d, clock=lambda:now[0])
            for index in range(2):
                timing.begin()
                timing.add(**{key:1. for key in FIELDS if key not in ('start_to_start_ms','active_cycle_ms')})
                now[0] += 10_000_000
                timing.policy_returned()
                now[0] += 2_000_000
                timing.environment_started()
                now[0] += 23_000_000
                timing.end()
                now[0] += 5_000_000
            report = timing.export()
            self.assertEqual(report['all_steps']['active_cycle_ms']['mean'],35.)
            self.assertEqual(report['all_steps']['start_to_start_ms']['mean'],40.)
            self.assertEqual(report['all_steps']['start_to_start_ms']['count'],1)
            self.assertEqual(report['all_steps']['pre_environment_checks_ms']['mean'],2.)
            self.assertFalse(report['realtime_acceptance'])
            with (Path(d)/'timing.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[-1]['start_to_start_ms'],'')

    def test_partial_tick_is_visible_and_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as d:
            timing = Timing(d)
            timing.begin()
            with self.assertRaises(RuntimeError):timing.begin()
            with self.assertRaises(RuntimeError):timing.add(environment_ms=float('nan'))
            report = timing.export()
            self.assertTrue(report['incomplete_tick'])
            self.assertEqual(report['completed_ticks'],0)

    def test_state_export_owns_data_and_has_exact_initial_and_final_states(self):
        import torch
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            data=NS(qpos=torch.tensor([[1.,2.]]),qvel=torch.tensor([[3.,4.]]),ctrl=torch.tensor([[5.]]))
            states=States(d,NS(sim=NS(data=data)),torch)
            states.capture()
            data.qpos.add_(10);states.capture()
            meta=states.export()
            self.assertEqual(meta['samples'],2)
            with np.load(meta['path'],allow_pickle=False) as archive:
                self.assertEqual(archive['qpos'].tolist(),[[1.,2.],[11.,12.]])
                self.assertEqual(archive['simulation_time_s'].tolist(),[0.,.02])
                self.assertEqual(archive['qpos'].dtype,np.float32)
            with self.assertRaises(FileExistsError):states.export()
