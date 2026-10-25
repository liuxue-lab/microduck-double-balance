"""Separate clocks and bounded state export; no actuator scheduling or rendering."""
import csv
import json
import math
from pathlib import Path
import time

from deploy_contract import require
from loop_contract import STEPS, measured

FIELDS = ('obs_copy_ms', 'request_call_ms', 'ipc_roundtrip_ms', 'ros_roundtrip_ms',
          'orin_inference_ms', 'reference_compare_ms', 'action_to_cuda_ms',
          'pre_environment_checks_ms', 'environment_ms', 'state_capture_ms',
          'active_cycle_ms', 'start_to_start_ms')


class Timing:
    def __init__(self, folder, clock=time.perf_counter_ns):
        self.folder, self.clock = Path(folder), clock
        self.rows, self.current = [], None
        self.previous_start = None

    def begin(self):
        require(self.current is None and len(self.rows) < STEPS, 'Unfinished or excess control tick')
        now = self.clock()
        if self.rows:
            self.rows[-1]['start_to_start_ms'] = (now - self.previous_start) / 1e6
        self.previous_start = now
        self.current = dict(step=len(self.rows) + 1, simulation_time_s=(len(self.rows) + 1) * .02,
                            started_ns=now, start_to_start_ms=None)
        return now

    def add(self, **values):
        require(self.current is not None, 'No active timing row')
        for key, value in values.items():
            require(key in FIELDS and type(value) in (float, int) and math.isfinite(value) and value >= 0,
                    'Invalid timing field: ' + key)
            self.current[key] = value

    def policy_returned(self):
        self.current['policy_returned_ns'] = self.clock()

    def environment_started(self):
        now = self.clock()
        self.add(pre_environment_checks_ms=(now - self.current['policy_returned_ns']) / 1e6)
        return now

    def end(self):
        self.add(active_cycle_ms=(self.clock() - self.current['started_ns']) / 1e6)
        require(all(key in self.current for key in FIELDS), 'Missing timing domain')
        row = {key: value for key, value in self.current.items() if not key.endswith('_ns')}
        self.rows.append(row)
        self.current = None

    def export(self):
        path = self.folder / 'timing.csv'
        with path.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=('step', 'simulation_time_s', *FIELDS))
            writer.writeheader()
            writer.writerows(self.rows)
        return self.summary()

    def summary(self):
        def summarize(rows):
            return {key: measured([r[key] for r in rows if r[key] is not None]) for key in FIELDS}
        return dict(all_steps=summarize(self.rows), after_initial_10_steps=summarize(self.rows[10:]),
            completed_ticks=len(self.rows), incomplete_tick=self.current is not None,
            csv_path=str(self.folder / 'timing.csv'), nominal_period_ms=20.,
            realtime_acceptance=False, hardware_acceptance=False,
            notes=[
                'Lockstep validation, without wall-clock release scheduling; timing is diagnostic.',
                'RPC/ROS/inference measurements are nested intervals and must not be added together.',
                'Orin inference uses the Orin local clock; RPC uses the PC clock, no clock subtraction across hosts.',
                'Environment time includes CUDA completion and every original 500Hz check.',
                'Active cycle includes reference parity and state capture, but not subsequent outer-loop checks.',
                'Start-to-start includes outer checks and progress logging; the final tick has no next start.',
                'The first 10 steps are included in all_steps; a separate steady-state view is diagnostic only.',
                'PC simulation duration is not Orin inference latency or physical-robot deployment acceptance.'])


class States:
    """At most 3001 post-reset/post-control states, for later plots or state replay."""
    def __init__(self, folder, raw, torch):
        import numpy as np
        self.np, self.folder, self.raw, self.torch = np, Path(folder), raw, torch
        self.rows = []
        data = raw.sim.data
        self.widths = [getattr(data, key)[0].numel() for key in ('qpos', 'qvel', 'ctrl')]
        require(sum(self.widths) * 4 * (STEPS + 1) <= 16 * 1024**2, 'State export exceeds bounded memory')

    def capture(self):
        require(len(self.rows) <= STEPS, 'State export budget exceeded')
        data = self.raw.sim.data
        fields = [getattr(data, key)[0].detach().reshape(-1) for key in ('qpos', 'qvel', 'ctrl')]
        require([v.numel() for v in fields] == self.widths and
                all(v.dtype == self.torch.float32 for v in fields), 'State schema changed')
        row = self.torch.cat(fields).cpu().numpy().copy()
        require(bool(self.np.isfinite(row).all()), 'Nonfinite exported state')
        self.rows.append(row)

    def export(self):
        np = self.np
        data = np.stack(self.rows) if self.rows else np.empty((0, sum(self.widths)), dtype=np.float32)
        q, v = self.widths[:2]
        path = self.folder / 'state-data.npz'
        with path.open('xb') as stream:
            np.savez_compressed(stream, qpos=data[:, :q], qvel=data[:, q:q+v], ctrl=data[:, q+v:],
                                control_step=np.arange(len(data)), simulation_time_s=np.arange(len(data)) * .02)
        meta = dict(path=str(path), samples=len(self.rows), fields=['qpos', 'qvel', 'ctrl'], widths=self.widths,
                    cadence_hz=50, initial_state_included=True, dtype='float32',
                    use='offline plots and state replay with the same frozen model',
                    warning='Recorded state replay is not a new closed-loop simulation or a simulator checkpoint.')
        with (self.folder / 'state-data.json').open('x') as stream:
            json.dump(meta, stream, indent=2)
        return meta
