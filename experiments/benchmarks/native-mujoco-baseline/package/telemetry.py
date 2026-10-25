"""Fixed-release 50Hz scheduling and native-state export; no frame dropping."""
import csv
import json
import time
from pathlib import Path

from deploy_contract import require
from loop_contract import STEPS, measured

FIELDS = ('release_lateness_ms', 'obs_copy_ms', 'request_call_ms', 'ipc_roundtrip_ms',
          'ros_roundtrip_ms', 'orin_inference_ms', 'reference_compare_ms', 'action_to_cpu_ms',
          'pre_environment_checks_ms', 'environment_ms', 'state_capture_ms',
          'active_cycle_ms', 'release_to_finish_ms', 'deadline_overrun_ms', 'start_to_start_ms')


class Timing:
    PERIOD_NS = 20_000_000

    def __init__(self, folder, clock=time.perf_counter_ns, sleep=time.sleep):
        self.folder, self.clock, self.sleep = Path(folder), clock, sleep
        self.rows, self.current, self.epoch, self.previous_start = [], None, None, None

    def begin(self):
        require(self.current is None and len(self.rows) < STEPS, 'Unfinished/excess tick')
        if self.epoch is None:
            self.epoch = self.clock()
        release = self.epoch + len(self.rows) * self.PERIOD_NS
        while True:
            now = self.clock()
            remaining = release - now
            if remaining <= 0:
                break
            self.sleep(remaining / 1e9)
        if self.rows:
            self.rows[-1]['start_to_start_ms'] = (now - self.previous_start) / 1e6
        self.previous_start = now
        self.current = dict(step=len(self.rows)+1, simulation_time_s=(len(self.rows)+1)*.02,
            release_ns=release, started_ns=now, deadline_ns=release+self.PERIOD_NS,
            release_lateness_ms=(now-release)/1e6, start_to_start_ms=None)
        return now

    def add(self, **values):
        import math
        require(self.current is not None, 'No active timing row')
        for key, value in values.items():
            require(key in FIELDS and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0,
                    'Invalid timing field: '+key)
            self.current[key] = value

    def policy_returned(self):
        self.current['policy_returned_ns'] = self.clock()

    def environment_started(self):
        now = self.clock()
        self.add(pre_environment_checks_ms=(now-self.current['policy_returned_ns'])/1e6)
        return now

    def end(self):
        now = self.clock()
        self.add(active_cycle_ms=(now-self.current['started_ns'])/1e6,
            release_to_finish_ms=(now-self.current['release_ns'])/1e6,
            deadline_overrun_ms=max(0, now-self.current['deadline_ns'])/1e6)
        require(all(key in self.current for key in FIELDS), 'Missing timing domain')
        self.current['finished_ns'] = now
        self.rows.append(dict(self.current))
        self.current = None

    def summary(self):
        complete = len(self.rows) == STEPS and self.current is None
        missed = sum(r['finished_ns'] > r['deadline_ns'] for r in self.rows)
        anchored = all(r['release_ns'] == self.epoch+i*self.PERIOD_NS for i,r in enumerate(self.rows))
        return dict(completed_ticks=len(self.rows), incomplete_tick=self.current is not None,
            all_steps={key: measured([r[key] for r in self.rows if r[key] is not None]) for key in FIELDS},
            deadline_misses=missed, release_schedule_anchored=anchored,
            nominal_period_ms=20., realtime_acceptance=complete and missed == 0 and anchored,
            first_tick_included=True, dropped_ticks=0, schedule_rebases=0,
            csv_path=str(self.folder/'timing.csv'),
            notes=['Deadlines include Orin RPC, reference comparison, physics, all bounds/force checks, state capture and progress logging.',
                   'All 3000 ticks count, including the first; no late ticks are skipped or hidden.',
                   'Inference, ROS and IPC intervals are nested and must not be summed.',
                   'This measures scheduled PC-Orin HIL only; no physical robot acceptance.'])

    def export(self):
        keys = ('step', 'simulation_time_s', 'release_ns', 'started_ns', 'deadline_ns',
                'policy_returned_ns', 'finished_ns', *FIELDS)
        with (self.folder/'timing.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.rows)
        return self.summary()


class States:
    def __init__(self, folder, raw, torch):
        import numpy as np
        import mujoco
        self.np, self.folder, self.raw = np, Path(folder), raw
        self.rows = []
        data = raw.sim.mj_data
        self.widths = [getattr(data, key).size for key in ('qpos', 'qvel', 'ctrl')]
        require(sum(self.widths)*8*(STEPS+1) <= 32*1024**2, 'State export exceeds memory budget')
        mujoco.mj_saveModel(raw.sim.native_model, str(self.folder/'native-model.mjb'), None)

    def capture(self):
        require(len(self.rows) <= STEPS, 'State export budget exceeded')
        data = self.raw.sim.mj_data
        row = self.np.concatenate([getattr(data, key) for key in ('qpos', 'qvel', 'ctrl')])
        require(self.np.isfinite(row).all(), 'Nonfinite exported native state')
        self.rows.append(row)

    def export(self):
        np = self.np
        array = np.stack(self.rows) if self.rows else np.empty((0, sum(self.widths)), dtype=np.float64)
        q,v = self.widths[:2]
        path = self.folder/'state-data.npz'
        with path.open('xb') as stream:
            np.savez_compressed(stream, qpos=array[:,:q], qvel=array[:,q:q+v], ctrl=array[:,q+v:],
                control_step=np.arange(len(array)), simulation_time_s=np.arange(len(array))*.02)
        result = dict(path=str(path), samples=len(self.rows), fields=['qpos','qvel','ctrl'],
            widths=self.widths, dtype='float64', cadence_hz=50, initial_state_included=True,
            model_path=str(self.folder/'native-model.mjb'), mujoco_version='3.10.0',
            use='Native model plus state replay; integration is not rerun during replay',
            exact_restart_checkpoint=False)
        (self.folder/'state-data.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
