"""Fixed-release 50Hz scheduling and native-state export; no frame dropping."""
import csv
import json
import resource
import time
from pathlib import Path

from deploy_contract import require
from loop_contract import STEPS, measured

FIELDS = ('release_lateness_ms', 'obs_copy_ms', 'request_call_ms', 'ipc_roundtrip_ms',
          'ros_roundtrip_ms', 'orin_inference_ms', 'reference_compare_ms', 'action_to_cpu_ms',
          'reference_inference_ms',
          'pre_environment_checks_ms', 'environment_ms', 'state_capture_ms',
          'post_environment_checks_ms', 'progress_emit_ms',
          'control_thread_cpu_ms', 'non_cpu_cycle_ms',
          'active_cycle_ms', 'release_to_finish_ms', 'deadline_overrun_ms', 'start_to_start_ms')

RESOURCE_FIELDS = ('major_page_faults', 'minor_page_faults', 'involuntary_context_switches')


def event_row(row):
    return {key: row[key] for key in ('step', 'simulation_time_s', *FIELDS, *RESOURCE_FIELDS)
            if key in row}


class Timing:
    PERIOD_NS = 20_000_000

    def __init__(self, folder, clock=time.perf_counter_ns, sleep=time.sleep,spin_ns=100_000):
        self.folder, self.clock, self.sleep = Path(folder), clock, sleep
        self.rows, self.current, self.epoch, self.previous_start = [], None, None, None
        self.spin_ns=spin_ns

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
            if remaining>self.spin_ns:self.sleep((remaining-self.spin_ns)/1e9)
        if self.rows:
            self.rows[-1]['start_to_start_ms'] = (now - self.previous_start) / 1e6
        self.previous_start = now
        self.current = dict(step=len(self.rows)+1, simulation_time_s=(len(self.rows)+1)*.02,
            release_ns=release, started_ns=now, deadline_ns=release+self.PERIOD_NS,
            thread_started_ns=time.thread_time_ns(),
            release_lateness_ms=(now-release)/1e6, start_to_start_ms=None)
        self.resources_start = resource.getrusage(resource.RUSAGE_THREAD)
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
        resources_end = resource.getrusage(resource.RUSAGE_THREAD)
        cpu_end=time.thread_time_ns()
        now = self.clock()
        cpu_ns=cpu_end-self.current['thread_started_ns']
        self.add(control_thread_cpu_ms=cpu_ns/1e6,
            non_cpu_cycle_ms=max(0,now-self.current['started_ns']-cpu_ns)/1e6)
        self.add(active_cycle_ms=(now-self.current['started_ns'])/1e6,
            release_to_finish_ms=(now-self.current['release_ns'])/1e6,
            deadline_overrun_ms=max(0, now-self.current['deadline_ns'])/1e6)
        require(all(key in self.current for key in FIELDS), 'Missing timing domain')
        self.current['finished_ns'] = now
        self.current.update(
            major_page_faults=resources_end.ru_majflt-self.resources_start.ru_majflt,
            minor_page_faults=resources_end.ru_minflt-self.resources_start.ru_minflt,
            involuntary_context_switches=resources_end.ru_nivcsw-self.resources_start.ru_nivcsw)
        self.rows.append(dict(self.current))
        self.current = None

    def summary(self):
        complete = len(self.rows) == STEPS and self.current is None
        missed = sum(r['finished_ns'] > r['deadline_ns'] for r in self.rows)
        anchored = all(r['release_ns'] == self.epoch+i*self.PERIOD_NS for i,r in enumerate(self.rows))
        late = [r for r in self.rows if r['finished_ns'] > r['deadline_ns']]
        worst = sorted(self.rows, key=lambda r: r['release_to_finish_ms'], reverse=True)
        return dict(completed_ticks=len(self.rows), incomplete_tick=self.current is not None,
            all_steps={key: measured([r[key] for r in self.rows if r[key] is not None]) for key in FIELDS},
            deadline_misses=missed, release_schedule_anchored=anchored,
            nominal_period_ms=20., realtime_acceptance=complete and missed == 0 and anchored,
            first_tick_included=True, dropped_ticks=0, schedule_rebases=0,
            first_tick=event_row(self.rows[0]) if self.rows else None,
            deadline_events=[event_row(r) for r in late[:32]],
            deadline_events_total=len(late), deadline_events_truncated=len(late)>32,
            worst_ticks=[event_row(r) for r in worst[:5]],
            peak_reference_tick=event_row(max(self.rows,key=lambda r:r['reference_inference_ms'])) if self.rows else None,
            resource_counters={key:sum(r[key] for r in self.rows) for key in RESOURCE_FIELDS},
            csv_path=str(self.folder/'timing.csv'),
            notes=['Deadlines include Orin RPC, reference comparison, physics, all bounds/force checks, state capture and live progress emission.',
                   'The supervisor persists progress files; the control process never fsyncs live display updates.',
                   'All 3000 ticks count, including the first; no late ticks are skipped or hidden.',
                   'Inference, ROS and IPC intervals are nested and must not be summed.',
                   'The unchanged local reference inference overlaps the Orin request; reference_compare_ms measures the subsequent parity check.',
                   'This measures scheduled PC-Orin HIL only; no physical robot acceptance.'])

    def export(self):
        keys = ('step', 'simulation_time_s', 'release_ns', 'started_ns', 'deadline_ns',
                'thread_started_ns', 'policy_returned_ns', 'finished_ns', *FIELDS, *RESOURCE_FIELDS)
        with (self.folder/'timing.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.rows)
        return self.summary()


class States:
    def __init__(self, folder, raw, torch, pending_models=None, live_timing=None):
        import numpy as np
        import mujoco
        self.np, self.folder, self.raw = np, Path(folder), raw
        self.count=0
        data = raw.sim.mj_data
        self.widths = [getattr(data, key).size for key in ('qpos', 'qvel', 'ctrl')]
        require(sum(self.widths)*8*(STEPS+1) <= 32*1024**2, 'State export exceeds memory budget')
        self.array=np.empty((STEPS+1,sum(self.widths)),dtype=np.float64)
        self.array.fill(0.)
        self.model_buffer=np.empty(mujoco.mj_sizeModel(raw.sim.native_model),dtype=np.uint8)
        mujoco.mj_saveModel(raw.sim.native_model,None,self.model_buffer)
        self.pending_models=pending_models
        import os, hashlib
        from live_stream import Publisher
        from live_support import MODEL_SHA
        require(self.widths == [35,32,14], 'Unexpected live state shape')
        require(hashlib.sha256(self.model_buffer).hexdigest() == MODEL_SHA,
                'New native visual model differs from accepted v3')
        require(live_timing is not None, 'Live publication must be timed')
        self.publisher = Publisher(int(os.environ['NATIVE_LIVE_FD']), int(self.folder.name[-2:]), live_timing, np)

    def capture(self):
        require(self.count <= STEPS, 'State export budget exceeded')
        data = self.raw.sim.mj_data
        row=self.array[self.count];offset=0
        for key,width in zip(('qpos','qvel','ctrl'),self.widths):
            self.np.copyto(row[offset:offset+width],getattr(data,key));offset+=width
        require(self.np.isfinite(row).all(), 'Nonfinite exported native state')
        self.publisher.publish(row, self.count)
        self.count+=1

    def export(self):
        np = self.np
        array = self.array[:self.count]
        q,v = self.widths[:2]
        path = self.folder/'state-data.npz'
        with path.open('xb') as stream:
            np.savez_compressed(stream, qpos=array[:,:q], qvel=array[:,q:q+v], ctrl=array[:,q+v:],
                control_step=np.arange(len(array)), simulation_time_s=np.arange(len(array))*.02)
        from realtime_io import export_models
        model=(self.folder/'native-model.mjb',self.model_buffer)
        if self.pending_models is None:export_models([model])
        else:self.pending_models.append(model)
        self.model_buffer=None
        result = dict(path=str(path), samples=self.count, fields=['qpos','qvel','ctrl'],
            widths=self.widths, dtype='float64', cadence_hz=50, initial_state_included=True,
            model_path=str(self.folder/'native-model.mjb'), mujoco_version='3.10.0',
            use='Native model plus state replay; integration is not rerun during replay',
            exact_restart_checkpoint=False, live_display=self.publisher.snapshot())
        (self.folder/'state-data.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
