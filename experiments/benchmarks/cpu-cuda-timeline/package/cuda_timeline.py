"""Short Kineto trace of the unchanged accepted runtime; never a speed claim.

Keep an unprofiled baseline. Only original environment-step marker intervals
are included in the summary: policy inference and outer parity trace copies
between steps may appear in the raw file but are excluded here.
"""
from contextlib import contextmanager, nullcontext
import functools
import json
import math
from pathlib import Path

from deploy_contract import require, sha
from phase_cache import Bindings

FIRST = 101
LAST = 108
WINDOWS = LAST - FIRST + 1


def union_length(intervals):
    total = 0.
    end = None
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
        elif stop > end:
            total += stop - end
        end = stop if end is None else max(end, stop)
    return total


def interval(event):
    start, duration = event.get('ts'), event.get('dur')
    require(type(start) in (int, float) and type(duration) in (int, float)
            and math.isfinite(start) and math.isfinite(duration) and duration >= 0,
            'Malformed timeline timestamp or duration')
    return start, start + duration


def summarize(document, expected_steps=tuple(range(FIRST, LAST + 1))):
    """Union CUDA intervals, do not sum overlapping kernels as elapsed time."""
    events = document.get('traceEvents')
    require(isinstance(events, list), 'Missing Chrome traceEvents')
    complete = [e for e in events if e.get('ph') == 'X']
    markers = []
    for e in complete:
        if e.get('cat') == 'user_annotation' and e.get('name', '').startswith('md12.env.'):
            step = int(e['name'].split('.')[-1])
            markers.append((step, *interval(e)))
    markers.sort()
    require(tuple(x[0] for x in markers) == tuple(expected_steps),
            'Incomplete or duplicate original environment markers')
    require(all(markers[i][2] <= markers[i+1][1] for i in range(len(markers)-1)),
            'Overlapping environment-step markers')
    windows = [(a, b) for _, a, b in markers]
    wall = sum(b-a for a,b in windows)
    require(wall > 0, 'Empty environment-step timeline')
    buckets = {k:{} for k in ('kernel','gpu_memcpy','gpu_memset','cuda_runtime','cuda_driver','cpu_op','user_annotation')}
    gpu_intervals = []
    kernel_intervals = []
    copy_intervals = []
    per_step = {step:[] for step, _, _ in markers}
    per_step_kernels = {step:0 for step, _, _ in markers}
    categories = {}
    for event in complete:
        category = event.get('cat', '')
        categories[category] = categories.get(category, 0) + 1
        if category not in buckets:
            continue
        a, b = interval(event)
        clips = [(step, max(a, left), min(b, right)) for step,left,right in markers
                 if min(b, right) > max(a, left)]
        if not clips:
            continue
        duration = sum(stop-start for _,start,stop in clips)
        name = str(event.get('name', '<unnamed>'))
        row = buckets[category].setdefault(name, dict(name=name, count=0, clipped_total_ms=0.))
        row['count'] += 1
        row['clipped_total_ms'] += duration / 1000.
        if category in ('kernel','gpu_memcpy','gpu_memset'):
            for step,start,stop in clips:
                gpu_intervals.append((start,stop)); per_step[step].append((start,stop))
                if category == 'kernel':
                    kernel_intervals.append((start,stop)); per_step_kernels[step] += 1
                if category == 'gpu_memcpy':copy_intervals.append((start,stop))
    kernel_count = sum(r['count'] for r in buckets['kernel'].values())
    scopes = buckets['user_annotation']
    required = ('physics.total','physics.original_sim_step','audit.before_read_and_check',
                'audit.after_read_and_check','audit.before_python','audit.after_python','scene.write_total')
    scopes_complete = all(scopes.get('md12.'+k,{}).get('count') == len(markers)*10 for k in required)
    kernels_complete = all(per_step_kernels.values())
    gpu = union_length(gpu_intervals)
    # The activity union is measured only inside the CPU environment markers.
    # Subtracting it does NOT identify exclusive Python time or GPU idle time.
    return dict(schema='stage12-cuda-timeline-summary-v1',
        status='CUDA_TIMELINE_CAPTURED' if kernel_count and kernels_complete and scopes_complete
               else 'CUDA_TIMELINE_INCOMPLETE_PRESERVE',
        recorded_environment_steps=[m[0] for m in markers],
        required_scope_counts_complete=scopes_complete, every_step_has_cuda_kernel=kernels_complete,
        total_trace_events=len(events), categories_in_raw_trace=categories,
        environment_window_total_ms=wall/1000.,
        cuda_activity_union_ms=gpu/1000., kernel_union_ms=union_length(kernel_intervals)/1000.,
        memcpy_union_ms=union_length(copy_intervals)/1000.,
        no_recorded_cuda_activity_in_environment_windows_ms=max(0.,wall-gpu)/1000.,
        cuda_kernel_events=kernel_count,
        scopes={name:row for name,row in scopes.items() if not name.startswith('md12.env.')},
        ranked_events={category:sorted(rows.values(), key=lambda r:(-r['clipped_total_ms'],r['name']))[:60]
                       for category,rows in buckets.items() if category!='user_annotation'},
        per_step=[dict(control_step=step, environment_window_ms=(b-a)/1000.,
            cuda_activity_union_ms=union_length(per_step[step])/1000.,
            cuda_kernel_events=per_step_kernels[step]) for step,a,b in markers],
        semantics=dict(profiler_overhead_included=True, performance_acceptance=False,
            raw_trace_contains_between_step_work=True, summary_excludes_between_step_work=True,
            sum_of_kernels_may_overlap=True, nested_host_scopes_are_not_additive=True,
            cuda_intervals_clipped_to_environment_markers=True,
            gap_is_not_exclusive_CPU_time_or_proof_of_device_idle=True,
            unavailable_CUPTI_or_missing_cuda_activity_is_not_a_pass=True,
            no_shapes_no_stacks_no_memory_profile=True))


class CudaTimeline(Bindings):
    def __init__(self, torch, destination):
        super().__init__()
        self.torch = torch
        self.destination = Path(destination)
        require(not self.destination.exists(), 'Refuse to overwrite a timeline')
        self.profiler = None
        self.active = self.started = self.stopped = self.closed = False
        self.completed_steps = []
        self.summary = None
        self.error = None

    def attach(self, obj, name, label):
        original = getattr(obj,name)
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            context = self.torch.profiler.record_function('md12.'+label) if self.active else nullcontext()
            with context:
                return original(*args, **kwargs)
        self.set(obj,name,wrapped)

    def attach_session(self, session):
        try:
            for obj,name,label in (
                (session.raw.sim,'step','physics.total'),
                (session,'physics_step_original','physics.original_sim_step'),
                (session.guard,'before_integration','audit.before_read_and_check'),
                (session.guard,'after_integration','audit.after_read_and_check'),
                (session.guard.audit,'before_integration','audit.before_python'),
                (session.guard.audit,'after_integration','audit.after_python'),
                (session,'assert_zero_wrench','force.assert_zero'),
                (session,'_clear_owned','force.clear_owned'),
                (session.raw.scene,'write_data_to_sim','scene.write_total'),
                (session.raw.action_manager,'apply_action','actions.apply'),
                (session.raw.observation_manager,'compute','observations'),
                (session.raw.metrics_manager,'compute','metrics'),
                (session.raw.termination_manager,'compute','termination'),
                (session.raw.sim,'forward','sim.forward'),
                (session.raw.sim,'sense','sim.sense'),
            ):
                self.attach(obj,name,label)
            for i,group in enumerate(session.guard.groups):
                for name in ('get_command','apply_delay','compute'):
                    self.attach(group,name,'actuator.%d.%s'%(i,name))
        except BaseException:
            self.close();raise

    def start(self):
        require(not self.started and not self.closed, 'Timeline already started or closed')
        api = self.torch.profiler
        require(api.ProfilerActivity.CUDA in api.supported_activities(),
                'CUDA profiler activity unavailable; preserve without installing dependencies')
        self.profiler = api.profile(activities=[api.ProfilerActivity.CPU,api.ProfilerActivity.CUDA],
                                   record_shapes=False, profile_memory=False, with_stack=False,
                                   with_flops=False, with_modules=False)
        self.profiler.start()
        self.started = self.active = True

    def finish(self):
        if not self.started or self.stopped:
            return
        self.active = False
        # A failed stop is not retried during teardown; never mask a primary failure.
        self.stopped = True
        try:
            self.profiler.stop()
            self.profiler.export_chrome_trace(str(self.destination))
            self.summary = summarize(json.loads(self.destination.read_bytes()))
        except BaseException as exc:
            self.error = repr(exc)
            raise

    @contextmanager
    def window(self, step):
        if step == FIRST:
            self.start()
        enabled = FIRST <= step <= LAST
        require(not enabled or self.active, 'Missing active CUDA timeline window')
        primary = None
        try:
            with self.torch.profiler.record_function('md12.env.%d'%step) if enabled else nullcontext():
                yield
            if enabled:self.completed_steps.append(step)
        except BaseException as exc:
            primary = exc
            raise
        finally:
            if step == LAST:
                try:self.finish()
                except BaseException:
                    if primary is None:raise

    def close(self):
        if self.closed:return
        try:self.finish()
        finally:
            super().close();self.closed = True

    def report(self):
        return dict(started=self.started, stopped=self.stopped, closed=self.closed,
            bindings_restored=not self.saved, active=self.active,
            completed_steps=list(self.completed_steps),error=self.error,summary=self.summary,
            trace=None if not self.destination.is_file() else dict(path=str(self.destination),
                bytes=self.destination.stat().st_size,sha256=sha(self.destination)))


def complete_timeline(report):
    return bool(report and report.get('started') and report.get('stopped') and report.get('closed')
        and report.get('bindings_restored') and not report.get('active') and report.get('error') is None
        and report.get('completed_steps') == list(range(FIRST,LAST+1))
        and (report.get('summary') or {}).get('status') == 'CUDA_TIMELINE_CAPTURED')
