"""Bounded host-wall profiling of the unchanged accepted evaluation pipeline.

No tensor reads, CUDA events/synchronizations, physics calls or RNG calls are
added here. Intervals include Python instrumentation and any existing GPU
waits. They cannot measure pure device kernel time or predict a speed gain.
"""
import cProfile
import functools
import pstats
import sys
import time

from deploy_contract import require
from loop_contract import measured
from phase_cache import Bindings


class HostProfile(Bindings):
    def __init__(self, clock=None):
        super().__init__()
        self.clock = clock or time.perf_counter_ns
        self.profiler = cProfile.Profile()
        self.enabled = False
        self.closed = False
        self.windows = 0
        self.stack = []
        self.samples = {}

    def attach(self, obj, name, label):
        require(not self.closed and not self.enabled, 'Cannot attach while profiling or closed')
        require(label not in self.samples, 'Duplicate profile scope: ' + label)
        original = getattr(obj, name)
        require(callable(original), 'Profile scope is not callable: ' + label)
        self.samples[label] = dict(inclusive=[], exclusive=[])

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            if not self.enabled:
                return original(*args, **kwargs)
            frame = [self.clock(), 0]
            self.stack.append(frame)
            try:
                return original(*args, **kwargs)
            finally:
                elapsed = self.clock() - frame[0]
                self.stack.pop()
                if self.stack:
                    self.stack[-1][1] += elapsed
                self.samples[label]['inclusive'].append(elapsed / 1e6)
                self.samples[label]['exclusive'].append(max(0, elapsed - frame[1]) / 1e6)
        self.set(obj, name, wrapped)

    def attach_session(self, session):
        require(not self.saved, 'Session scopes already attached')
        try:
            for obj, name, label in (
                (session.raw.sim, 'step', 'physics.total'),
                (session, 'physics_step_original', 'physics.original_sim_step'),
                (session.guard, 'before_integration', 'audit.before_read_and_check'),
                (session.guard, 'after_integration', 'audit.after_read_and_check'),
                (session.guard.audit, 'before_integration', 'audit.before_python'),
                (session.guard.audit, 'after_integration', 'audit.after_python'),
                (session, 'assert_zero_wrench', 'force.assert_zero'),
                (session, '_clear_owned', 'force.clear_owned'),
                (session.raw.scene, 'write_data_to_sim', 'scene.write_total'),
            ):
                self.attach(obj, name, label)
            for i, group in enumerate(session.guard.groups):
                for name in ('get_command', 'apply_delay', 'compute'):
                    self.attach(group, name, 'actuator.%d.%s' % (i, name))
        except BaseException:
            self.close()
            raise

    def start(self):
        require(not self.closed and not self.enabled and not self.stack,
                'Profile window is already active or closed')
        # Do not replace another user profiler/debugging hook.
        require(sys.getprofile() is None, 'Existing Python profiling hook; preserve it')
        self.profiler.enable()
        self.enabled = True
        self.windows += 1

    def stop(self):
        if self.enabled:
            self.profiler.disable()
            self.enabled = False
        require(not self.stack, 'Profile stopped inside a nested scope')

    def close(self):
        self.stop()
        super().close()
        self.closed = True

    def report(self, destination=None):
        require(not self.enabled and not self.stack, 'Cannot serialize an active profile')
        divisor = max(1, self.windows)
        scopes = {}
        for label, samples in self.samples.items():
            inc, exc = samples['inclusive'], samples['exclusive']
            scopes[label] = dict(inclusive_ms=measured(inc), exclusive_ms=measured(exc),
                inclusive_sum_ms=sum(inc), exclusive_sum_ms=sum(exc),
                inclusive_ms_per_control=sum(inc)/divisor,
                exclusive_ms_per_control=sum(exc)/divisor)
        data = pstats.Stats(self.profiler).stats if self.windows else {}
        if destination is not None and self.windows:
            self.profiler.dump_stats(str(destination))
        key = lambda k: dict(file=k[0], line=k[1], function=k[2])
        rows = []
        for loc, (primitive, calls, own, total, callers) in data.items():
            # cProfile caller edges use (total calls, primitive calls, own, cumulative),
            # unlike the function-row tuple above.
            # for each caller edge. Keep the full call graph in JSON, so the
            # report alone suffices; the optional pstats file stays on the PC.
            edges = [dict(**key(k), primitive_calls=v[1], total_calls=v[0],
                          self_ms=v[2]*1000, cumulative_ms=v[3]*1000)
                     for k, v in callers.items()]
            edges.sort(key=lambda v: (-v['cumulative_ms'], v['file'], v['line'], v['function']))
            rows.append(dict(**key(loc), primitive_calls=primitive, total_calls=calls,
                self_ms=own*1000, cumulative_ms=total*1000,
                self_ms_per_control=own*1000/divisor,
                cumulative_ms_per_control=total*1000/divisor, callers=edges))
        rows.sort(key=lambda v: (-v['self_ms'], v['file'], v['line'], v['function']))
        ranked = sorted(rows, key=lambda v: (-v['cumulative_ms'], v['file'], v['line'], v['function']))
        return dict(schema='stage12-host-profile-v1', measured_control_windows=self.windows,
            bindings_restored=not self.saved, profiler_disabled=not self.enabled,
            stack_empty=not self.stack, closed=self.closed, scopes=scopes,
            cprofile=dict(function_count=len(rows), total_self_ms=sum(r['self_ms'] for r in rows),
                functions=rows, top_cumulative_functions=[key((r['file'],r['line'],r['function'])) for r in ranked[:60]],
                local_pstats_path=None if destination is None else str(destination)),
            timing_semantics=dict(
                clock='host wall time; existing asynchronous waits may be charged to later calls',
                exclusive='scope interval minus directly nested instrumented scope intervals only; includes wrapper overhead',
                nested_values_are_not_additive=True, cprofile_and_scopes_are_not_additive=True,
                pure_GPU_kernel_times=False, profiler_overhead_is_included=True,
                profile_speedup_is_not_an_optimization_result=True,
                policy_inference_and_external_comparison_trace_excluded=True,
                original_in_step_audit_and_trace_retained=True,
                no_new_tensor_reads_or_synchronization=True))


def complete_profile(report, controls, groups):
    """Require every measured integration and every original actuator call."""
    if not report or report.get('measured_control_windows') != controls:
        return False
    if not all(report.get(k) is True for k in ('bindings_restored','profiler_disabled','stack_empty','closed')):
        return False
    if report.get('cprofile', {}).get('function_count', 0) == 0:
        return False
    scopes = report.get('scopes', {})
    labels = ['physics.total','physics.original_sim_step','audit.before_read_and_check',
              'audit.after_read_and_check','audit.before_python','audit.after_python',
              'force.clear_owned','scene.write_total']
    labels += ['actuator.%d.%s' % (i, name) for i in range(groups)
               for name in ('get_command','apply_delay','compute')]
    return (groups > 0 and all(scopes.get(k, {}).get('inclusive_ms', {}).get('count') == controls*10
                              for k in labels)
            and scopes.get('force.assert_zero', {}).get('inclusive_ms', {}).get('count', 0) >= controls*10)
