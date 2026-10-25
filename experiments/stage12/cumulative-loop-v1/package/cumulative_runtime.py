"""The exact qualified adapters, scoped to one unchanged evaluation episode."""
from bootstrap import require
from action_graph import ActionGraph, CHECK_CALLS
from sync_adapters import StaticQuaternion
from scalar_audit import ScalarAudit
from packed_checks import PackedChecks
from phase_cache import PhaseCache
from bam_graph import BamGraphs, SHADOW_CHECK_CALLS
from inference_only import InferenceOnly
from loop_contract import STEPS
from telemetry import States

PIPELINE = 'InferenceOnly + PhaseCache + BAM Graph v2 + StaticQuaternion + ScalarAudit + PackedChecks + ActionGraph'


def adapter_checks(s, steps=STEPS):
    a, q = s.get('action_graph') or {}, s.get('static_quaternion') or {}
    p, c = s.get('packed_checks') or {}, s.get('scalar_audit') or {}
    slim, graphs = s.get('inference_only') or {}, s.get('bam_graphs') or []
    return dict(
        restored=s.get('runtime_bindings_restored') is True,
        cached=(s.get('phase_cache') or {}).get('cache_hits', 0) > 0,
        bam=bool(graphs) and all(g and g.get('replays') == steps * 10 and
            g.get('same_input_checked_calls') == SHADOW_CHECK_CALLS for g in graphs),
        action=a.get('closed') is True and a.get('bindings_restored') is True and
            a.get('replays') == steps * 10 and a.get('controls') == a.get('staged_controls') == steps and
            a.get('same_input_checked_calls') == CHECK_CALLS and a.get('comparison_failure') is None,
        quaternion=q.get('closed') is True and q.get('checked_calls') == CHECK_CALLS and
            q.get('calls', 0) >= CHECK_CALLS,
        packed=all(p.get(k) == steps * 10 for k in ('zero_wrench_checks_before_write',
            'packed_reads_before_integration', 'packed_reads_after_integration')),
        scalar=all(c.get('calls', {}).get(k) == steps * 10 and
            c.get('same_input_checked_calls', {}).get(k) == CHECK_CALLS
            for k in ('before_integration', 'after_integration')) and c.get('last_comparison_failure') is None,
        inference=slim.get('omitted_reward_calls') == steps and slim.get('actor_group_calls') == steps
            and slim.get('unexpected_critic_calls') == 0,
        state_export=(s.get('state_export') or {}).get('samples') == steps + 1,
        cleanup=not s.get('close_errors'))


def session_type(base, stats):
    class Session(base.EvaluationSession):
        def __init__(self, *args, **kwargs):
            self.cache = self.graphs = self.slim = self.static_quat = None
            self.scalar = self.packed = self.action_graph = self.states = None
            self.exported = self.adapters_closed = False
            super().__init__(*args, **kwargs)
            # Count sessions without retaining the first case's full CUDA plant
            # through a stats -> session -> closure -> stats reference cycle.
            stats['sessions'].append(True)
            try:
                self.cache = PhaseCache(self.raw, base.mdp, self.probe)
            except BaseException:
                self.close()
                raise

        def after_reset(self, episode):
            super().after_reset(episode)
            # Same construction order and method bodies as the measured cumulative arm.
            self.graphs = BamGraphs(self.guard.groups, base.torch)
            self.slim = InferenceOnly(self.raw, base.torch)
            self.static_quat = StaticQuaternion(base.torch)
            self.scalar = ScalarAudit(self.guard.audit)
            self.packed = PackedChecks(self, base.torch)
            self.action_graph = ActionGraph(self.raw, base.torch)
            self.states = States(self.run, self.raw, base.torch)
            self.states.capture()

        def physics_step(self):
            if self.packed is not None:
                return self.packed.physics_step()
            return super().physics_step()

        def step(self, env, actions):
            timing = stats['timing']
            base.torch.cuda.synchronize()
            begun = timing.environment_started()
            answer = super().step(env, actions)
            base.torch.cuda.synchronize()
            end = timing.clock()
            timing.add(environment_ms=(end - begun) / 1e6)
            self.states.capture()
            timing.add(state_capture_ms=(timing.clock() - end) / 1e6)
            timing.end()
            return answer

        def close(self):
            if self.adapters_closed:
                return
            self.adapters_closed = True
            errors = []
            for item in (self.action_graph, self.static_quat, self.scalar, self.slim, self.graphs, self.cache):
                if item is not None:
                    try:
                        item.close()
                    except BaseException as exc:
                        errors.append(repr(exc))
            try:
                super().close()
            except BaseException as exc:
                errors.append(repr(exc))
            export = None
            if self.states is not None:
                try:
                    export = self.states.export()
                    self.exported = True
                except BaseException as exc:
                    errors.append(repr(exc))
            stats['adapters'] = dict(pipeline=PIPELINE,
                phase_cache=None if self.cache is None else self.cache.stats,
                bam_graphs=None if self.graphs is None else self.graphs.stats(),
                inference_only=None if self.slim is None else self.slim.stats(),
                static_quaternion=None if self.static_quat is None else self.static_quat.stats(),
                scalar_audit=None if self.scalar is None else self.scalar.stats(),
                packed_checks=None if self.packed is None else dict(self.packed.stats(),
                    unchanged_CPU_Audit=False, original_Audit_method_bytecode_retained=True,
                    scalar_helpers_specialized=True),
                action_graph=None if self.action_graph is None else self.action_graph.stats(),
                runtime_bindings_restored=all(item is None or not item.saved for item in
                    (self.action_graph, self.static_quat, self.scalar, self.slim, self.graphs))
                    and (self.cache is None or not self.cache.bindings.saved),
                state_export=export, close_errors=errors)
            require(not errors, 'Cumulative adapter cleanup failed: ' + '; '.join(errors))
    return Session
