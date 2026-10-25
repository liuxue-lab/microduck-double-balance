"""CPU-only adapters around the unchanged task observations and native solver."""
from bootstrap import require
from native_delay import OriginalDelayClock
from native_inference import InferenceOnly
from phase_cache import PhaseCache
from telemetry import States
from realtime_io import ControlGC
from sync_adapters import StaticQuaternion
from native_properties import PublishedProperties

PIPELINE = 'Native MuJoCo CPU 5ms + original BAM/2ms delay clock + 50Hz actor/Orin loop'


from native_checks import adapter_checks


def session_type(base, stats):
    class Session(base.EvaluationSession):
        def __init__(self, *args, **kwargs):
            self.cache = self.slim = self.delay = self.states = self.control_gc = self.static_quat = None
            self.properties = None
            self.adapters_closed = False
            super().__init__(*args, **kwargs)
            stats['sessions'].append(True)
            try:
                self.cache = PhaseCache(self.raw, base.mdp, self.probe)
            except BaseException:
                self.close()
                raise

        def after_reset(self, episode):
            super().after_reset(episode)
            self.delay = OriginalDelayClock(self.raw)
            self.slim = InferenceOnly(self.raw, base.torch)
            self.static_quat = StaticQuaternion(base.torch)
            # Warm the compiled selection kernel without consuming its 40 live
            # same-input comparisons, policy state or random samples.
            warm_matrix=self.raw.sim.data.site_xmat[:,self.raw.scene['robot'].indexing.site_ids]
            for _ in range(40):self.static_quat.calculate(warm_matrix)
            self.properties = PublishedProperties(self.raw,base.torch)
            self.states = States(self.run, self.raw, base.torch,stats.get('pending_models'))
            self.states.capture()

        def arm_control(self):
            require(self.control_gc is None, 'Control was already armed')
            self.control_gc = ControlGC()

        def step(self, env, actions):
            timing = stats['timing']
            begun = timing.environment_started()
            returned = super().step(env, actions)
            end = timing.clock()
            timing.add(environment_ms=(end-begun)/1e6)
            self.states.capture()
            timing.add(state_capture_ms=(timing.clock()-end)/1e6)
            # evaluate_one ends the tick after finite/done checks and logging.
            return returned

        def close(self):
            if self.adapters_closed:
                return
            self.adapters_closed = True
            if self.control_gc is not None:self.control_gc.close()
            result = dict(backend=self.raw.sim.snapshot(),
                delay=self.delay.snapshot() if self.delay else {},
                inference_only=self.slim.stats() if self.slim else {})
            errors = []
            for item in (self.properties,self.static_quat,self.slim,self.delay,self.cache):
                if item is not None:
                    try:
                        item.close()
                    except BaseException as exc:
                        errors.append(repr(exc))
            try:
                super().close()
                self.raw.sim.data.push()
                require((self.raw.sim.mj_data.xfrc_applied==0).all(), 'Native wrench not cleared')
            except BaseException as exc:
                errors.append(repr(exc))
            if self.states is not None:
                try:
                    result['state_export'] = self.states.export()
                except BaseException as exc:
                    errors.append(repr(exc))
            result.update(runtime_bindings_restored=not errors, close_errors=errors,
                trace_export=self.trace.snapshot(),
                static_quaternion=self.static_quat.stats() if self.static_quat else {},
                published_properties=self.properties.stats() if self.properties else {},
                control_gc=self.control_gc.snapshot() if self.control_gc else {})
            stats['adapters'] = result
            require(not errors,'Native session cleanup failed: '+str(errors))
    return Session
