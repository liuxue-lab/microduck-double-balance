"""CPU-only adapters around the unchanged task observations and native solver."""
from bootstrap import require
from native_delay import OriginalDelayClock
from native_inference import InferenceOnly
from phase_cache import PhaseCache
from telemetry import States

PIPELINE = 'Native MuJoCo CPU 5ms + original BAM/2ms delay clock + 50Hz actor/Orin loop'


def adapter_checks(s, steps=3000):
    backend, delay, slim = s.get('backend',{}), s.get('delay',{}), s.get('inference_only',{})
    return dict(native_physics=backend.get('backend')=='mujoco.mj_step' and
                    backend.get('native_step_calls')==steps*4 and backend.get('physics_dt')==.005,
        cpu_only=backend.get('device')=='cpu' and backend.get('warp_physics_calls')==0 and
                    backend.get('cuda_graphs') is False,
        # VecEnvWrapper resets once at construction, then evaluate_one performs
        # the explicit seeded reset. Neither is an episode retry.
        native_refresh=backend.get('native_forward_calls')==steps+4 and
                    backend.get('sense_calls')==steps+2 and backend.get('reset_calls')==2 and
                    backend.get('warning_counts')==[0]*7,
        original_delay_clock=bool(delay.get('native_delay_calls')) and
                    all(n==steps*4 for n in delay['native_delay_calls']) and
                    all(n==steps*10 for n in delay.get('original_delay_samples',[])),
        actor_preserved=slim.get('actor_group_calls')==steps and slim.get('unexpected_critic_calls')==0,
        states=s.get('state_export',{}).get('samples')==steps+1,
        cleanup=s.get('runtime_bindings_restored') is True and not s.get('close_errors'))


def session_type(base, stats):
    class Session(base.EvaluationSession):
        def __init__(self, *args, **kwargs):
            self.cache = self.slim = self.delay = self.states = None
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
            self.states = States(self.run, self.raw, base.torch)
            self.states.capture()

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
            result = dict(backend=self.raw.sim.snapshot(),
                delay=self.delay.snapshot() if self.delay else {},
                inference_only=self.slim.stats() if self.slim else {})
            errors = []
            for item in (self.slim,self.delay,self.cache):
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
            result.update(runtime_bindings_restored=not errors, close_errors=errors)
            stats['adapters'] = result
            require(not errors,'Native session cleanup failed: '+str(errors))
    return Session
