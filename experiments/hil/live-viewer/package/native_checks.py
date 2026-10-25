"""Pure report predicates; intentionally no numerical/runtime imports."""
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

