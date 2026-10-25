"""Warm the actual frozen reference before the first release, then reset it.

No plant step, transport request, training update or formal tick occurs here.
The public RSL-RL reset API restores the initial LSTM state to None. The
original actor object, parameters, normalizer and inference path are retained.
"""
import resource
import time

from bootstrap import require
from fingerprint import fingerprint

WARMUP_CALLS = 128
RESET_EVERY = 32


def prepare_reference(policy, observations, torch, boundary=lambda: None):
    require(policy.is_recurrent and not policy.training and
            not policy.obs_normalizer.training, 'Reference must be frozen recurrent inference')
    require(policy.get_hidden_state() is None, 'Prepare only the fresh, reset reference')
    require(tuple(observations['actor'].shape) == (1, 61), 'Unexpected warmup observation layout')
    require(all(value.device.type == 'cpu' for value in policy.state_dict().values()),
            'Reference preparation must stay on CPU')
    before = fingerprint(policy.state_dict())
    observation_before = fingerprint(observations)
    rng = torch.get_rng_state().clone()
    duration, cpu_duration = [], []
    usage = resource.getrusage(resource.RUSAGE_THREAD)
    begun = time.perf_counter_ns()
    try:
        with torch.inference_mode(), torch.random.fork_rng(devices=[]):
            for index in range(WARMUP_CALLS):
                if index % RESET_EVERY == 0:
                    boundary()
                    policy.reset()
                start = time.perf_counter_ns()
                cpu_start = time.thread_time_ns()
                action = policy(observations.to('cpu'), stochastic_output=False)
                cpu_duration.append((time.thread_time_ns() - cpu_start) / 1e6)
                duration.append((time.perf_counter_ns() - start) / 1e6)
                require(tuple(action.shape) == (1, 14) and bool(torch.isfinite(action).all()),
                        'Invalid action during reference preparation')
    finally:
        # This executes on stop, exceptions and normal completion alike.
        policy.reset()
        require(policy.get_hidden_state() is None, 'Reference reset did not clear LSTM state')
        require(torch.equal(rng, torch.get_rng_state()), 'Preparation changed CPU RNG state')
        require(fingerprint(policy.state_dict()) == before, 'Preparation changed actor/normalizer')
        require(fingerprint(observations) == observation_before, 'Preparation changed observations')
    after = resource.getrusage(resource.RUSAGE_THREAD)
    return dict(schema='stage12-actual-reference-preparation-v1', same_actor_instance=True,
        status='PREPARED_AND_RESET', warmup_calls=len(duration), reset_sequences=WARMUP_CALLS//RESET_EVERY,
        input='actual initial observations; original CPU TensorDict conversion',
        elapsed_ms=(time.perf_counter_ns()-begun)/1e6,
        first_inference_ms=duration[0], maximum_inference_ms=max(duration),
        last_32_maximum_inference_ms=max(duration[-RESET_EVERY:]),
        maximum_thread_cpu_ms=max(cpu_duration),
        major_page_faults=after.ru_majflt-usage.ru_majflt,
        minor_page_faults=after.ru_minflt-usage.ru_minflt,
        actor_state_sha256=before, policy_and_normalizer_unchanged=True,
        initial_observations_unchanged=True, rng_restored=True, lstm_state_reset_to_none=True,
        formal_control_ticks=0, Orin_requests=0, physics_steps=0, training_updates=0,
        first_formal_tick_still_included=True)
