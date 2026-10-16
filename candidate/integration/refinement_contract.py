"""Predeclared v7 numerical experiment. No threshold changes or dynamic tiers."""
TIERS = (32, 64, 128, 256, 512)
TRIALS = len(TIERS) + 1  # All candidates plus one mandatory accepted replay.
MAX_INTEGRATIONS = sum(TIERS) + TIERS[-1]
MAX_FORWARD = 4 * MAX_INTEGRATIONS + TRIALS + 1


def macro_cost():
    from macro_state_bridge import MACRO_LIMITS
    result = dict(MACRO_LIMITS)
    result.update(state_snapshot=1, state_restore=TRIALS+1,
        integrations=MAX_INTEGRATIONS, RK4_integrator=MAX_INTEGRATIONS,
        GPU_forward=MAX_FORWARD, position_write=MAX_INTEGRATIONS,
        sense=TRIALS+1, endpoint_sensor_read=TRIALS+1,
        private_clock_write=TRIALS, model_option_write=2*TRIALS,
        public_clock_write=1)
    return result
