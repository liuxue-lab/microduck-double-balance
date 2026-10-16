"""Terminal-preserving adapter for one manual-reset native world per process."""
from .durable import require
import math


def snapshot(value):
    if hasattr(value, "clone"):
        return value.clone()
    if isinstance(value, dict):
        return {k: snapshot(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(snapshot(v) for v in value)
    if isinstance(value, list):
        return [snapshot(v) for v in value]
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise RuntimeError("Unsupported mutable transition value: " + type(value).__name__)


class TerminalBridge:
    """Owns exactly one reset; obtains terminal obs before any reset mutates it.

    No observation_manager.compute call is added. A done transition computes
    terminal observations once in native manual step and initial observations
    once in native reset. These are different states and distinct RNG draws.
    """
    def __init__(self, env, torch, tensor_dict, max_resets):
        require(env.num_envs == 1 and env.cfg.auto_reset is False,
                "One manual-reset world per process required")
        require(type(max_resets) is int and max_resets >= 1, "Reset budget missing")
        self.env, self.torch, self.td = env, torch, tensor_dict
        self.num_envs, self.num_actions = 1, 14
        self.device, self.max_episode_length = env.device, env.max_episode_length
        self.resets, self.max_resets, self.stopped = 0, max_resets, False
        self.observations = None
        self.reset()

    def obs(self, values):
        require(set(values.keys()) == {"actor", "critic"}, "Observation groups changed")
        require(tuple(values["actor"].shape) == (1, 61) and
                tuple(values["critic"].shape) == (1, 85), "Observation dimensions changed")
        require(all(bool(self.torch.isfinite(v).all()) for v in values.values()),
                "Nonfinite observation")
        return self.td({k: v.clone() for k, v in values.items()}, batch_size=[1])

    def get_observations(self):
        require(not self.stopped and self.observations is not None, "Bridge unavailable")
        return self.observations.clone()

    def reset(self):
        require(not self.stopped and self.resets < self.max_resets, "Reset budget exhausted/stopped")
        self.resets += 1  # Attempt retained even if native reset fails.
        try:
            values, extras = self.env.reset(env_ids=[0])
            self.observations = self.obs(values)
            return self.get_observations(), snapshot(extras)
        except BaseException:
            self.stopped = True
            raise

    def step(self, actions):
        require(not self.stopped, "Failed transition cannot retry")
        require(tuple(actions.shape) == (1, 14) and bool(self.torch.isfinite(actions).all()),
                "Invalid action")
        try:
            values, reward, terminated, truncated, extras = self.env.step(actions)
            terminal = self.obs(values)
            reward, terminated, truncated, extras = map(snapshot, (reward, terminated, truncated, extras))
            require(tuple(reward.shape) == tuple(terminated.shape) == tuple(truncated.shape) == (1,),
                    "Transition batch layout changed")
            require(bool(self.torch.isfinite(reward).all()), "Nonfinite reward")
            require(terminated.dtype == self.torch.bool and truncated.dtype == self.torch.bool,
                    "Termination flags must be bool tensors")
            done = self.torch.logical_or(terminated, truncated)
            bootstrap = self.torch.logical_and(truncated, self.torch.logical_not(terminated))
            # Snapshot EVERYTHING above before reset clears manager-owned buffers.
            if bool(done.any()):
                next_obs, reset_extras = self.reset()
                extras["reset_extras"] = reset_extras
            else:
                self.observations = terminal.clone()
                next_obs = self.get_observations()
            require("stage12_transition" not in extras, "Reserved extras key collision")
            extras.pop("time_outs", None)  # The dedicated algorithm bridge handles this once.
            extras["stage12_transition"] = dict(terminal_observation=terminal, terminated=terminated,
                truncated=truncated, bootstrap_mask=bootstrap, bootstrap_applied=False)
            return next_obs, reward, done.long(), extras
        except BaseException:
            self.stopped = True
            raise


def process_transition(algorithm, next_obs, rewards, dones, extras):
    """Installed RSL 5.0.1 adds V(s_t) for time_outs; use V(terminal) instead.

    Current critic is feed-forward. A recurrent critic is explicitly rejected.
    Call under torch.inference_mode in the dedicated rollout loop. Statistics
    update remains exactly once in algorithm.process_env_step(next_obs,...).
    """
    require(not getattr(algorithm.critic, "is_recurrent", False), "Recurrent terminal critic unreviewed")
    require("time_outs" not in extras, "Duplicate stock timeout bootstrap path")
    row = extras["stage12_transition"]
    require(row["bootstrap_applied"] is False, "Transition was already bootstrapped")
    corrected = rewards.clone()
    mask = row["bootstrap_mask"]
    if bool(mask.any()):
        values = algorithm.critic(row["terminal_observation"]).detach().reshape(-1)
        require(tuple(values.shape) == tuple(rewards.shape), "Terminal critic value layout differs")
        require(all(math.isfinite(float(v)) for v in values.cpu().tolist()) and
                math.isfinite(algorithm.gamma) and 0 <= algorithm.gamma <= 1,
                "Nonfinite terminal value or invalid discount")
        corrected = corrected + algorithm.gamma * values * mask.to(rewards.dtype)
    row["bootstrap_applied"] = True  # Failed storage cannot be retried as a new transition.
    clean = {k: v for k, v in extras.items() if k != "stage12_transition"}
    algorithm.process_env_step(next_obs, corrected, dones, clean)
    return corrected
