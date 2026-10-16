"""Read committed observations without a second noise/history/delay update.

This adapter is a candidate. It is tested with a fake CPU environment only.
Reset/step delegate once to the installed wrapper, then snapshot observations.
Reward/done/timeout outputs are unchanged; training timeout policy is unreleased.
"""
from .contracts import require


def snapshot_observations(obs, num_envs, tensor_dict):
    # TensorDict iterates over its first batch dimension, unlike dict. Use the
    # explicit mapping keys for both native TensorDict outputs and dict caches.
    require(set(obs.keys()) == {"actor", "critic"}, "Unexpected observation groups")
    require(tuple(obs["actor"].shape) == (num_envs, 61), "Actor layout differs")
    require(tuple(obs["critic"].shape) == (num_envs, 85), "Critic layout differs")
    # Clones prevent a normalizer or caller from writing into the committed cache.
    return tensor_dict({k: v.clone() for k, v in obs.items()}, batch_size=[num_envs])


def cached_observations(env, tensor_dict):
    require(env.cfg.auto_reset is True, "Training wrapper requires explicit auto-reset semantics")
    return snapshot_observations(env.obs_buf, env.num_envs, tensor_dict)


def make_wrapper_class(base_wrapper, tensor_dict):
    class Stage12CachedVecEnv(base_wrapper):
        def get_observations(self):
            return cached_observations(self.unwrapped, tensor_dict)
        def reset(self):
            obs, extras = super().reset()
            return snapshot_observations(obs, self.num_envs, tensor_dict), extras
        def step(self, actions):
            obs, rewards, dones, extras = super().step(actions)
            return snapshot_observations(obs, self.num_envs, tensor_dict), rewards, dones, extras
    return Stage12CachedVecEnv
