# Model selection and velocity rewards

[Project](../../../README.md) · [Experiments](../../../experiments/README.md) · **English** · [简体中文](README.zh-CN.md)

This study compares recurrent PPO checkpoints and velocity-reward refinements for
the coupled balance task. Its selected policy provided the starting point for the
later [joint-margin](../joint-margin-penalty/README.md) and
[disturbance-aware training](../disturbance-policy/README.md).

## Workflow

- [Candidate evaluation](evaluate_candidates.py)
- [Result collection](collect_training_results.py)
- [Selected-policy retrieval and replay](fetch_and_replay_selected_policy.py)
- [Training and recovery workflows](workflows/)
- [Detailed study record — Chinese](research-record.zh-CN.md)

The [model card](../../../docs/double-balance/model-card.md) identifies the final published policy.
