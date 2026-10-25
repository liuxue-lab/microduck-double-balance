# Training

[Project](../../README.md) · [Experiments](../../experiments/README.md) · **English** · [简体中文](README.zh-CN.md)

PPO training, reward design and policy selection for the coupled balance task.

| Component | Focus |
| --- | --- |
| [Disturbance-aware policy training](disturbance-policy/README.md) | Recurrent PPO refinement with periodic horizontal force pulses; the source workflow for the released policy. |
| [Model selection and velocity rewards](model-selection/README.md) | Earlier baseline training, velocity-reward comparisons and candidate selection. |
| [Joint-margin rewards](joint-margin-penalty/README.md) | Position-based penalties that encourage the policy to retain margin inside joint limits. |
| [Predictive joint-margin rewards](joint-margin-prediction/README.md) | Position and velocity-based costs for approaching joint limits, with per-physics-step statistics. |
| [Training statistics and checkpoint saving](joint-margin-logging/README.md) | Inference-tensor-compatible logging and checkpoint saving around training updates. |
| [Training dependency recovery](joint-margin-dependencies/README.md) | Restoration and validation of the reward-module dependency set for a prepared training run. |
| [Reward preflight checks](disturbance-reward-preflight/README.md) | Reward-scale checks and recovery for the disturbance-training workflow. |

[Demonstrations](../double-balance/README.md) · [Model card](../../docs/double-balance/model-card.md) · [Code guide](../../docs/double-balance/code-guide.md)
