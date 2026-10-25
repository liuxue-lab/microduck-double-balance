# 训练

[项目首页](../../README.zh-CN.md) · [实验导航](../../experiments/README.zh-CN.md) · [English](README.md) · **简体中文**

双球耦合平衡任务的 PPO 训练、奖励设计与策略选择。

| 组件 | 内容 |
| --- | --- |
| [抗扰策略训练](disturbance-policy/README.zh-CN.md) | 包含周期水平外力脉冲的循环 PPO 微调，是发布策略对应的训练工作流。 |
| [模型选择与速度奖励](model-selection/README.zh-CN.md) | 早期基线训练、速度奖励对照和候选策略选择。 |
| [关节余量奖励](joint-margin-penalty/README.zh-CN.md) | 通过位置成本引导策略在关节边界内保留余量。 |
| [预测关节余量奖励](joint-margin-prediction/README.zh-CN.md) | 结合位置与速度预测的关节边界成本，以及逐物理步统计。 |
| [训练统计与检查点保存](joint-margin-logging/README.zh-CN.md) | 兼容推理张量的训练统计，以及更新后的检查点保存。 |
| [训练依赖恢复](joint-margin-dependencies/README.zh-CN.md) | 已准备训练运行的奖励模块依赖恢复与检查。 |
| [奖励预检](disturbance-reward-preflight/README.zh-CN.md) | 抗扰训练工作流的奖励尺度预检与恢复。 |

[实验演示](../double-balance/README.zh-CN.md) · [模型说明](../../docs/double-balance/model-card.md) · [代码导览](../../docs/double-balance/code-guide.zh-CN.md)
