# 模型选择与速度奖励

[项目首页](../../../README.zh-CN.md) · [实验导航](../../../experiments/README.zh-CN.md) · [English](README.md) · **简体中文**

本研究对比循环 PPO 检查点与速度奖励微调结果。选定策略作为后续
[关节余量训练](../joint-margin-penalty/README.zh-CN.md)及
[抗扰训练](../disturbance-policy/README.zh-CN.md)的起点。

## 工作流

- [候选评估](evaluate_candidates.py)
- [结果收集](collect_training_results.py)
- [选定策略取回与回放](fetch_and_replay_selected_policy.py)
- [训练与恢复实现](workflows/)
- [详细研究记录](research-record.zh-CN.md)

最终发布策略的身份见[模型说明](../../../docs/double-balance/model-card.md)。
