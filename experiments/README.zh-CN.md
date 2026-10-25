# 实验

[项目首页](../README.zh-CN.md) · [项目文档](../docs/README.zh-CN.md) · [English](README.md) · **简体中文**

## 双球平衡

[![Microduck 双球平衡](double-balance/media/balance-demo.webp)](double-balance/README.zh-CN.md)

脚踩篮球，头顶小球。PPO 策略接入 50 Hz PC–Jetson 闭环，两组各 60 秒演示展示自然平衡与抗扰恢复。

[查看实验](double-balance/README.zh-CN.md) · [模型说明](../docs/double-balance/model-card.md)

## Microduck Playground 原有实验

| 实验 | 内容 |
| --- | --- |
| [篮球](basketball/README.md) | 站在自由滚动的篮球上保持平衡 |
| [秋千](swing/README.md) | 从静止开始自主荡秋千 |
| [奔跑](running/README.md) | 学习策略控制的快速移动 |
| [高跷](stilts/README.md) | 使用参数化鞋底和高跷行走 |

以上实验来自 Hannes von Essen / Vottivott 的 Microduck Playground，详见[原项目介绍](../docs/upstream/README.md)。

## 系统实现

| 方向 | 内容 |
| --- | --- |
| [训练](training/README.zh-CN.md) | PPO 工作流、奖励设计与模型选择 |
| [部署](deployment/README.zh-CN.md) | ONNX 导出、一致性与 ROS2 配置 |
| [硬件在环](hil/README.zh-CN.md) | 原生控制、实时窗口与状态回放 |
| [评估](evaluation/README.zh-CN.md) | 候选选择与独立策略检查 |
| [性能实验](benchmarks/README.zh-CN.md) | 运行分析与定向性能对照 |
| [原生物理系统](physics-validation/README.zh-CN.md) | 接触、状态所有权、重置与恢复 |

[代码导览](../docs/double-balance/code-guide.zh-CN.md)将这些组件与项目源码连接起来。
