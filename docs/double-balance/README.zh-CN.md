# 双球平衡项目文档

[项目首页](../../README.zh-CN.md) · [实验演示](../../experiments/double-balance/README.zh-CN.md) ·
[English](README.md) · **简体中文**

项目将循环 PPO 策略接入 50 Hz 控制闭环：PC 执行原生 MuJoCo 动力学，
Jetson Orin Nano Super 执行 ONNX 策略推理。
从这里可以进一步阅读演示背后的系统实现与运行数据。

## 理解系统

| 文档 | 重点 |
| --- | --- |
| [实验演示](../../experiments/double-balance/README.zh-CN.md) | 任务、演示与测试设置 |
| [开始使用](getting-started.zh-CN.md) | 安装源码环境，了解部署前提 |
| [模型说明](model-card.md) | 循环 actor、观测/动作接口与模型身份 |
| [代码导览](code-guide.zh-CN.md) | 训练任务、控制循环、ROS 通信与显示 |
| [项目报告](project-report.md) | 训练过程与系统测量详情 |
| [复现说明](reproduction-guide.md) | 运行和查看既有 PC–Jetson 系统 |

## 阅读数据

- [实时窗口测试](evidence/reports/live-control-report.json)：完整周期耗时、动作对照及显示统计。
- [数据索引](evidence/README.md)：报告、模型身份与来源记录。
- [媒体说明](../../experiments/double-balance/media/README.zh-CN.md)：演示来源与画面处理方法。
- [资产记录](archive-scope.md)：检查点、运行包与文件清单。

## 项目维护

[参与贡献](../../CONTRIBUTING.md) · [CI 记录](ci-followup.md) ·
[仓库页面设置](repository-settings.md) · [上游贡献说明](upstream-contribution.md)

发布与开发历史保留在[发布记录](publication.md)、[验证记录](validation/README.md)
和[研究与维护记录](../handoffs/)中。维护操作沿用
[本地 SSH 流程](../handoffs/microduck-local-ssh-push-protocol.md)
和[计算与资料传输规范](../handoffs/microduck-cloud-gpu-and-transfer-protocol.md)。
