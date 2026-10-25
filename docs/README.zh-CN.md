# 文档

[项目首页](../README.zh-CN.md) · [English](README.md) · **简体中文**

从双球平衡任务出发，了解学习策略、部署链路与 PC–Jetson 控制系统。

## 从这里开始

| 入口 | 内容 |
| --- | --- |
| [实验演示](../experiments/double-balance/README.zh-CN.md) | 自然平衡、抗扰恢复与实测结果 |
| [双球项目文档](double-balance/README.zh-CN.md) | 系统、模型与实现的阅读入口 |
| [开始使用](double-balance/getting-started.zh-CN.md) | 源码安装、CPU 检查与 HIL 运行条件 |
| [模型说明](double-balance/model-card.md) | 策略结构、模型身份与可获取范围 |
| [代码导览](double-balance/code-guide.zh-CN.md) | 训练、控制、推理和显示的代码位置 |

## 技术资料

- [项目报告](double-balance/project-report.md)：训练、周期耗时、扰动与评估定义。
- [复现说明](double-balance/reproduction-guide.md)：既有 PC–Jetson 环境与运行命令。
- [数据索引](double-balance/evidence/README.md)：原始报告与来源记录。
- [媒体说明](../experiments/double-balance/media/README.zh-CN.md)：演示来源与处理方法。
- [资产目录](double-balance/archive-scope.md)：模型与运行资产记录。

## 开发资料

[参与贡献](../CONTRIBUTING.md) · [CI 记录](double-balance/ci-followup.md) · [引用信息](../CITATION.cff)

[原项目介绍](upstream/README.md)汇集 Microduck Playground 的原有实验与硬件设计。
开发记录位于 [audits/](audits/) 与 [handoffs/](handoffs/)。
维护操作见[本地 SSH 流程](handoffs/microduck-local-ssh-push-protocol.md)
和[计算与资料传输规范](handoffs/microduck-cloud-gpu-and-transfer-protocol.md)。
