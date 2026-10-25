# GitHub 发布、贡献上游与官方展示

核对日期：2026-10-25。这里的“原项目”指
[Vottivott/microduck-playground](https://github.com/Vottivott/microduck-playground)，
“官方”指 Pollen Robotics。

## 原作者是怎样发布的

原项目 README 明确说明，它是基于 Pollen Robotics 的 `microduck_rl` 的独立实验延续，
不是 Pollen Robotics 官方发行版。作者在自己的 GitHub 仓库展示实验，
并将部分模型发布在自己的 Hugging Face 账号下。
在个人仓库发布 README 和视频，不需要先让机器人官方接收整个项目。

同时，原作者确实贡献过一个具体的官方功能：

| 项目 | 已核实事实 |
| --- | --- |
| 官方 PR | [pollen-robotics/microduck #231](https://github.com/pollen-robotics/microduck/pull/231) |
| 标题 | Support LSTM ONNX policies with recurrent state handling |
| 作者 | `Vottivott` |
| 状态 | 已合并；2026-09-10 09:57:31 UTC |
| 含义 | 官方接收了循环 ONNX 策略的运行时支持；不等于接收整个 Playground 仓库或为全部实验背书 |

本次查看原项目、官方 Microduck 产品页和官方 RL 文档，没有找到足以确认
“整个 Playground 项目提交后被列入官方产品主页”的证据。
这不是断言它从未在任何官方渠道被提及；只能说上述页面和已合并 PR 不能证明这种发布流程。

## 几种不同的发布方式

| 目标 | 实际动作 | 是否等于官方接纳 |
| --- | --- | --- |
| 自己的 GitHub 项目主页 | 推送自己的分支，设置默认分支，完善 README、证据与许可证 | 否；是维护者自行发布 |
| 贡献官方代码 | 针对具体功能向对应官方仓库提交 PR，由维护者审查 | 合并后，仅该项代码贡献被接纳 |
| 分享可下载策略 | 在模型托管平台发布权重、部署合同、模型卡和评估证据 | 否；可使用作者自己的账号 |
| 官方网站或社区精选 | 向官方社区介绍成果，由其维护者决定是否收录或转载 | 需要单独确认，推送或 PR 合并不会自动触发 |

官方 [Microduck 页面](https://pollen-robotics.com/microduck/) 提供社区 Discord 入口，
也鼓励分享策略。官方 [microduck_rl 文档](https://github.com/pollen-robotics/microduck_rl)
介绍通过 `uv run publish` 向 Hugging Face 发布策略与 manifest/model card 的机制。
这是当前官方工具的说明，不代表本仓库冻结环境已集成该命令，也不代表本双球策略已完成官方真机部署兼容性验证。

## 本项目当前适合怎样对外介绍

使用“Microduck 双球平衡：PPO 训练与 50 Hz PC–Jetson 仿真硬件在环”这一定位，
展示两个有界测试、回放预览、冻结模型身份和完整周期计时。
明确机器人与球的动力学仍在仿真中，真实硬件仅承担推理和通信。

如以后希望贡献官方，先挑选可独立审阅的通用功能，并对照对应仓库的
[贡献说明](https://github.com/pollen-robotics/microduck/blob/main/CONTRIBUTING.md)。
若仅希望展示成果，项目页和证据链接本身已经可以用于社区交流；官方是否展示需另行确认。
当前整理不包含提交官方 PR、上传公共模型、联系官方或发布社区消息。

关联：[原项目说明](../upstream/README.md) · [模型可获取范围](model-card.md) · [仓库页面设置](repository-settings.md)
