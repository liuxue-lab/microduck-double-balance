<h1 align="center">Microduck 双球平衡</h1>

<p align="center">脚踩篮球，头顶小球。</p>

<p align="center">
  <a href="README.md">English</a> · <strong>简体中文</strong>
</p>

<p align="center">
  <a href="experiments/double-balance/README.zh-CN.md">演示</a> ·
  <a href="docs/double-balance/README.zh-CN.md">项目文档</a> ·
  <a href="docs/double-balance/model-card.md">模型</a> ·
  <a href="#开始使用">开始使用</a>
</p>

<p align="center">
  <a href="https://raw.githubusercontent.com/liuxue-lab/microduck-double-balance/refs/heads/double-balance/experiments/double-balance/media/balance.mp4">
    <img src="experiments/double-balance/media/balance-demo.webp" alt="Microduck 脚踩篮球，通过全身协调保持头顶小球平衡">
  </a>
</p>

<p align="center"><sub>12 秒预览 · <a href="https://raw.githubusercontent.com/liuxue-lab/microduck-double-balance/refs/heads/double-balance/experiments/double-balance/media/balance.mp4">下载高清 MP4 · 1440p · 60 秒 · 50 fps</a></sub></p>

<p align="center"><strong>50 Hz 控制</strong> &nbsp;·&nbsp; <strong>200 Hz 物理</strong> &nbsp;·&nbsp; <strong>6,000 个实测控制周期</strong></p>

[![CI](https://github.com/liuxue-lab/microduck-double-balance/actions/workflows/ci.yml/badge.svg?branch=double-balance)](https://github.com/liuxue-lab/microduck-double-balance/actions/workflows/ci.yml?query=branch%3Adouble-balance)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Microduck 双球平衡贯通了 **PPO 训练、循环策略全身控制与 Jetson 部署**。
机器人站在可自由滚动的篮球上，同时保持直接接触原头壳的上方小球平衡。
硬件在环系统由 PC 上的原生 MuJoCo 计算机器人与球的动力学，
Jetson Orin Nano Super 通过有线 ROS2 Humble 链路返回控制动作。

## 项目亮点

- **双球协同平衡**：学习策略协调 14 个关节，同时控制下方支撑与上方小球。
- **训练到部署的完整链路**：循环 actor 以 FP32 ONNX 在 Jetson 上运行，保留观测归一化与循环状态。
- **完整闭环实时测量**：两组各 60 秒测试均无 20 ms 期限超时，计时覆盖通信、物理与动作对照。
- **实时三维显示与运行数据**：独立窗口展示控制过程，配套结果报告、模型身份和媒体来源。

## 演示

实验展示自然平衡，以及面对重复水平扰动时的平衡恢复。

**[查看两组演示 →](experiments/double-balance/README.zh-CN.md)**

## 实测性能

开启实时三维窗口后，两组 60 秒测试的完整控制周期耗时：

| 工况 | 均值 | p99 | 最大值 | 超时周期 |
| --- | ---: | ---: | ---: | ---: |
| 自然平衡 | 10.888 ms | 13.279 ms | 15.310 ms | 0 / 3,000 |
| 抗扰恢复 | 11.317 ms | 12.634 ms | 14.676 ms | 0 / 3,000 |

抗扰工况采用相当于机器人自重 10% 的水平力脉冲，两组测试均完成 60 秒双球保持。
[实验设置](experiments/double-balance/README.zh-CN.md#实验设置) ·
[原始计时数据](docs/double-balance/evidence/reports/live-control-report.json)

## 开始使用

检出项目分支，通过 `uv` 安装锁定环境：

```bash
git clone --branch double-balance git@github.com:liuxue-lab/microduck-double-balance.git
cd microduck-double-balance
uv sync --frozen
uv run --frozen python -m pytest tests/ -q
```

以上命令运行项目的 CPU 测试。PC–Jetson 环境配置及外部模型资产见
[开始使用指南](docs/double-balance/getting-started.zh-CN.md)和[模型说明](docs/double-balance/model-card.md)。

## 项目导航

| 入口 | 内容 |
| --- | --- |
| [实验演示](experiments/double-balance/README.zh-CN.md) | 自然平衡、抗扰恢复与测试设置 |
| [项目文档](docs/double-balance/README.zh-CN.md) | 系统、模型、实现与复现说明 |
| [代码导览](docs/double-balance/code-guide.zh-CN.md) | 训练任务、控制循环、推理与显示 |
| [实验集合](experiments/README.md) | 双球任务与原 Playground 实验 |
| [参与贡献](CONTRIBUTING.md) | 开发流程和检查方法 |

## 后续研究方向

如何将轨迹跟踪与全身控制相结合，使 Microduck 在脚踩篮球、头顶小球的双重平衡约束下沿给定路径运动？

## 致谢与贡献

项目基于 Hannes von Essen / Vottivott 的 [Microduck Playground](https://github.com/Vottivott/microduck-playground)、
[Pollen Robotics Microduck RL](https://github.com/pollen-robotics/microduck_rl)、
[mjlab](https://github.com/mujocolab/mjlab) 和 [BAM](https://github.com/Rhoban/bam) 构建。
[原项目介绍](docs/upstream/README.md)保留了原有实验与硬件设计。

## 许可证

软件采用 [Apache-2.0](LICENSE)，继承的硬件设计采用 [CC BY-NC-SA 4.0](LICENSE-HARDWARE)。
署名与引用：[NOTICE](NOTICE) · [CITATION.cff](CITATION.cff)。
