# Experiments

[Home](../README.md) · [Documentation](../docs/README.md) · **English** · [简体中文](README.zh-CN.md)

## Double balance

[![Microduck double-ball balance](double-balance/media/balance-demo.webp)](double-balance/README.md)

Balance on a basketball while supporting a second ball on the head.
PPO training connects to a 50 Hz PC–Jetson loop, with two 60-second evaluations
covering balance and disturbance recovery.

[Explore the experiment →](double-balance/README.md) · [中文](double-balance/README.zh-CN.md) ·
[Model card](../docs/double-balance/model-card.md)

## From Microduck Playground

The original experiments by Hannes von Essen / Vottivott remain part of this codebase.

| Experiment | Description | Media |
| --- | --- | --- |
| [Basketball](basketball/README.md) | Proprioceptive balancing on a free-rolling basketball | [Video](basketball/media/preview.mp4) |
| [Swing](swing/README.md) | Self-pumped motion from a stationary start | [Video](swing/media/alpha050_seed27.mp4) |
| [Running](running/README.md) | Fast locomotion with a learned policy | [Video](running/media/preview.mp4) |
| [Stilts](stilts/README.md) | Walking with parametric replacement soles and stilts | [Video](stilts/media/preview.mp4) |

[Upstream overview and hardware galleries](../docs/upstream/README.md)

## Implementation

| Area | What to explore |
| --- | --- |
| [Training](training/README.md) | PPO workflows, reward design and model selection |
| [Deployment](deployment/README.md) | ONNX export, numerical parity and ROS2 setup |
| [Hardware in the loop](hil/README.md) | Native control, live visualization and recorded-state replay |
| [Evaluation](evaluation/README.md) | Candidate selection and independent policy checks |
| [Runtime benchmarks](benchmarks/README.md) | Profiling and focused performance comparisons |
| [Native physics integration](physics-validation/README.md) | Contact, state ownership, reset and recovery |

The [code guide](../docs/double-balance/code-guide.md) connects these components to the project source.
