# Experiments

[Home](../README.md) · [Documentation](../docs/README.md)

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

## Implementation history

The [code guide](../docs/double-balance/code-guide.md) identifies the final control,
inference and visualization modules. Versioned development packages remain in
[Stage11](stage11/) and [Stage12](stage12/).
