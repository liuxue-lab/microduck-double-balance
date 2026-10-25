# Microduck Double Balance v1.0.0

Release notes prepared for the first public release.

Microduck Double Balance combines recurrent PPO training, native MuJoCo simulation
and Jetson inference for a robot balancing on a basketball while retaining a second
ball directly on its original head shell.

## Included

- Training, deployment, evaluation and visualization source with locked dependencies.
- English and Chinese project pages and experiment guides.
- Complete original 60-second recordings for balance and disturbance recovery.
- Recorded PC–Jetson HIL results: 50 Hz control, 200 Hz physics, and 6,000 measured
  cycles with no missed 20 ms deadlines across two cases.
- Model identities, evaluation records and the deployment contract.

The selected policy uses FP32 ONNX inference with recurrent state and observation
normalization. Model files are distributed separately from the source checkout.
See the [model card](model-card.md) for verified identities and availability.

## Research outlook

How can trajectory tracking be integrated with whole-body control so that Microduck
follows a desired path while balancing on a basketball and keeping a second ball on
its head?

[English project page](../../README.md) · [中文项目页](../../README.zh-CN.md)
