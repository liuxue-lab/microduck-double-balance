# Getting started

[Home](../../README.md) · [Demonstrations](../../experiments/double-balance/README.md) · [简体中文](getting-started.zh-CN.md)

Explore the demonstrations, inspect the policy and control system, or set up the
source environment for development.

## Explore the project

| Start with | What you will find |
| --- | --- |
| [Demonstrations](../../experiments/double-balance/README.md) | Balance and disturbance tests, with the main results |
| [Code guide](code-guide.md) | Training, MuJoCo dynamics, ROS2 communication and Jetson inference |
| [Model card](model-card.md) | Policy architecture, deployment inputs and artifact identities |
| [Detailed report — Chinese](project-report.md) | Training, test conditions and complete timing measurements |

## Set up the source environment

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). With a configured GitHub SSH identity:

```bash
git clone --branch double-balance git@github.com:liuxue-lab/microduck-double-balance.git
cd microduck-double-balance
uv sync --frozen
```

The lockfile supplies the project dependencies, including the training stack.
Run the CPU regression suite from the repository root:

```bash
uv run --frozen python -m pytest tests/ -q
```

This is the test selection used by CI. The root `pytest.ini` also selects `tests/`
by default. Research packages have their own dependencies and can be inspected
separately through the [code guide](code-guide.md).

## Understand the control system

The PC runs native MuJoCo at 200 Hz. A real Jetson Orin Nano Super executes the
recurrent FP32 ONNX policy and returns actions over wired ROS2 Humble at 50 Hz.
The deployed actor uses 61 observations and produces 14 actions.

Start from the [runtime and training map](code-guide.md), then consult the
[model card](model-card.md) for the normalization, recurrent state and model files.

## Run the PC–Jetson setup

The HIL setup uses the fixed checkpoint, ONNX export, deployment contract, ROS2
configuration and associated runtime packages. The
[existing-environment guide — Chinese](reproduction-guide.md) gives their locations
and the commands for viewing recordings or running the installed system.

The repository includes source, previews, result reports and both complete original
60-second recordings: [balance](../../experiments/double-balance/media/balance.mp4)
and [disturbance recovery](../../experiments/double-balance/media/disturbance-recovery.mp4).
The policy weights are distributed separately; see
[model availability](model-card.md#artifacts-and-availability) for their current status
and the required deployment assets.

On the original workstation, [view the full recordings — Chinese](reproduction-guide.md#4-只查看已完成结果)
using the existing artifact directory. The [dependency map — Chinese](reproduction-guide.md#3-必须保留的依赖链)
documents the packages used by the installed system.
