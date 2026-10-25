# Double-balance documentation

[Home](../../README.md) · [Demonstrations](../../experiments/double-balance/README.md) ·
**English** · [简体中文](README.zh-CN.md)

The project connects a recurrent PPO policy to a 50 Hz control loop, with native
MuJoCo dynamics on the PC and ONNX inference on a Jetson Orin Nano Super.
Use this page to move from the demonstrations to the implementation and recorded measurements.

## Learn the system

| Read | Focus |
| --- | --- |
| [Experiment](../../experiments/double-balance/README.md) | Task, demonstrations and evaluation settings |
| [Getting started](getting-started.md) | Install the source environment and locate deployment prerequisites |
| [Model card](model-card.md) | Recurrent actor, observation/action interfaces and model identities |
| [Code guide](code-guide.md) | Training tasks, the control loop, ROS communication and viewer |
| [Technical report (中文)](project-report.md) | Detailed training and system measurements |
| [Reproduction guide (中文)](reproduction-guide.md) | Run and inspect the established PC–Jetson system |

## Read the measurements

- [Live-viewer evaluation](evidence/reports/live-control-report.json): complete-cycle timing, action checks and visualization statistics.
- [Data index (Chinese)](evidence/README.md): reports, model identities and their source records.
- [Media notes](../../experiments/double-balance/media/README.md): demonstration sources and rendering details.
- [Asset records (Chinese)](archive-scope.md): the checkpoint, runtime packages and inventory.

## Maintain the project

[Contributing](../../CONTRIBUTING.md) · [CI history (Chinese)](ci-followup.md) ·
[Repository settings](repository-settings.md) · [Upstream contribution notes (Chinese)](upstream-contribution.md)

Publication and development history are preserved in the [publication record (Chinese)](publication.md),
[validation records (Chinese)](validation/README.md) and [research and maintenance records](../handoffs/).
The maintainer workflow follows the [local SSH (Chinese)](../handoffs/microduck-local-ssh-push-protocol.md)
and [compute/transfer (Chinese)](../handoffs/microduck-cloud-gpu-and-transfer-protocol.md) guides.
