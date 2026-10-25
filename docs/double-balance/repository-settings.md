# Repository presentation settings

[Documentation](README.md) · [Home](../../README.md)

The default branch is **`double-balance`**, confirmed on 25 October 2026.
GitHub opens the English `README.md`; the language link leads to `README.zh-CN.md`.

## About

The repository's **Code → About → Edit repository details** panel can use:

| Field | Value |
| --- | --- |
| Description | `Recurrent PPO for Microduck double-ball balance, with real-time PC-Jetson hardware-in-the-loop control.` |
| Website | `https://github.com/liuxue-lab/microduck-double-balance/tree/double-balance/experiments/double-balance` |
| Topics | `microduck` `robotics` `reinforcement-learning` `ppo` `mujoco` `robot-learning` `hardware-in-the-loop` `jetson` `ros2` `onnx` |

The Website field links directly to the demonstration page.
About metadata is edited separately from the Git-tracked README files.

## Branch and release workflow

Continue development on `double-balance`, preserving the existing SSH remote.
The [local SSH workflow](../handoffs/microduck-local-ssh-push-protocol.md) records
the maintainer's checkout, incremental bundle import and publication procedure.

The [CI workflow](../../.github/workflows/ci.yml) checks the exact pushed commit.
The presentation baseline `575e7b4` passed both its
[branch run](https://github.com/liuxue-lab/microduck-double-balance/actions/runs/40190552481)
and [tag run](https://github.com/liuxue-lab/microduck-double-balance/actions/runs/40190558697).
