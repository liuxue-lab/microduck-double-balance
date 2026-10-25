# Repository presentation settings

[Documentation](README.md) · [Home](../../README.md)

The default branch is **`double-balance`**, confirmed on 25 October 2026.
GitHub opens the English `README.md`; the language link leads to `README.zh-CN.md`.

## About

The repository's **Code → About → Edit repository details** panel can use:

| Field | Value |
| --- | --- |
| Description | `Learning whole-body control for Microduck to balance on a basketball while keeping a second ball on its head, with real-time Jetson hardware-in-the-loop validation.` |
| Website | `https://github.com/liuxue-lab/microduck-double-balance/tree/double-balance/experiments/double-balance` |
| Topics | `microduck` `robotics` `reinforcement-learning` `ppo` `mujoco` `robot-learning` `hardware-in-the-loop` `jetson` `ros2` `onnx` |

The Website field links directly to the demonstration page.
About metadata is edited separately from the Git-tracked README files.

## Branch and release workflow

Continue development on `double-balance`, preserving the existing SSH remote.
The [local SSH workflow](../handoffs/microduck-local-ssh-push-protocol.md) records
the maintainer's checkout, incremental bundle import and publication procedure.

The [CI workflow](../../.github/workflows/ci.yml) checks each pushed commit.
Use the current [Actions page](https://github.com/liuxue-lab/microduck-double-balance/actions)
for release validation. Earlier CI records belong to the private history repository.
