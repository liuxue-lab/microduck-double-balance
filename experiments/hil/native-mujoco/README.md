# Native MuJoCo control

[Project](../../../README.md) · [Experiments](../../../experiments/README.md) · **English** · [简体中文](README.zh-CN.md)

Native CPU physics at 200 Hz, Jetson action requests and complete-cycle timing.

## Control system

Native `mujoco.mj_step` advances the robot and ball dynamics every **5 ms**.
The recurrent FP32 ONNX policy runs on the Jetson and returns joint actions every
**20 ms**. The runtime retains the BAM actuator model, observation normalization,
recurrent state, joint limits and force checks.

The two recorded 60-second cases completed 6,000 control cycles with zero missed
deadlines. The maximum complete-cycle times were **14.519 ms** for balance and
**14.471 ms** for disturbance recovery. The [headless control report](../../../docs/double-balance/evidence/reports/headless-control-report.json)
contains the measurements.

Each case exports timing data, native states, the model image and audit records.
The [recorded-state replay](../recorded-state-replay/README.md) uses those saved states
to produce the original demonstration videos.

## Source map

- [Run controller](controller.py)
- [Implementation](package/)
- [Focused tests](tests/)

[Setup and assets](../../../docs/double-balance/getting-started.md) · [Detailed implementation record](README.txt)
