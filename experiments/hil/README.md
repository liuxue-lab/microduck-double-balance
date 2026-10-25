# Hardware in the loop

[Project](../../README.md) · [Experiments](../../experiments/README.md) · **English** · [简体中文](README.zh-CN.md)

The demonstrated system combines native MuJoCo dynamics on the PC with recurrent ONNX inference on a real Jetson Orin Nano Super.

| Component | Focus |
| --- | --- |
| [Live visualization](live-viewer/README.md) | The 50 Hz PC–Jetson control loop with an independent interactive 3D viewer. |
| [Native MuJoCo control](native-mujoco/README.md) | Native CPU physics at 200 Hz, Jetson action requests and complete-cycle timing. |
| [Recorded-state replay](recorded-state-replay/README.md) | Video rendering from saved native states, with synchronized timing data. |
| [MuJoCo Warp control loop](warp-loop/README.md) | The earlier GPU-simulation loop used to establish PC–Jetson integration. |
| [Optimized Warp control loop](optimized-warp-loop/README.md) | A combined runtime experiment applying the selected GPU-path optimizations. |

[Demonstrations](../double-balance/README.md) · [Model card](../../docs/double-balance/model-card.md) · [Code guide](../../docs/double-balance/code-guide.md)
