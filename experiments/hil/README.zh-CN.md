# 硬件在环

[项目首页](../../README.zh-CN.md) · [实验导航](../../experiments/README.zh-CN.md) · [English](README.md) · **简体中文**

演示系统将 PC 上的原生 MuJoCo 动力学与真实 Jetson Orin Nano Super 上的循环 ONNX 推理连接起来。

| 组件 | 内容 |
| --- | --- |
| [实时可视化](live-viewer/README.zh-CN.md) | 带独立交互式三维窗口的 50 Hz PC–Jetson 控制闭环。 |
| [原生 MuJoCo 控制](native-mujoco/README.zh-CN.md) | 200 Hz 原生 CPU 物理、Jetson 动作请求和完整周期计时。 |
| [状态记录回放](recorded-state-replay/README.zh-CN.md) | 从保存的原生状态生成视频，并同步展示周期计时数据。 |
| [MuJoCo Warp 控制闭环](warp-loop/README.zh-CN.md) | 用于建立 PC–Jetson 接口的早期 GPU 仿真闭环。 |
| [Warp 闭环优化](optimized-warp-loop/README.zh-CN.md) | 组合选定 GPU 路径优化措施的运行实验。 |

[实验演示](../double-balance/README.zh-CN.md) · [模型说明](../../docs/double-balance/model-card.md) · [代码导览](../../docs/double-balance/code-guide.zh-CN.md)
