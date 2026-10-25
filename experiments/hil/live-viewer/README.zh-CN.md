# 实时可视化

[项目首页](../../../README.zh-CN.md) · [实验导航](../../../experiments/README.zh-CN.md) · [English](README.md) · **简体中文**

带独立交互式三维窗口的 50 Hz PC–Jetson 控制闭环。

## 控制与显示

PC 以 **200 Hz** 推进原生 MuJoCo，并以 **50 Hz** 与 Jetson 交换控制动作。
独立显示进程读取状态流，使渲染与控制、物理计算分别运行。

两组各 60 秒运行分别完成 3,000 个控制周期，均无 20 ms 期限超时。
自然平衡与抗扰恢复的完整周期最大耗时分别为 **15.310 ms** 和 **14.676 ms**，
详见[原始测量](../../../docs/double-balance/evidence/reports/live-control-report.json)。

窗口提供全身、头部视角，以及跟随、旋转、平移和缩放。
状态流与窗口实现见 [live_stream.py](package/live_stream.py) 和 [live_viewer.py](package/live_viewer.py)。

## 源码导航

- [运行控制器](controller.py)
- [实现代码](package/)
- [定向测试](tests/)

[环境与资产](../../../docs/double-balance/getting-started.zh-CN.md) · [详细实施记录](README.txt)
