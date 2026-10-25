# 原生 MuJoCo 控制

[项目首页](../../../README.zh-CN.md) · [实验导航](../../../experiments/README.zh-CN.md) · [English](README.md) · **简体中文**

200 Hz 原生 CPU 物理、Jetson 动作请求和完整周期计时。

## 控制系统

原生 `mujoco.mj_step` 每 **5 ms** 推进机器人与球的动力学。
循环 FP32 ONNX 策略在 Jetson 上运行，每 **20 ms** 返回关节动作。
运行系统保留 BAM 执行器模型、观测归一化、循环状态、关节边界及外力检查。

两组各 60 秒运行共完成 6,000 个控制周期，均无期限超时。
自然平衡与抗扰恢复的完整周期最大耗时分别为 **14.519 ms** 和 **14.471 ms**，
测量见[无窗口控制报告](../../../docs/double-balance/evidence/reports/headless-control-report.json)。

每组保存周期计时、原生状态、模型映像与检查记录。
[状态记录回放](../recorded-state-replay/README.zh-CN.md)读取这些状态，生成原始演示视频。

## 源码导航

- [运行控制器](controller.py)
- [实现代码](package/)
- [定向测试](tests/)

[环境与资产](../../../docs/double-balance/getting-started.zh-CN.md) · [详细实施记录](README.txt)
