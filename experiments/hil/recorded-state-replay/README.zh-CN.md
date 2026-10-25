# 状态记录回放

[项目首页](../../../README.zh-CN.md) · [实验导航](../../../experiments/README.zh-CN.md) · [English](README.md) · **简体中文**

从保存的原生状态生成视频，并同步展示周期计时数据。

## 原始演示

每段录像为 **60 秒、3,000 帧、1280 × 480、50 fps**。
并排画面分别展示全身与上方小球。

- [自然平衡：原始 MP4](../../double-balance/media/balance.mp4)
- [抗扰恢复：原始 MP4](../../double-balance/media/disturbance-recovery.mp4)

渲染器读取已保存的原生模型与状态数组，更新几何位置，并让视频与周期计时对应。
回放使用既有运行记录，不推进物理积分，也不请求新的策略动作。

[回放报告](../../../docs/double-balance/evidence/reports/recording-replay-report.json)
记录了输入来源、帧检查与计时数据核对。

## 源码导航

- [运行控制器](controller.py)
- [实现代码](package/)
- [定向测试](tests/)

[环境与资产](../../../docs/double-balance/getting-started.zh-CN.md) · [详细实施记录](README.txt)
