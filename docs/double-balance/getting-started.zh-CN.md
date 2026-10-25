# 开始使用

[项目首页](../../README.zh-CN.md) · [实验演示](../../experiments/double-balance/README.zh-CN.md) · [English](getting-started.md)

从演示和结果了解项目，沿代码导览阅读策略与控制系统，或配置源码环境开展开发。

## 选择阅读入口

| 入口 | 内容 |
| --- | --- |
| [实验演示](../../experiments/double-balance/README.zh-CN.md) | 双球平衡、抗扰演示与主要结果 |
| [代码导览](code-guide.zh-CN.md) | 训练、MuJoCo 动力学、ROS2 通信和 Jetson 推理 |
| [模型说明（英文）](model-card.md) | 策略结构、部署输入和模型文件身份 |
| [详细报告](project-report.md) | 训练过程、测试条件和完整周期计时 |

## 配置源码环境

使用 Python 3.12 和 [uv](https://docs.astral.sh/uv/)。配置好 GitHub SSH 身份后执行：

```bash
git clone --branch double-balance git@github.com:liuxue-lab/microduck-double-balance.git
cd microduck-double-balance
uv sync --frozen
```

锁定文件提供项目依赖，其中包含训练环境。在仓库根目录运行 CPU 回归测试：

```bash
uv run --frozen python -m pytest tests/ -q
```

CI 使用相同的测试范围。根目录 `pytest.ini` 默认也选择 `tests/`。
历史运行包具有各自的依赖，可通过[代码导览](code-guide.zh-CN.md)单独阅读和检查。

## 理解控制系统

PC 以 200 Hz 运行原生 MuJoCo；真实 Jetson Orin Nano Super 执行循环 FP32 ONNX
策略，通过有线 ROS2 Humble 以 50 Hz 返回动作。部署的 actor 接收 61 维观测，
输出 14 维动作。

先从[运行和训练代码地图](code-guide.zh-CN.md)阅读实现，再查[模型说明（英文）](model-card.md)
中的归一化、循环状态和模型文件。

## 运行 PC–Jetson 系统

HIL 环境使用固定检查点、ONNX 导出、部署合同、ROS2 配置及配套运行包。
[原环境运行指南](reproduction-guide.md)列出了这些文件的位置，以及查看录像和运行既有系统的命令。

仓库已包含源码、预览、结果报告，以及两段完整的 1440p、60 秒演示视频：
[自然平衡](https://raw.githubusercontent.com/liuxue-lab/microduck-double-balance/refs/heads/double-balance/experiments/double-balance/media/balance.mp4)和
[抗扰恢复](https://raw.githubusercontent.com/liuxue-lab/microduck-double-balance/refs/heads/double-balance/experiments/double-balance/media/disturbance-recovery.mp4)。
策略权重单独分发，当前获取状态和部署所需资产见
[模型可获取范围（英文）](model-card.md#artifacts-and-availability)。

原工作站可利用已有资产目录[查看完整录像](reproduction-guide.md#4-只查看已完成结果)。
[依赖关系说明](reproduction-guide.md#3-必须保留的依赖链)记录了该系统使用的运行包。
