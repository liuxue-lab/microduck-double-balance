# 代码导览

[项目首页](../../README.zh-CN.md) · [开始使用](getting-started.zh-CN.md) · [English](code-guide.md)

项目由 PPO 训练部分和 PC–Jetson 控制运行系统组成。
沿下面的代码地图，可以追踪 MuJoCo 观测如何进入策略，以及动作如何回到仿真机器人。

## 控制与推理

| 环节 | 源码 |
| --- | --- |
| 运行入口与包加载 | [bootstrap.py](../../experiments/stage12/native-live-v1/package/bootstrap.py) |
| 原生 MuJoCo 动力学 | [native_backend.py](../../experiments/stage12/native-live-v1/package/native_backend.py) |
| 控制循环与会话 | [native_runtime.py](../../experiments/stage12/native-live-v1/package/native_runtime.py) · [native_session.py](../../experiments/stage12/native-live-v1/package/native_session.py) |
| ROS2 观测与动作回传 | [loop_ros.py](../../experiments/stage12/native-live-v1/package/loop_ros.py) · [loop_contract.py](../../experiments/stage12/native-live-v1/package/loop_contract.py) |
| 策略推理与 PyTorch 参考 | [native_inference.py](../../experiments/stage12/native-live-v1/package/native_inference.py) · [reference_overlap.py](../../experiments/stage12/native-live-v1/package/reference_overlap.py) |
| 任务判据与模型约束 | [native_criteria.py](../../experiments/stage12/native-live-v1/package/native_criteria.py) · [native_model_limits.py](../../experiments/stage12/native-live-v1/package/native_model_limits.py) |
| 状态传输与实时显示 | [live_stream.py](../../experiments/stage12/native-live-v1/package/live_stream.py) · [live_viewer.py](../../experiments/stage12/native-live-v1/package/live_viewer.py) |
| 记录回放与可视化 | [回放包](../../experiments/stage12/hil-replay-v1/package/) |

运行实现保存在带版本的包中，使报告和模型身份持续指向对应源码。
`bootstrap.py` 负责加载关联模块和资产；[依赖关系说明](reproduction-guide.md#3-必须保留的依赖链)
给出了具体配置。

## 训练与机器人模型

| 源码 | 阅读内容 |
| --- | --- |
| [任务定义](../../src/mjlab_microduck/tasks/) | 环境配置、观测、奖励、终止条件和随机化 |
| [双球训练](../../src/mjlab_microduck/double_balance_training.py) | 项目源码中的训练工具 |
| [原生训练工作流](../../experiments/stage12/native-workflow/README.md) | 原生工作流入口和保留的训练过程 |
| [BAM 执行器模型](../../src/mjlab_microduck/actuator/) | 机器人的执行器动力学 |
| [机器人资产](../../src/mjlab_microduck/robot/) | MJCF、关节布局、碰撞和硬件几何 |
| [CPU 回归测试](../../tests/) | 配置检查与行为约束 |

顶层源码同时包含继承任务和早期双球实现。阅读演示系统时，可从上面的运行代码地图
及其配套清单进入。[训练报告](project-report.md#3-训练与模型冻结)说明了训练过程与最终模型文件的关系。

## 结果与配套资料

- [模型说明（英文）](model-card.md)：actor 输入、动作输出、推理格式和文件哈希。
- [实验演示](../../experiments/double-balance/README.zh-CN.md)：任务效果与实测周期耗时。
- [原始报告](evidence/reports/)：机器可读的性能和评估记录。
- [详细报告](project-report.md)：方法、测试条件及源码版本索引。
- [审计记录](../audits/)：固定配置、模型身份和开发检查。

准备修改项目时可阅读[贡献说明（英文）](../../CONTRIBUTING.md)。
源码导入和资产引用沿用上表中的包路径。
