# 运行性能实验

[项目首页](../../README.zh-CN.md) · [实验导航](../../experiments/README.zh-CN.md) · [English](README.md) · **简体中文**

围绕仿真、推理、同步和数据传输开展的定向对照。各运行包保留其测试设置与测量，最终完整闭环结果见下方链接。

| 组件 | 内容 |
| --- | --- |
| [运行分析](runtime-profile/README.zh-CN.md) | 既定控制运行配置的主机调用分析。 |
| [CPU 与 CUDA 时间线](cpu-cuda-timeline/README.zh-CN.md) | 关联 CPU 与 CUDA 活动轨迹，定位控制循环开销。 |
| [隔离 CUPTI 支持](cupti-isolation/README.zh-CN.md) | 用于分析工作站 GPU 活动采集的独立 CUPTI 库配置。 |
| [Jetson 推理](jetson-inference/README.zh-CN.md) | Jetson 部署的推理、通信与优化测量。 |
| [运动学缓存](kinematics-cache/README.zh-CN.md) | 复用当前计算阶段的运动学结果，并与原始计算对照。 |
| [BAM 图输出所有权](bam-graph-shared-output/README.zh-CN.md) | BAM 执行器计算的共享输出 CUDA 图候选实现。 |
| [BAM 图输入快照](bam-graph-snapshot/README.zh-CN.md) | 为 BAM 图重放分离输入、图输出与已提交状态缓冲。 |
| [动作写入图](action-write-graph/README.zh-CN.md) | 动作应用过程的 CUDA 图捕获与目标实时状态检查。 |
| [头部运动学图](head-kinematics-graph/README.zh-CN.md) | 头部几何计算的 CUDA 图捕获。 |
| [纯推理运行路径](inference-only-loop/README.zh-CN.md) | 固定策略推理时只计算 actor 观测，并省去训练奖励计算。 |
| [合并检查数据读取](packed-audit-reads/README.zh-CN.md) | 在保留完整检查项的前提下合并读取检查数据。 |
| [标量检查处理](scalar-audit/README.zh-CN.md) | 控制路径的标量检查与合并读取对照。 |
| [张量读取图](tensor-read-graph/README.zh-CN.md) | 从仿真张量读取检查数据的图捕获实验。 |
| [仿真数据桥接读取](bridged-read-graph/README.zh-CN.md) | 使用仿真数组包装对象的共享张量视图执行图读取。 |
| [int64 索引命令视图](command-views-int64/README.zh-CN.md) | 基于 int64 地址假设的连续命令字段视图。 |
| [混合索引命令视图](command-views-mixed-index/README.zh-CN.md) | 保留实际 int32 与 int64 索引布局的连续命令视图。 |
| [同步开销](synchronization/README.zh-CN.md) | 同步、四元数与延迟处理开销的对照。 |
| [原生 MuJoCo 基线](native-mujoco-baseline/README.zh-CN.md) | 采用既有策略和 ROS2 通信的原生 CPU 物理接入基线。 |
| [带缓冲的原生控制](native-mujoco-buffered/README.zh-CN.md) | 原生控制的预分配记录存储与独立进度上报。 |

[实验演示](../double-balance/README.zh-CN.md) · [模型说明](../../docs/double-balance/model-card.md) · [代码导览](../../docs/double-balance/code-guide.zh-CN.md)
