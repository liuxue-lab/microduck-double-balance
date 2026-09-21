# Stage 07 云 GPU 辅助课程训练计划与容量预检

状态：原 A800 部署收到用户返回的 `Stage07CloudSetup=PASS`；用户随后克隆到另一台同配置
A800（端口 26497）。部署入口先验证该新实例的环境复用，再启动实际容量测量。
正式训练尚未启动。本文件不是 Stage 07 完成交接。

## 基线和硬件

- Stage 06 注释标签提交：`e3f26e3cbf633c70c75d4d073dbeb8c47c4f5bd1`。
- Stage 06 已验收 64 环境、5 更新、保存重载；不重复该 smoke。
- 当前云实例：AutoDL，`NVIDIA A800-SXM4-80GB`，81920 MiB，驱动 580.126.09。
- 当前新实例 SSH：`ssh -p 26497 root@connect.bjb1.seetacloud.com`；此前 52528 为已替换实例。
- 原实例安装 PASS 已收到；新实例型号、完整环境和 CPU 配额将随部署命令读回。
  如果克隆未带入数据盘上的项目/环境，入口停止并报告缺失，不自动重新下载。
- 正式训练只在云 GPU 执行，本机 RTX 5060 继续用于仿真，不启动长期训练。
- A800 实际单价尚未提供。先前 5090 的单价不能套用到 A800。

## 初始化及计数规则

唯一默认输入为 Stage 05 迁移检查点，SHA256：

`548758afc546e11d8f3f446a4bcc846283a669ff20f4048da7a3a7bb1332b98e`

本机：`/home/lx/microduck-double-balance/artifacts/double-balance-stage05/checkpoint.pt`。
云端：`/root/autodl-tmp/microduck-double-balance/artifacts/double-balance-stage05/checkpoint.pt`。

严格加载 actor、critic、观测归一化器和 Stage 05 重建的空 Adam 状态。
源 `iter=6999`、`common_step_counter=168048` 只记录为来源，不作为新阶段进度。
加载后将 runner iteration、环境 common counter、物理步计数全部重置为 0。
初始 PPO 和 Adam 学习率均为 `1e-3`，沿用原 adaptive KL 调度。
Stage 06 smoke 检查点被 SHA256 门禁排除。

辅助课程保持原表 `[1, 0.5, 0.25, 0.1, 0.03, 0]`；逐环境等级重置到 0、hold=1。
该辅助作用于下球和鸭身，上球始终自由。真实 episode 存活达到 6 秒才升级，
短于 1.5 秒降级。初始 episode 长度为 0，不启用 runner 的随机 episode 年龄，
避免用虚构的存活时长提前触发升级。原有奖励权重、指令范围等全局课程从 step 0 运行。

完成 N 次 PPO 更新后，RSL-RL 保存的末次循环索引为 N-1，下次起始索引应为 N；
环境计数为 N×24，而不是 N×24×环境数。正式训练恢复入口仍待实现，须在启动正式
训练前明确保存并恢复 actor/critic/归一化器、Adam、调度器 LR、已完成更新数、
全局计数、逐环境辅助等级和 hold；模拟器、RNG、episode 及 rollout 隐状态重新开始，
不得称为逐轨迹无缝恢复。强制重置 episode 不得改变恢复的课程等级。

## 实际容量预检

入口：`scripts/measure_stage07_capacity.py`，调用
`mjlab_microduck.double_balance_stage07` 中的注册 runner 工作进程。

- 按 512、1024、2048、4096 环境升档，每档使用独立进程并从 Stage 05 重新初始化。
- 每档 3 次预热更新、12 次计时更新；每次 24 步、原 PPO epochs/minibatches。
- 保持 2 ms 仿真步长、decimation=10、20 ms 控制周期及 Stage 03/04 定义。
- 检查真实 rollout、return、loss、梯度、模型、Adam 有限值及 nan_state 终止。
- 每档记录逐次更新耗时、transition/s、辅助等级分布、Torch 峰值分配。
- 父进程以至少 0.5 秒间隔调用 nvidia-smi，记录整张分配 GPU 的显存和利用率；
  该统计覆盖 Warp，也可能包括其他 GPU 进程。它是采样峰值，不是瞬时峰值保证。
- 同步记录工作进程 RSS。对最多使用 85% 总显存且正常完成的档位提出训练规模建议。
- 同一档最多 15 分钟，整个容量任务最多 40 分钟；这是容量预检限时，不是正式训练时长。
- 任何失败保留日志、采样 CSV、运行报告，并停止继续升档；不以配置通过替代实测。
- 计时排除首三次更新，但并非保证所有惰性编译均已结束。报告保留每次耗时供复核。
- 容量阶段关闭 TensorBoard/检查点/ONNX 保存，计时不包括后续保存与独立评估。
  其预测时长只覆盖当前带有限值检查的 rollout/PPO，不作为账单承诺。
- 辅助课程后期的接触状态可能改变内存和速度，正式训练仍须监控。

容量预检产生的策略全部丢弃。正式训练须再次加载 Stage 05 初始化输入。
不会从某个容量预检末尾的策略继续。

## 正式训练候选方案（容量和预算结果出来后落实）

首段目标 2000 次 PPO 更新；若采用 4096 环境，共 196608000 条 transition。
每 100 更新保存，每 500 更新做阶段检查和评估，保留初始/阶段/最终检查点。
总体先按不超过 8 小时规划；实际更新数、可用时长与人民币预算依 A800 单价、
已耗时和容量实测确定。定时关机尚未确认配置，不声称已设置。

保留训练奖励、各惩罚项、episode 长度、辅助等级比例和 Stage 04 指标。
独立评估使用冻结的 play 配置及连续末尾 5 秒的无辅助成功定义。
确定性名义初态重复运行不能解释为多个独立随机任务样本。
如果未学会无辅助双重平衡，按实际结果交接，不改写成功阈值或物理参数。

## 部署和记录

本次代码仅添加 Stage 07 工具，不修改 Stage 03/04 文件或依赖锁。
新增 CPU 定向测试 7 项通过，覆盖计数复位、虚假 episode 年龄、计时预热排除、
autoreset 隐藏 NaN、GPU 总显存余量及失败进程不可被成功报告掩盖。
测试使用隔离的 Torch 2.9.1 CPU 环境；该环境缺少 NumPy 的提示不代表云环境缺包。
尚未在开发环境执行 CUDA 容量试跑，不以 CPU 测试替代云端实测。
预检启动前检查 Stage 06 标签、祖先关系及冻结路径相对基线未变化。
每个工作进程报告当前提交和运行模块哈希，结果存入仓库外
`artifacts/double-balance-stage07/capacity-...`。

`scripts/deploy_stage07_capacity_local.sh` 在本机执行，把增量 Git bundle 与已校验的
Stage 05 文件通过一次 SSH 连接送入当前 A800；云端以 ff-only 导入，然后通过
tmux 启动容量测量。它不重新安装依赖、不运行 Stage 06 smoke，也不启动正式训练。

长期沿用 [本地提交与 SSH 推送规范](../handoffs/microduck-local-ssh-push-protocol.md)。
固定下载目录 `/home/lx/下载`，本机仓库 `/home/lx/microduck-double-balance/workspace`，
分支 `double-balance`，本机 origin 继续使用原 SSH 地址。
增量 bundle 名称为 `microduck-stage-07-from-stage-06.bundle`。
Stage 07 完成时才在本机创建 `stage-07-complete` 注释标签并执行 SSH push；
正常推送成功后不重复完整远端核验。
