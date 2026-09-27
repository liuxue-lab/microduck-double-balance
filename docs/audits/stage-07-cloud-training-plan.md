# Stage 07 云 GPU 辅助课程训练计划与容量预检

状态：A800（端口 26497）复用环境后，用户返回 `Stage07Capacity=PASS`。
完整容量汇总已收到并归档：四档均通过且满足显存余量门槛，正式规模选择 4096。
正式入口与恢复/评估工具已实现，等待本机导入和云端启动；不预先声称训练已运行。
本文件不是 Stage 07 完成交接。

## 基线和硬件

- Stage 06 注释标签提交：`e3f26e3cbf633c70c75d4d073dbeb8c47c4f5bd1`。
- Stage 06 已验收 64 环境、5 更新、保存重载；不重复该 smoke。
- 当前云目标：AutoDL，`NVIDIA A800-SXM4-80GB`，容量实测 81920 MiB；
  驱动 580.126.09 是前一 A800 的记录，新实例完整环境报告尚未回传。
- 当前新实例 SSH：`ssh -p 26497 root@connect.bjb1.seetacloud.com`；此前 52528 为已替换实例。
- 原实例安装 PASS、新实例环境复用及容量 PASS 已收到；新实例完整环境和 CPU 配额
  已由部署入口写入云端 clone-environment 报告，完整内容尚未回传。
  如果克隆未带入数据盘上的项目/环境，入口停止并报告缺失，不自动重新下载。
- 正式训练只在云 GPU 执行，本机 RTX 5060 继续用于仿真，不启动长期训练。
- 用户给定约 24 小时的规划上限，允许小幅浮动，训练效果优先；不设置 24 小时强制超时。
  A800 实际单价尚未提供，不代填人民币费用；按这次明确的时长范围推进。
- 完整容量数据见 `stage-07-cloud-capacity-summary.json`：来自用户粘贴的完整 JSON，
  为语义归档，不声称原始文件字节/SHA 已核验。过程来源见 `stage-07-capacity-progress.json`。

| 环境数 | 采样显存峰值 MiB | transition/s | 每次更新秒数 |
|---|---:|---:|---:|
| 512 | 3615 | 4964.43 | 2.4752 |
| 1024 | 6281 | 8208.72 | 2.9939 |
| 2048 | 11937 | 14465.47 | 3.3979 |
| 4096 | 23323 | 22342.05 | 4.4000 |

4096 的采样显存峰值约 22.78 GiB，余量约 71.5%；它在本次实测中吞吐率最高。

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
环境计数为 N×24，而不是 N×24×环境数。正式入口保存并恢复 actor/critic/归一化器、
Adam、调度器 LR、已完成更新数、全局/物理计数、逐环境辅助等级和 hold。
检查点命名为 `update_NNNNNN.pt`，N 表示已完成更新数；内置 iter 仍保存 N-1。
恢复时采用原规模和 seed，先在恢复的全局课程计数下重置 episode，再恢复每环境等级，
避免以新 episode 的零年龄将课程误降一级。模拟器、RNG、episode 及 rollout 隐状态
重新开始，不称为逐轨迹无缝恢复。Adam 和 PPO 学习率必须一致。

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

## 原项目训练量与正式方案

已核对的是作者开源实验记录，没有据此杜撰论文中的统一训练轮数：

- `experiments/basketball/TRAINING.md`：b9 从 b6_4250 续训 3000 次，末次 7249
  表现退化，验证选取较早的 6500；b11 再从 b9_6500 续训 500 次，4096 环境，
  发布编号 6999。这个编号不能解释为一次从零开始训练的精确总更新数。
- `src/mjlab_microduck/tasks/microduck_basketball_env_cfg.py`：新训练默认 6000 次。
  Stage 04 双平衡配置继承这个值。
- 我们从 Stage 05 迁移权重开始，新任务进度从零计算；既不是复刻原作者最后 500 次
  续训，也不把 Stage 05 的源编号 6999 加到 Stage 07 已完成更新数中。

首段设为 **4096 环境 × 6000 次 PPO 更新**。每次 24 步，共 589824000 条 transition。
先前提出的 2000 次改为中途评估节点。每 100 次保存、每 500 次评估，保留初始、
所有周期检查点、最佳名义评估模型指针和最终模型。以下纯训练预测均不含保存/评估：

| 累计更新目标 | transition 数 | 纯训练小时 |
|---|---:|---:|
| 2000（中途节点） | 196608000 | 2.44 |
| 6000（首段） | 589824000 | 7.33 |
| 10000（按结果续训） | 983040000 | 12.22 |
| 16000（按结果续训） | 1572864000 | 19.56 |

首段暂留约 8–10 小时，包括启动、保存和评估；这是估计，不是收敛或账单保证。
6000 次后检查辅助等级是否下降、episode 是否延长、顶球误差是否减小、无辅助名义评估
是否改善，再决定续训到 10000 或 16000 次。如果指标持续退化或课程长期不前进，
先检查原因，不以追加轮次代替诊断。约 24 小时只是用户给定的规划范围；
训练脚本按累计更新目标执行，不设置 `timeout 24h`、不配置自动关机。

保留训练奖励、各惩罚项、episode 长度、辅助等级比例和 Stage 04 指标。
独立评估使用冻结的 play 配置及连续末尾 5 秒的无辅助成功定义。
每次运行 64 个确定性名义初态副本、10 秒 horizon，仅统计各环境首个 episode。
终止前直接读取已计算的 Stage 04 指标，避免重复累积稳定时间或被 autoreset 清空。
确定性名义初态副本不能解释为多个独立随机任务样本或随机鲁棒性成功率。
选择顺序为名义成功比例、存活时长、稳定时间占比、顶球中心误差；完全并列保留较早模型。
该 best 指针仅覆盖当前训练段；续训段与旧段的所有候选在最终归档时一起比较。
如果未学会无辅助双重平衡，按实际结果交接，不改写成功阈值或物理参数。
本阶段仍为带辅助课程训练；不自动展开 Stage 08 的独立撤辅助训练或广泛随机评估。

## 部署和记录

本次仅修改 Stage 07 工具，不修改 Stage 03/04 文件或依赖锁。
Stage 07 定向测试共 19 项通过（包含 7 项容量工具测试），覆盖计数复位、虚假 episode
年龄、计时、NaN、GPU 余量、错误进程、恢复的课程/LR/Adam、一致性拒绝、初始和训练
检查点索引、终止前指标、回退模型选择及带起始偏移的更新回调。
测试使用隔离的 Torch 2.9.1 CPU 环境，未运行 Stage 06 smoke。
容量 CUDA 实测已由用户回传；新增正式入口仍须在所选云实例实际执行。
预检启动前检查 Stage 06 标签、祖先关系及冻结路径相对基线未变化。
每个工作进程报告当前提交和运行模块哈希，结果存入仓库外
`artifacts/double-balance-stage07/capacity-...`。

`scripts/deploy_stage07_capacity_local.sh` 在本机执行，把增量 Git bundle 与已校验的
Stage 05 文件通过一次 SSH 连接送入当前 A800；云端以 ff-only 导入，然后通过
tmux 启动容量测量。它不重新安装依赖、不运行 Stage 06 smoke，也不启动正式训练。

容量通过后使用 `scripts/deploy_stage07_training_local.sh 26497`，只传新 bundle，
复用已安装环境、Stage 05 检查点和容量报告；后台会话为 `microduck-stage07-training`。
启动时重新检查实际 GPU、版本和冻结文件，然后保存并独立评估初始检查点，最后开始
正式 PPO 更新。评估子进程失败则保留检查点并停止本段，不把评估失败解释成零成功率。
完整操作与恢复说明见 `docs/handoffs/microduck-stage-07-training-runbook.md`。

长期沿用 [本地提交与 SSH 推送规范](../handoffs/microduck-local-ssh-push-protocol.md)。
固定下载目录 `/home/lx/下载`，本机仓库 `/home/lx/microduck-double-balance/workspace`，
分支 `double-balance`，本机 origin 继续使用原 SSH 地址。
增量 bundle 名称为 `microduck-stage-07-from-stage-06.bundle`。
Stage 07 完成时才在本机创建 `stage-07-complete` 注释标签并执行 SSH push；
正常推送成功后不重复完整远端核验。
