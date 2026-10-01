# Microduck Stage 10：初始化批次 C 审查与下一步

本批次三组初始化轨迹已审查。第二、三组完成原计划的末尾校验，第一组保留原始失败及缺失后检查；不把它改成完整 PASS。Stage 10 未完成，未训练、未评估新候选策略、未拆板。

## 输入、运行及证据保留

| 分卷 | 字节数 | SHA256 |
|---|---:|---|
| review-20261001T144102Z-3cf83d-part01.zip | 30,865,261 | `0aee59d5d9c9e78f8379229705f41b75acd9fab1c1476fec0676d58d94959dbb` |
| review-20261001T144102Z-3cf83d-part02.zip | 18,039,887 | `c30685597865d998849fd58342b74b3b94dfe26c16c9609eabdc08e99e710d24` |

两卷共 85 个条目：84 项清单文件加清单自身。逐项大小与 SHA256、ZIP 完整性及清单覆盖核验通过；原首组审查的 28 项文件未变化。回传恢复工具与交付版本逐字节一致。

| 组 | 尝试目录 | 结果 |
|---|---|---|
| C1 | `01-initialization/attempt-772ee431` | 复用 33 个快照，原状态仍为 FAILED_REVIEW_REQUIRED |
| C2 | `02-initialization/attempt-8181eb75` | 33 个快照，末尾检查完成 |
| C3 | `03-initialization/attempt-41fdfd96` | 33 个快照，末尾检查完成 |

每组 507 个独立数组，合计逐组验证 1,521 项；跨组去重后为 605 个。99 个快照共 91,428 次数组引用，数组内容哈希、dtype、shape、文件位置与引用一致；已保存数值数组均有限。每组原始数值数据 61,295,000 字节，含 NPY 头为 61,359,896 字节，二者不能混称。

三组都准确记录原有最终 reset 一次、forward 两次、BAM compute 一次、sense 一次、observation compute 两次、policy.reset 一次。rollout 为零、加载后的策略推理为零、新增 PPO 为零。构造期原有 CUDA graph 路径记录到 Python 层 mjwarp.step/forward/reset_data 各三次；不能据“零 rollout”声称零内部物理调用，也不能将这些入口计数当作完整 GPU kernel/graph 次数。

C2/C3 的 backend_before/after 相同，读取器保留的是实时 getter 对象而非缓存数值，未写精度设置。模型参数、完整 state_dict、Adam、进度/LR 及 101 项冻结源码末尾检查完成。初态数据、任务配置、依赖类源码、加载模型摘要及运行时身份三组一致。

C1 没有末尾 backend、参数/完整模型状态、Adam、LR/迭代、零 twist、observer.rows、最终 source/plan 校验；后续校验不回填为其历史结果。仍使用 `INITIALIZATION_C_REVIEWED_WITH_FIRST_RUN_POSTCHECK_GAP`。

## 独立比较结论

本次没有仅采用脚本的比较结论：直接读取数组，独立对齐全部事件和字段，重算首次差异及幅值，结果与回传 `comparison.json` 一致。

三对比较在全部 33 个对齐快照中，下列已采字段始终逐字节相同：

- 11 类显式状态输入：time、qpos、qvel、act、ctrl、qacc_warmstart、qfrc_applied、xfrc_applied、eq_active、mocap_pos、mocap_quat。
- 已采随机数状态、BAM 直接状态及调用参数、返回观测。
- 接触前缀；所有快照接触总数 nacon 均为零。

这些不是“全部模拟器输入完全相同”的证明。派生动力学数组、约束工作区、未采内部状态及运行时模型字段仍需分别考虑。

在首个 `loaded_before_eval_counter_reset` 快照中，三对已经出现 qacc、qfrc_bias、质量矩阵、子树质心等派生量差异。该快照位于环境/runner 构造、加载完成后，不能称为“从进程启动即无差异”，也不能把差异起点归到之后才执行的本次最终 reset。

最终 reset 完成时的关键差异：

| 比较 | qacc 最大绝对差 | qfrc_bias 最大绝对差 | 运行时 dof_frictionloss 最大绝对差 |
|---|---:|---:|---:|
| C1–C2 | 1.52587890625e-05 | 4.76837158203125e-07 | 9.313225746154785e-10 |
| C1–C3 | 1.52587890625e-05 | 4.76837158203125e-07 | 9.313225746154785e-10 |
| C2–C3 | 2.288818359375e-05 | 4.76837158203125e-07 | 0（逐字节一致） |

qacc 为广义加速度，含不同自由度类型；上述最大值按各分量原单位计算，不能写成“下球速度差”或统一标成 m/s。

两段值得保留的观测证据：

1. C1–C2 在第一次 forward 前，已采输入与可比模型字段一致，但部分派生量/约束工作区已有差异。第一次 forward 后出现新的质量矩阵和加速度差异。不能忽略调用前已有差异而断言 forward 是唯一原因。
2. C2–C3 在第二次 forward 前，已采输入和可比模型字段一致；第二次 forward 后 qacc 最大差为 2.288818359375e-05，子树质心等也再次出现差异。这支持继续审查前向动力学实现，但尚未识别具体 kernel，也没有证明完整隐藏状态相等。

本次没有 rollout，不能建立这些差异与 10 秒成功标签、下球 0.15 m/s 违例或 Stage 09 训练退化之间的定量因果关系。

## BAM 摩擦与约束的解释修正

回传比较器将 `model/*` 统一归类为 `compiled_model`，其中 **GPU 运行时 `dof_frictionloss` 不是不可变的编译参数**。此分类只表示存储位置，不代表静态物理定义。

本机回传的 `bam.mjlab.BamActuator` 实际源码显示：compute 从 qfrc_bias、qfrc_constraint、qfrc_actuator 和有效 EFC 摩擦行读取上一轮求解状态，形成外部/电机负载；随后计算 BAM 摩擦预算，经 `_write_frictions` 写入每世界 dof_frictionloss/dof_damping。返回的电机力矩与摩擦预算是不同输出路径。

C1–C2 的 dof_frictionloss 首次跨组差异出现在 `bam.compute#1.after`，只有 env 74 的一个分量，差 9.313225746154785e-10；此时已记录的调用参数、BAM 直接状态与返回力矩相同，而其读取的动力学量并非全部相同。源码存在相应传播路径，但这里未复算完整 BAM 摩擦公式，不能宣称已确定全部因果。

C2–C3 的摩擦字段原先有微差，在本次 BAM 调用后变为逐字节相同，之后 qacc 仍不同。因此“摩擦字段不同”不能单独解释全部初始化差异，也不能把该字段直接归为随机化参数漂移。

只检查每世界 `[0:nefc]` 有效约束行后，BAM 前/后及第二次 forward 后：nefc=nf=14，ne=nl=0，ntendon=0，nacon=0；有效行 type code 为 1，14 个 id 与受控 BAM DOF 一一对应。安装的 Data 源码将 nf 定义为摩擦约束数。这些记录对应 DOF 摩擦行，不是 14 个球—地面接触。填充工作区不参与这个结论。

## 解释边界

15 项模型/工作区采集缺口继续保留，包括较大的 BVH、EFC/J、未枚举的模型内部结构。BAM 的部分 Parameter 对象未被直接状态采集递归展开，不能称“BAM 所有内部参数都已逐项比较”。全数组哈希一致只证明已保存字节一致。

读回数组会同步 GPU，三组属于带观测的诊断条件；不能直接把它们当作无观测运行的完全替代。尚无依据修改 TF32、确定性开关、CUDA graph、求解器参数、奖励或验收阈值。

## 下一步范围：一次只读源码审计

现有证据足以停止扩展同条件 GPU 重放。下一步先补齐**实际安装版本**的 MuJoCo-Warp `forward/smooth/solver/constraint` 实现、相关 Warp 路径，以及 BAM 模块和参数读取实现。现有回传只包含主要类源码，缺少这些模块级函数和 kernel 定义，不能靠另一个版本的在线源码代替。

已交付 `microduck-stage10-source-audit.py`。它使用现有 `.venv`，只通过标准库读分发元数据和明确范围的 Python 源文件；不 import torch/warp/mujoco/mjlab/bam，不构造环境、不初始化 CUDA、不调用物理或策略、不训练、不安装依赖、不访问云端。核对原批次七个固定元数据文件和 101 项源码，保存各源文件 SHA256 后输出一个审查 ZIP。

读取范围：MuJoCo-Warp 的 `_src` Python 文件；BAM Python 实现；mjlab sim/env/entity/scene/manager/actuator 模块；Warp 的少量上下文/代码生成文件。跳过测试，最多 1,000 个文件、合计 24 MiB 源码；缺失源文件或版本变化如实记录，不自动下载安装。零环境、零 rollout、零策略推理、零 PPO、云预算 0。

将文件下载到 `/home/lx/下载` 后执行：

```bash
python3 '/home/lx/下载/microduck-stage10-source-audit.py'
```

输出在 `/home/lx/microduck-double-balance/artifacts/double-balance-stage10/source-audit/`。上传打印的单个 review ZIP。该采集程序已做语法、路径白名单、合成安装目录文件拷贝及打包清单核验；这里没有本机依赖安装，也未宣称已读到缺失 kernel 源码。

源码审计重点：是否存在跨 body/DOF 的浮点累加路径、哪些派生/约束量由 reset 保留或 forward 重算、BAM 模型写入如何进入已捕获的 CUDA graph。源码里存在某类操作仍只是机制线索，不等于证明此次运行的具体因果。

审计后收敛 Stage 10：若仍不能定位根因，保留“同机重放不一致已复现、根因未定”的正式结论，不自动扩大 GPU 实验矩阵。任何改变精度/graph/求解器的对照需先单列变量、原因和预算；物理、奖励、约束及验收变化必须先由用户确认。

## 头顶接触与拆板范围保持

Batch A 的头顶接触迁移静态审计仍有效，尚未运行任何迁移变体。上球当前只能接触托盘与地面，直接删板不成立。

保留此前待确认的隔离审计提案：唯一定位真实 head shell 碰撞面；仅开放上球—头壳及地面接触；校验真实曲面支承和有效摩擦；正式移除托盘 body/geom 及显式 18 g 惯量；复核整机/头部质量、质心和惯量；新增无质量头固连参考 site；迁移初态、61/85 维观测语义、奖励和验收参考系。旧托盘任务及结果保留，允许调整时转头，不锁头、不默认全程朝前。

材料/几何台架的固定头壳测试只用于审计，不能算无辅助平衡成功；接触参数、减重对照、完整任务迁移均不在本次源码收集的授权或执行范围内。正式实施前仍需具体确认。

## 持续状态与交接要求

本机仓库 `/home/lx/microduck-double-balance/workspace`；下载 `/home/lx/下载`；分支 `double-balance`；HEAD `00e34c2038771c5d4ad49fe45dff828c0232e60c`。`stage-09-complete` 仍指向技术收尾提交 `9c51a133b40e4342985bab696e54fc02cd45f615`，不移动、不重复完整远端核验。

主模型仍为 Stage 08 `E/20260929/update_004000.pt`，SHA256 `86d55c3703c18fcf4817db49c6e4dcf839b3f5195d68ae2c559db7e222bded97`。历史独立严格成功率保持 652/768=84.90%，不混入开发重复结果。旧标准仍为 10 秒回合终局连续稳定至少 5 秒、无辅助；程序检查通过不代表策略成功。

Stage 09 保留 PPO 2000 次，含断电损失重做的实际尝试为 2025 次；Stage 10 新增 PPO 0。本机 RTX 5060 仅代码、MuJoCo 仿真、推理与视频，不重跑 Stage 06 smoke。

云 RTX 5090 状态 `OFF_CONFIRMED_BY_USER`。不启动/停止实例，不调用旧关机守护；云端尚未全部回传的 Stage 09 原始轨迹继续保留。未来云工作及必要回传 SHA256 完成后立即提醒用户手动关机，提醒与实际确认分别记录。

后续交接必须保留 `docs/handoffs/microduck-local-ssh-push-protocol.md` 和 `docs/handoffs/microduck-cloud-gpu-and-transfer-protocol.md`。传输默认 Git SSH，代码与模型归档分支分开、不合并归档分支；本机创建提交和注释标签，由用户 SSH 推送，正常成功后不重复完整远端核验。本次没有创建本机提交/标签。

阶段结束前仍需整合 Batch A/B/C 评估与差异审计、已有视频审查、必要文件归档、正式最终交接和本机提交。此次初始化没有视频，视频审查记为不适用，不能声称新做了完整行为评估。
