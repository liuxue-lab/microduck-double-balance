# Microduck Stage 10：重放差异诊断与头顶接触迁移审计交接

审查日期：2026-10-01。证据基线：`double-balance@00e34c2038771c5d4ad49fe45dff828c0232e60c`。

Stage 10 的技术审查已完成，允许以明确保留问题的方式收尾。本文件生成时，本机归档核验、提交、注释标签和 SSH 推送尚待执行；只有收尾工具实际输出的 receipt 才证明本机步骤完成，正常成功的用户 SSH 推送输出才证明发布完成。不得把这份准备文件当作已执行回执。

双球平衡目标尚未完成；没有产生新合格策略。继续保留 Stage 08 主模型 E/20260929/update_004000.pt，SHA256 `86d55c3703c18fcf4817db49c6e4dcf839b3f5195d68ae2c559db7e222bded97`，历史独立严格成功率仍为 652/768（84.90%）。Stage 10 的重复开发集测试不能与该结果合并。

## 1. 本阶段范围和完成证据

全阶段新增 PPO 为零，云 GPU 用时为零，未重跑 Stage 06 smoke，未修改 Stage 03 物理、Stage 04 任务/验收、奖励或约束。头部纠偏暂停。旧标准保持：10 秒回合终局连续稳定至少 5 秒、无辅助，下球速度上限 0.15 m/s。程序检查 PASS 不代表策略成功。

| 批次 | 已完成工作 | 结论与边界 |
|---|---|---|
| A 重放 | 同一 RTX 5060、主模型、128 个开发初态，六次 10 秒重放；三段 env 99 视频 | 成功数依次 109、110、112、110、109、105；41 个初态的成功标签在重复间变化。属于重复开发诊断，没有新增独立基准 |
| B 子步 | 三个新进程，各 128 环境、10 控制步、100 个 2 ms 物理步；共 38,400 环境物理步、1,683 事件 | 第一个物理步后 qvel 已不同；B1 全轨迹保留但事后检查缺失，B2/B3 完整检查通过。0.2 秒不作成功率评估 |
| C 初始化 | 三个新进程、各 33 事件；没有 rollout、没有加载后策略推理 | 显式输入相同而部分派生动力学字段不同；C1 原始失败及事后缺口保留，C2/C3 完整检查通过 |
| D 安装源码 | 133 份已安装源码、142 项清单文件；纯 CPU float32 算术枚举 | 并行浮点累加是与质心差异相符的机制，未证明实际调度或全部轨迹差异的根因 |

审查清单合计 411 项：A 117、B 68、C 84、D 142；六个回传 ZIP 的 SHA256 固定在 `docs/audits/stage-10-evidence-inventory.json`。A 批的 review 是原始轨迹的投影，另有 80 项原始文件保留在本机，收尾工具也会核验。第三方安装源码、原始数组、视频和模型保留在 artifacts，不加入正式代码分支。

Stage 09 已保留四组各 500 次 PPO，共 2000 次；F2 断电丢失后重做 25 次，实际尝试计数为 2025 次。四组均无合格候选，此结论没有被本阶段改变。

## 2. 重放差异定位到哪里

A 批首个策略动作逐位相同，50 Hz 记录中 qvel 在 0.02 秒首次不同；同为无渲染的重复执行也不同，不能仅归因于 GPU 型号或渲染。B 批把 qvel 差异定位到第一个 2 ms 物理步之后，最大差约 5.96e-8；BAM/ctrl 在第二个物理步之前开始不同。三对比较中 qpos 首次在第 2/3/2 个物理步后不同，策略动作首次在第 2/2/3 个控制步不同。广义坐标混合平移、旋转和关节单位，不能把所有 qvel 差都写成 m/s。

B 的 reset_ready 已有 qacc、qfrc_bias、qfrc_constraint 差异；当时记录的 qpos、qvel、ctrl、warmstart、观测、BAM 直接状态、RNG 和首动作相同。C 从更早的初始化边界继续观察：最早保存的 loaded_before_eval_counter_reset 已有派生字段差异。所有 33 个对齐事件的 11 项显式状态输入逐位一致，记录的 RNG、观测、BAM 直接参数/调用参数一致；这不等于完整隐藏状态一致，也没有定位到构造之前的起始差异。

C 第二次 forward 后 qacc 的最大两两差为 1.52587890625e-5、1.52587890625e-5、2.288818359375e-5；qfrc_bias 最大差为 4.76837158203125e-7。某次 BAM 调用出现一个 dof_frictionloss 值差 9.313225746e-10，但输出电机力矩仍相同。另一些模型字段差异在该调用后消失。因此 BAM 是可能传递差异的环节，不能写成唯一根因。

C 相关边界的 nefc=nf=14、ne=nl=0、ntendon=0、nacon=0。有效行是 14 个 type 1 DOF 摩擦约束，ID 与 BAM DOF 对应；不是 14 个球接触。分析只使用有效行，没有把 EFC 容量尾部当物理约束。B 的初始接触存储变化一度只是排列变化；canonical 内容首次差异在第 15/16/15 步。

## 3. 安装源码机制审查

证据版本：mujoco-warp 3.8.1、mjlab 1.3.0、better-actuator-models 1.0.1、warp-lang 1.12.0。Simulation、ManagerBasedRlEnv、BamActuator 三个类正文与 C 运行时捕获源码一致。C 未在运行时固定每个内核模块的文件哈希，所以这份后续安装源码不能证明所有内核字节从未变化。

`smooth.py` 的 `_subtree_com_acc` 对同层并行 body 使用 `wp.atomic_add` 累加子树加权位置，再除以子树质量；`cinert`、`cdof` 随之计算。`_crb_accumulate` 对组合刚体惯量也有浮点原子累加。`forward.py` 中，这些位置/惯量计算位于动力学和求解器之前。

对 C 保存的相同 xipos、float32 body_mass、body_subtreemass、parentid 做纯 CPU 标量加法顺序枚举：第一次 forward 后三个发生跨运行变化的 subtree_com 分量、第二次 forward 后两个分量，各次实测值均属于可能的 float32 加法顺序结果集合。该结果增强了并行累加顺序敏感的解释；每个标量独立枚举，未重建统一向量/内核调度，未复现 qacc，也未证明 10 秒标签的因果链。

`reset_data` 清除显式状态和部分工作区，并非把所有派生动力学、约束存储和模型 dof_frictionloss 全部清零。BAM 读取上一次动力学载荷和有效摩擦约束，扣除自身干摩擦贡献，再原位更新每环境摩擦/阻尼；输出力矩与摩擦预算是不同量。Simulation 在扩展模型数组时会清缓存并重捕获图，不能仅凭原位写入就断言 CUDA graph 地址陈旧。

完整根因仍未确定。15 项模型/工作区捕获缺口（包括 BVH、EFC Jacobian 等）、14 个 BAM _mjs_actuator 对象和未完整枚举的 Parameter 对象值继续列为限制。到此结束 Stage 10 重复诊断，不继续无明确判别目标的 GPU 重跑，不默认改 TF32、求解器或确定性设置。

## 4. 原失败与检查缺口

- A 初始 probe 因混用 TF32 新旧读取 API 停止；修复的是诊断读取，不是训练/仿真精度配置。原日志保留。
- B 首次预检误把版本元数据 2.9.1 与运行时 2.9.1+cu128 当成不同版本；在构造前停止。后续改用实际运行时版本。
- B1 的完整 561 事件轨迹已产生，事后读取 `cudnn.rnn` 时受 LSTM 模块覆盖影响而失败。保留原失败，仅复用原轨迹；backend_after 与该次最终源码/plan 检查缺失，不能由 B2/B3 或后来核验倒填。
- C1 已捕获 33 事件后，错误地在尚未 process_actions 的初始化时刻要求辅助缓存为 Tensor；实际 `_force=None`，其余三个未使用缓存尚未建立。恢复工具仅修复此初始化观测器：同时核验辅助级别为零、动作宽度为零、hold/xfrc/qfrc 有限且为零，不写入辅助状态、不增加 reset/forward。保留 C1 原失败，不伪造 completed。
- C1 缺少最后的 twist、observer.rows、模型参数/full state_dict、Adam、学习率/计数、backend_after 和源码/plan 事后检查。C2/C3 完整通过不能消除该历史缺口。

旧 producer 的 PENDING、STOPPED、stage10_complete=false 是历史快照，不能覆盖或删除。最终审查可以结束技术阶段，但不会把原失败改成成功。当前本机收尾核验也不是过去进程退出时的状态证明。

## 5. 视频、验收和策略失败

A 三段 env 99 视频各 10 秒、250 帧、25 fps、1280×720；全部帧可解码，审看 150 个均匀样本及三个终帧，并交叉检查完整 50 Hz 指标。三次 env 99 均按旧标准成功，机器人与双球保持支撑，托盘可见；头部快速转约 90° 并保持偏转，仍有细小调整。终局头部 roll 约 25.4° 的姿态问题未解决。单视角和抽帧不能排除短时事件，也不能验证精确接触法线、滑移或穿透；失败 env 66/93 和边界 env 63 在本批没有视频。

下球速度超过 0.15 m/s 仍是主要直接违例条件，奖励因果未确定。env 63 的 250 个终局稳定样本可能对应 float32 累计计时 4.999996185302734；保留原程序 FAIL，仅记录影子计数诊断，不能静默放宽阈值。程序 PASS、三段成功视频和重复集成功数均不能宣布策略目标完成。

## 6. 拆板迁移审计与待确认方案

当前上球碰撞 mask 只允许托盘和地面，不允许头壳，直接删除托盘不可行。当前编译的头壳 visual geom 48、collision geom 49 只是本次索引；新实现应按明确的名称/身份定位。

HOME 下旧承球中心投影约 x=13.84 mm、y=0，真实壳面在该处约 z=150.105 mm，旧托盘顶面约 156.95 mm，差约 6.85 mm；整个壳的最高点 154.433 mm 不能代替该处接触高度。局部坡度中心约 4.56°、y=±12 mm 约 8°、±20 mm 约 16.4°。这是静态网格/凸包/射线分析，不是半径 20 mm 球的真实支持接触求解，也没有证明动态稳定。

上球接触 priority=1、托盘/头壳为 0，当前有效 condim=4（滑动和扭转），不是包含滚动摩擦的 condim=6。后续不能靠偷偷提高摩擦或添加隐藏平板达到结果。

拆除 18 g 显式惯量托盘后，机器人质量从 0.75524318 kg 变为 0.73724318 kg；jaw+tray 从 0.206766 kg 变为 0.188766 kg，组合质心在 jaw 坐标中的移动约 (-3.373,+0.105,-0.839) mm。完整惯量矩阵见 head-contact-migration-audit，必须重新编译核对，不能只改总质量。

下一阶段若用户确认，建议先单独实现保留旧任务的头顶接触变体和零 PPO 物理验收：真实曲面碰撞、上球只接触头壳/地面、去掉托盘 body/geom/18 g 惯量；分别测接触点、法线、穿透、滑移、有效摩擦、质量/质心/惯量。静态固定头材料测试只能验证物理，不能当无辅助策略成功，更不能永久锁死头颈。

正式任务迁移还需独立确认：承球参考点、初态采样、观测、奖励和验收坐标系。旧托盘参考系涉及 6D actor、9D critic 的上球状态及 61/85 总维度；维度不变不代表语义兼容。旧 `Rᵀ(v_ball-v_site)`、含 `-ω×r` 的旋转坐标导数、真实接触滑移是三个不同量。须先决定采用哪一个、对应阈值如何解释，再落代码。旧任务/模型/结果始终保留。

用户尚未授权上述物理或正式任务变更。本交接提供可审查范围，不是执行许可。头部未来允许调整时转向、稳定后回正，不默认全程朝前，不锁死头颈。

## 7. 下一次训练前必须明确的合同

目前无新增训练授权或云预算。若未来提出训练，先列出：初始化 checkpoint SHA、模型和观测归一化恢复规则、Adam 是完整恢复还是清空、学习率数值/调度、runner iteration/common_step_counter/辅助课程进度恢复或重置、辅助动作和缓存、实验组/种子/每组 PPO 增量、保存/评估间隔、云 GPU 型号/小时/费用上限及停止条件，供用户确认。

C 的诊断恢复记录是模型/归一化和 Adam 完整恢复，学习率 5.062500000000001e-05、iteration 4999、common_step_counter 120000，评估计数按旧流程清零并使用原 policy.reset；没有执行 optimizer step。该记录不能自动成为未来训练配置，特别是迁移观测语义之后。

## 8. 固定路径、Git、关机与数据保留

- 本机仓库 `/home/lx/microduck-double-balance/workspace`；下载 `/home/lx/下载`。
- A：`/home/lx/microduck-double-balance/artifacts/double-balance-stage10/replay/20260930T221521Z-e410f3f1`。
- B：`/home/lx/microduck-double-balance/artifacts/double-balance-stage10/substep/20261001T071355Z-6050d852`。
- C：`/home/lx/microduck-double-balance/artifacts/double-balance-stage10/initialization/20261001T141949Z-37e84aba`。
- D：`/home/lx/microduck-double-balance/artifacts/double-balance-stage10/source-audit/20261001T150302Z-59a20649`。
- 主模型：`/home/lx/microduck-double-balance/artifacts/double-balance-stage08/returned/5d998fc322345102/evidence-20260930T093645-8955/extension/E-20260929/segment-001/checkpoints/update_004000.pt`。
- 原开发初态：`/home/lx/microduck-double-balance/artifacts/double-balance-stage09/video-review/20260930T210037Z-5a95cdc3/initial-states/dev.pt`，SHA256 `253123669696a953830caeaf587e1300e023c12293d608e38e655a38cdc00f58`。
- 本机收尾回执：`/home/lx/microduck-double-balance/artifacts/double-balance-stage10/finalization/<commit>/finalization-receipt.json`。

必须继续遵守 [本机 SSH 推送规范](microduck-local-ssh-push-protocol.md) 和 [云 GPU 与传输规范](microduck-cloud-gpu-and-transfer-protocol.md)。正式代码在 double-balance；模型/大文件走独立阶段归档分支、独立 Git SSH 缓存检出并核对提交与文件 SHA，绝不合并归档分支。Stage 09 历史上传 SSH、下载 HTTPS 的事实保留，未来默认取文件也使用 Git SSH。

Stage 09 的 13 个必要检查点、18 份云端评估、10 段本机视频及 175 项审查文件已核验，不重复完整远端检查。`stage-09-complete` 仍指向 `9c51a133b40e4342985bab696e54fc02cd45f615`，后续关机确认文档在 00e34c2，不移动旧标签。

云 RTX 5090 为 **OFF_CONFIRMED_BY_USER**，依据 Stage 09 后续关机确认和本次用户说明，优先于早期 UNCONFIRMED/PENDING。Stage 10 未启用云实例，没有新关机操作或新提醒事件。完整 Stage 09 云端原始轨迹尚未全部回传，禁止删除云端数据。不得自行启停实例，不运行旧失败自动关机守护。未来云计算与必要回传 SHA 完成后立即提醒用户手动关机，提醒时间和实际确认分别记录。

## 9. 本机收尾命令与常见停止原因

下载最终收尾脚本后执行：

```bash
python3 '/home/lx/下载/microduck-stage10-finalize.py'
```

该工具只做标准库文件核验和本机 Git 操作：固定 101 项运行源码、411 项审查文件、80 项 A 原始文件、六个回传 ZIP、主模型和开发初态；写入明确列出的 Stage 10 新文件，提交并创建注释标签 `stage-10-complete`。不联网、不运行仿真/训练、不改包、不推送、不删除历史证据。先执行 `--verify-only` 可只核验、不写入。

成功后按输出执行：

```bash
cd /home/lx/microduck-double-balance/workspace
git push origin double-balance
git push origin stage-10-complete
```

正常成功的 SSH 输出足够，不再次要求完整远端核验。若 HEAD、标签、目标文件、暂存区或证据 SHA 与固定值冲突，工具停止并保留文件；回传错误消息，不使用 reset、force-tag 或覆盖来绕过。提交后中断、尚未打标签时，可重新执行同一收尾脚本，它会核验自己已经生成的提交再完成标签。已有同名其他标签绝不强制移动。用户 Git 身份/签名配置异常时保留现场，由用户按原规范解决。

阶段收尾后下一对话进入 Stage 11，先读取本交接、current-handoff、final-acceptance、final-evaluation-review、failure-analysis、source-mechanism-review、head-contact-migration-audit、local-finalization 与实际 receipt，再提出最小的头顶接触物理验证变更供用户确认。不要再次把 Stage 10 的重复初始化诊断当作默认下一步。
