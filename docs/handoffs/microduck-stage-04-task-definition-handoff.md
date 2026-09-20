# Microduck 双重平衡项目：Stage 04 任务定义交接文档

日期：2026-09-20

分支：`double-balance`

阶段标签：`stage-04-complete`

本文件供下一次对话开始 Stage 05 使用。Stage 04 已冻结双重平衡任务的观测、
奖励、终止、成功判据、指标和无辅助评估配置，但没有迁移检查点、没有运行 PPO、
没有得到策略成功率，也没有导出 ONNX。

## 1. 阶段结论

```text
Stage03ImplementationAudit=PASS
Stage04PlanAudit=PASS
ActorObservationDimension=61
CriticObservationDimension=85
ActorTopBallSlotDimension=6
CriticPrivilegedTopBallDimension=9
RewardSignAndAntiCheatChecks=PASS
TerminationBoundaryChecks=PASS
ContinuousSuccessCheck=PASS
DeterministicUnassistedEvalProfile=PASS
ManagerEnvironmentSmoke=PASS
Stage03PhysicsRegression=PASS
OriginalBasketballRegression=PASS
NoCheckpointMigration=PASS
NoPpoTraining=PASS
```

Stage 04 最重要的修正不是把 Actor 从 `61` 扩展到 `61+k`，而是确认
`k=0`：复用原篮球盲策略中恒为零的 6 维 `body_command` 槽，保持整个策略族的
61 维运行时契约。

## 2. 接手基线与前置审计修复

Stage 04 从以下已发布标签开始审计：

```text
Stage03Tag=37f991869297beeb18dc93260b9e2faab101c4a6
Stage03TagName=stage-03-complete
```

正式实现 Stage 04 前，先形成独立审计修复提交：

```text
AuditFixCommit=11a270f55f76fcac416b787edd170b3bd73cdbf6
AuditFixMessage=fix: close Stage 03 audit gaps
```

审计发现并修复：

1. 继承的 `nan_state` 只检查机器人，没有检查两个自由球体。双重平衡配置现在
   同时检查 `robot、ball、top_ball`，其他任务的默认行为不变。
2. 原 Stage 04 计划中的 `61+k` 与仓库共享 Actor 不变量冲突。计划已改为复用
   6 维预留槽，不扩展 Actor。
3. 官方评估必须排除下球 hold、随机推力和域随机化，避免把带辅助结果写成
   无辅助成功率。

审计报告：

```text
docs/audits/stage-03-implementation-and-stage-04-plan-audit.md
```

## 3. 传感器与可部署性审计

当前 Microduck 的 61 维可部署基础观测来自 IMU、关节编码器、上一动作和命令。
现有硬件接口没有直接测量头顶小球相对托片的三维位置与速度。

因此 Stage 04 的 6 维上球输入具有明确边界：

- 它是仿真中的 oracle 状态基线。
- 它同时定义了未来视觉或状态估计器必须输出的统一接口。
- 它不是“当前硬件已经具备的传感器输入”。
- 本项目当前只做仿真，因此可以用该基线分离控制问题与视觉估计问题。
- 若未来进入实物，必须由相机或其他估计器生成同顺序、同尺度的 6 维量，并
  重新做延迟、噪声和丢帧验证，不能直接把仿真真值用于实物结论。

没有把上球角速度放入 Actor。小球是均匀球体，姿态本身对接触几何无意义；
角速度可以帮助 critic 估值，但不是第一版控制所必需的可部署输入。

## 4. Actor 观测契约

Actor 总维数仍为：

```text
ActorDimension=61
```

前 55 维保持原策略顺序：

| 0-based 索引 | 内容 | 维数 |
|---:|---|---:|
| 0–2 | 机体角速度 | 3 |
| 3–5 | 投影重力 | 3 |
| 6–19 | 14 个关节位置 | 14 |
| 20–33 | 14 个关节速度 | 14 |
| 34–47 | 上一动作 | 14 |
| 48–50 | twist 命令 | 3 |
| 51–54 | 原 head command 零填充 | 4 |

原 `body_command` 的 6 维槽被原位替换为上球状态：

| 0-based 索引 | 数值 | 物理含义 | 归一化尺度 |
|---:|---|---|---:|
| 55 | `x_tray / 0.05` | 上球球心相对托片中心的局部 x | `20 m⁻¹` |
| 56 | `y_tray / 0.04` | 上球球心相对托片中心的局部 y | `25 m⁻¹` |
| 57 | `(z_tray - R) / 0.02` | 球心高度相对理想半径高度的误差 | `50 m⁻¹` |
| 58 | `vx_tray / 0.5` | 上球相对托片的局部 x 速度 | `2 s/m` |
| 59 | `vy_tray / 0.5` | 上球相对托片的局部 y 速度 | `2 s/m` |
| 60 | `vz_tray / 0.5` | 上球相对托片的局部 z 速度 | `2 s/m` |

其中 `R=0.020 m`。理想中心静止状态的新增 6 维正好为全零，这一点对 Stage 05
的旧策略动作一致性检查很重要。

训练配置在归一化空间对六维统一加入 `[-0.02, 0.02]` 均匀噪声，约对应：

```text
x position: ±1.0 mm
y position: ±0.8 mm
z position: ±0.4 mm
velocity: ±0.01 m/s
```

官方确定性评估配置关闭该随机噪声。双重平衡任务固定 `history=1`，继续使用
LSTM 内部状态，不允许通过观测堆叠改变 61 维输入形状。

## 5. Critic 特权状态

原 critic 为 76 维，继续保留 6 维下方篮球状态。Stage 04 在末尾新增 9 维：

```text
CriticDimension=76+9=85
```

前 6 维与 Actor 上球槽完全相同，后 3 维是上球角速度在托片坐标系中的分量：

```text
[wx_tray, wy_tray, wz_tray] * 0.2
```

即 `5 rad/s` 归一化为 `1`。这 3 维只供 critic 使用，不得在文档、导出或后续
实物讨论中冒充 Actor 可部署输入。

## 6. 奖励定义

Stage 04 保留原篮球任务的下层平衡奖励和正则项，并新增三项：

| 奖励名 | 权重 | 函数输出符号 | 含义 |
|---|---:|---|---|
| `double_balance` | `+4.0` | `[0,1]` | 上下两层同时稳定的乘法复合奖励 |
| `top_ball_speed` | `-0.10` | `≥0` | 上球相对托片线速度平方和 |
| `tray_tilt` | `-0.50` | `≥0` | 托片法向水平分量平方和 |

`double_balance` 不是三个独立正奖励的简单相加。其上球部分为：

```text
exp(-(x/0.020)^2 -(y/0.016)^2)
× exp(-((z-R)/0.006)^2)
× exp(-||v_rel||^2 / 0.12^2)
```

下层部分为机器人直立、根部相对下球居中和根部目标高度三个 Gaussian 的乘积。
最终奖励为上下两部分相乘。任一层明显失败时，复合奖励都会塌缩，避免机器人
倒地后仅靠“上球暂时靠近托片”继续获取大部分正奖励。

本阶段只冻结设计和初始权重，没有用训练曲线证明权重已最优。权重调整必须基于
Stage 06 以后的真实日志，且每个 penalty 的加权 episode reward 必须保持非正。

## 7. 终止条件

双重平衡任务包含：

| 终止名 | 条件 |
|---|---|
| `time_out` | 继承的回合时间上限 |
| `out_of_terrain_bounds` | 继承的场地边界 |
| `nan_state` | 机器人、下球或上球出现非有限状态 |
| `fell` | 继承的下层篮球跌落条件 |
| `top_ball_lost` | 上球超出托片可恢复几何包络 |

`top_ball_lost` 使用托片局部坐标：

```text
abs(x) > 0.050 + 0.020 = 0.070 m
or abs(y) > 0.040 + 0.020 = 0.060 m
or z < -0.020 m
```

阈值包含球半径，因此球心刚越过托片边缘时不会立刻终止；只有整个球已超出
可接触包络或落到托片下方后才回收环境。没有添加隐藏墙、焊接、磁吸或上球外力。

## 8. 成功判据

瞬时稳定必须同时满足：

```text
TopBallPlanarCenterError <= 0.012 m
TopBallHeightError <= 0.006 m
TopBallRelativeSpeed <= 0.08 m/s
RobotLowerBallPlanarOffset <= 0.060 m
0.200 m <= RobotRootHeightAboveLowerBallCenter <= 0.290 m
RobotTilt <= 20 deg
LowerBallPlanarSpeed <= 0.15 m/s
LowerBallHold <= 1e-6
```

正式成功不是单帧命中，而是以上全部条件连续保持：

```text
StableDuration=5.0 s
```

任一条件中断，连续计时立即清零。最后一条明确禁止在下球 hold 辅助尚未撤除时
记录成功，避免训练课程的辅助状态污染官方成功率。

## 9. 评估指标

配置新增：

| 指标 | reduce | 含义 |
|---|---|---|
| `top_ball_center_error_m` | mean | 上球平面中心误差均值 |
| `top_ball_relative_speed_m_s` | mean | 上球相对托片速度均值 |
| `double_balance_stable_fraction` | mean | 回合中满足瞬时无辅助稳定的时间比例 |
| `double_balance_success` | last | 回合结束时是否已连续稳定 5 秒 |

不能只报告 success。后续批量评估至少同时报告：

```text
SuccessRate
EpisodeLength
TopBallCenterErrorMean
TopBallRelativeSpeedMean
StableFractionMean
TerminationBreakdown
```

## 10. 官方无辅助评估配置

`make_microduck_double_balance_env_cfg(play=True)` 被固定为确定性、无辅助的名义
评估配置：

```text
LowerBallHold=(0.0,)
TwistCommand=(0.0, 0.0, 0.0)
Curriculum=None
RandomPush=None
DomainRandomization=None
ActorTopBallNoise=None
ResetXYNoise=0
ResetYaw=0
ResetTiltNoise=0
ResetJointNoise=0
ResetBallVelocityNoise=0
ResetTopBallXYNoise=0
```

只保留三个必要事件：

```text
reset_double_balance
reset_action_history
expand_bam_friction_fields
```

域随机化和扰动测试不是被取消，而是必须在后续单独的 robustness battery 中显式
开启并单独报告，不能混入名义成功率。

## 11. 变更文件

- `src/mjlab_microduck/tasks/mdp.py`
  - 新增上球相对托片运动学、Actor/critic 观测、复合奖励、惩罚、丢球终止、
    瞬时稳定、连续成功和评估指标函数。
- `src/mjlab_microduck/tasks/microduck_double_balance_env_cfg.py`
  - 冻结 61/85 维观测、奖励权重、终止、指标和确定性无辅助 play 配置。
- `tests/test_double_balance_cfg.py`
  - 将 Stage 03 的 61 维测试更新为“原位复用 6 维槽”。
- `tests/test_double_balance_task_definition.py`
  - 新增 Stage 04 的布局、尺度、噪声、符号、不可作弊、终止边界、连续成功、
    评估隔离和真实 Manager 维度测试。
- `scripts/validate_double_balance_task_definition.py`
  - 新增可重复 Stage 04 任务定义验收脚本。
- `docs/handoffs/microduck-stage-04-task-definition-handoff.md`
  - 本交接文档。

## 12. 验收结果

可重复命令：

```bash
uv sync --locked
python -m py_compile \
  src/mjlab_microduck/tasks/mdp.py \
  src/mjlab_microduck/tasks/microduck_double_balance_env_cfg.py \
  scripts/validate_double_balance_task_definition.py \
  tests/test_double_balance_task_definition.py
uv run list-envs | rg 'Mjlab-(Basketball|DoubleBalance)-MicroDuck'
uv run --with pytest pytest -q \
  tests/test_double_balance_cfg.py \
  tests/test_double_balance_task_definition.py \
  tests/test_basketball_cfg.py
uv run python scripts/validate_double_balance_task_definition.py
uv run python scripts/validate_double_balance_physics.py
uv run --with pytest pytest -q
git diff --check
```

本次结果：

```text
UvSyncLocked=PASS
PythonCompileCheck=PASS
TaskRegistryCheck=PASS
Stage04TargetedTests=26 passed
DoubleBalanceTaskDefinitionAcceptance=PASS
DoubleBalancePhysicsRegression=PASS
ActorDimension=61
CriticDimension=85
OneStepFinite=PASS
FullCpuTestSuite=239 passed, 2 skipped
GitDiffCheck=PASS
```

两个跳过项是仓库既有的平台条件测试。已知 actuator/site 正则警告仍来自 mjlab
同时检查 joint 和 site 命名空间；执行器传动类型没有改变，托片 site 不会成为
执行器目标。

Stage 03 的物理回归仍得到：

```text
MuJoCo2msMaxPenetration=0.8526506679 mm
MJWarp2msMaxPenetration=0.8503664285 mm
ContactSequence=tray -> terrain
OriginalBasketballActorDimension=61
OriginalBasketballFiniteSmoke=PASS
```

## 13. Stage 04 明确没有完成的内容

- 没有迁移原篮球 PyTorch checkpoint。
- 没有修改或扩展 Actor 第一层权重。
- 没有扩展 critic checkpoint 的 `76 -> 85` 输入层。
- 没有运行 64 环境、5 iteration PPO smoke。
- 没有启动本机或云端正式训练。
- 没有报告双重平衡成功率。
- 没有声称当前硬件能直接测量上球 6 维状态。
- 没有导出新 ONNX。
- 没有进行实物实验。

任务定义脚本 PASS 只表示接口和判据能正确构建，不表示零动作策略或旧篮球策略
已经能完成双重平衡。

## 14. Stage 05 唯一边界

下一阶段名称：

```text
Stage 05：迁移原篮球检查点并验证新旧策略动作一致性
```

Stage 05 应完成：

1. 定位并校验 Stage 02 使用的原发布 PyTorch checkpoint，记录路径、大小和
   SHA-256；模型文件仍不进入 Git。
2. 审计 checkpoint 中 Actor、critic、normalizer 和 LSTM 状态字典的真实键名与
   形状，不根据经验猜测。
3. Actor 输入仍为 61 维，应直接保持原形状。理想中心静止上球状态为六个零，
   在其他输入和 LSTM 状态相同时，新旧 Actor 动作必须数值一致。
4. critic 输入从 76 增至 85。保留旧 76 列，新增 9 列显式零初始化；不能静默
   丢弃旧 critic，也不能把 Actor 一致性误写成整个 PPO checkpoint 完全一致。
5. 对 normalizer 做同样的 61/85 维审计和迁移，新增 critic 维采用零均值、单位
   方差或与框架状态一致的中性初始化，并用测试锁定。
6. 生成合成观测序列，比较旧 Actor 与迁移后 Actor 的逐步动作和 LSTM 隐状态。
7. 只做加载、迁移和数值一致性，不运行 PPO；训练 smoke 属于 Stage 06。

## 15. Stage 05 验收建议

```text
CheckpointLocatedAndHashed=PASS
StateDictSchemaAudited=PASS
ActorInputStill61D=PASS
CriticInputMigrated76To85=PASS
NormalizerMigration=PASS
ZeroTopBallActorActionParity=PASS
LstmHiddenStateParity=PASS
OriginalCheckpointUnmodified=PASS
NoPpoTraining=PASS
FullCpuTestSuite=PASS
Stage05HandoffCreated=PASS
```

## 16. 固定阶段封口与远端状态

Stage 04 应把代码、测试、脚本和本交接文档放在同一个阶段提交中，再创建注释
标签：

```text
stage-04-complete
```

本自动化环境能够读取仓库，但普通 HTTPS 推送没有交互式凭据，已连接的 GitHub
应用对该仓库写操作返回 `403 Resource not accessible by integration`。因此本地
提交和标签可以完成，但远端快进必须在具有写权限的环境中执行：

```bash
git push origin double-balance
git push origin stage-04-complete
```

推送后只读核验：

```bash
git fetch origin double-balance tag stage-04-complete
git rev-list --left-right --count origin/double-balance...double-balance
git rev-parse double-balance
git rev-parse origin/double-balance
git rev-parse 'stage-04-complete^{}'
git merge-base --is-ancestor stage-03-complete stage-04-complete
git status --short --branch
```

期望：

```text
BranchDivergence=0 0
LocalBranchEqualsRemote=PASS
Stage04TagEqualsBranch=PASS
Stage03TagAncestor=PASS
WorkingTreeCleanAfterCommit=PASS
```

## 17. 新对话接续提示词

```text
请完整阅读《Microduck 双重平衡项目：Stage 04 任务定义交接文档》，开始 Stage 05。

本阶段只迁移并审计原篮球 checkpoint：Actor 继续保持 61 维，在理想中心静止的六维上球输入为零时验证新旧动作和 LSTM 隐状态一致；critic 与其 normalizer 从 76 维显式扩展到 85 维并将新增列中性初始化。

不要运行 PPO，不要修改 Stage 03 物理参数，不要改变 Stage 04 的观测顺序、尺度、奖励、终止和成功判据。完成 Stage 05 验收、交接文档、提交、注释标签和远端推送后结束对话。
```

交接结束。
