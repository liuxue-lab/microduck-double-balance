# Microduck 双重平衡项目：Stage 03 物理场景与重置机制交接文档

日期：2026-09-20

分支：`double-balance`

阶段标签：`stage-03-complete`

本文件供下一次对话开始 Stage 04 使用。Stage 03 已建立独立双重平衡物理
场景、固定托片、自由上球、接触过滤和完整重置，但没有设计新任务奖励、没有
扩展 Actor 输入、没有迁移检查点，也没有训练策略。

## 1. 阶段结论

Stage 03 的实现和本地验收结论如下：

```text
OriginalBasketballTaskUnchanged=PASS
DoubleBalanceTaskRegistered=PASS
TrayMassAndInertiaCheck=PASS
TopBallFreeJointCheck=PASS
ExpectedContactPairsCheck=PASS
UnexpectedSupportCheck=PASS
TopBallNaturalRollAndDropCheck=PASS
ResetRepeatabilityCheck=PASS
WarpSceneSmokeCheck=PASS
ActorObservationStill61D=PASS
Stage03HandoffCreated=PASS
```

本阶段新增任务 ID：

```text
Mjlab-DoubleBalance-MicroDuck
```

原任务 ID 保持不变：

```text
Mjlab-Basketball-MicroDuck
```

## 2. 接手基线

Stage 03 从以下已发布分支状态开始：

```text
Branch=double-balance
ParentCommit=9a3816541cafea72e9eb0ae2e4388d9a980eed6e
Stage02Tag=c094dc93173afa4df84bed1e1f04870af58524c5
Stage02TagAncestor=PASS
```

`stage-02-complete` 标签没有移动。Stage 02 标签之后的 `9a38165` 是上一轮
审计修正和交接文档提交。

依赖仍由现有锁文件固定：

```text
Python=3.12.14
uv=0.12.15
mjlab=1.3.0
mujoco=3.10.0
mujoco-warp=3.8.1
torch=2.9.1
warp-lang=1.12.0
```

## 3. 变更文件

Stage 03 的实现集中在以下文件：

- `src/mjlab_microduck/tasks/microduck_double_balance_env_cfg.py`
  - 独立任务工厂和注册配置。
  - 双重平衡专用派生机器人 `MjSpec`。
  - 固定托片、自由上球和碰撞位定义。
  - `2 ms` 默认物理步长和 `20 ms` 控制周期约束。
- `src/mjlab_microduck/tasks/mdp.py`
  - 新增 `reset_double_balance`，严格执行下层栈重置、前向运动学、上球重置。
- `src/mjlab_microduck/tasks/__init__.py`
  - 注册 `Mjlab-DoubleBalance-MicroDuck`。
- `tests/test_double_balance_cfg.py`
  - 配置、质量、惯量、自由关节、接触矩阵、重置、自然掉落和时间步测试。
- `scripts/validate_double_balance_physics.py`
  - 可重复的 CPU MuJoCo、MJWarp、reset 和原篮球固定种子验收。
- `docs/handoffs/microduck-stage-03-double-balance-physics-handoff.md`
  - 本交接文档。

共享文件 `src/mjlab_microduck/robot/microduck/robot_allcollisions.xml` 没有修改。
原篮球配置文件也没有修改。

## 4. 机器人和头部几何审计

托片安装目标是最终头部连杆：

```text
jaw_soft
```

在 HOME 姿态下测得 `top_head_shell` 碰撞网格的世界轴包围盒约为：

```text
FullExtentX=122.69 mm
FullExtentY=91.76 mm
TopZRelativeToRoot=154.433 mm
```

`jaw_soft` 在 HOME 姿态相对根连杆的位姿约为：

```text
Position=(-0.009003, 0.000000, 0.112599) m
Quaternion=(-0.707107, 0.000000, 0.707107, 0.000000)
```

该连杆的局部轴不能直接作为水平托片轴使用。派生 `MjSpec` 因此显式使用：

```text
TrayPositionInJaw=(0.04335135, 0.0, -0.02284261) m
TrayQuaternionInJaw=(sqrt(0.5), 0.0, sqrt(0.5), 0.0)
```

编译后的 HOME 托片坐标系为水平面，其中心相对根连杆为：

```text
TrayCenterRelativeToRoot=(0.01384, 0.0, 0.15595) m
```

托片下表面与审计得到的头顶包围盒之间约保留 `0.5 mm` 几何间隙。

## 5. 托片最终物理参数

托片是 `jaw_soft` 的无关节固定子 body：

```text
BodyName=double_balance_tray
GeomName=double_balance_tray_surface
Shape=box
FullSize=100 x 80 x 2 mm
MuJoCoHalfSize=(0.050, 0.040, 0.001) m
Mass=0.018 kg
Friction=(0.6, 0.005, 0.0001)
ContactDimension=4
Rim=None
```

代码和文档严格区分完整尺寸与 MuJoCo `box.size` 半尺寸。

托片 body 使用显式质心和对角惯量，不依赖编译器从可视几何估算：

```text
InertialPosition=(0.0, 0.0, 0.0) m
Ixx=9.6060e-06 kg*m^2
Iyy=1.5006e-05 kg*m^2
Izz=2.4600e-05 kg*m^2
```

编译质量审计：

```text
RobotMassBeforeTray=0.73724318 kg
RobotMassWithTray=0.75524318 kg
MassDelta=0.01800000 kg
```

HOME 姿态下机器人子树质心相对根连杆的审计值为：

```text
BeforeTray=(0.0005608, -0.0000136, 0.0217486) m
WithTray=(0.0008773, -0.0000133, 0.0249471) m
```

## 6. 自由上球最终物理参数

上球是独立实体，不是机器人子 body：

```text
EntityName=top_ball
BodyName=top_ball
JointName=top_ball_freejoint
GeomName=top_ball_sphere
Radius=0.020 m
Diameter=0.040 m
Mass=0.010 kg
CompiledDiagonalInertia=(1.6e-06, 1.6e-06, 1.6e-06) kg*m^2
Friction=(0.6, 0.005, 0.0001)
ContactDimension=4
```

编译模型中该实体只有一个真正的 `freejoint`：

```text
nq=7
nv=6
dof_damping=0
dof_frictionloss=0
dof_armature=0
joint_stiffness=0
EqualityConstraintCount=0
```

上球没有 action term、没有外力事件、没有焊接、没有磁吸、没有隐藏墙。

上球接触使用：

```text
solref=(0.003, 1.0)
solimp=(0.95, 0.99, 0.001, 0.5, 2.0)
```

该参数使所选 `2 ms` 样例中的最大动态穿透低于 `1 mm`。`5 ms` 会触发
MuJoCo 的安全时间常数限制，也是没有选择 `5 ms` 作为默认值的原因之一。

## 7. 接触矩阵

碰撞位仅在双重平衡派生场景中重分配：

| 类别 | `contype` | `conaffinity` | 说明 |
|---|---:|---:|---|
| 地面 | 1 | 1 | 保持场景默认 |
| 机器人普通碰撞 | 4 | 5 | 可碰地面和同类 |
| 下方篮球 | 4 | 5 | 可碰地面和机器人 |
| 机器人自碰撞专用 | 8 | 8 | 只碰同类自碰撞几何 |
| 托片 | 16 | 32 | 只接受上球 |
| 上球 | 32 | 17 | 只接受地面和托片 |

接触真值表：

| 接触对 | 结果 |
|---|---|
| 上球—托片 | 允许 |
| 上球—地面 | 允许 |
| 上球—下方篮球 | 禁止 |
| 上球—机器人普通碰撞 | 禁止 |
| 上球—机器人自碰撞几何 | 禁止 |
| 上球—视觉几何 | 禁止 |
| 下方篮球—地面 | 允许 |
| 下方篮球—机器人 | 允许 |

测试遍历了编译模型中的全部几何体，而不是只抽查足部。除托片和地面外，
没有任何几何体能与上球生成接触。

派生机器人不再运行继承的 `FULL_COLLISION` 后处理器。该后处理器只匹配
`.*_collision`，会把新托片静默改成 `contype=0, conaffinity=0`。Stage 03
改为在独立派生 `MjSpec` 内为每类几何显式设置位掩码，并保留原足底的
`condim=3`、摩擦系数和接触优先级。共享 XML 和原任务不受影响。

## 8. 重置顺序

新事件名：

```text
reset_double_balance
```

顺序固定为：

1. 重置下方篮球位置、姿态和速度。
2. 重置机器人根位姿、根速度、默认关节位置和关节速度。
3. 执行一次中间 `env.sim.forward()`，刷新 `jaw_soft` 和托片的真实世界位姿。
4. 从 `double_balance_tray_frame` 的局部坐标计算上球世界位置。
5. 把球心放在托片上表面法向 `20.5 mm` 处。
6. 将上球平移速度和角速度全部清零。
7. 由环境正常 reset 流程执行最终 forward 和观测计算。

中间 forward 是必要的；机器人根和关节刚写入 qpos 后，`site_xpos` 和
`site_xmat` 仍是上一回合的派生量。若直接读取，会使用旧头部姿态放置上球。

真实 Manager reset 验收结果：

```text
Seed=20260920
ActorDimension=61
TrayToTopBallCenter=0.0205000043 m
TopBallResidualSpeed=0.0
RepeatedResetPoseMaxAbsError=0.0
FiniteState=PASS
```

中心静止初始化还检查了：

```text
InitialPenetration=0.0 mm
InitialSpeed=0.0 m/s
FirstFiveStepsUpwardExcursion=0.0 mm
FiniteState=PASS
```

## 9. 时间步比较

所有比较都保持策略控制周期 `20 ms`：

| 后端 | 物理步长 | decimation | 首次接触地面 | 最大穿透 | 上球接触顺序 | NaN |
|---|---:|---:|---:|---:|---|---|
| CPU MuJoCo | 5 ms | 4 | 0.435 s | 7.022 mm | 托片 → 地面 | 否 |
| CPU MuJoCo | 2 ms | 10 | 0.412 s | 0.853 mm | 托片 → 地面 | 否 |
| CPU MuJoCo | 1 ms | 20 | 0.418 s | 2.890 mm | 托片 → 地面 | 否 |
| MJWarp | 2 ms | 10 | 0.412 s | 0.850 mm | 托片 → 地面 | 否 |

测试初始条件不是中心静止：上球在托片局部 x 方向偏移 `20 mm`，并获得
`0.18 m/s` 切向初速度。三档物理步长都让它自然接触托片、滚离边缘并落到
地面，没有任何脚本式掉落或支撑撤除。

默认选择：

```text
PhysicsTimestep=0.002 s
Decimation=10
ControlPeriod=0.020 s
```

`1 ms` 在这个单一样例中没有比 `2 ms` 获得更小的最大穿透，因此没有为
“数值更小看起来更安全”而盲目增加 2 倍计算量。`2 ms` 同时给出最低穿透、
稳定接触顺序和 CPU/MJWarp 一致结论。

## 10. CPU 与 MJWarp 说明

验收脚本同时使用两条独立路径：

- CPU MuJoCo：直接调用 `mujoco.mj_step`。
- MJWarp：通过 `mjlab.sim.Simulation` 调用 `mujoco_warp.step`。

当前自动化容器没有 NVIDIA 驱动，因此本次 MJWarp 验收运行在 Warp CPU
设备。它不是用 CPU MuJoCo 结果冒充 Warp 结果；输出后端明确为 `MJWarp`。

在有 NVIDIA GPU 的工作站上，同一脚本默认自动选择 `cuda:0`，也可显式运行：

```bash
uv run python scripts/validate_double_balance_physics.py --warp-device cuda:0
```

CPU 与 MJWarp 不要求逐点轨迹完全相等。Stage 03 检查的是接触顺序、自然掉落
结论、首次落地时间、最大穿透和数值有限性。本次两后端在 `2 ms` 下的首次
落地时间相同，穿透分别为 `0.853 mm` 和 `0.850 mm`。

## 11. 原篮球任务回归

双重平衡工厂通过深拷贝原篮球配置创建独立任务。原任务继续满足：

```text
Entities=(robot, ball)
PhysicsTimestep=0.005 s
Decimation=4
ActorDimension=61
OriginalBallCollisionMask=(1, 1)
ResetEvent=reset_basketball
```

固定种子真实 Manager smoke：

```text
Task=Mjlab-Basketball-MicroDuck
Seed=20260920
ControlSteps=3
ActorDimension=61
FiniteObservationsAndRewards=PASS
```

Stage 03 没有修改：

- `microduck_basketball_env_cfg.py`
- `robot_allcollisions.xml`
- 原篮球观察、奖励、终止、hold curriculum
- 原发布 LSTM 配置

## 12. 可重复验收命令

从仓库根目录运行：

```bash
uv sync --locked
python -m py_compile \
  src/mjlab_microduck/tasks/microduck_double_balance_env_cfg.py \
  src/mjlab_microduck/tasks/mdp.py \
  src/mjlab_microduck/tasks/__init__.py \
  scripts/validate_double_balance_physics.py \
  tests/test_double_balance_cfg.py
uv run list-envs | rg 'Mjlab-(Basketball|DoubleBalance)-MicroDuck'
uv run --with pytest pytest -q tests/test_double_balance_cfg.py tests/test_basketball_cfg.py
uv run python scripts/validate_double_balance_physics.py
uv run --with pytest pytest -q
git diff --check
```

有 GPU 时把物理验收命令替换为：

```bash
uv run python scripts/validate_double_balance_physics.py --warp-device cuda:0
```

本次本地结果：

```text
UvSyncLocked=PASS
PythonCompileCheck=PASS
TaskRegistryCheck=PASS
Stage03TargetedTests=8 passed
DoubleBalancePhysicsAcceptance=PASS
FullCpuTestSuite=230 passed, 2 skipped
GitDiffCheck=PASS
```

两个跳过项是仓库既有的平台条件测试，不是 Stage 03 失败。测试输出中的
actuator/site 警告来自 mjlab 对同一正则在不同命名空间的提示；actuator 的传动
类型仍明确为 joint，新增托片 frame 不会成为执行器目标。

## 13. Stage 03 明确没有完成的内容

以下内容不得从本阶段结果推断：

- 没有为上球添加 Actor 或 critic 观测。
- 没有冻结 `61+k` 中的 `k`。
- 没有增加上球奖励、终止条件或成功判据。
- 没有迁移旧 LSTM 输入权重。
- 没有运行 PPO smoke 或正式训练。
- 没有评估双重平衡策略成功率。
- 没有导出双重平衡 ONNX。
- 没有进行真实机器人试验。

新任务当前继承原篮球任务的奖励和 61 维观察，只是为了保持已知配置骨架和
完成物理场景构建；这些继承项没有被写成双重平衡奖励设计。禁止在 Stage 04
冻结观察和任务定义之前启动训练。

## 14. Stage 04 唯一边界

下一阶段名称：

```text
Stage 04：定义新任务观测、奖励、终止和评估条件
```

Stage 04 应完成：

1. 基于 Stage 03 已验证的托片坐标和自由上球状态，列出候选新增观测。
2. 冻结 Actor 输入从 `61` 扩展到 `61+k` 时的 `k`、顺序、尺度和噪声。
3. 明确 critic 特权状态，不能把 critic 设计冒充可部署 Actor 输入。
4. 设计双层平衡奖励、终止条件、成功判据和评估指标。
5. 为奖励符号、观察布局、终止阈值和不可作弊条件添加 CPU 测试。
6. 仍不迁移检查点、不训练策略；检查点迁移属于 Stage 05。

Stage 04 不得修改 Stage 03 已冻结的核心物理事实，除非新证据能复现并明确记录
问题。尤其不能通过隐藏墙、约束、磁吸或对上球施力来降低任务难度。

## 15. 固定阶段封口规则

Stage 03 的代码、测试和本交接文档必须在同一个阶段提交中出现，然后创建注释
标签：

```text
stage-03-complete
```

标签创建后应推送开发分支和标签，并只读核验：

```bash
git fetch origin double-balance tag stage-03-complete
git rev-list --left-right --count origin/double-balance...double-balance
git rev-parse double-balance
git rev-parse origin/double-balance
git rev-parse 'stage-03-complete^{}'
git merge-base --is-ancestor stage-02-complete stage-03-complete
git status --short --branch
```

验收期望：

```text
BranchDivergence=0 0
LocalBranchEqualsRemote=PASS
Stage03TagEqualsBranch=PASS
Stage02TagAncestor=PASS
WorkingTreeCleanAfterCommit=PASS
GitHubPushStatus=PASS
```

## 16. 新对话接续提示词

完成本阶段远端封口后，在新对话中使用：

```text
请完整阅读《Microduck 双重平衡项目：Stage 03 物理场景与重置机制交接文档》，开始 Stage 04。

本阶段只定义双重平衡任务的 Actor/critic 观测、奖励、终止、成功判据和评估指标。先审计可部署传感器，再冻结 61+k 的 k、布局和尺度。

不要迁移检查点，不要训练策略，不要修改 Stage 03 已验证的托片、自由上球、接触矩阵和 2 ms 物理步长。完成 Stage 04 验收、交接文档、提交、注释标签和 GitHub 推送后结束对话。
```

交接结束。
