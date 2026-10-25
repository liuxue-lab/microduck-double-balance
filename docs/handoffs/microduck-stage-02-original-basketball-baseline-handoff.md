# Microduck 双重平衡项目：Stage 02 原篮球任务基线复现交接文档

交接日期：2026 年 9 月 18 日

阶段状态：`PASS`

项目仓库：<https://github.com/liuxue-lab/microduck-double-balance>

开发分支：`double-balance`

本文件用于在新对话中开始 Stage 03。Stage 03 之前不得重复安装环境、重新下载已校验模型，或直接开始双重平衡训练。

## 1. 项目目标与当前边界

项目最终目标是：让 Microduck 站在自由滚动的篮球上，通过头颈、机身和双腿的全身协调，使安装在头部无围边薄托片上的自由小球保持平衡，并能在规定扰动下恢复。

已经确定的实施约束：

- 本机负责代码开发、MuJoCo 仿真、推理、测试和视频渲染。
- RTX 5060 Laptop 不承担正式强化学习训练。
- PPO 正式训练在云 GPU 上进行。
- 仿真平台沿用 MuJoCo、MuJoCo Warp 和 mjlab。
- 继承原篮球策略的单层 LSTM 与后续 MLP。
- 保持原 14 维动作顺序、缩放和 BAM 执行器语义。
- 原 `Mjlab-Basketball-MicroDuck` 任务必须保持兼容，不在其上直接覆盖新任务。
- 双重平衡任务使用独立任务 ID、配置、测试和导出接口。
- 当前阶段没有修改算法、动作空间或正式训练配置。

## 2. 阶段目标

Stage 02 的任务不是训练双重平衡策略，而是建立可信的原篮球任务本机基线，确认以下闭环成立：

1. 原篮球任务已注册并可加载。
2. 固定版本的训练检查点和 ONNX 模型已下载并通过校验。
3. PyTorch 与 ONNX 循环策略输出一致。
4. 原篮球策略能在本机 MuJoCo 环境中完成单环境短时推理。
5. 视频渲染入口可用。
6. 源码、修复和阶段状态已进入 Git/GitHub。

完成这一阶段后，Stage 03 才能在不混淆“原策略问题”和“新增上球物理问题”的前提下添加托片与自由小球。

## 3. 已冻结的软件与源码

### 3.1 本机环境

| 项目 | 已验收值 |
|---|---|
| 用户 | `lx` |
| 主机 | `lxlab` |
| 操作系统 | Ubuntu 22.04.4 LTS |
| 内核 | `6.8.0-138-generic` |
| 架构 | `x86_64` |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，8151 MiB |
| NVIDIA 驱动 | `595.84` |
| 项目根目录 | `/home/lx/microduck-double-balance` |
| 工作区 | `/home/lx/microduck-double-balance/workspace` |
| Python | uv 管理的 CPython 3.12.14 |
| uv | `0.12.15` |
| 虚拟环境 | `/home/lx/microduck-double-balance/workspace/.venv` |

原有 Conda 环境 `base` 和 `robot-il` 未被修改；已有 `/home/lx/robomimic` 项目未被覆盖。

### 3.2 上游冻结版本

| 项目 | 固定值 |
|---|---|
| Hugging Face 发布 revision | `6d8f74b97c75b1597efead1754ff54ca2af4900c` |
| 发布 manifest 中的训练代码提交 | `aa5bd7909873a807fb9f4045f491519c443142e7` |
| 早期资料核查 GitHub 提交 | `e7c82578852683978783e751ecf1e8e909d32a5b` |
| 本地导入基线提交 | `1d5b0373d31c81a1e3126c9be0d3cfa8de335486` |
| 本地冻结标签 | `upstream-hf-6d8f74b` |

三个上游标识分别代表模型发布 revision、manifest 记录的训练源码提交和资料核查提交，不能视为同一提交。

源码归档：

```text
/home/lx/microduck-double-balance/upstream/source-release-6d8f74b97c75.tar.gz
```

源码归档 SHA-256：

```text
9cd29c21a8d6ae4edce7b4f4e9d0e9e05a3a153a90923f016ba474d9163f9a27
```

### 3.3 锁定依赖

核心版本已通过导入检查：

| 组件 | 版本 |
|---|---:|
| `mjlab-microduck` | 0.1.0 |
| `mjlab` | 1.3.0 |
| `mujoco` | 3.10.0 |
| `mujoco-warp` | 3.8.1 |
| `torch` | 2.9.1+cu128 |
| `torchvision` | 0.24.1 |
| `warp-lang` | 1.12.0 |
| `better-actuator-models` | 1.0.1 |

环境同步后：

```text
UvLockHash=affaa39d97463c26d149043058c500a79bb2e78e89dfe37da8dce4ce5555dcdb
PyprojectHash=5c1a3430c1f41a20ea255649c53a49dad2d7e840650fafe7526626d2e6f7fd6f
UvLockUnchanged=PASS
PyprojectUnchanged=PASS
CoreImports=PASS
```

## 4. 原任务注册与接口事实

任务注册表包含：

```text
Mjlab-Basketball-MicroDuck
```

发布的 `b11_6999` 策略契约为：

| 项目 | 原任务约定 |
|---|---|
| Actor 观测 | 61 维 |
| 篮球状态进入 Actor | 否，blind policy |
| 网络 | 单层 LSTM，隐藏维度 256，后接 512/256/128 MLP |
| 动作 | 14 维 |
| 物理步长 | 0.005 s |
| 控制降采样 | 4 |
| 策略频率 | 50 Hz |
| 循环状态 | `h`、`c`，回合边界清零 |

本机运行时显式采用：

```text
MICRODUCK_BB_BLIND=1
MICRODUCK_BB_HISTORY=1
```

`HISTORY=1` 保持发布策略的 61 维 Actor 输入。Stage 03 不能在原任务上直接追加观测；新增观测将在独立双重平衡任务中实现。

## 5. 模型制品与完整性

固定模型目录：

```text
/home/lx/microduck-double-balance/artifacts/basketball-release-6d8f74b
```

已取得并校验的文件：

```text
checkpoint.pt
policy.onnx
manifest.json
validation.json
SHA256SUMS
params/agent.yaml
params/env.yaml
source.tar.gz
```

核心制品：

| 文件 | 大小 | SHA-256 |
|---|---:|---|
| `checkpoint.pt` | 约 9.6 MiB | `57a322eff092cb71e7cba831791232f0cc4fd45ede0441cad6d87813b0033e41` |
| `policy.onnx` | 约 2.4 MiB | `e105148b160b3a86170621648215be38dd5018e955930ce182394f4513c7569b` |
| `source.tar.gz` | 约 63 MiB | `9cd29c21a8d6ae4edce7b4f4e9d0e9e05a3a153a90923f016ba474d9163f9a27` |

模型目录约 75 MiB，不进入 Git 仓库。GitHub 中只保存源码、配置、测试、报告和制品校验信息。

### 5.1 下载故障与恢复

Python `huggingface_hub.snapshot_download` 初次失败，原因是本机环境存在：

```text
ALL_PROXY=socks://127.0.0.1:7897
```

当前 `httpx` 将该代理 URL 识别为未知 scheme，并抛出：

```text
ValueError: Unknown scheme for proxy URL URL('socks://127.0.0.1:7897/')
```

恢复方式是使用固定 revision 的直接文件地址与 `curl` 下载，再用发布仓库的 `SHA256SUMS` 校验。所有选定文件均得到 `OK`，恢复状态为 `PASS`。

后续若再次使用 Python 下载接口，应先处理代理 scheme，或沿用固定地址加校验和的下载方式；不得绕过哈希验证。

## 6. ONNX 一致性验收

使用固定检查点与固定 ONNX 执行 40 步循环策略比较，并在第 20 步重置 LSTM 状态：

```json
{
  "steps": 40,
  "reset_at": 20,
  "max_abs_error": 1.430511474609375e-06
}
```

验收结论：

```text
MicroduckStage02OnnxParityStatus=PASS
```

该结果只证明原 `b11_6999` PyTorch 与 ONNX 策略在此测试输入序列下数值一致。Stage 04/05 扩展观测后必须重新导出并重新验证，不能复用该结果作为新策略证据。

## 7. 单环境仿真烟雾测试

已运行参数：

| 参数 | 数值 |
|---|---|
| 检查点 | 固定 `b11_6999` 检查点 |
| seed | 101 |
| reset seed | 10101 |
| blind | `true` |
| command scale | 0.0 |
| 环境数 | 1 |
| 时长 | 5.0 s |
| 水平速度扰动 | x/y 均为 `[-0.09, 0.09] m/s` |
| 扰动间隔 | `[0.5, 1.0] s` |

测试结果：

```text
survival_fraction=1.0
fell_only_survival_fraction=1.0
time_out=0
out_of_terrain_bounds=0
nan_state=0
fell=0
failures=[]
MicroduckStage02SimulationSmokeStatus=PASS
```

存活期间指标：

| 指标 | 数值 |
|---|---:|
| forward MAE | 0.0442366 m/s |
| lateral MAE | 0.0776483 m/s |
| yaw MAE | 1.3845022 rad/s |
| mean tilt | 0.0314616 rad，约 1.80° |
| mean offset | 0.0091771 m |
| root above ball | 0.2344460 m |
| ball speed | 0.1159308 m/s |
| action delta RMS | 0.2257813 |

这只是 1 个环境、5 秒的流程烟雾测试，不是原任务的正式复现实验，也不能替代发布方三种子、3072 回合、60 秒评估。

## 8. 视频渲染模块修复

### 8.1 原始故障

首次执行：

```text
scripts/render_checkpoint.py --help
```

出现：

```text
ModuleNotFoundError: No module named 'mjlab_microduck.video_effects'
```

`render_checkpoint.py` 引用了 `mjlab_microduck.video_effects`，但发布源码归档的主包路径缺少该模块。

### 8.2 修复来源与完整性

从核查提交 `e7c82578852683978783e751ecf1e8e909d32a5b` 中归档的完整模块恢复：

```text
experiments/desk-climb/source/src/mjlab_microduck/video_effects.py
```

恢复到：

```text
src/mjlab_microduck/video_effects.py
```

Git blob 校验：

```text
ExpectedGitBlob=77cf4cb8c4180593354a199a56fd6044720cec40
ActualGitBlob=77cf4cb8c4180593354a199a56fd6044720cec40
DownloadedModuleIntegrity=PASS
```

模块导入通过：

```text
CrashEffects=PASS
configure_video_cfg=PASS
fix_render_shadows=PASS
VideoEffectsImport=PASS
RenderHelpCheck=PASS
```

修复提交：

```text
c094dc93173afa4df84bed1e1f04870af58524c5
fix: restore missing video effects module
```

这是 Stage 02 唯一修改的源码文件。该文件是从已核查的上游内容恢复，不是重新设计视频效果实现。

## 9. 视频渲染烟雾测试

视频烟雾测试采用：

```text
Task=Mjlab-Basketball-MicroDuck
Checkpoint=b11_6999
Blind=1
History=1
CommandScale=0
Duration=5s
NumEnvs=1
Seed=101
Resolution=1280x720
Effects=disabled
StopOnDone=enabled
MuJoCoGL=egl
```

返回验收状态：

```text
WorkspaceUnchanged=PASS
MicroduckStage02RenderSmokeStatus=PASS
```

用户返回的日志没有包含渲染文件最终路径、文件大小、视频元数据和 SHA-256，因此本交接文档不填写这些值。若 Stage 03 需要引用该视频，应从本机 `runs/stage02-render/` 下重新只读定位并计算哈希，不应猜测。

## 10. Git 与 GitHub 状态

### 10.1 本地 Git

| 项目 | 数值 |
|---|---|
| 开发分支 | `double-balance` |
| 当前阶段代码提交 | `c094dc93173afa4df84bed1e1f04870af58524c5` |
| 基线分支 | `main` |
| 基线提交 | `1d5b0373d31c81a1e3126c9be0d3cfa8de335486` |
| Git 用户 | `liuxue-lab` |
| Git 邮箱 | `296450086+liuxue-lab@users.noreply.github.com` |
| 工作树 | clean |

远端：

```text
origin  git@github.com:liuxue-lab/microduck-double-balance.git
```

### 10.2 远端核验

远端分支：

```text
refs/heads/main
1d5b0373d31c81a1e3126c9be0d3cfa8de335486

refs/heads/double-balance
c094dc93173afa4df84bed1e1f04870af58524c5
```

远端标签：

```text
refs/tags/upstream-hf-6d8f74b
TagObject=9b0cb1d16696bd0fe2361f855c5d6d2a2354c088

refs/tags/stage-02-complete
TagObject=405fc1910db01d7f6a4307327d266b103e002fc7
PeeledCommit=c094dc93173afa4df84bed1e1f04870af58524c5
```

Stage 02 源码和标签已上传 GitHub。

### 10.3 推送过程注意事项

首次推送命令在交互式终端中使用了：

```bash
set -euo pipefail
```

终端曾表现为自动退出。恢复审计显示两个分支和两个标签实际上均已成功上传，工作树也保持干净。

以后提供给交互式终端直接粘贴的命令，不再使用可能关闭当前 Shell 的 `set -e`。若需要严格错误处理，应在独立脚本或子 Shell 中运行，并先说明行为。

## 11. Stage 02 验收矩阵

| 验收项 | 状态 | 证据摘要 |
|---|---|---|
| 固定源码完整性 | PASS | 源码归档 SHA-256 已确认 |
| Python 3.12 独立环境 | PASS | CPython 3.12.14、`uv sync --locked` |
| 核心依赖导入 | PASS | Torch、MuJoCo、Warp、mjlab 导入成功 |
| 原任务注册 | PASS | `Mjlab-Basketball-MicroDuck` 可见 |
| 模型文件完整性 | PASS | 发布校验和通过 |
| PyTorch/ONNX一致性 | PASS | 最大绝对误差 `1.430511474609375e-06` |
| 单环境仿真 | PASS | 5 秒存活，无跌倒、无 NaN |
| 渲染入口 | PASS | 缺失模块已恢复，帮助信息可加载 |
| 视频烟雾测试 | PASS | 5 秒单环境渲染验收通过 |
| 原任务兼容性 | PASS | Actor 仍为 blind、61维、14动作 |
| Git工作树 | PASS | clean |
| GitHub远端 | PASS | `main`、`double-balance` 和标签已核验 |

Stage 02 总结论：

```text
MicroduckStage02Status=PASS
```

## 12. 未完成事项与禁止误读

Stage 02 没有完成以下工作：

- 没有添加头部托片。
- 没有添加自由上球。
- 没有建立双重平衡任务 ID。
- 没有扩展 Actor 输入。
- 没有修改奖励函数或终止条件。
- 没有迁移 LSTM 输入权重。
- 没有运行任何双重平衡训练。
- 没有得到双重平衡成功率。
- 没有导出新任务 ONNX。

因此，原任务 5 秒存活、原发布 97.01% 成绩和原 ONNX 一致性都不能写成双重平衡项目成果。

## 13. Stage 03 的唯一任务边界

下一阶段名称：

```text
Stage 03：双重平衡物理场景与重置机制
```

Stage 03 只处理物理建模，不训练策略，也不扩展 Actor 输入。具体任务：

1. 审计原篮球场景、机器人 XML、头部 body/geom、实体配置和任务注册方式。
2. 建立独立双重平衡场景，不覆盖 `Mjlab-Basketball-MicroDuck`。
3. 在头部添加可见、平整、无围边薄托片。
4. 为托片设置明确的质量、惯量和安装关系。
5. 添加带 `freejoint` 的自由上球。
6. 设置碰撞组和接触过滤，只允许预期接触。
7. 实现机器人、篮球、托片、上球和相关状态的完整重置。
8. 验证上球能自然滚动、离面和掉落，不使用隐藏墙、磁吸或焊接约束。
9. 比较必要的物理步长与 CPU/Warp 接触行为。
10. 确认原篮球任务和现有测试不受影响。

Stage 03 不包含：

- PPO 正式训练。
- 检查点迁移。
- LSTM 输入扩展。
- 奖励权重调参。
- ONNX 导出。
- 低速移动和正式扰动评估。

## 14. Stage 03 建议验收条件

进入 Stage 04 前至少满足：

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
WorkingTreeCleanAfterCommit=PASS
GitHubPushStatus=PASS
Stage03HandoffCreated=PASS
```

具体尺寸、质量、摩擦和接触参数应在 Stage 03 根据模型审计后确定。旧概念图中的托片 `80 × 60 × 2 mm`、上球半径 `20 mm`、质量 `10 g` 只是初始候选，不得直接写成已经验证的最终参数。

## 15. 固定阶段工作规则

从 Stage 03 开始，每个对话只处理一个阶段。阶段结束顺序固定为：

1. 完成该阶段实现。
2. 运行阶段验收。
3. 生成阶段交接文档。
4. 将代码、测试和交接文档一起提交。
5. 创建 `stage-XX-complete` 注释标签。
6. 推送开发分支和标签到 GitHub。
7. 只读核验远端分支、标签与工作树。
8. 结束当前对话。
9. 在新对话中开始下一阶段。

模型检查点、下载缓存和 `runs/` 视频默认不进入 Git；仓库中记录其来源、相对用途、SHA-256 和可复现命令。

本 Stage 02 交接文档是在 `stage-02-complete` 标签发布后补写。为避免重写已经公开的标签，不移动或强制覆盖该标签。将本文件追加提交到 `double-balance` 分支即可。从 Stage 03 起，必须先提交交接文档，再创建阶段完成标签。

## 16. 新对话接续提示词

在新对话中上传或粘贴本文件，并使用：

```text
请完整阅读《Microduck 双重平衡项目：Stage 02 原篮球任务基线复现交接文档》，开始 Stage 03。

本阶段只建立双重平衡物理场景和重置机制：添加无围边薄托片与自由上球，核查质量、惯量、碰撞、自然滚落、重置和 MuJoCo Warp 兼容性。

不要训练策略，不要扩展 Actor 输入，不要修改原 Mjlab-Basketball-MicroDuck 任务。一次只推进一个可验证小步骤，不使用会让交互式终端自动退出的 set -e。阶段结束时生成 Stage 03 交接文档，提交并上传 GitHub，然后结束该对话。
```

## 17. 接手时首先核对的状态

新对话第一步应只读检查：

```text
Workspace=/home/lx/microduck-double-balance/workspace
ExpectedBranch=double-balance
ExpectedMinimumHEAD=c094dc93173afa4df84bed1e1f04870af58524c5
ExpectedRemote=git@github.com:liuxue-lab/microduck-double-balance.git
ExpectedStage02TagCommit=c094dc93173afa4df84bed1e1f04870af58524c5
ExpectedWorkingTree=CLEAN
```

若交接文档已提交，`double-balance` 的 HEAD 会晚于 `c094dc9`，这是正常情况；应检查 `c094dc9` 是否为其祖先，而不是要求 HEAD 必须完全相等。

## 18. Stage 02 发布后审计修正

在进入 Stage 03 前对远端分支、标签和调用接口进行了只读复核，确认原
`stage-02-complete` 标签仍指向 `c094dc93173afa4df84bed1e1f04870af58524c5`，
且该标签不得移动或强制覆盖。

审计发现 `scripts/render_checkpoint.py` 在启用 `--effects` 时向
`CrashEffects` 传入了其构造器不支持的 `ground_only` 参数。当时的 Stage 02
视频烟雾测试明确关闭了特效，因此原无特效渲染结果仍然有效，但不能据此证明
特效路径可用。修正采用最小变更：移除无效实参，不改变恢复的上游特效实现，
并新增调用关键字与构造器签名的回归测试。

同时修正了原篮球配置中 `BALL_FRICTION` 的注释顺序。MuJoCo 的三项顺序为
`sliding、torsional、rolling`。本次只修正文档语义，不改变数值，不改变原
篮球动力学，也不使用该修正冒充新的物理复现实验。

本交接文档在发布后纳入仓库。上述修正形成 `stage-02-complete` 之后的新提交；
接手时仍应验证旧标签提交是当前 HEAD 的祖先，而不是要求两者完全相等。

由于仓库不包含发布检查点和既有 `runs/` 视频，本次提交只验证代码接口和 CPU
测试；不得补写不存在的视频文件大小、媒体元数据或哈希。需要正式引用 Stage 02
视频时，仍须在用户本机定位原文件并单独生成证据记录。

本次审计修正的可重复验收结果为：

```text
UvSyncLocked=PASS
PythonCompileCheck=PASS
TargetedVideoEffectsAndBasketballTests=10 passed
FullCpuTestSuite=222 passed, 2 skipped
RenderHelpCheck=PASS
GitDiffCheck=PASS
```

两个跳过项分别需要 Linux AArch64 GPU 和本环境未安装的 Rust 编译器，不属于
本次 Python 接口修正失败。仓库中新增的回归测试会静态比较
`render_checkpoint.py` 调用 `CrashEffects` 使用的关键字与实际构造器签名。

## 19. 统一阶段编号

后续仓库阶段统一为九个阶段，不再与早期“五个概念阶段”混用：

1. Stage 01：固定源码、环境和项目仓库。
2. Stage 02：复现原篮球任务基线。
3. Stage 03：建立托片、自由上球、接触和重置机制。
4. Stage 04：定义新任务观测、奖励、终止和评估条件。
5. Stage 05：迁移检查点并验证新旧策略动作一致性。
6. Stage 06：运行本机烟雾训练和云端规模预检。
7. Stage 07：进行带辅助的课程训练。
8. Stage 08：撤除辅助，完成正式训练和批量评估。
9. Stage 09：完成消融、ONNX、视频与最终文档。

## 20. Stage 03 强制实现约束

Stage 03 在原验收矩阵基础上增加以下不可省略的约束：

1. 新任务 ID 固定为 `Mjlab-DoubleBalance-MicroDuck`。
2. 不直接编辑共享的 `robot_allcollisions.xml`；使用独立派生 `MjSpec` 添加托片。
3. 托片安装在最终头部连杆 `jaw_soft` 的固定子 body 上，并具有显式质量和惯量。
4. MuJoCo `box` 的 `size` 使用半尺寸；文档中的托片尺寸始终记录完整尺寸。
5. 上球使用真正的 `freejoint`，不得继承关节阻尼、摩擦、刚度或电枢参数。
6. 明确并测试接触矩阵：上球—托片和上球—地面允许；上球不得由机器人其他
   部位、隐藏几何、下方篮球、约束或外力提供支撑。
7. 上球重置位置由托片坐标系计算；机器人和下球先重置，再更新托片位姿并放置
   上球。检查初始穿透、残留速度及前若干物理步的异常弹射。
8. 除中心静止样例外，必须使用初始偏移、切向速度或小角度倾斜验证自然滚动和
   掉落，避免把被动静止误写成控制成功。
9. 比较 `5 ms、2 ms、1 ms` 物理步长时，控制周期保持 `20 ms`；记录接触对象、
   首次掉落时间、最大穿透和 NaN 状态。
10. CPU MuJoCo 与 Warp 不要求轨迹逐点相等，但接触序列、自然掉落结论和数值
    稳定性必须一致。
11. 新任务实现后重新运行原篮球配置测试和固定种子烟雾测试，确认原任务没有被
    派生场景修改。
12. Stage 03 不添加 Actor 观测；`61+k` 的 `k` 在 Stage 04 冻结后才能进入
    检查点迁移。

交接结束。
