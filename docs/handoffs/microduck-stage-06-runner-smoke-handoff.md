# Microduck 双重平衡项目：Stage 06 真实 runner 加载与 PPO smoke 交接文档

验收记录时间（UTC）：2026-09-21T04:52:10.556809+00:00

分支：`double-balance`

阶段标签：`stage-06-complete`，由本机收尾命令创建，指向包含本文件及归档证据的最终提交。

Stage 06 已完成迁移检查点的真实 runner 严格加载、64 环境 × 5 iteration PPO smoke、检查点保存与重载，以及 4096 环境的配置规模预检。没有启动正式长期训练，没有测得正式双重平衡成功率。实际云 GPU 容量尚未测量，留到 Stage 07 云端启动预检。

## 1. 阶段验收结论

```text
Stage05AnnotatedTag=PASS
Stage05MigrationCheckpointSHA256=PASS
RegisteredRunnerStrictActorLoad=PASS
RegisteredRunnerStrictCriticLoad=PASS
RebuiltAdamLoad=PASS
ActorCriticActionDimensions=61/85/14
RestoredIteration=6999
RestoredCommonStepCounter=168048
SmokeEnvironments=64
SmokePpoUpdates=5
SmokeVectorSteps=120
SmokeTransitions=7680
ReturnPasses=5
AdamMiniBatchUpdates=100
NanStateTerminations=0
RolloutReturnLossGradientModelOptimizerFinite=PASS
SavedCheckpointAndSHA256=PASS
FreshRunnerStrictReload=PASS
RecurrentActionHiddenAndValueEquality=PASS
MigrationInputUnmodified=PASS
Cloud4096ConfigurationPreflight=PASS
CloudGpuCapacity=NOT_MEASURED_ON_CLOUD_GPU
FormalLongTrainingStarted=False
FormalDoubleBalanceSuccessRate=NOT_EVALUATED
Stage06Acceptance=PASS
```

本机完整回归结果：`254 passed, 2 skipped, 3 warnings in 19.95s`。两个跳过项为原有平台条件测试，三个 warning 为既有 actuator 正则同时匹配 joint/site 的提示。没有通过修改物理或任务定义来消除提示。

## 2. Git 基线与提交

```text
Stage05TagCommit=c5b0aee66de10aa0c8aea0bdc85f1a917fada1df
Stage06StartingCommit=55e41f8a5d4a791af5625f45366191a68eaa015b
AcceptedSmokeImplementationCommit=06ec7241766d4e6ef67b90580855344dfa9748d3
FinalizationToolCommit=41b4c5cc286633c964598ad47e302933883af9d4
```

最终文档提交的 SHA 由本机 `git rev-parse HEAD` 输出，`stage-06-complete` 解引用到该提交。文件不预填自身提交 SHA，以免形成自引用。SSH 远端推送状态以实际 `git push` 输出为准，本文件不预先声称远端已更新。

## 3. 检查点与原始证据

### 3.1 Stage 05 迁移输入

```text
Path=/home/lx/microduck-double-balance/artifacts/double-balance-stage05/checkpoint.pt
SizeBytes=3342581
SHA256=548758afc546e11d8f3f446a4bcc846283a669ff20f4048da7a3a7bb1332b98e
```

本机最初缺少该迁移文件。使用既有迁移工具、固定发布源检查点重新生成后，SHA-256 和独立数值验收均通过。检查点不进入 Git，导入代码 bundle 不会同时带入模型文件。

固定原篮球源检查点仍为：

```text
/home/lx/microduck-double-balance/artifacts/basketball-release-6d8f74b/checkpoint.pt
SHA256=57a322eff092cb71e7cba831791232f0cc4fd45ede0441cad6d87813b0033e41
```

### 3.2 Stage 06 smoke 输出

```text
Path=/home/lx/microduck-double-balance/artifacts/double-balance-stage06/smoke-20260921-125127/logs/model_7003.pt
SizeBytes=10020981
SHA256=ba124dd0385f3697521d2c53f2e3261394290c2f2a83c2a149a69b1c3db51d23
FinalIteration=7003
FinalCommonStepCounter=168168
```

该文件只作为训练链路验证制品，不能作为正式性能结果或部署策略发布。

### 3.3 验收记录

```text
OriginalReport=/home/lx/microduck-double-balance/artifacts/double-balance-stage06/smoke-20260921-125127/acceptance.json
OriginalReportSHA256=38b385a2e51eddd4b9c571ec09e087a1b15616033aa6fc7f0ec7566c8371d1e4
ArchivedReport=docs/audits/stage-06-cuda-acceptance.json
ArchivedPytestLog=docs/audits/stage-06-local-pytest.txt
```

归档 JSON 和测试日志保留本机文件的原始字节；收尾工具在写入文档前复核 smoke 检查点实际文件大小及 SHA-256。路径、哈希、设备和各轮 loss 均从本机报告读取，没有使用预期值代填。完整控制台和 TensorBoard 日志继续保存在对应 artifacts 运行目录及其旁边的日志文件中。

## 4. 实际运行配置

设备：`NVIDIA GeForce RTX 5060 Laptop GPU`。

真实 runner：`mjlab_microduck.tasks.MicroduckOnPolicyRunner`。

| 组件 | 实际版本 |
|---|---|
| mjlab | 1.3.0 |
| mujoco | 3.10.0 |
| rsl-rl-lib | 5.0.1 |
| torch | 2.9.1 |
| warp-lang | 1.12.0 |

- 任务：`Mjlab-DoubleBalance-MicroDuck`。
- 随机种子：`20260921`。
- 64 个环境，每环境每次 rollout 24 步；仅 5 次 PPO update。
- LSTM Actor 输入 61 维，critic 输入 85 维，动作 14 维。
- 初始 Adam moments 为空，初始学习率 `1e-3`，沿用原 adaptive 调度。
- 仿真步长 2 ms，decimation 10，控制周期 20 ms。
- 继承的篮球辅助课程 levels：`[1.0, 0.5, 0.25, 0.1, 0.03, 0.0]`。
- 只修改本次运行的环境数量、seed、日志和有限运行参数；Stage 03 物理和 Stage 04 任务定义保持不变。
- 使用本地 TensorBoard，关闭外部模型上传。

## 5. 五轮运行记录

| PPO 更新 | runner iteration | 环境计数 | 学习率 | entropy | surrogate | value |
|---|---|---|---|---|---|---|
| 1 | 6999 | 168072 | 1e-05 | 2.69380597 | 0.358744807 | 29.7784858 |
| 2 | 7000 | 168096 | 1e-05 | 2.71953903 | -0.0351978171 | 36.2783651 |
| 3 | 7001 | 168120 | 1e-05 | 2.72231275 | -0.0328609534 | 28.5036461 |
| 4 | 7002 | 168144 | 1e-05 | 2.72154875 | -0.0313083829 | 25.2234006 |
| 5 | 7003 | 168168 | 1e-05 | 2.72001985 | -0.0308178884 | 22.3445962 |

Smoke 记录耗时 `6.090 s`，该耗时包含验收检查和保存等开销，不作为正式训练吞吐率基准。

Actor 新六列权重最大绝对值：`0.00458024954`；critic 新九列权重最大绝对值：`0.00542293675`。它们从迁移时的零权重变为非零，证明新输入参与了短程学习，不代表上球控制技能已经学成。

Torch 测得峰值 allocated：`66189824 bytes`；峰值 reserved：`94371840 bytes`。它们不包含所有 Warp/模拟器分配，不能当作整个进程的总显存。

## 6. 保存、重载与计数语义

RSL-RL 5.0.1 的 `learn(5)` 从恢复的 6999 开始，执行索引 6999、7000、7001、7002、7003。保存的是最后一次循环索引，因此最终检查点为 `model_7003.pt`。独立计数确认实际执行了 5 次更新；环境计数从 168048 增至 168168。

真实 runner 保存后，重新构造全新的模型、storage 和 Adam，通过 `strict=True` 重载；模型和 optimizer 状态逐项精确一致。将环境总计数先设为 0 再加载，确认确实恢复为 168168；重置 LSTM 后，对同一观测序列比较四步动作、隐状态和 value，全部精确一致。

RSL-RL 会恢复 Adam param-group 的学习率，但不会自动同步 PPO 的 `learning_rate` 属性。因此新 runner 使用 checkpoint 中保存的学习率构建，再严格加载。本次重载后两者均为 `1e-05`，避免重启调度时静默跳回 `1e-3`。

检查点只保存 `common_step_counter` 等已列明信息，不保存逐环境辅助课程级别、模拟器状态、episode 状态、RNG 或 rollout LSTM 状态。因此本阶段不宣称逐轨迹无缝续跑。

## 7. 数值异常与回归

验收工具检查实际 rollout、return、advantage、loss、梯度、模型参数和 Adam moments；Adam moment 形状与参数对应，并累计 100 次 mini-batch 更新。额外检查 `nan_state` 终止次数为 0，避免环境自动重置为有限观测而掩盖 NaN。

开发侧完整 CPU 回归为 253 passed、2 skipped；之后补入 autoreset 掩盖 NaN 的测试，Stage 06 定向测试 10 passed。本机已验收运行对应的完整回归为前述 254 passed、2 skipped。其后仅添加标准库收尾工具、文档模板及 7 项归档测试，开发侧该 7 项测试全部通过，已验收的训练实现未改变。收尾工具不再次执行测试或训练。

## 8. 云端规模预检与 ONNX 副产物

4096 环境的配置规模预检通过：每 iteration 98304 条 transition、循环 mini-batch 可整分、物理和控制周期保持冻结值。已知 rollout 与 LSTM 缓冲区合计 `277315584 bytes`，约 264.5 MiB。

该数值不包含模拟器、接触缓存、梯度激活、CUDA/Warp 额外分配。没有在云 GPU 上构建 4096 环境，没有完成实际云端容量或吞吐率测量，没有提交云训练作业。

真实 runner 保存时自动生成的 ONNX 副产物如下：

- `/home/lx/microduck-double-balance/artifacts/double-balance-stage06/smoke-20260921-125127/logs/logs.onnx`

这些文件未做部署验收，不作为已学会双重平衡的策略发布。

## 9. 复现入口与故障处理

代码入口为 `scripts/validate_double_balance_smoke.py`，默认 `--mode load` 只构建和加载；`--mode smoke` 才运行固定 64 环境、5 次更新。

正常接续直接使用已保存的验收记录，不重复训练。仅在明确需要复现或排错时使用：

```bash
cd /home/lx/microduck-double-balance/workspace
uv run --locked python scripts/validate_double_balance_smoke.py \
  /home/lx/microduck-double-balance/artifacts/double-balance-stage05/checkpoint.pt \
  --mode smoke \
  --output /home/lx/microduck-double-balance/artifacts/double-balance-stage06/reproduction-new-run
```

输出目录必须是仓库外的新目录；不覆盖已有运行。不带任何 `MICRODUCK_*` 配置覆盖变量。失败时查看本次 `failure.json` 和控制台日志，保留失败记录。SHA 不一致时停止，不加 `--force` 或修改验收预期绕过。

## 10. 固定本地提交、下载与 SSH 推送规范

后续所有阶段继续保留并遵守：

```text
docs/handoffs/microduck-local-ssh-push-protocol.md
LocalRepository=/home/lx/microduck-double-balance/workspace
DownloadDirectory=/home/lx/下载
OriginSSH=git@github.com:liuxue-lab/microduck-double-balance.git
Branch=double-balance
Stage06Bundle=microduck-stage-06-from-stage-05.bundle
```

网页工作区的增量 bundle 下载到 `/home/lx/下载` 后，fetch 到临时远端跟踪引用再 fast-forward 合并，不能直接 fetch 覆盖当前检出的分支。本阶段收尾版本的 bundle 沿用同一文件名，需要以最新文件替换旧下载。

```bash
cd /home/lx/microduck-double-balance/workspace
git bundle verify /home/lx/下载/microduck-stage-06-from-stage-05.bundle
git fetch /home/lx/下载/microduck-stage-06-from-stage-05.bundle \
  refs/heads/double-balance:refs/remotes/stage06/double-balance
git switch double-balance
git merge --ff-only refs/remotes/stage06/double-balance
```

本阶段最终文档和真实验收 JSON 在本机归档并提交，本机创建 `stage-06-complete` 注释标签，再按既有方式推送：

```bash
git push origin double-balance
git push origin stage-06-complete
```

正常分支更新与 `[new tag]` 输出即接受推送成功，不重复要求完整远端核验。仅在 push 报错、non-fast-forward、标签冲突、SHA 不一致、导入结果不确定或用户明确要求时核验一次。

`Everything up-to-date` 但本地仍旧、标签不存在、HTTPS 密码提示、SSH key 未加载、口令缓存、non-fast-forward、标签冲突和下载路径错误，继续按长期规范处理。不切换 HTTPS，不索取 PAT，不 force push，不覆盖阶段标签，不将模型文件提交到 Git。

## 11. 下一阶段与接续提示

Stage 07：首次带辅助课程的云 GPU 正式训练。本阶段到此停止。

Stage 07 首先核对当前阶段标签和归档证据，再明确云 GPU、预算、训练规模、辅助课程、初始化检查点与进度计数的保留/重置规则。4096 环境的真实云端容量仍需在所选设备上预检。Stage 05 迁移检查点是可重复的初始化基线，不能未经说明就把 Stage 06 smoke 制品当成正式训练起点。

```text
请完整读取 docs/handoffs/microduck-stage-06-runner-smoke-handoff.md 和归档验收 JSON，先核验 stage-06-complete 标签及工作树，接续 Stage 07。

Stage 06 已通过真实 runner 严格加载、64 环境 5 iteration PPO smoke、检查点保存重载和配置规模预检。本机最终回归为 254 passed、2 skipped。不要重复 Stage 06 训练，也不要把 smoke 结果写成正式成功率。

正式训练使用云 GPU，本机用于代码和仿真。先规划辅助课程、检查点初始化、计数规则和预算，并完成所选云 GPU 的实际容量预检。继续保留 docs/handoffs/microduck-local-ssh-push-protocol.md；提交、阶段标签和 SSH 推送按既有本机流程完成，正常推送成功后不重复完整远端核验。
```

交接结束。
