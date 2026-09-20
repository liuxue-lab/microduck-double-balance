# Microduck 双重平衡项目：Stage 05 检查点迁移交接文档

日期：2026-09-20

分支：`double-balance`

阶段标签：`stage-05-complete`

本文件供下一次对话开始 Stage 06 使用。Stage 05 只完成固定原篮球检查点的
schema 审计、确定性迁移、数值一致性验证和完整回归测试；没有构建或运行 PPO，
没有进行训练 smoke，没有得到双重平衡成功率，也没有导出 ONNX。

## 1. 阶段结论

```text
Stage04ImplementationAudit=PASS
Stage05PlanCorrected=PASS
CheckpointLocatedAndHashed=PASS
StateDictSchemaAudited=PASS
ActorInputStill61D=PASS
ActorNormalizerMigration=PASS
ActorLstmNewInputColumnsZeroed=PASS
CriticInputMigrated76To85=PASS
CriticNormalizerMigration=PASS
OptimizerRebuilt=PASS
ZeroTopBallActorActionParity=PASS
LstmHiddenStateParity=PASS
ActorNewInputIsolation=PASS
CriticValueParity=PASS
StrictActorLoad=PASS
StrictCriticLoad=PASS
RebuiltOptimizerLoad=PASS
OriginalCheckpointUnmodified=PASS
DoubleBalanceTaskDefinitionRegression=PASS
DoubleBalancePhysicsRegression=PASS
OriginalBasketballRegression=PASS
FullCpuTestSuite=PASS
NoPpoTraining=PASS
```

## 2. 接手基线与提交

Stage 05 从以下已发布基线开始：

```text
Stage04Tag=stage-04-complete
Stage04Commit=ee3f1dd71f00479c74c00644cea2028396e408bc
```

审计计划修正提交：

```text
AuditFixCommit=395664b
AuditFixMessage=docs: correct Stage 05 checkpoint migration plan
```

迁移实现提交：

```text
MigrationCommit=4124e5d72f045bf9edb2a2471da6b52cca76b3d8
MigrationMessage=feat: migrate double-balance checkpoint
```

审计报告：

```text
docs/audits/stage-04-implementation-and-stage-05-plan-audit.md
```

## 3. 前置审计结论

Stage 04 的物理和任务定义实现通过审计，没有回改以下冻结内容：

- Actor 61 维顺序与尺度；
- critic 85 维顺序与尺度；
- 14 维动作；
- 托片和上球物理参数；
- 奖励、终止、成功判据和评估指标；
- 原篮球任务兼容性。

原 Stage 05 计划发现三个缺陷并已修正：

1. Actor 索引 55–60 原来恒接收零，但相应 LSTM 输入权重仍是随机初始化值。
   若直接复用，非零上球状态会任意扰动旧策略。
2. 这六列原 normalizer 的方差和标准差均为 0。直接接入新状态时，RSL-RL
   的 `std + 0.01` 会造成约 100 倍额外放大。
3. 原 Adam 含 21 项旧任务 moments，学习率已衰减到 `2e-5`；critic 变宽后
   不能直接续用这些状态，也不应把末期学习率静默带到新任务。

## 4. 固定源检查点

用户本机固定位置：

```text
/home/lx/microduck-double-balance/artifacts/basketball-release-6d8f74b/checkpoint.pt
```

来源：

```text
HuggingFaceRepo=HannesVonEssen/microduck-basketball
Revision=6d8f74b97c75b1597efead1754ff54ca2af4900c
```

完整性：

```text
SourceSizeBytes=9965749
SourceSHA256=57a322eff092cb71e7cba831791232f0cc4fd45ede0441cad6d87813b0033e41
SourceIteration=6999
SourceCommonStepCounter=168048
```

检查点包含顶层键：

```text
actor_state_dict
critic_state_dict
optimizer_state_dict
iter
infos
```

## 5. 源 schema 审计

### 5.1 Actor

```text
StateDictKeys=17
NormalizerMeanShape=1x61
NormalizerVarShape=1x61
NormalizerStdShape=1x61
LstmInputWeightShape=1024x61
LstmHiddenWeightShape=1024x256
LstmLayers=1
LstmHiddenDimension=256
MlpHiddenDimensions=512,256,128
ActionDimension=14
```

Actor 索引 55–60 的原 normalizer 统计为：

```text
mean=0
variance=0
std=0
```

这些列对应发布 blind basketball 策略中的恒零 `body_command` 填充。

### 5.2 critic

```text
StateDictKeys=12
NormalizerMeanShape=1x76
NormalizerVarShape=1x76
NormalizerStdShape=1x76
FirstLayerWeightShape=512x76
MlpHiddenDimensions=512,256,128
OutputDimension=1
```

### 5.3 optimizer

```text
Type=Adam
ParamGroups=1
StateEntries=21
StoredLearningRate=2e-5
```

## 6. 迁移规则

### 6.1 Actor 仍为 61 维

旧索引 0–54 的全部参数和 normalizer 统计逐元素保留。

新语义索引 55–60 采用：

```text
NormalizerMean=0
NormalizerVariance=1
NormalizerStd=1
LstmInputWeights=0
```

这不是把 Actor 扩展为 `61+k`。Actor 形状仍为 `1024×61`，只对原恒零槽做
语义迁移。零上球输入下，原归一化输出和迁移后归一化输出都严格为零，因此旧
Actor 动作与 LSTM 状态能够逐步完全一致。

将对应 LSTM 列置零还有一个更强的性质：迁移初始时，任意非零上球输入不会通过
未训练的随机旧权重改变篮球策略。后续训练再从零学习这六维的控制作用。

### 6.2 critic 从 76 扩展到 85 维

critic 原索引 0–75 的权重和 normalizer 统计逐元素保留。在末尾追加 9 列：

```text
AppendedIndices=76..84
NormalizerMean=0
NormalizerVariance=1
NormalizerStd=1
FirstLayerWeights=0
```

因此迁移初始时，无论新增 9 维取何值，critic 第一层都忽略它们，旧 76 维相同
时 value 输出保持一致。后续训练可以从零学习新的特权状态权重。

### 6.3 optimizer 显式重建

迁移保留原参数分组和参数 ID 顺序，但执行：

```text
AdamMoments={}
LearningRate=1e-3
```

这会移除 21 项旧任务 moments，避免 critic 形状不一致和旧任务梯度历史污染。
学习率恢复为 `MicroduckDoubleBalanceRlCfg` 当前配置值。

### 6.4 保留的进度信息

```text
Iteration=6999
CommonStepCounter=168048
```

两者用于保留来源和既有环境课程进度，不等于 Stage 05 进行了训练。迁移元数据
额外写入 `infos.double_balance_checkpoint_migration`，其中固定记录
`ppo_executed=false`。

## 7. 迁移后制品

推荐用户本机位置：

```text
/home/lx/microduck-double-balance/artifacts/double-balance-stage05/checkpoint.pt
```

本次确定性迁移结果：

```text
MigratedSizeBytes=3342581
MigratedSHA256=548758afc546e11d8f3f446a4bcc846283a669ff20f4048da7a3a7bb1332b98e
ActorInputWeightShape=1024x61
ActorNormalizerShape=1x61
CriticInputWeightShape=512x85
CriticNormalizerShape=1x85
OptimizerStateEntries=0
OptimizerLearningRate=1e-3
```

模型文件不进入 Git。Git 只保存迁移工具、验证工具、测试、审计报告和本交接文档。

## 8. 数值一致性结果

固定种子 `20260920`，生成 16 步合成 Actor 序列，并在第 8 步同时重置两个
LSTM。Actor 索引 55–60 设为零。

```text
ActorZeroTopBallActionMaxAbsError=0.0
ActorLstmHiddenMaxAbsError=0.0
```

对迁移后 Actor 使用相同前 55 维、不同的任意新六维输入：

```text
ActorNewInputIsolationMaxAbsError=0.0
```

对 32 组随机旧 critic 观测和任意新增 9 维观测：

```text
CriticValueMaxAbsError=0.0
```

严格加载：

```text
StrictActorLoad=PASS
StrictCriticLoad=PASS
RebuiltOptimizerLoad=PASS
SourceUnmodified=PASS
```

这里的“Actor 新输入隔离”只描述迁移初始状态。正式训练后这六列权重应当学习为
非零，否则策略无法利用上球状态。

## 9. 可重复命令

用户本机仓库结构中执行：

```bash
cd /home/lx/microduck-double-balance/workspace

uv sync --locked

uv run python scripts/migrate_double_balance_checkpoint.py \
  /home/lx/microduck-double-balance/artifacts/basketball-release-6d8f74b/checkpoint.pt \
  /home/lx/microduck-double-balance/artifacts/double-balance-stage05/checkpoint.pt

uv run python scripts/validate_double_balance_checkpoint_migration.py \
  /home/lx/microduck-double-balance/artifacts/basketball-release-6d8f74b/checkpoint.pt \
  /home/lx/microduck-double-balance/artifacts/double-balance-stage05/checkpoint.pt \
  --expected-migrated-sha256 \
  548758afc546e11d8f3f446a4bcc846283a669ff20f4048da7a3a7bb1332b98e
```

迁移脚本默认拒绝：

- 源文件 SHA-256 不匹配；
- 源文件与输出文件为同一路径；
- 输出文件已存在且没有显式传入 `--force`；
- Actor/critic/normalizer schema 与固定发布检查点不一致。

因此脚本不会静默覆盖源检查点。

## 10. 测试与回归结果

定向测试：

```text
Stage03To05TargetedTests=31 passed
```

全量 CPU 测试：

```text
FullCpuTestSuite=244 passed, 2 skipped
```

两个跳过项是仓库既有的平台条件测试。三个 warning 是 mjlab 同时在 joint 和
site 命名空间检查 actuator 正则产生的既有提示；执行器传动类型没有改变，托片
site 不会成为执行器目标。

Stage 04 任务定义回归：

```text
DoubleBalanceTaskDefinitionAcceptance=PASS
ActorDimension=61
CriticDimension=85
OneStepFinite=PASS
```

Stage 03 物理回归：

```text
DoubleBalancePhysicsAcceptance=PASS
MuJoCo2msMaxPenetration=0.8526506679 mm
MJWarp2msMaxPenetration=0.8503664285 mm
ContactSequence=tray -> terrain
OriginalBasketballActorDimension=61
OriginalBasketballFiniteSmoke=PASS
```

## 11. 变更文件

- `docs/audits/stage-04-implementation-and-stage-05-plan-audit.md`
  - 记录 Stage 04 实现审计和 Stage 05 原计划缺陷。
- `docs/handoffs/microduck-stage-04-task-definition-handoff.md`
  - 修正 Stage 05 迁移契约和验收项。
- `src/mjlab_microduck/double_balance_checkpoint.py`
  - 实现 schema 审计、Actor/critic/normalizer 迁移、optimizer 重建与 provenance。
- `src/mjlab_microduck/double_balance_checkpoint_validation.py`
  - 实现严格模型加载、循环状态、动作、value 和 optimizer 数值验证。
- `scripts/migrate_double_balance_checkpoint.py`
  - 固定哈希保护、原子写入和机器可读迁移报告。
- `scripts/validate_double_balance_checkpoint_migration.py`
  - 对源文件和迁移文件执行独立验收。
- `tests/test_double_balance_checkpoint_migration.py`
  - 锁定维度、非原地修改、迁移规则、数值一致性和非法参数拒绝。
- `docs/handoffs/microduck-stage-05-checkpoint-migration-handoff.md`
  - 本交接文档。

## 12. Stage 05 明确没有完成的内容

- 没有构建或运行 PPO。
- 没有运行 64 环境、5 iteration 训练 smoke。
- 没有用本机 RTX 5060 进行正式训练。
- 没有启动云 GPU 正式训练。
- 没有更新 Actor 新六列的训练后权重。
- 没有得到双重平衡成功率。
- 没有导出双重平衡 ONNX。
- 没有声称迁移后未训练策略能完成双重平衡。
- 没有把源或迁移后的模型文件提交到 Git。

动作、隐状态和 value 一致性只证明迁移在指定零耦合条件下保持旧函数，不代表
迁移后的策略已经学会控制上球。

## 13. Stage 06 唯一边界

下一阶段名称：

```text
Stage 06：迁移检查点加载与 64 环境、5 iteration PPO smoke
```

Stage 06 应完成：

1. 重新核验迁移检查点 SHA-256 为
   `548758afc546e11d8f3f446a4bcc846283a669ff20f4048da7a3a7bb1332b98e`。
2. 用真实 `Mjlab-DoubleBalance-MicroDuck` runner 严格加载 Actor、critic 和
   重建 optimizer。
3. 先只构建环境和 runner，确认观测 61/85、动作 14、迭代 6999 和环境进度
   恢复符合预期。
4. 运行 64 环境、5 iteration PPO smoke，验证 rollout、return、update、保存和
   重载链路无形状错误、NaN 或 optimizer 错误。
5. 保存 smoke 检查点并记录路径、大小、SHA-256、最终 iteration 和关键日志。
6. 做云端规模预检，但不启动正式长期训练。

Stage 06 只验证训练链路可运行，不能把 5 iteration 的结果写成正式训练性能，
也不能报告双重平衡正式成功率。首次带辅助课程训练属于 Stage 07。

## 14. 下一对话接续提示词

```text
请完整读取 microduck-stage-05-checkpoint-migration-handoff.md，先核验 stage-05-complete 标签、迁移检查点 SHA-256 和工作树状态，然后开始 Stage 06。

本阶段只使用迁移检查点完成真实 runner 严格加载、64 环境 5 iteration PPO smoke、保存与重载检查点及云端规模预检。不要启动正式长期训练，不要修改 Stage 03 物理参数或 Stage 04 任务定义，不要把 smoke 结果写成双重平衡成功率。

完成 Stage 06 全部验收、交接文档、提交、stage-06-complete 标签和远端推送后结束对话。
```

交接结束。
