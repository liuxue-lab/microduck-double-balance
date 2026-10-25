# Stage 04 实现与 Stage 05 计划审计

日期：2026-09-20

审计基线：`stage-04-complete`，提交
`ee3f1dd71f00479c74c00644cea2028396e408bc`

## 1. 审计范围

本审计只检查以下内容：

- Stage 04 的双重平衡任务实现是否满足已冻结的 61 维 Actor、85 维 critic、
  14 维动作、独立任务和无辅助评估契约；
- Stage 05 计划是否能把固定发布检查点安全迁移为可由新任务严格加载的状态；
- 是否存在会破坏继承策略数值连续性或后续训练可用性的隐含状态。

没有运行 PPO，没有修改 Stage 03 物理参数，也没有改变 Stage 04 的观测顺序、
尺度、奖励、终止或成功判据。

## 2. Stage 04 实现结论

Stage 04 实现审计通过：

- 新任务 ID 独立于原篮球任务；
- Actor 仍为 61 维，上球 6 维原位占用索引 55–60；
- critic 在原 76 维末尾追加 9 维，合计 85 维；
- 动作为原 14 维；
- play 配置关闭下球 hold、随机推力、域随机化和观测噪声；
- 原篮球配置和 Stage 03 物理回归由既有测试覆盖。

未发现需要回改 Stage 04 任务定义或物理实现的缺陷。

## 3. 固定发布检查点事实

固定源文件：

```text
artifacts/basketball-release-6d8f74b/checkpoint.pt
```

```text
SizeBytes=9965749
SHA256=57a322eff092cb71e7cba831791232f0cc4fd45ede0441cad6d87813b0033e41
Iteration=6999
ActorInputWeightShape=1024x61
CriticInputWeightShape=512x76
ActorNormalizerShape=1x61
CriticNormalizerShape=1x76
OptimizerStateEntries=21
OptimizerLearningRate=2e-5
```

Actor 索引 55–60 的 normalizer 均值、方差和标准差全部为 0，因为发布策略在这
六列只接收恒零填充。对应 LSTM 输入权重不是 0，而是未被有效训练的初始化值。

## 4. 原 Stage 05 计划缺陷

### 4.1 新语义接入随机旧权重

仅保持 61 维形状并验证“六维输入为零时动作一致”还不够。非零上球状态会经过
原先没有接受有效数据的随机 LSTM 输入列，从迁移第一步起任意扰动篮球策略。

修正：将 LSTM 输入权重第 55–60 列显式置零。这样继承策略初始时忽略新信号，
后续训练再从零学习其作用。

### 4.2 新输入会被异常放大

原六列 normalizer 的标准差为 0。RSL-RL 前向计算使用
`(x - mean) / (std + 0.01)`，直接接入新状态会产生约 100 倍额外放大。

修正：将这六列重置为均值 0、方差 1、标准差 1。零输入的归一化结果仍严格为
零，因此不破坏零上球状态下的新旧动作一致性。

### 4.3 optimizer 状态不适合直接续用

原计划未明确处理 optimizer。固定检查点包含 21 项旧 Adam moments，critic
第一层仍是 512×76；同时保存的学习率已从配置值衰减到 `2e-5`。直接加载会让
变宽 critic 的 optimizer 状态形状不一致，并把新任务绑定到旧任务末期学习率。

修正：显式清空 Adam moments，保留参数分组，将学习率恢复为当前配置值
`1e-3`。critic 权重本身不丢弃：旧 76 列逐元素保留，新增 9 列为 0。

## 5. 修正后的 Stage 05 契约

```text
ActorDimension=61
ActorInheritedColumns=0..54 unchanged
ActorTopBallColumns=55..60 normalizer neutral, LSTM weights zero
CriticDimension=76 -> 85
CriticInheritedColumns=0..75 unchanged
CriticAppendedColumns=76..84 normalizer neutral, weights zero
OptimizerMoments=reset
OptimizerLearningRate=1e-3
Iteration=preserved
EnvironmentProgress=preserved
PPOExecuted=false
```

必须验证：

1. 零上球输入序列下 Actor 每一步动作误差为 0；
2. 同一序列下 LSTM 隐状态误差为 0；
3. 迁移初始时非零上球输入不会通过随机旧权重改变动作；
4. 任意新增 9 维 critic 输入下，新旧 value 一致；
5. Actor、critic 和重建 optimizer 均能严格加载；
6. 源检查点 SHA-256 在迁移前后不变；
7. 不构建、不运行 PPO。

## 6. 审计结论

```text
Stage04ImplementationAudit=PASS
Stage05OriginalPlanAudit=FAIL
Stage05PlanCorrected=PASS
NoPpoTraining=PASS
```
