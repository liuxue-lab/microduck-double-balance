# Stage 06：真实 runner 加载与有限 PPO smoke 验收实现

日期：2026-09-21

状态：代码与 CPU 预检已完成；用户本机 CUDA 验收结果待回收。本文不是 Stage 06 完成交接，当前不创建 `stage-06-complete` 标签。

## 1. 已核验基线

- 分支：`double-balance`。
- 本阶段起点：`55e41f8a5d4a791af5625f45366191a68eaa015b`。
- `stage-05-complete` 为注释标签，指向 `c5b0aee66de10aa0c8aea0bdc85f1a917fada1df`。
- 用户本机工作树：`CLEAN`。
- 本机迁移检查点最初缺失；使用现有 Stage 05 工具恢复后，用户返回 `Stage06CheckpointRecovery=PASS`。
- 迁移文件 SHA-256：`548758afc546e11d8f3f446a4bcc846283a669ff20f4048da7a3a7bb1332b98e`。

## 2. 实现边界

入口：`scripts/validate_double_balance_smoke.py`。

实现：`src/mjlab_microduck/double_balance_smoke.py`。

测试：`tests/test_double_balance_smoke.py`。

使用注册任务 `Mjlab-DoubleBalance-MicroDuck` 和真实 `MicroduckOnPolicyRunner`，不修改 Stage 03 物理、Stage 04 观测/奖励/终止/成功判据。环境数量固定 64，PPO 更新固定 5 次，每次每环境采集 24 步；入口没有扩大训练时长的参数。

默认 `--mode load` 只构建和加载。`--mode smoke` 在严格加载检查全部通过后自动运行短程 PPO。新输出目录必须在仓库外，且事先不存在，避免覆盖旧制品。运行前拒绝 `MICRODUCK_*` 环境变量覆盖，确保验收使用冻结的默认任务。日志使用本地 TensorBoard，关闭模型上传。

## 3. 运行时验收

1. 输入检查点 SHA-256 和迁移 schema。
2. 真实训练环境和注册 runner 构建；观测 `64×61` / `64×85`，动作 `64×14`。
3. Actor、critic、Adam 严格加载，并与检查点逐项精确比较。
4. 恢复 iteration `6999`、`common_step_counter=168048`、空 Adam moments、学习率 `1e-3`。
5. 五轮真实 rollout → return → update。逐步检查观测、动作、奖励，逐轮检查存储、return、advantage、loss、梯度、模型和 optimizer。
6. 额外检查 `nan_state` 终止次数为 0，避免 autoreset 返回有限观测而掩盖物理异常。
7. 验证 120 次 vector step、7680 条 transition、5 次 return / PPO update；当前配置下 Adam 共执行 100 次 mini-batch 更新。
8. 保存后记录检查点路径、大小和 SHA-256；确认 Actor 新六列和 critic 新九列已有非零权重。
9. 构造全新的注册 runner，严格重载模型、Adam 和计数，比较四步循环推理的动作、LSTM 隐状态及 critic value。
10. 确认原迁移输入的 SHA-256 未改变。

## 4. 两个恢复语义

### 4.1 iteration 是最后一次循环索引

当前 `rsl-rl-lib==5.0.1` 的 `learn(5)` 从已恢复的 6999 开始，循环索引依次为 6999、7000、7001、7002、7003。预期最终检查点是 `model_7003.pt`，环境计数是 `168168`。实际更新数通过独立计数核验为 5，不能从检查点编号差值误判为 4。

### 4.2 学习率与环境进度的恢复范围

RSL-RL 的 `load` 恢复 Adam param-group 的学习率，却不更新 PPO 对象上的 `learning_rate`。在 smoke 后构建新 runner 时，验收工具从保存的 Adam 状态读取学习率并用于构建 PPO，随后严格加载；这样两个值保持一致，不会静默跳回初始 `1e-3`。

源检查点的环境状态只有 `common_step_counter`。它不包含逐环境篮球辅助课程级别、物理状态、episode 进度或 RNG 状态；本阶段不宣称无缝复原原 rollout。任务继承的辅助课程和物理参数均保持原样。重载比较从重置 LSTM 隐状态后的同一观测序列开始。

## 5. 云端规模配置预检

单独检查 4096 环境配置、循环 mini-batch 可分性、24 步 rollout、2 ms 仿真步长和 20 ms 控制周期，不分配云端模拟器，不启动长期训练。

4096 环境、24 步下，已知 rollout 张量和逐步 LSTM h/c 缓冲区合计 `277315584 bytes`，约 264.5 MiB。该数值不含模拟器、接触缓存、训练激活、CUDA/Warp 额外分配，不能当作总显存需求。实际云 GPU 的容量状态明确记录为 `NOT_MEASURED_ON_CLOUD_GPU`。

## 6. CPU 预检证据

在隔离 CPU 验证环境中使用 Python 3.12.14、Torch 2.9.1+cpu、mjlab 1.3.0、RSL-RL 5.0.1、Warp 1.12.0、MuJoCo 3.10.0 和锁文件对应的主要依赖。这里的 CPU wheel 不替代用户本机 CUDA 环境。

- 独立下载固定发布源检查点并核验 SHA-256，通过现有迁移脚本再次得到完全相同的迁移文件 SHA-256。
- 真实注册任务、单环境 CPU runner 严格加载 Actor、critic 和 Adam：PASS。
- 61/85/14 维度、6999/168048 进度、空 Adam 和学习率恢复：PASS。
- 4096 环境配置预检：PASS。
- 全量 CPU 回归：253 passed、2 skipped、3 个既有 actuator 正则提示。
- 在上述全量运行后，补入 autoreset 掩盖 NaN 的测试；Stage 06 定向运行结果：10 passed。
- CPU 真实环境预检没有执行 PPO；64 环境 CUDA smoke 尚待用户本机运行。

## 7. 本机命令

只加载：

```bash
cd /home/lx/microduck-double-balance/workspace
uv run --locked python scripts/validate_double_balance_smoke.py \
  /home/lx/microduck-double-balance/artifacts/double-balance-stage05/checkpoint.pt \
  --mode load \
  --output /home/lx/microduck-double-balance/artifacts/double-balance-stage06-load
```

完整短程验收：

```bash
cd /home/lx/microduck-double-balance/workspace
uv run --locked pytest -q tests
uv run --locked python scripts/validate_double_balance_smoke.py \
  /home/lx/microduck-double-balance/artifacts/double-balance-stage05/checkpoint.pt \
  --mode smoke \
  --output /home/lx/microduck-double-balance/artifacts/double-balance-stage06-smoke
```

重试时使用新的输出目录，不删除旧记录。`acceptance.json` 保存完整结果，`load-report.json` 保存训练前的加载结果；异常写入 `failure.json`。注册 runner 保存 checkpoint 时可能自动产生 ONNX 副产物，其路径被记录，但未经过部署验收，不作为双重平衡发布策略。

验收输出为 `Stage06Acceptance=PASS` 且本机回归通过后，再补齐最终交接文档、提交、注释标签和本机 SSH 推送。任何 5 iteration 日志和奖励值都不能写成正式训练性能或双重平衡成功率。

## 8. 固定下载与 SSH 推送规范

长期沿用 `docs/handoffs/microduck-local-ssh-push-protocol.md`：

```text
LocalRepository=/home/lx/microduck-double-balance/workspace
DownloadDirectory=/home/lx/下载
OriginSSH=git@github.com:liuxue-lab/microduck-double-balance.git
Branch=double-balance
Bundle=microduck-stage-06-from-stage-05.bundle
```

网页工作区提供增量 bundle，本机 fetch 到 `refs/remotes/stage06/double-balance` 后 fast-forward 合并。实际 GPU 验收完成前不创建完成标签。最终在本机创建注释标签 `stage-06-complete`，沿用 `git push origin double-balance` 和 `git push origin stage-06-complete`。正常推送成功后不重复远端全套核验；常见故障处理继续参照长期规范。
