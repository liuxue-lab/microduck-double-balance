# Stage 08 运行入口与云端操作交接

更新：2026-09-29（北京时间）

状态：运行入口已实现，原 43 项 CPU 定向测试通过。用户已从本机连接新实例，但首轮只读预检因 `python3: command not found` 退出；尚未完成部署、执行 CUDA 容量验证或正式训练。兼容修复另通过 11 项部署/预算定向回归，其中 3 项为新增。实机修复效果待重试。本文件不是 Stage 08 最终交接。

完整依据见 `docs/audits/stage-08-execution-plan.md`。固定下载与 SSH 推送规范继续沿用 `docs/handoffs/microduck-local-ssh-push-protocol.md`。

## 1. 本轮实现

| 入口 | 用途 |
|---|---|
| `scripts/deploy_stage08_local.sh` | 本机只读预检、传送已提交源码及两个已知检查点、启动后台环境准备 |
| `scripts/setup_stage08_autodl.sh` | 重用锁定依赖、建立或重用累计预算、运行 CPU 检查；不启动 PPO |
| `python -m mjlab_microduck.double_balance_stage08 capacity` | 只针对新 5090 的 4096 环境验证，D/E 各 3 次预热＋12 次测量 |
| `scripts/run_stage08_batch.py` | 串行执行首轮、三种子复核或条件扩训；遇到暂停/失败停止批次 |
| `python -m mjlab_microduck.double_balance_stage08_review` | 用开发评估生成带证据哈希的扩训决定 |
| `scripts/freeze_stage08_selection.py` | 仅按 N/R-dev 固定每种子的候选，加入第 1000/6000 次参照 |
| `scripts/evaluate_stage08_batch.py` | 固定模型表的 N、R-dev、三个 R-test 初态集批量评估与配对比较 |
| `scripts/render_stage08_local.py` | 本机推理/视频，固定 env0、稳定时长中位、最差速度及成功回合 |
| `scripts/archive_stage08_results.py` | 归档必要模型和证据，回传后逐文件核验并提醒关机 |

Stage 03/04、依赖文件及复用的 Stage 07 训练/恢复实现未改写。B–E 是独立的 Stage 08 训练覆盖。没有增加下球速度奖励，没有放宽严格标准。

## 2. 恢复与计数

所有新分支来源是第 1000 次，SHA-256 为 `c667f96607b68383047f23956ba58805920434465245ef8e32d1148b17fc65b7`。第 6000 次 SHA 为 `a5ad0aedb500555b649c5d4b11aded833cbc0f932c1fff1a747e1492534c495c`，仅用于评估。

- 恢复 actor/critic、归一化统计、Adam 一二阶矩和 step；源学习率为 `2.2500000000000008e-5`，之后续训恢复各自实际 adaptive LR。
- 完成 Stage 08 新增 k 次后，实验计数为 k，累计计数为 1000+k，下一迭代为 1000+k，保存的末次迭代为 999+k；common counter 为 `(1000+k)×24`，sim counter 再乘 10，Adam step 为 `(1000+k)×20`。
- 新 episode 的年龄与 LSTM 隐状态清零，重新初始化 RNG，不声称逐轨迹恢复。
- reset 会计算课程。A–D 在 reset 后恢复源逐环境 hold/level；E 在 reset 后设 level=5、hold=0，并重新计算/覆盖篮球和鸭身两套缓存及实际施加的外力/力矩。
- 原全局课程使用 `>`，在之后发生 reset 的课程计算中生效；不会改成 `>=`。零命令组每个控制步检查实际命令；E 同时检查缓存和仿真外力，运行中每次更新记录 live manager 状态。
- 恢复不能跨实验组、种子或预算 campaign；容量验证模型标记为 `capacity`，正式恢复入口拒绝使用它们。

## 3. 本机先导入代码

下载最新版 `microduck-stage-08-from-stage-07.bundle` 到 `/home/lx/下载`。此前已导入准备提交 `c7f7c4c` 的本机同样可以快进；本包包含从 Stage 07 开始的全部 Stage 08 提交。工作区有未提交修改时先处理差异，不重置。

```bash
cd /home/lx/microduck-double-balance/workspace
git status --short
git bundle verify /home/lx/下载/microduck-stage-08-from-stage-07.bundle
git fetch /home/lx/下载/microduck-stage-08-from-stage-07.bundle \
  refs/heads/double-balance:refs/remotes/stage08/double-balance
git switch double-balance
git merge --ff-only refs/remotes/stage08/double-balance
```

现在不创建 `stage-08-complete` 标签。无需为部署先向 GitHub 推送，云端接收本机已提交代码的完整 bundle。

## 4. 选定云机后部署

优先标准 RTX 5090 32GB，60 小时；A800 80GB，24 小时为另一条路线。用户已提供新实例 `connect.bjb2.seetacloud.com:45743`。实际 GPU 身份尚待预检确认，不沿用 Stage 07 的旧地址。

用户已明确无需手动记录时间：本机执行下列命令即可，首次修复版部署自动生成时间起点并保存在仓库外。相同目标重试复用该起点，云端已有预算也不归零。部署前的真实开机时间未知，不把自动起点称为平台开机时间。

```bash
cd /home/lx/microduck-double-balance/workspace
bash scripts/deploy_stage08_local.sh connect.bjb2.seetacloud.com 45743 5090
```

脚本先读取云机硬件和已有文件；拒绝型号不符、已有 GPU 计算进程、脏工作区、非快进或检查点哈希冲突。源码在隔离的 incoming 目录传送；已有 Stage 07 模型可在云盘复制复用，目标已存在且哈希正确时不重复上传。没有密码或私钥写入配置。

预检之前先通过纯 Bash 查找 Python >=3.10，检查 PATH、系统和常见 Conda 路径，使用选中的绝对路径贯穿预检、模型处理及预算初始化，不依赖交互式 Conda 激活。查找不安装软件、不导入 CUDA；若所有候选均不可用，退出并输出 `Stage08Bootstrap=NO_USABLE_PYTHON`，此时先检查镜像实际环境，不绕过预检。正式项目仍按锁文件安装 Python 3.12.14。

环境准备在 `microduck-stage08-setup` tmux 会话中运行，SSH 断开不会中断准备。日志位于：

```text
/root/autodl-tmp/microduck-double-balance/artifacts/double-balance-stage08/setup/latest.log
```

准备成功只说明源码、依赖与 CPU 检查通过。5090 的真实仿真/PPO 路径仍须下一步验证。A800 不重跑 Stage 06 smoke 或容量扫描。

## 5. 预算与云端容量

预算文件固定为 `artifacts/double-balance-stage08/budget.json`。计时起点为自动保留的部署开始时间。新实验、进程重启和安装重试都重用它；起点后的空闲以及离线间隔也保守计入，不通过重建文件归零。

父进程监督器与训练进程分别检查预算。A800 在累计 20h、5090 在累计 52h 停止训练工作，分别留 4h/8h 收尾。训练在更新边界保存；监督器会对挂住的进程先发 SIGTERM，再在宽限期后终止进程组，此时只能保留之前已验证的检查点。总额度前保留退出裕量，绝不保证被强杀的未完成更新能恢复。

在云端进入项目环境，建议用独立 tmux 会话执行后续长任务：

```bash
cd /root/autodl-tmp/microduck-double-balance/workspace
export MUJOCO_GL=egl
stage08_artifacts=/root/autodl-tmp/microduck-double-balance/artifacts/double-balance-stage08
```

5090 路线：

```bash
.venv/bin/python -m mjlab_microduck.double_balance_stage08 capacity \
  --gpu 5090 --ledger "$stage08_artifacts/budget.json" \
  --checkpoint "$stage08_artifacts/references/update_001000.pt" \
  --datasets "$stage08_artifacts/initial-states" \
  --output "$stage08_artifacts/capacity-5090"
stage08_capacity="$stage08_artifacts/capacity-5090/capacity-summary.json"
```

两个路径均须通过数值、保存重载和至少 15% 实测总显存余量检查。容量测试也覆盖训练环境仍驻留时的 N/R-dev 子进程，避免正式评估时额外占用显存才暴露 OOM。单个路径最多 15 分钟，加有限退出宽限；两个路径均从第 1000 次重新开始，优化结果弃用。

A800 路线直接使用仓库已归档证据：

```bash
stage08_capacity="$PWD/docs/audits/stage-07-cloud-capacity-summary.json"
```

若 5090 未通过，不自动降成 2048、不使用容量模型续训；先检查失败原因，再决定 A800 回退或另行讨论统一批量调整。不能将两条路线的机时自动相加。

## 6. 首轮五组

下面示例为 5090；A800 把 `--gpu` 改为 `A800`，使用上述已有容量文件。

```bash
.venv/bin/python scripts/run_stage08_batch.py screening \
  --gpu 5090 --ledger "$stage08_artifacts/budget.json" \
  --source "$stage08_artifacts/references/update_001000.pt" \
  --capacity-summary "$stage08_capacity" \
  --datasets "$stage08_artifacts/initial-states" \
  --output "$stage08_artifacts/screening"
```

A–E 各新增 500 次，共 2500 次；种子为 20260928。每 100 次保存，并额外保存 250 次；在 0/250/500 运行 N/R-dev。先评估保存/加载路径才运行第一轮 PPO。每个段的 `training.json`、`iterations.jsonl`、`gpu-samples.jsonl` 与 `watchdog.json` 保留实际状态。

暂停后用同一命令增加 `--resume-batch`，会从相应段最后已验证检查点继续；有异常先看日志处理原因，不自动循环重试。新段启动新的 episode/RNG/LSTM 状态，保留 Adam/LR、课程和辅助。

选择采用固定排序：R-dev 成功率 → N 成功率 → 存活时间 → 末尾稳定时长中位数 → 最长稳定时长中位数 → 较低下球越界率 → 较早检查点。同训练量比较必须另看各组 500 次终点，不能用各自最佳点代替因果对照。

## 7. 复核与条件扩训

超过 500 次，入口要求提供带原始报告哈希的 `decision.json`。生成器不启动训练：

```text
python -m mjlab_microduck.double_balance_stage08_review
  --kind replication
  --campaign-id <budget.json 中的 campaign_id>
  --profiles D E
  --baseline <源第 1000 次或实验第 0 次的 dev/evaluation.json>
  --reports <候选 250 次 dev 报告> <候选 500 次 dev 报告>
  --output <replication-decision.json>
```

门槛：最新点出现高于源模型的严格成功，且物理终止比例增加不超过 5 个百分点；或两个相邻点的末尾/最长稳定时长均提高，下球超限比例降低至少 25%，存活均不低于 9.5s。这些是预算筛选规则，不是统计显著性或新成功定义。

使用 `run_stage08_batch.py replication`，除首轮通用参数外传入 `--decision` 和 `--previous-batch "$stage08_artifacts/screening/batch.json"`，输出新目录 `replication`。两配置各三种子；种子 20260928 恢复自己 500 次终点并续至新增 1500，另两种子各从源训练新增 1500。核心总新增量最多 10500 次。首轮全部无改善则先分析，不强行生成复核决定。

5090 条件扩训要求同一配置三个种子在 1250/1500 点均通过改善门槛。用 `--kind extension`、一个 profile 和六份报告生成新决定；`run_stage08_batch.py extension` 从复核三条 1500 次终点续到最多各新增 4000，总量最多 18000。A800 入口拒绝该扩训路线。

复核/扩训出现连续两个评估点退化时保存暂停。后备的新奖励、弱辅助门控或优化器重置不在本实现中，仍须明确提出内容、理由和影响后由用户确认。

## 8. 固定候选与最终批量评估

`freeze_stage08_selection.py --profile <胜出组> --runs <相关运行根目录...> --source <第1000次> --stage07-final <第6000次> --output <final-selection.json>` 会按预定排序固定三个种子各自最佳点，并加入两个参照。只读 N/R-dev，不用 R-test 选优。

若首轮无改善而停止，可用 `--screening-only --stop-reason '具体原因'` 固定单个训练种子的最好候选，明确没有完成多种子复核，仍完成最终推理评估；不能把这种情况记为策略目标达成。

```bash
.venv/bin/python scripts/evaluate_stage08_batch.py \
  --gpu 5090 --ledger "$stage08_artifacts/budget.json" \
  --selection "$stage08_artifacts/final-selection.json" \
  --datasets "$stage08_artifacts/initial-states" \
  --output "$stage08_artifacts/final-evaluation"
```

名义 64 个确定性副本单独报告，不作为独立试验计算区间。R-dev 为 seed 8101、128 回合；R-test 为 8201/8202/8203，各 256 回合。实际初态张量及哈希保存一次，之后所有模型必须逐位复现同一初态；开发与留出集禁止重复。若换硬件引起初态差异，明确报错，不偷偷新建一批。

每模型的 768 回合分别给出成功数/总数、Wilson 区间和相对源模型的配对成功差。不同训练种子不直接混成一个独立样本池。逐回合报告包含速度分位数、独立/共现违例、终止原因、末尾及最长稳定时长、动作差分；`trace.pt` 保存指标计算时刻的完整轨迹。技术 PASS 与策略表现分开。

## 9. 回传、视频与关机

云端任务停止后，用 `archive_stage08_results.py pack --root <Stage08目录> --source <第1000次> --stage07-final <第6000次> --output <目录外的结果.tar.gz>` 生成归档与哈希回执。保留源、各段终点、开发最佳、最终候选、配置、逐步日志和初态；未选中的周期模型逐项列入 `MANIFEST.json`，继续留在云盘。若最终评估尚未完成，必须提供 `--partial-reason`，只生成部分归档，不伪造完工状态。

通过已有 SSH/scp 下载归档和回执到 `/home/lx/下载`，再在本机运行：

```text
.venv/bin/python scripts/archive_stage08_results.py verify
  --archive /home/lx/下载/<结果.tar.gz>
  --sha256 <云端回执中的 SHA-256>
  --output /home/lx/microduck-double-balance/artifacts/double-balance-stage08/<新归档目录>
```

检查归档总哈希、成员清单、每个文件哈希与解包后哈希；拒绝路径穿越、链接、重复成员和覆盖已有目录。校验成功后提醒用户关闭云机，不自动关机、不删除云盘。若仍有其他任务或必要文件尚未回传，先处理后再关机。

本机视频命令为 `render_stage08_local.py --checkpoint <本机模型> --reference-report <对应 N 或 R-dev 报告> --datasets <已回传初态目录> --output <新视频目录> --device cuda:0`。只运行推理，RTX 5060 不运行 PPO。视频保留完整 10 秒或物理终止前回合，叠加下球速度、阈值、顶球偏移与连续稳定时间；视频生成不等于已经目视检查。

完成失败分析、视频检查、归档与结果汇总后，再生成最终阶段交接和 final-acceptance。若严格目标仍未达到，明确记 `PolicyObjective=UNMET`。最终注释标签在本机创建，分支与标签沿用 SSH 推送；正常推送成功后不再完整远端核验。
