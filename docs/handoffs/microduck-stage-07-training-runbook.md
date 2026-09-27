# Microduck Stage 07 正式训练操作与恢复

状态：2026-09-28 02:56 用户回传 TRAINING_COMPLETE、6000/6000 和 finite_checks=PASS，末次评估也已 PASS。全部 14 条名义评估的严格成功率均为 0%；跨段最佳候选是原段第 1000 次（稳定时间 90.18%），第 6000 次为 35.16%。训练段已完成，原始证据回传、诊断/视频、检查点归档和阶段交接仍待完成，不自动追加训练。证据见 `docs/audits/stage-07-training-progress.json` 及 `docs/audits/stage-07-update-6000-review.md`。
本文件是运行说明，不是 `stage-07-complete` 交接。

始终沿用 [本机提交与 SSH 推送规范](microduck-local-ssh-push-protocol.md)。
本机仓库 `/home/lx/microduck-double-balance/workspace`，下载目录 `/home/lx/下载`，
分支 `double-balance`。Stage 07 完成前不创建完成标签；推送在本机执行。

## 训练规模与来源

当前目标 SSH：`ssh -p 26497 root@connect.bjb1.seetacloud.com`。
使用单张 A800 80GB，4096 环境、每次 rollout 24 步，首段 6000 次 PPO 更新。
容量实测约 4.40 秒/更新，纯训练约 7.33 小时；暂留 8–10 小时包括启动和评估。
训练后期接触状态、课程和日志开销可能改变速度。

用户给定约 24 小时的规划范围、允许小幅浮动、效果优先；脚本没有 24 小时强制截止。
先完成 6000 次，根据完整曲线和评估决定是否续训到累计 10000 或 16000 次。
2000 次是中途评估节点，不作为原先建议的最终训练量。

原项目默认 6000 次；发布编号 6999 经多段续训得到，最后 b11 只续训 500 次。
作者曾因末次模型退化而选较早检查点，详见 `experiments/basketball/TRAINING.md`。
这些是开源实验记录，不能改称为论文中规定的固定训练轮数。

初始化固定为 Stage 05 迁移文件，SHA256：
`548758afc546e11d8f3f446a4bcc846283a669ff20f4048da7a3a7bb1332b98e`。
新训练严格加载权重和归一化器，使用迁移文件中的空 Adam，辅助 level=0、hold=1，
全局/物理/更新计数归零，PPO 和 Adam LR 均为 1e-3，之后沿用 adaptive KL。
Stage 06 smoke 和容量预检产生的策略不用于正式初始化。

## 本机导入和云端启动

将本次新 bundle 保存为 `/home/lx/下载/microduck-stage-07-from-stage-06.bundle`，
替换同名旧版，核对当次交付消息中的 SHA256 后导入：

```bash
cd /home/lx/microduck-double-balance/workspace
test -z "$(git status --porcelain)"
git bundle verify /home/lx/下载/microduck-stage-07-from-stage-06.bundle
git fetch /home/lx/下载/microduck-stage-07-from-stage-06.bundle \
  refs/heads/double-balance:refs/remotes/stage07/double-balance
git switch double-balance
git merge --ff-only refs/remotes/stage07/double-balance
bash scripts/deploy_stage07_training_local.sh 26497
```

部署通过一次 SSH 验证传入新 bundle，不重新下载依赖、不重跑容量或 Stage 06 smoke。
如果有正在运行的训练/容量会话，部署在修改源代码前停止；不覆盖正在运行的任务。
GPU、版本、冻结路径、容量报告和输入检查点仍在云端启动时核对。

`Stage07TrainingJob=STARTED` 表示 tmux 已启动，不等于整个初始化已经通过。
随后云端输出 `Stage07Initialization=PASS`、初始名义评估，以及
`Stage07Training=RUNNING`，才开始 PPO 更新。初始评估日志在新运行目录的
`evaluations/update_000000.log`，首次可能需要编译内核。

## 查看进度

```bash
ssh -t -p 26497 root@connect.bjb1.seetacloud.com \
  'tail -n 30 -F /root/autodl-tmp/microduck-double-balance/artifacts/double-balance-stage07/latest-training.log'
```

按 Ctrl+C 只退出这个日志查看命令；后台 tmux 训练继续。
每个更新输出 `Stage07TrainingUpdate`，包含已完成更新数、LR、loss、辅助等级分布和
episode 指标。每 100 次保存，每 500 次评估；评估期间 PPO 暂停。

本机读取当前状态（只读，不启动任何新作业）：

```bash
ssh -p 26497 root@connect.bjb1.seetacloud.com \
  'cat /root/autodl-tmp/microduck-double-balance/artifacts/double-balance-stage07/latest-training/training.json'
```

主要制品都在云端仓库外的 `artifacts/double-balance-stage07/train-.../`：

- `training.json`：初始化、来源、当前进度、末次评估、完成或失败原因。
- `iterations.jsonl`、`tensorboard/`：逐更新损失、episode 指标、奖励和辅助分布。
- `gpu-samples.jsonl`：每 10 次更新的整卡显存/利用率采样，不是精确瞬时峰值。
- `checkpoints/update_NNNNNN.pt` 及同名 `.json`：完整训练状态和 SHA256 收据。
- `latest.pt`：最近完成且经保存检查的检查点。
- `best.json`：当前训练段中最佳名义评估检查点；各周期文件均保留。
- `evaluations/update_NNNNNN/evaluation.json`：64 个名义副本的首个 episode 结果。
- `params/`、`capacity-summary.json`：运行配置与作为规模依据的容量报告。

## 评估含义

每次独立进程严格加载刚保存的模型，使用冻结的 Stage 04 play 配置：无辅助、
零命令、无随机推力/DR，10 秒 horizon，采用连续末尾 5 秒的成功定义。
64 个确定性名义初态副本不是 64 个独立随机任务样本；该比例不代表扰动鲁棒性。
训练只读取已计算的终止前指标，终止视为失败，不以曾经短暂达标冒充最终成功。
评估通过表示评估过程正常完成，成功比例可能为零。

候选排序为：成功比例、存活时间、稳定比例、中心误差（更低优先）；并列保留较早模型。
若进入续训段，其 best 指针只覆盖该段，阶段归档时需要连同旧段一起比较。
本阶段不改变 Stage 03/04 定义，不自动展开 Stage 08 的独立撤辅助训练。

## 暂停和恢复

需要有序暂停时，在另一个本机终端执行：

```bash
ssh -p 26497 root@connect.bjb1.seetacloud.com \
  'touch /root/autodl-tmp/microduck-double-balance/artifacts/double-balance-stage07/latest-training/STOP'
```

程序在完整 PPO 更新结束后保存，并报告 `Stage07Training=PAUSED`。
如果刚好在评估，等待当前评估和随后更新边界处理。数值异常时不保存部分更新状态，
保留最近有效检查点。直接关闭实例仍可能丢失上次周期保存后的更新。

确认原进程已经结束后，继续到累计 10000 次的示例：

```bash
ssh -p 26497 root@connect.bjb1.seetacloud.com \
  'cd /root/autodl-tmp/microduck-double-balance/workspace && bash scripts/launch_stage07_training.sh --resume /root/autodl-tmp/microduck-double-balance/artifacts/double-balance-stage07/latest-training/latest.pt --target-updates 10000'
```

`10000` 是累计目标，不是额外再训练 10000 次。入口先解析旧检查点绝对路径，再更新
latest-training 指针。每次恢复创建新目录，保留旧日志和所有检查点。
保存的末次循环索引为 N-1，恢复从 N 开始；保留 Adam/LR/全局计数/逐环境课程。
模拟器、RNG、episode 和循环隐藏状态重新开始，不承诺逐轨迹无缝恢复。

首次代码版本可能需要依据云端返回修复。若要导入修订 bundle 并恢复，可在本机执行：

```bash
bash scripts/deploy_stage07_training_local.sh 26497 \
  --resume /root/autodl-tmp/microduck-double-balance/artifacts/double-balance-stage07/latest-training/latest.pt \
  --target-updates 6000
```

不要仅为获得更高轮次而自动延长。如果辅助不退、顶球经常立即丢失、奖励与无辅助
评估走势相反，先保留证据并检查原因。已经完成的训练、评估和周期检查点均保留。

## 完成状态与剩余工作

`Stage07Training=TRAINING_COMPLETE` 只表示本段目标更新完成；Stage 07 还需选择模型、
回传归档、检查评估/视频、生成完整交接并提交。
当前正式训练工具通过 19 项 CPU 定向测试与 shell/语法检查；云端 6000 次训练、断点恢复、
最终名义评估与 finite_checks=PASS 已得到用户回传确认；原始证据与结果归档仍待验收。
容量四档已实测通过；不将这两类证据混为一次正式训练验收。
最终 `stage-07-complete` 注释标签和 SSH 推送在本机执行，正常推送成功后不重复完整核验。

## 训练结束后的只读回传

将 `scripts/collect_stage07_results.py` 下载到固定目录 `/home/lx/下载`，在本机执行：

```bash
python3 /home/lx/下载/collect_stage07_results.py --port 26497
```

只需系统 Python 标准库和 SSH，不需要部署新 bundle 或重新安装项目环境。
脚本将自身通过 SSH 标准输入送到云端运行，只读结果文件并输出归档；不会写入云端仓库、
启动训练、加载模型执行推理或修改课程。先确认最新训练段为 6000 次完成状态、有限值检查
和末次评估通过，再按现有排序跨全部训练段选择最佳名义模型，同时保留最终训练状态。
云端检查模型文件、保存收据与对应评估记录的 SHA-256 一致后才传输。

回传内容包括两个训练段的逐更新记录、配置、评估报告、日志、全部检查点收据及
最佳/最终两个 `.pt`，不传其他周期模型和 TensorBoard 事件。所有云端原件均保留。
下载包使用唯一名称 `microduck-stage07-results-<时间>-<随机后缀>.tar.gz`，放在固定下载目录；
本机核对归档中全部文件的大小和 SHA-256 后，解包到仓库外的
`/home/lx/microduck-double-balance/artifacts/double-balance-stage07/` 新目录。
脚本打印 `Stage07ResultsDownload=PASS`、归档路径和校验值；失败时不报告成功。
随后把该归档回传给审查端，以检查原始指标、课程和模型。这里的下载校验不替代视频或策略验收。

收集工具已通过 6 项标准库测试：跨段选择与回传校验、拒绝未完成训练、拒绝损坏检查点、
拒绝缺失的最终评估、拒绝不安全归档路径以及拒绝内容传输校验不一致。
这仅是工具验证；实际云端回传尚待用户执行。
