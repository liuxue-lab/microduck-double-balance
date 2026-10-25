# Stage 08 实施准备包

状态：**仅实施准备，Stage 08 尚未完成，正式训练尚未开始。**

完整证据、实验矩阵、恢复规则与预算见
`docs/audits/stage-08-execution-plan.md`。2026-09-29 已继续实现训练恢复、独立预算监督、
批量训练/评估、初态集、选模、部署、视频和归档入口。最新操作顺序与验证范围见
`docs/handoffs/microduck-stage-08-runtime-and-cloud-runbook.md`；当前仍待目标云机验证，未开训。

## 预算与实验

优先 5090 60 小时；A800 24 小时为替代路线。按用户最新要求自动从部署开始累计，不再要求手填开机时间。
5090 留最后 8 小时收尾，A800 留 4 小时；时长不随新进程或新实验清零。

五组都从已归档的 Stage 07 第 1000 次检查点初始化，首轮各新增 500 次。
A 为原配方；B 全零命令；C 在 B 上固定动作变化权重 -0.4；D 在 C 上固定
躯干/头部重心随机化为 ±5 mm；E 在 D 上彻底撤除辅助。只比较相邻组时可以隔离
该配置背景下的单项差异。第 6000 次模型仅参与推理评估。

先完成 2500 次筛选；根据开发评估选两配置、三种子复核，核心总量最多 10500 次。
5090 有证据且预算允许时再扩至最多 18000 次总更新。总量跨模型累加，不能当成
单个模型训练长度。全部无改善时转失败分析，不能机械补足更新数。

## 本机导入

沿用 `docs/handoffs/microduck-local-ssh-push-protocol.md`。
下载文件保存为 `/home/lx/下载/microduck-stage-08-from-stage-07.bundle`。
本机工作区须干净；遇到非快进或本机已有修改时先处理差异，不重置用户工作。

```bash
cd /home/lx/microduck-double-balance/workspace
git status --short
git bundle verify /home/lx/下载/microduck-stage-08-from-stage-07.bundle
git fetch /home/lx/下载/microduck-stage-08-from-stage-07.bundle \
  refs/heads/double-balance:refs/remotes/stage08/double-balance
git switch double-balance
git merge --ff-only refs/remotes/stage08/double-balance
```

本包没有 `stage-08-complete` 标签。完成训练、评估、回传和阶段交接后，才在本机
创建注释标签并按既有 SSH 规范推送。正常成功后不重复完整远端核验。

## 只读云端预检

若目标实例已经运行，可以在本机将下面的 `PORT`、`HOST` 换成平台实际值。
不沿用 Stage 07 旧主机/端口猜测新实例，也不提交密码或私钥。

```bash
cd /home/lx/microduck-double-balance/workspace
mkdir -p /home/lx/microduck-double-balance/artifacts/double-balance-stage08
stage08_python=$(ssh -p PORT root@HOST 'bash -s -- --resolve-python' \
  < scripts/stage08_bootstrap.sh) &&
[[ "$stage08_python" =~ ^/[A-Za-z0-9_./+-]+$ ]] &&
ssh -p PORT root@HOST "'$stage08_python' - --gpu 5090" \
  < scripts/preflight_stage08_cloud.py \
  > /home/lx/microduck-double-balance/artifacts/double-balance-stage08/preflight-5090.json
```

A800 路线将 `--gpu 5090` 替换为 `--gpu A800`，并修改报告文件名。
报告会读取 GPU 型号/显存/驱动、CPU 与内存限额、磁盘空间、已有仓库 HEAD、
虚拟环境版本和第 1000/6000 次检查点哈希。它不安装依赖、不导入 Torch、
不执行 CUDA 或 PPO；退出码 2 表示硬件身份需要检查。运行入口现已接入，可在选定
实例后按最新运行交接执行部署和验证。

## 运行接入状态

1. 已接入训练分支的 Adam/LR/归一化器、计数与辅助状态恢复。A–D 在 reset 后恢复
   源辅助；E 在 reset 后设 level=5、hold=0，并验证下球及鸭身辅助力/力矩全零。
2. 已接入实时累计预算台账、更新边界保存退出、独立父进程监督、批量评估及逐回合诊断。
   `budget_decision` 本身仍只负责准入计算，运行监管由预算台账和监督器完成。
3. 已完成 43 项 CPU 定向检查，覆盖真实命令、动作、课程和指标管理器，以及 Adam/LR、
   计数、预算和归档；不代替云端 CUDA、渲染与真实训练验证。
   新实例首轮 SSH 预检遇到 `python3` 不在命令路径，兼容修复另通过 11 项部署/预算回归（3 项新增），等待本机导入后实机重试。
4. 复用目标机器已有环境和文件。旧 Stage 07 启动器限定 A800，安装脚本限定旧
   提交，不能直接套用或删除断言绕过。
5. A800 沿用已测 4096 容量；5090 只做 4096 的有界新硬件验证，要求至少 15%
   实测显存余量。未通过时暂停选择回退，不能自动减半批量。不要重跑 Stage 06 smoke。

本机 RTX 5060 不运行 PPO。Stage 03 物理、Stage 04 原任务文件、冻结严格评估
定义与项目依赖均未改写。额外奖励/物理修改须先说明并获用户确认。

## CPU 验证

```bash
.venv/bin/python -m pytest -q tests/test_stage08_*.py
```

清单生成命令仅写文件：

```bash
python3 scripts/prepare_stage08_campaign.py --gpu 5090 \
  --output /home/lx/microduck-double-balance/artifacts/double-balance-stage08/campaign-5090.json
```

生成器拒绝覆盖已有文件。预生成清单位于 `configs/stage08/`，技术检查记录位于
`docs/audits/stage-08-runtime-checks.json`；原 preparation-checks 保留上轮历史记录。
这些记录不表示策略成功或正式训练已开始。

云端结束后，必要文件回传、清单/哈希核对与可读性检查完成，再明确提醒用户关机。
