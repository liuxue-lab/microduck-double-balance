# Microduck Stage12 原生仿真结果与阶段封口

日期：2026-10-20。训练、择优、必要文件取回、逐回合汇总审计及本机视频已完成。当前模型为速度奖励微调 +500，253/256（98.828125%），最长连续稳定时长中位 9.48 秒。阶段标签的范围是本轮原生仿真工作流与已验证结果的归档，不包含尚未执行的最终留出集、旧 G11 严格物理验收或实机验收。

本文件在封口包准备时生成。本机提交和标签是否完成，以本机入口产生的 `LOCAL_STAGE12_COMPLETE_READY_TO_PUSH` 报告为准；是否发布，以用户正常 SSH push 输出为准。不能仅凭本文件存在声称已发布。

## 固定状态

- 本机仓库 `/home/lx/microduck-double-balance/workspace`，下载 `/home/lx/下载`。
- 分支 `double-balance`；SSH 远端 `git@github.com:liuxue-lab/microduck-double-balance.git`。
- 准备时读取的远端基线 `73ee01930a031d76a9d57419afdfd4464a547ab3`；当时远端尚无 `stage-12-complete`。
- 最佳检查点 `native-speed-refine-v1/refine_000500.pt`：原 4500 + 速度奖励 500，策略累计 5000；不是原训练 5000，也不是原奖励微调 5000。
- 模型 SHA256 `51821a28880aba74b80e6d5a388dfd7a82098bd3dca04a1e8947b4b2a8d5539a`，10036828 字节。
- 本机模型：`/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-selected-speed500-v1/checkpoints/refine_000500.pt`。
- 原 4500 保留：`native-selected-4500/checkpoints/update_004500.pt`，SHA256 `df03202acf010680b63d3277ec879f63e2d4808e5d1286653923a314f125ac6a`。
- 本机录像：`/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-replay-speed500-v1/index.html`。

## 结果与限制

原训练 6000 次完成后，4500 的 212/256 优于 6000 的 165/256。原奖励微调 500 次未改善；速度奖励微调完成 1000 次预算，四个候选为 246、253、245、250 个成功首回合（每个 256）。保留中途最佳 +500。各轮预算与旧失败尝试记录保留，不重开或清零。详细代码、报告与说明见[实验归档](../../experiments/stage12/native-workflow/README.md)。

评估要求首回合连续 250 个 50 Hz 控制时刻稳定，且无物理终止；每回合最多 10 秒。共同种子标签不等于初态严格配对。成功率用于模型选择，未执行最终留出集。

选定模型的两份 128 回合文件审计为 127/128 与 126/128；五项汇总审计精确相等，无浮点容差放行，三个失败回合原样保留。本机单环境视频种子 20261021、20261022 各一段，均存活 10 秒且最长连续稳定 9.84 秒。视频是视觉样例，不是云批次严格轨迹重放。

两段 MP4 均 H.264、1280×480、25 fps、250 帧；完整解码通过，并按约 0.5 秒间隔抽样审查及查看末帧。可见保持脚下球支撑与上球顶头，有小幅姿态调整和缓慢平移。未据抽样画面宣称高频抖动消失或接触力、毫米级接触精度已通过验收。

当前不继续加训、不重评、不重录。4096 环境性能设置已恢复，若以后确需新训练继续保留该基准，并重新明确预算。旧 G11 严格物理路线未被本轮原生结果追认通过。项目当前目标为仿真，不新增实机部署为阶段必做项。

## 本机归档与发布

下载已交付的 `microduck-stage12-finalize-v1-20261020.py` 到 `/home/lx/下载` 后：

```bash
python3 "/home/lx/下载/microduck-stage12-finalize-v1-20261020.py" --finalize
```

入口仅使用标准库和 Git。先核对分支、SSH origin、365 份冻结源码、选定模型/评估/回放文件 SHA 与当前 Git 状态；随后新增实验归档和本交接，将远端尚缺的 48 份现有冻结源码及 `mdp.py` 的已验证版本一并纳入提交，检查仓库文档相对链接，创建本机提交与注释标签 `stage-12-complete`。源码只读取和暂存，不改写工作区字节。检查点、视频、完整轨迹不进入代码分支。不会运行 Torch、PPO、仿真、依赖安装、云 SSH 或 push。

本机 HEAD 可为上述基线或它的后继，365 份源码在文件系统中必须全部匹配冻结 SHA。允许纳入提交的 49 个源码路径及原 Git 对象明确固定在清单中；它们的 HEAD 版本必须是已核对的 Stage11 原版本或目标冻结版本，未知差异会停止。其余源码的 HEAD 必须已经匹配。实验归档和本交接必须是新增路径。无关已暂存改动会阻止提交，保留现场；不 stash/reset/clean 或覆盖。精确匹配的本次中断归档、提交或标签可重入；冲突停止，不强制覆盖。已有本机 Git hooks/签名配置正常生效。

成功状态必须为 `LOCAL_STAGE12_COMPLETE_READY_TO_PUSH`。提交、标签及 bundle 身份记入小 JSON。增量 bundle 在本机生成：

```text
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-stage12-finalization-v1/microduck-stage-12-from-stage-11.bundle
```

本机仓库已经包含提交，无需再导入该备份 bundle。脚本停在推送之前，再按长期规范执行：

```bash
cd /home/lx/microduck-double-balance/workspace
git push origin double-balance
git push origin stage-12-complete
```

正常成功输出足以确认发布，不要求再做完整 fetch/分歧核对。未收到输出前发布状态仍未确认。

## 恢复与云端

只读预检：将上述 `--finalize` 换成 `--check`。归档/提交过程意外中断可重复同一 `--finalize`；如果报告 `LOCAL_FINALIZATION_STOPPED_PRESERVE`，回传该小 JSON。不要删除现场、强制覆盖标签或重开任何已完成训练脚本。现有无关已暂存内容需先由用户按其意图提交或取消暂存，入口不会擅自处理。

push 若报 non-fast-forward、标签冲突或 SHA 不符，按[固定推送规范](microduck-local-ssh-push-protocol.md)做最小核验，禁止 force push，不改 HTTPS/PAT。旧 SSH 共享连接中断和评估浮点汇总恢复的完整历史保留在实验归档的 `history-guide-before-finalization.txt`，不授权再次启动旧预算。

必要云端计算与本次六份文件取回已完成，无排定的新云任务。其余候选、完整日志与旧证据仍在云端，不能宣称完整备份。2026-10-20 接收新模型回放结果后已提醒：两小时内预计复用则保持 5090 开机，否则手动关机并保留存储。平台电源状态仍为 UNCONFIRMED，未自动开关机。

继续遵循[本机提交与 SSH 推送固定规范](microduck-local-ssh-push-protocol.md)和[云 GPU 关机提醒与文件传输规范](microduck-cloud-gpu-and-transfer-protocol.md)。本次完成的云文件回传使用会话已批准的小包 SSH 方式；以后不据此更改长期 Git SSH 传输规则。
