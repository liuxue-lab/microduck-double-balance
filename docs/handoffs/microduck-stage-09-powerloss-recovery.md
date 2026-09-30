# Stage 09 欠费关机恢复记录

用户于 2026-10-01 02:36:08 Asia/Shanghai 批准一次性补偿 F2 丢失的 25 次更新。

F2 已计尝试 175，最后有效保存 150；恢复源为 pilot/F2/segment-773f5dc5/checkpoints/update_000150.pt。已失更新及其消耗保留。F2 尝试上限为 525，F0/F1/F3 仍为 500，总尝试上限为 2025；四组模型进度仍最多 500。原四小时作业起点和 45 分钟结束预留不变，停机期间仍计入墙钟时间，不执行实例电源操作。

恢复完整 actor/critic、观测归一化、Adam 一二阶矩和 step、保存的学习率及进度。第150次检查点 Adam step 必须为103000。回合、RNG、RNN及动作延迟按既有规则重置，辅助保持零；这不是精确轨迹续接。

原本机零PPO验证报告、原批准方案和已完成评估不改写。旧运行时哈希清单保存在 stage-09-runtime-before-powerloss.json；新清单仅允许审计列出的预算/入口/版本兼容改动及恢复脚本、监控、测试与文档新增。campaign 保留原 runtime_manifest_sha256，并另记 active_git_head / active_runtime_manifest_sha256。

恢复程序先核对 GPU 无任务、作业锁、F0/F1 各500及评估报告SHA、F2断点与账本、F3尚未开始和剩余时间。检查通过后备份原账本及状态，原子添加一次性授权并启动原 campaign。计数不清零，重复补偿或再次丢失更新不自动扩大额度。

本机安装器运行零PPO合约测试，本机提交，使用小型增量 Git bundle 经 SSH 更新云端代码。无需重新导出模型或初态，无 Stage06 smoke。恢复后实际训练、评估、视频检查、检查点回传、最终失败分析与阶段交接仍待完成。

本机仓库 /home/lx/microduck-double-balance/workspace，下载目录 /home/lx/下载。阶段结束时本机创建注释标签并沿用 docs/handoffs/microduck-local-ssh-push-protocol.md 通过 SSH 推送，正常成功后不重复完整远端核验。
