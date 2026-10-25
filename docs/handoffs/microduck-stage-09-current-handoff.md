# Microduck Stage 09 当前状态

技术审查完成；小试没有合格候选，保留 Stage 08 主模型。本机归档核验、提交和注释标签已完成；用户 SSH 推送尚待成功输出。阶段技术完成提交 `9c51a133b40e4342985bab696e54fc02cd45f615`，注释标签 `stage-09-complete` 保持指向该提交；后续关机确认作为文档补充提交，不移动标签。

完整交接见 [Stage 09 头部诊断、小试与平衡优先交接](microduck-stage-09-head-diagnosis-and-balance-handoff.md)。长期保留 [本机 SSH 推送规范](microduck-local-ssh-push-protocol.md) 和 [云 GPU 与文件传输规范](microduck-cloud-gpu-and-transfer-protocol.md)。

- 四组保留新增 PPO 2000 次；F2 断电丢失重做 25 次，实际计尝试/日志更新 2025 次。本机新增 PPO 为零，不追加训练。
- 18 份云端评估已审查；13 个必要检查点已本机 SHA256 核验；10 段视频完成解码及 225 帧目视抽样检查。原 producer 的 REVIEW_PENDING 保留为历史状态，不必重录。
- 云端原始统计与本机重放不同，且同一 5060 的重复运行也不同；原因未定。严格成功仍主要受下球速度超限影响，未证明奖励因果。
- 头部纠偏暂停；未来允许调整过程中转头、稳定后回正，具体方案未批准。
- 用户询问拆板时机：下一阶段可先提零 PPO 头部接触审计，当前不改物理/任务、不拆板。上球当前不与头壳碰撞，直接删板不可行。
- 云端必要工作结束，2026-10-01 05:03:28 +08 已提醒手动关机；用户于 2026-10-01 05:39:34 +08 确认“5090已关机”，状态 `OFF_CONFIRMED_BY_USER`；该时间是收到确认的时间，实际关机时刻未提供。保留云端未回传的完整原始轨迹，不操作实例电源。
- 本次已完成回传实际是 SSH 上传/HTTPS 下载；未来新脚本默认 Git SSH 双向传输，正常推送成功后不重复全量远端核验。

固定仓库 `/home/lx/microduck-double-balance/workspace`，下载目录 `/home/lx/下载`。下一条命令：

```bash
cd /home/lx/microduck-double-balance/workspace
git push origin double-balance
git push origin stage-09-complete
```

收尾脚本已成功，无须再次执行。归档/关机确认见 `docs/audits/stage-09-local-completion-and-shutdown-confirmation.json`；历史脚本打印的 UNCONFIRMED 是确认前快照。保留已有注释标签，由用户推送；正常成功后不重复全量远端核验。不要重跑历史 preparation/recovery 文档中的已完成命令，不重跑 Stage 06 smoke。

本文件记录推送前状态；用户随后提供正常成功的分支和标签推送输出，即可确认 Stage 09 发布完成，不为更新 PENDING 字段再创建补充提交，也不重复全量远端核验。
