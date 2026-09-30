# Microduck Stage 09 当前状态

技术审查完成；小试没有合格候选，保留 Stage 08 主模型。本机阶段归档核验、提交、注释标签和用户 SSH 推送须以收尾脚本及终端回执为准，当前交付时尚未完成。

完整交接见 [Stage 09 头部诊断、小试与平衡优先交接](microduck-stage-09-head-diagnosis-and-balance-handoff.md)。长期保留 [本机 SSH 推送规范](microduck-local-ssh-push-protocol.md) 和 [云 GPU 与文件传输规范](microduck-cloud-gpu-and-transfer-protocol.md)。

- 四组保留新增 PPO 2000 次；F2 断电丢失重做 25 次，实际计尝试/日志更新 2025 次。本机新增 PPO 为零，不追加训练。
- 18 份云端评估已审查；13 个必要检查点已本机 SHA256 核验；10 段视频完成解码及 225 帧目视抽样检查。原 producer 的 REVIEW_PENDING 保留为历史状态，不必重录。
- 云端原始统计与本机重放不同，且同一 5060 的重复运行也不同；原因未定。严格成功仍主要受下球速度超限影响，未证明奖励因果。
- 头部纠偏暂停；未来允许调整过程中转头、稳定后回正，具体方案未批准。
- 用户询问拆板时机：下一阶段可先提零 PPO 头部接触审计，当前不改物理/任务、不拆板。上球当前不与头壳碰撞，直接删板不可行。
- 云端必要工作结束，2026-10-01 05:03:28 +08 已提醒手动关机；实际电源状态 UNCONFIRMED。保留云端未回传的完整原始轨迹，不操作实例电源。
- 本次已完成回传实际是 SSH 上传/HTTPS 下载；未来新脚本默认 Git SSH 双向传输，正常推送成功后不重复全量远端核验。

固定仓库 `/home/lx/microduck-double-balance/workspace`，下载目录 `/home/lx/下载`。下一条命令：

```bash
python3 /home/lx/下载/microduck-stage09-finalize.py
```

脚本完成后由用户执行它打印的 `git push origin double-balance` 和 `git push origin stage-09-complete`。本机注释标签不强制覆盖，不自动推送。不要重跑历史 preparation/recovery 文档中的已完成命令，不重跑 Stage 06 smoke。
