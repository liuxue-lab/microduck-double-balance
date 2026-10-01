# Stage 10 当前入口

先读 [完整交接](microduck-stage-10-replay-and-head-contact-handoff.md) 和 `docs/audits/stage-10-final-acceptance.json`。技术诊断与视频/源码审查已完成，策略目标未完成；本文件生成时本机提交/标签与用户 SSH 推送仍待执行。实际完成状态以本机 finalization receipt 和后续正常成功的 SSH 推送输出为准。

当前源基线 double-balance@00e34c2038771c5d4ad49fe45dff828c0232e60c；Stage 09 注释标签仍指向 9c51a133b40e4342985bab696e54fc02cd45f615。收尾工具将新建 stage-10-complete，不移动旧标签。Stage 08 E/20260929/update_004000.pt 继续作为主模型，SHA256 `86d55c3703c18fcf4817db49c6e4dcf839b3f5195d68ae2c559db7e222bded97`，历史独立 652/768（84.90%）。Stage 10 零 PPO、无云计算、未重跑 Stage 06 smoke。

同 GPU 重复标签有差异；B 定位到首物理步、C 发现初始化派生字段差异，D 的安装源码和 CPU 加法顺序枚举支持浮点累加顺序敏感的解释，但完整根因未证实。B1/C1 事后检查缺口保留，不能倒填。旧标准仍为 10 秒回合终局连续稳定至少 5 秒、无辅助；程序 PASS 不等于策略成功。下球超速仍是直接问题，奖励因果未知。

头部纠偏暂停，允许将来调整时转头、稳定后回正，不锁头颈。拆板需要真实头顶碰撞、18 g 质量/惯量去除及观测/初态/奖励/验收参考系迁移。方案已静态审计，尚未授权正式修改或训练；保留旧托盘任务和旧结果。

本机仓库 `/home/lx/microduck-double-balance/workspace`，下载 `/home/lx/下载`，证据统一在 `/home/lx/microduck-double-balance/artifacts/double-balance-stage10`，各固定子路径、SHA、错误处理见完整交接。下一命令：

```bash
python3 '/home/lx/下载/microduck-stage10-finalize.py'
```

成功后用户 SSH 推送分支与新注释标签，不重复完整远端核验。收尾工具遇到冲突会停止保留现场，禁止强制覆盖。已生成正确提交但中断未打标签时，同一脚本可安全续接。

继续遵守 [本机 SSH 推送规范](microduck-local-ssh-push-protocol.md) 与 [云 GPU 与传输规范](microduck-cloud-gpu-and-transfer-protocol.md)。RTX 5090 已由用户确认手动关闭；早期 UNCONFIRMED/PENDING 为历史快照。完整云端原始轨迹未全回传，禁止删除。不得自行启停云实例或启动旧失败关机守护。以后云计算和必要回传 SHA 完成后立即提醒手动关机，提醒和实际确认分别记录。正式代码/模型归档分支分开、不合并，上传和取文件默认 Git SSH。
