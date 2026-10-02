# Stage 10 当前入口

## 2026-10-02 发布与 CI 最终补记（优先于下文旧快照）

用户已通过既有 Git SSH 成功推送 `double-balance`：`1a11f30..293900b`。此前按用户明确要求完成的只读远端检查确认分支为 `293900b9ca73ef90fd8152e215f3676cbc401db6`；注释标签 `stage-10-complete` 保持指向 `1a11f30cc7264d00dec7bce8f11449ded9164169`，没有移动。此后不重复完整远端核验。

本次仅补查该修复提交的 [CI 工作流](https://github.com/liuxue-lab/microduck-double-balance/actions/runs/36892445739)：run `36892445739`、job `110471132738` 均为 `completed / success`，工作流更新时间为北京时间 **2026-10-02 00:39:08**。CPU 测试日志为 **416 passed、6 skipped、3 warnings、17 subtests passed**；构建、CLI 发现、策略导出、ONNX rollout parity、现有结构生成检查和 136 项相对文档链接检查均通过。保留跳过和警告数，不写成全部测试无条件通过。

原来的两次 CI 失败由恢复工具归档遗漏 JSON 依赖引起，已由 `293900b` 修复并获上述远端验证。旧失败日志和旧 PENDING 记录保留，作为历史快照；它们不覆盖本补记。详细证据见 [发布确认](../audits/stage-10-publication-confirmation.json) 和 [CI 补充交接](microduck-stage-10-ci-followup.md)。

Stage 10 技术诊断、审查、归档、主收尾提交/标签、用户推送及 CI 修复已闭环；双球平衡策略目标仍未完成。零 PPO、无新模型、无正式拆板或训练授权，B1/C1 事后检查缺口及根因不确定性不变。上述绿色 CI 仅对应 `293900b`，不预先认证此后文档补记或其他提交。

本补记由助手交付，本机是否应用、生成何种补充提交及是否推送，以随后本机输出为准；不得伪造新提交 SHA。应用本补记后，仅推送 `double-balance`，保留原阶段标签。不要重跑旧 finalize、ci-fix 或 A/B/C 诊断脚本。

下一对话进入 Stage 11，先审查头顶接触物理验证方案；正式改物理、任务/验收、奖励或约束前仍须用户确认。完整开场见 [Stage 11 开场文本](microduck-stage-11-opening-prompt.txt)。下文原收尾命令仅为历史追溯，不再是下一步操作。

## 下一阶段的顺序

1. 阅读完整交接和 head-contact-migration-audit，提出最小头顶接触物理验证方案，列明改动、理由、影响与判据，等待用户确认。
2. 获确认后在独立变体完成零 PPO 物理验证，保留旧托盘任务；先验证接触和质量/惯量，再讨论正式任务迁移。
3. 正式迁移时单独审查上球观测、初态、奖励和验收参考系；经确认再实施。头部纠偏暂停，允许调整转向、稳定后回正。
4. 训练不是默认下一步；训练前另行确认恢复合同、实验规模和云预算。

## 固定状态与操作

本机 `/home/lx/microduck-double-balance/workspace`，下载 `/home/lx/下载`，证据根目录 `/home/lx/microduck-double-balance/artifacts/double-balance-stage10`。完整 A/B/C/D 路径、清单与主模型路径见 [完整交接](microduck-stage-10-replay-and-head-contact-handoff.md)。主模型仍为 Stage 08 E/20260929/update_004000.pt，SHA256 `86d55c3703c18fcf4817db49c6e4dcf839b3f5195d68ae2c559db7e222bded97`，历史独立 652/768（84.90%）。旧 10 秒/终局连续 5 秒/无辅助标准和下球 0.15 m/s 上限保留。

遵守 [本机 SSH 推送规范](microduck-local-ssh-push-protocol.md) 和 [云 GPU/传输规范](microduck-cloud-gpu-and-transfer-protocol.md)。正式代码/模型归档分支分开、不合并，上传取文件默认 Git SSH。本机 RTX 5060 仅开发、MuJoCo 仿真、推理和视频，不重跑 Stage 06 smoke。

RTX 5090 为 OFF_CONFIRMED_BY_USER；Stage 10 没有新云会话、关机操作或提醒事件。Stage 09 完整云原始轨迹未全回传，禁止删除。不得自行启停云实例或运行旧失败关机守护。未来云计算与必要回传 SHA 完成后立即提醒手动关机，提醒与用户实际确认分别记录。

本次文档补记交付脚本的精确命令：

```bash
python3 '/home/lx/下载/microduck-stage10-handoff-update.py'
```

它只核对本机 Git/目标文档、更新五份文档并创建补充提交，不联网、不运行仿真/训练、不推送、不移动标签。可加 `--verify-only` 只检查；重复执行已完成的自身提交时只核验。若 HEAD、目标文件、暂存区或标签冲突，停止并保留现场，不 reset/force/覆盖。无关未跟踪文件不纳入提交。

脚本成功后用户执行：

```bash
cd /home/lx/microduck-double-balance/workspace
git push origin double-balance
```

正常成功输出即足够，不重推标签，不重复整套远端检查。新文档提交的 CI 若触发，与已验证的 293900b CI 分开记录。下一对话按 [Stage 11 开场](microduck-stage-11-opening-prompt.txt) 开始，第一步为审查，不运行旧 finalize/ci-fix/A/B/C 工具。
