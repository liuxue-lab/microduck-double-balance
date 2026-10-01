# Stage 10 CI 补充交接

Stage 10 收尾提交为 `1a11f30cc7264d00dec7bce8f11449ded9164169`。double-balance 与 stage-10-complete 的两次 push 分别触发 CI，均在 Run CPU tests 的收集阶段失败。代码/标签已传到 GitHub，当前 CI 尚未通过；本次补充修复仍属于 Stage 10。

## 原因与责任

Stage 10 收尾工具只归档了恢复工具目录中的 Python 文件，遗漏 import 时需要的 recovery-reference.json 等非 Python 文件。根目录 pytest 自动收集 test_recovery.py，导入 stage10_recovery_common 时出现 FileNotFoundError。先前 14 项收尾工具隔离测试检查了写文件、Git 提交/标签和冲突保护，但没有覆盖整个仓库 pytest 的自动收集。这是归档集成遗漏。

已通过只读 GitHub API 取得完整失败日志：

- [分支 CI](https://github.com/liuxue-lab/microduck-double-balance/actions/runs/36890165087)：job 110463470818，1 error in 20.02s。
- [标签 CI](https://github.com/liuxue-lab/microduck-double-balance/actions/runs/36890169180)：job 110463485222，1 error in 19.47s。

两次都已成功安装 frozen environment，后续构建、CLI、导出/parity、结构生成和文档链接步骤因前面的失败被跳过。邮件显示 3 annotations 不代表有三个独立根因；本次停止原因是缺少 recovery-reference.json。Node action 版本警告没有导致这次收集异常，暂不扩展修改 CI 依赖。

## 修复范围

从已核验 C 批的 initialization-recovery-v1/tool 恢复四份原始 JSON（recovery-reference、reference-b、reference-runtime、reviewed-first-trace）、原 README 和 manifest。初次归档的 worker 去掉了末尾多余空行，因此新归档 manifest 仅更新这一条的 SHA/字节数，使清单与已有归档文件一致。原始执行包、原 manifest、所有数组/回执和旧失败结果保持原样。

test_recovery.py 保持原字节，没有删除测试、改名避开 pytest、增加跳过条件或禁用工作流。新增 tests/test_stage10_archive_package.py，在新进程验证导入与完整包清单，并确认没有导入 torch、mujoco 或 mjlab。

没有修改 101 项冻结运行文件、Stage 03 物理、Stage 04 任务/验收、奖励、辅助、模型或训练状态。零 PPO、零新仿真、零云操作。B1/C1 原有事后检查缺口保留。

## 验证与限制

- 修复前在原交付目录复现同一 FileNotFoundError：收集退出码 2，未收集到测试。
- 修复后，全部归档测试与新增完整性回归：28 passed、5 skipped、13 subtests passed。
- 五个跳过项本来就要求历史轨迹素材，GitHub runner 上没有该素材。另在审查环境用原始返回证据的临时副本运行恢复测试：10/10 通过、0 skipped；控制器使用 mock，没有生成新仿真轨迹。
- 文档相对链接检查通过，冻结运行文件 101/101 不变。
- 审查环境缺完整仿真依赖，没有运行全部仓库 CI。最终远端结果须等本机补充提交并由用户 SSH 推送后检查；不提前标记绿灯。

## 本机应用与发布

本机仓库 `/home/lx/microduck-double-balance/workspace`；下载 `/home/lx/下载`。下载 microduck-stage10-ci-fix.py 后运行：

```bash
python3 '/home/lx/下载/microduck-stage10-ci-fix.py'
```

工具仅做文件校验、增补归档/测试/文档并创建本机补充提交。它要求分支 double-balance、预期 HEAD、原始 SSH remote、固定目标文件 SHA 和 stage-10-complete 指向 1a11f30；不联网、不安装依赖、不启动仿真/训练、不推送。遇到冲突保留现场，不强制覆盖。`--verify-only` 可只检查。

成功后只推送分支：

```bash
cd /home/lx/microduck-double-balance/workspace
git push origin double-balance
```

保留 stage-10-complete 在原收尾提交，不移动或重建标签，不重复推送标签。正常成功的 SSH 输出足够；本次因为有明确 CI 错误，只检查新提交对应的工作流结果，不做整套远端标签/历史重复核验。若新 CI 出现其他失败，读取实际失败步骤再处理。不要重跑旧 stage10_finalize.py。

## 长期状态与后续

完整阶段交接见 [Stage 10 交接](microduck-stage-10-replay-and-head-contact-handoff.md)，并继续遵守 [本机 SSH 规范](microduck-local-ssh-push-protocol.md) 与 [云 GPU/传输规范](microduck-cloud-gpu-and-transfer-protocol.md)。本机 RTX 5060 只做开发、仿真、推理/视频；正式训练须另行确认云预算和模型/Adam/LR/进度/辅助恢复规则，不重跑 Stage 06 smoke。代码与模型归档分支分开，不合并；上传和取文件默认 Git SSH。

RTX 5090 为 OFF_CONFIRMED_BY_USER，本阶段没有新云会话；不自行启停云实例，不运行旧失败自动关机守护。完整 Stage 09 云原始轨迹未全回传，禁止删除。未来云计算与必要回传 SHA 完成后立即提醒用户手动关机，提醒与实际确认分开记录。

主模型仍为 Stage 08 E/20260929/update_004000.pt，SHA256 `86d55c3703c18fcf4817db49c6e4dcf839b3f5195d68ae2c559db7e222bded97`，历史独立 652/768（84.90%）。旧 10 秒回合终局连续稳定至少 5 秒、无辅助标准不变。当前数值机制推断及策略未达目标的结论不变。CI 修复不构成训练或拆板授权；Stage 11 仍先提出头顶接触物理验证方案供用户确认。
