# Stage 10 诊断归档

这些文件保存实际运行/审查版本。Stage 10 已完成的诊断不需要再执行。脚本包含当时的 HEAD、固定路径和包版本检查；提交后旧启动器会因 HEAD 改变而停止，这是预期保护，不能删检查后重跑。离线分析脚本的原提取目录是审查工作区路径，重现分析时须显式映射到已核验的本机证据；归档不代表它们可在任意目录直接运行。

substep 的原失败与修复快照、initialization 的原失败和独立恢复版本保留于本机原始证据目录，manifest 已固定。初始化恢复只修复观测器和前置断言，没有修改环境、物理、任务、奖励、辅助或验收。

本机收尾入口为 `scripts/stage10_finalize.py`，仅标准库文件/Git操作；不导入训练或仿真包。训练、拆板和正式任务迁移均未授权。新阶段须先提出具体方案，不能拿本归档当启动许可。

归档的 initialization_recovery/stage10_initialization_worker.py 仅去除末尾多余空行，以满足 Git 空白检查；AST 一致，前后 SHA 见 stage-10-archive-normalization.json。实际执行的原字节继续保留在 C 批工具目录与原始 manifest 中。

## Stage 10 CI 补充修复

初次归档遗漏了初始化恢复包的 JSON 依赖，导致 pytest 在收集 test_recovery.py 时失败。现补齐原包的四份参考/审查 JSON、README 与 manifest；manifest 仅把已在初次归档中去除末尾空行的 worker 条目更新为实际归档 SHA。原始 manifest 和执行字节继续保留在 C 批审查证据中。没有删除、改名或排除原测试，物理、策略和验收不变。

CI 可执行不依赖本机轨迹的测试；已有五项需要历史轨迹的测试按原逻辑跳过。审查环境用原始证据的临时副本补验了恢复测试 10/10，未启动仿真。补充提交后不要重跑旧 stage10_finalize.py（其固定基线保护会拒绝新 HEAD）。详见 docs/handoffs/microduck-stage-10-ci-followup.md。
