Microduck Stage12 PC CPU/CUDA timeline v1 — 2026-10-24 (Asia/Shanghai)

范围已按用户要求收敛为ROS2 Humble + Orin Nano Super硬件在环。
真实传感器/电机接入和真机演示不再是本项目剩余任务；仍保留Orin实时闭环验收。
完整交接：docs/handoffs/microduck-stage12-hil-handoff-20261024.md。

最新auditfast-v1报告数值对照通过，但平均提速3.50%、p95提速3.14%，未达5%门槛。
保留InferenceOnly + PhaseCache + BAM Graph v2 + BatchedAuditReads。
PC基准约25.03ms/p95 26.21ms；50Hz完整闭环尚未验收，不重新训练。

本轮两组各160控制步/1600物理步，固定种子，每组一次10%自重0.1秒扰动。
第二组只在第101–108步收集CPU/CUDA时间线，保留所有原500Hz检查。
无分析器基准用于正常耗时；时间线用于定位，不生成“提升多少”的采用结论。
逐步核对22类状态、完整Audit/Trial/扰动；分析器缺GPU事件或计数不足则失败保留。
详细报告内含GPU活动区间并集、核函数/拷贝/运行时统计和原始trace路径/哈希。
无新PPO、云连接、Orin连接、电机I/O、依赖安装、生产激活或自动重试。

下载脚本到/home/lx/下载。每条成功后执行下一条：
python3 "/home/lx/下载/microduck-stage12-pc-timeline-v1-20261024.py" --install-code
git -C /home/lx/microduck-double-balance/workspace push origin double-balance
python3 "/home/lx/下载/microduck-stage12-pc-timeline-v1-20261024.py" --run

仅使用电脑RTX5060；5090保持关闭，本轮无需Orin；保持主运行终端开启。
预计数分钟，每组worker上限230秒。5秒心跳；步数/百分比显示运行进度。
另开终端 --watch 每2秒刷新且结束后不退出；--status单次查看；--stop停止本轮。
已有报告时--run不重做，--retry-run仅用于明确决定后的重跑。
发回PC_TIMELINE_REPORT指向的report.json：
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-pc-timeline-v1/report.json
PC_TIMELINE_MEASURED只表示诊断和状态对照完成，不表示50Hz或最终项目通过。

本轮父提交5d80475823dfc41e662f62752e5919d5523ac5bc；新提交/包哈希见build-receipt.json。
只新增experiments/stage12/pc-timeline-v1，保护本机改动，不reset/stash/clean/force push。
本工作区CPU测试含真实PyTorch CPU profiler；CUDA行仅在测试夹具中模拟，不能当GPU实测。
测试数量与结果以cpu-tests.log、validation-summary.json为准。

长期协议原文保留：
docs/handoffs/microduck-local-ssh-push-protocol.md
docs/handoffs/microduck-cloud-gpu-and-transfer-protocol.md
固定本地、SSH、有线配置、模型哈希、回传状态、历史错误和剩余四项任务见完整交接。
