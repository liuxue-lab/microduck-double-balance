# 证据索引

本目录保留三份原始 JSON 报告，字节内容与用户交付版本一致。完整 SHA256、字节数和
原笔记本运行目录记录在 [provenance.json](provenance.json) 中。
报告中的本机路径、网络地址、PID 和 RUNNING 快照属于历史会话，不表示当前进程状态。

| 原始报告 | 主要用途 |
| --- | --- |
| [native-mujoco-v3-report.json](reports/headless-control-report.json) | 无窗口的 200 Hz 原生物理、完整周期时间和动作对照 |
| [native-live-v1-report.json](reports/live-control-report.json) | 两组有窗口运行、期限与显示统计、状态一致性、进程退出 |
| [hil-replay-v1-report.json](reports/recording-replay-report.json) | 无窗口控制记录回放来源、视频解码核对和 CSV 统计重算 |

## 本机资产核对

2026-10-25 16:32:55（Asia/Shanghai）的采集记录支持以下文件：

- [150 项初始资产期望](../assets-known.json)。
- [229 项已核对资产](../assets-verified.json)，路径相对于 `/home/lx/microduck-double-balance/`。
- [归档审查](../validation/archive-review.json)：229 项通过，历史扫描达到 50,000 条上限。
- [145 项 Python 分发包元数据](../environment-distributions.json)，属于 PC 项目环境。
- [132 项未跟踪状态条目](../untracked-worktree-entries.txt)，仅记录路径，未取得内容或备份证明。

原始回传文件为 `microduck-stage12-archive-inventory.zip`，SHA256 为
`5fd447696e39c7f3d08fe1f43690f3bf6a7147c7009d719b444f00bf2ea411ac`。
它包含 `report.json`、`metadata-inventory.json`、`expanded-expectations.json` 和说明。
本机采集目录为：

```text
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/final-archive-inventory-v1/20261025T083253Z-d68c1a99
```

该 ZIP 和完整目录元数据保留在原资料中，不复制到源码分支。
JSON 内的 `evidence/archive-inventory-received.zip` 是此前材料包中的位置，
不是当前源码目录下的文件。`full_backup_verified=false` 保留。

## 历史记录与直接测量

用户已确认的代码推送、离线视频审查、云 GPU 关闭，以及训练历史和取消范围，
来自原交接资料。其 SHA256 和选定范围声明记录于 [provenance.json](provenance.json)。
该文件为来源摘要，未冒充原始交接全文；相关历史结论没有在本次重新执行验证。

[早期证据检查](../validation/evidence-verification.json) 和
[归档审查](../validation/archive-review.json) 都早于本次文档提交。
本次验证范围见 [验证说明](../validation/README.md)。

源码中的 JSON 和清单可用于阅读、核对身份及追溯结论。
两段完整原始 MP4 已随源码发布。权重、原始 MJB/NPZ/CSV、全部日志和环境镜像仍需独立资料副本，清单本身不是备份。
