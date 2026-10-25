# Microduck Stage12 统一复现说明

适用日期：2026 年 10 月 25 日（Asia/Shanghai）

本说明适用于已经完成 Stage12 安装的原笔记本与 Orin。当前结果均已通过，默认只查看保存结果。仅当用户明确要求新一次演示时，才使用本说明中的重新运行入口。现有单文件发布脚本是增量安装器，不能在空白电脑上单独还原全部环境。

## 1 固定目录与版本

| 内容 | 位置或身份 |
| --- | --- |
| 项目根目录 | `/home/lx/microduck-double-balance` |
| 源码仓库 | `/home/lx/microduck-double-balance/workspace` |
| 下载目录 | `/home/lx/下载` |
| 产物目录 | `/home/lx/microduck-double-balance/artifacts/`，位于源码仓库外 |
| 源码分支 | `double-balance` |
| 最新功能提交 | `056657aa36158fb81dca6fd2092e525222d6e632` |
| Git SSH 远端 | `git@github.com:liuxue-lab/microduck-double-balance.git` |

模型、轨迹和视频不加入 `double-balance` 源码分支。保留已有工作树，不因材料整理执行 reset、clean、stash 或 force push。此前用户已确认推送成功，目前不需要再次做完整远端验收。

## 2 已验证的环境

| 位置 | 已记录配置 | 用途 |
| --- | --- | --- |
| PC | x86_64，Ubuntu 22.04.4 | 原生物理、参考对照、显示与回放 |
| PC 项目 Python | 仓库 `.venv`，Python 3.12 | Torch、MuJoCo、任务和显示 |
| PC 系统 Python | `/usr/bin/python3`，3.10.12 | 监督、状态汇总、ROS2 Humble 桥 |
| 项目依赖 | torch 2.9.1；rsl-rl-lib 5.0.1；mjlab 1.3.0；mujoco 3.10.0 | 原有任务、策略及原生物理 |
| 保留依赖 | mujoco-warp 3.8.1；warp-lang 1.12.0 | 环境依赖保留，本次不进行 Warp/CUDA 物理积分 |
| Orin | aarch64，Ubuntu 22.04.5；L4T 36.4.3；MAXN_SUPER | 实际推理硬件 |
| Orin 推理 | 系统 Python 3.10.12；onnxruntime 1.22.0；numpy 1.26.4；CPUExecutionProvider | 冻结 FP32 ONNX |
| 通信 | ROS2 Humble、Domain 62、rmw_fastrtps_cpp / Fast DDS | 有线限定、reliable、depth 1、volatile |
| 图形与媒体 | 原有 GLFW 桌面环境；离线回放使用 EGL；已有 ffmpeg | 实时窗口及 H.264 回放 |

本表是成功运行时的已知配置，不是完整 lockfile。BAM、任务源码、资源文件和其他 Python/系统包也必须保留。2026 年 10 月 25 日 16:32:55 的本机采集补充了 [145 项项目环境分发包元数据](environment-distributions.json)，其中项目 `.venv` 的 numpy 为 2.4.1、onnx 为 1.22.0、onnxruntime 为 1.24.4、tensordict 为 0.10.0、glfw 为 2.10.0。这些是 PC 项目环境版本，不是 PC 系统 ROS Python 或 Orin 的版本。不能仅据包版本号认定空白系统可复现。

PC 系统 Python 没有 onnxruntime 的旧探测记录不构成故障：PC 系统 Python 用于 ROS 桥，ONNX 在 Orin 上执行。不要为此向系统 ROS Python 安装 Torch 或改动已验收环境。

## 3 必须保留的依赖链

以下目录均相对于 `/home/lx/microduck-double-balance/artifacts/double-balance-stage12/`。

| 资产 | 必要性 |
| --- | --- |
| `native-live-v1/package/`、固定入口及当前 run | 最新实时显示运行代码和记录 |
| `native-mujoco-v3/package/`、固定入口及已通过 run | 无窗口结果；live 读取其 MJB 和 NPZ 作为显示及状态对照来源 |
| `native-hil-replay-v1/package/`、固定入口及已完成 run | 离线视频与页面 |
| `native-pc-timeline-v1/package/` | live/v3 的 bootstrap 实际导入的固定基线；完整身份见下文 |
| `native-mujoco-v1/package/`、`code-installed.json`、报告及报告指向的导出 | 后续入口会复核其历史证据，包括曾经失败的汇总记录 |
| `native-mujoco-v2/runs/20261025T041428Z-9b2774b5/` | 需要原 report、两组 timing.csv 和 state-data.npz 做历史对照 |
| `native-ros2-humble-v1/frozen-export/` | ONNX、contract、golden 和 export-report |
| `native-ros2-humble-v4/report.json` | 已通过的 ROS 链路证据 |
| 其他被固定引用的旧报告 | `native-cumulative-loop-v1`、`native-pc-actiongraph-v1`、`native-pc-sync-v1`、`native-pc-timeline-fixed-v2`、`native-ros2-humble-loop-v1` |
| `native-push10-final6h-v1-client/returned/0cbee30e5a83904371076ca09a8cfb4afd1fa3f4/` | 冻结检查点、候选回执、账本、任务源码和回传索引 |

基线包完整身份：`ba0b16e7acde7bc8484d9c9519c490110a3556fe528d9276f0cf611732456b54`；其 manifest SHA256 为 `c320743d583532893a7d7069770f94cfaa6a0818afcc6dbcaa161dc9a61a3bc1`。依赖关系已从已发布 bundle 中的源码核对。旧版失败报告仍是当前入口读取的依赖；保留它们不会改变当前成功结论。

不能仅保留“最新脚本＋ONNX”后删除其他旧目录。本机采集已展开固定 manifest 和旧报告中的文件列表，共核对通过 229 项。其他历史目录只有部分元数据，不能将其自动认定为可恢复备份。

## 4 只查看已完成结果

在笔记本终端查看最新状态：

```bash
python3 /home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-live-v1/microduck-stage12-native-live-v1.py --status
```

打开已完成的视频与耗时曲线，无需 Orin：

```bash
python3 /home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-hil-replay-v1/microduck-stage12-hil-replay-v1.py --open
```

正确的已完成回放目录为：

```text
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-hil-replay-v1/runs/20261025T051929Z-dc4ca58d
```

其中 `index.html` 是浏览入口，`quiet-2407-recorded-hil.mp4` 和 `regression-2500-recorded-hil.mp4` 是两段 v3 记录回放视频。视频已经生成并通过人工审查，无需重生成。

## 5 有明确新演示需求时再运行

保持 5090 关闭。将笔记本与 Orin 用网线连接，在笔记本图形桌面的终端启动，保留主终端和显示窗口：

```bash
python3 /home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-live-v1/microduck-stage12-native-live-v1.py --retry-run
```

该命令新建独立 run，执行两组各 60 秒的 HIL，不覆盖旧结果。初始化和清理另需时间，因此整个进程墙钟时间并非精确 120 秒。`--run` 在已有报告时复用结果；`--retry-run` 才明确请求新一次运行。`--apply` 是原发布包首次导入并启动的入口，当前无需重复执行。

| 操作 | 作用 |
| --- | --- |
| 窗口按 `1` / `2` | 全身 / 头部视角 |
| 窗口按 `F` | 跟随 |
| 鼠标 | 旋转、平移、缩放视角 |
| `--watch` | 每 2 秒观察进度，Ctrl+C 仅退出该观察终端 |
| `--stop` | 请求停止所属闭环，保留已写日志和数据，不关闭 Orin |
| 主运行终端 Ctrl+C | 触发受控停止和清理 |
| Esc 或关闭显示窗口 | 只结束显示，控制及原始记录继续；该轮显示验收可能不通过 |

需要停止整个闭环时，另开终端执行：

```bash
python3 /home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-live-v1/microduck-stage12-native-live-v1.py --stop
```

入口不提供暂停控制、鼠标施力或复位功能。关闭主终端或收到 SIGHUP 会触发清理，因此这不是可脱离终端长期运行的后台服务。突然断电或 SIGKILL 可能使尚未导出的内存缓冲丢失。

## 6 固定网络

| 设备 | 网口 | 连接配置 | 地址 |
| --- | --- | --- | --- |
| PC | `enp4s0` | `jetson-direct` | `192.168.50.1/24` |
| Orin | `enP8p1s0` | `jetson-wired` | `192.168.50.2/24` |

Orin SSH 为 `jetson@192.168.50.2:22`。使用既有共享会话和身份，不在材料中保存密码或私钥。PC 网线连接不设置互联网默认网关和 DNS，外网由 `wlo1` 承担。不要在 PC 上操作 Orin 的 `enP8p1s0`。

只有出现直连故障时才检查：

```bash
ip -br link show enp4s0
ip -br -4 addr show enp4s0
ip -4 route get 192.168.50.2
ping -n -I enp4s0 -c 3 -W 1 192.168.50.2
ip neigh show 192.168.50.2 dev enp4s0
```

已有 `jetson-direct` 配置但未激活时，可按需执行：

```bash
sudo nmcli --wait 15 connection up id jetson-direct ifname enp4s0
```

当前无网络故障，无需更改配置。图形桌面缺失导致窗口启动失败，与 Orin 推理失败是不同问题。

## 7 阅读运行结果

新运行后先看根报告及 `simulator.cases`，确认实际状态、错误列表、周期数、期限超时、逐动作对照和子步审计，不以终端中的旧 RUNNING 或单独的推理耗时作为通过证据。

| 数据 | 用途与限制 |
| --- | --- |
| `report.json` | 该次运行的功能、实时、显示和退出汇总 |
| `case-*/timing.csv` | 每个正式控制周期的原始计时，首周期包含在内 |
| `case-*/state-data.npz` | 3,001 个状态：qpos[3001,35]、qvel[3001,32]、ctrl[3001,14] 等 |
| `case-*/native-model.mjb` | 原生模型快照，已知每份 116,664,189 字节 |
| `case-*/push10-trace.jsonl` | 扰动与回合记录 |
| `loop-transcript.jsonl` | 通信记录文件；文件 SHA 与协议链式摘要是不同概念 |
| 其他 JSON/CSV/日志 | 初始条件、逐子步审计、力计划、进程及依赖证据 |

NPZ/MJB 缺少完整 BAM 与积分器内部恢复状态，支持结果回放和核对，但不是可从任意一步无损恢复的完整仿真检查点。

## 8 本机清单已核对

本机只读采集已完成，229 项固定资产全部通过。当前无需再次执行采集器。该次结果保存在：

```text
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/final-archive-inventory-v1/20261025T083253Z-d68c1a99
```

采集器只读取既有文件并另存新报告，使用标准库及禁止网络协议的 Git 只读命令。该次没有运行仿真、推理或训练，没有连接 Orin、云主机或 GitHub。原始回传清单的文件身份与本机位置见 [证据索引](evidence/README.md)；其完整目录元数据 ZIP 不加入源码分支。

`LOCAL_PINNED_ASSETS_VERIFIED` 表示固定清单文件在采集时与期望一致。一般目录扫描达到 50,000 条上限，工作树有 132 项未跟踪条目；这两项范围边界已经记录，当前不为材料整理扩大扫描或重复核对已通过资产。完整异地备份尚未确认，不能只凭该状态认定可从空白系统恢复。

若以后文件迁移或恢复后明确需要重新核对，可使用原交付的 `microduck-stage12-archive-inventory-20261025.py --collect`；脚本 SHA256 记录在归档审查 JSON 中。该脚本仍保留在原下载资料内，不是本轮待执行任务。

## 9 发布与恢复规则

本目录为文档入库版本；导入方法见 [发布说明](publication.md)。文档增量包只更新源码中的说明和证据，随后由用户通过既有 SSH 身份推送；正常推送确认后不反复做远端核验。

新增大型资料上传遵循独立资料分支、仓库外缓存、完整提交 SHA 和 SHA256 核对；每个压缩包上限 49 MiB，大于上限的资产按既有方案分包或分卷。不将资料分支合并进代码分支。

完整规范：[本地提交与 SSH 推送](../handoffs/microduck-local-ssh-push-protocol.md)、[云 GPU 与传输](../handoffs/microduck-cloud-gpu-and-transfer-protocol.md)。规范中的 Stage09 主机、端口和旧状态是历史记录，不应复用于当前阶段。当前剩余云端计算为 NONE，5090 已由用户确认关闭。
