# Microduck Stage12 总项目交接：ROS2 Humble + Orin 硬件在环

更新日期：2026-10-24，Asia/Shanghai。
当前阶段：PC 实时性能定位；待用户运行 pc-timeline-v1。项目尚未最终验收。

## 1. 最新范围决定（后续交接必须保留）

用户明确要求：“去掉5 6没有真机演示 继续下一步 并声称交接文档”。
这里按“生成交接文档”执行。

最终交付范围是：电脑运行 Microduck 双球平衡仿真，通过有线 ROS2 Humble
与 Jetson Orin Nano Super 交换观测和动作，Orin 执行固定 FP32 ONNX 策略，
完成硬件在环功能、实时性、故障处理和可复现交付。

以下项目已从剩余任务中移除，不是等待以后补做：

- 真实机器人传感器与 14 个伺服电机的接入、标定、控制接口。
- 真机任务验收、安全试验及真机演示。

本范围没有电机 I/O，不需要提供舵机端口、零位或 IMU 安装信息。
Orin 仍是硬件在环的真实推理设备；“没有真机演示”不等于取消 Orin。
Stage13/14 地面轨迹行走仍已取消。任务仍为脚踩篮球、顶球直接接触原头壳。
不追加训练；不重新导出模型；5090 保持用户确认的关闭状态。
历史源码/报告中的 hardware_acceptance=false 记录保留，不修改成真机通过。
历史文档中的真机待办不再定义本次交付范围，以本节最新用户决定为准。

## 2. 已完成内容与证据边界

| 内容 | 已有证据 | 不能据此宣称 |
|---|---|---|
| 最后一次 6 小时预算训练及模型回传 | final6h_005304.pt；5304 次新增完成更新，累计11314；400文件/5归档核验 | 云端全部原始轨迹均已备份 |
| 最终保留验证 | heldout 18/18：2组无外力、16组随机抗扰，每组60秒 | 所有随机序列必通过；整个campaign 27/27（实际24/27） |
| 抗扰条件 | 10%机器人自重的水平随机力；2,4,…,58秒各施力0.1秒；每回合60秒 | 真机抗扰通过，或严格连续接触通过 |
| ONNX 导出 | 归一化烘焙；128金标准；actor61、critic85、action14；LSTM h/c=[1,1,256] | 可以改关节顺序、补滤波/裁剪或重新归一化 |
| ROS2 有线通信及功能闭环 | 2个固定场景各3000控制步/30000物理步；6000次ONNX对照；CLOSED_LOOP_FUNCTIONAL_PASSED | 50Hz实时闭环已通过；原完整周期约39.9ms |
| Orin 推理与通信测量 | FP32推理平均0.352815ms，p95 0.388907ms；JSON调度到验证平均2.593678ms，p95 3.052529ms | 通信回放达到要求就代表完整仿真闭环达到要求 |
| 通信故障注入 | 延迟、静默、重复消息测试通过；超期动作丢弃、不自动重启、不复用旧动作 | 真实电机硬件故障验收 |

保留验证允许位置漂移，strict_contact_acceptance=false。
顶球保持/模型边界的有限测试通过，不表述为严格持续接触或全随机总体保证。
量化、剪枝均为可选方法：INT8 已测到动作误差而被拒绝（最大约0.04030）；
没有启用剪枝或未经验证的 TensorRT FP16。目前保留 JSON + FP32。

## 3. 剩余四项任务

| 顺序 | 任务 | 完成条件/下一动作 |
|---|---|---|
| 1 | PC 实时性能优化 | 先完成本轮时间线诊断，定位计算、传输、提交和同步成本；再选择改动。保持原状态、500Hz边界检查和扰动顺序；以不带分析器的对照判断收益。 |
| 2 | 最终 ROS2/Orin 实时硬件在环验收 | 使用最终选定代码、模型与有线连接；重新运行完整60秒场景，核对动作、双球保持、模型边界、扰动时序与截止处理；报告完整周期均值/p95/p99/最大值/超过20ms次数，依既定50Hz截止门槛判定。 |
| 3 | 部署定版 | 固定版本与哈希，统一启动/状态/停止/恢复/日志入口，验证重启与异常中断后保留记录；明确哪些任务能退出终端继续、哪些需保持主终端。 |
| 4 | 最终材料与归档 | 汇总功能、性能、故障、对照及范围证据；生成电脑仿真与Orin运行状态的演示材料、复现说明、依赖/文件清单；完成阶段提交及最终交接。无需真机演示。 |

顺序1的短程分析不是新的抗扰验收，不以其中的3.2秒替代60秒。
50Hz控制预算为20ms，但完整闭环还包含通信、Orin推理及其他开销；
不能只证明PC环境步低于20ms，也不能只看均值，就宣称端到端实时通过。
本轮不降低控制频率、物理频率或检查频率来制造达标结果。

## 4. 最新性能结论与本轮选择

最新报告 report(20261024-120056).json：PC_AUDITFAST_MEASURED。
SHA256 c46fe69f73ad7f8c23e610af5b904a723aed61107719a220bb04e68a938c831d
已运行代码提交：5d80475823dfc41e662f62752e5919d5523ac5bc。
包身份：6e80fb13e0f421999d724a9918259dbc727ea21f831a76157a15b756c0594f2c。

| 方案 | PC环境步均值(ms) | p95(ms) | 结论 |
|---|---:|---:|---|
| 已保留推理方案 | 25.034763 | 26.213595 | 作为基准；150/150计时步超过20ms |
| ScalarAudit + PackedChecks | 24.158626 | 25.389597 | 均值改善3.50%、p95改善3.14%，均未到5%采用门槛；不采用 |

该次两组各160控制步/1600物理步；73440个共同状态标量差异全部为0；
完整Audit/Trial/扰动相同；退出、恢复、模型和历史记录检查通过。
数值通过与性能采用是不同结论。KEEP_INFERENCE_ONLY，未自动激活生产方案。

保留运行组合：InferenceOnly + PhaseCache + BAM Graph v2 + BatchedAuditReads。
之前 CommandViews 修复类型问题后仍略慢；ReadGraph、PackedChecks单项未达到采用门槛；
HeadGraph没有收益；这些都没有因本轮诊断而重新启用。

上轮cProfile把均值从约24.71ms放大到118.51ms，不能用其时间占比推断正常瓶颈。
普通宿主计时也可能把CUDA等待算到后续读取调用；积分入口返回快不代表GPU积分计算快。
因此本轮收集CPU/CUDA时间线，暂不再押注一个未经定位的小幅优化。

## 5. 本轮 pc-timeline-v1 的具体行为

两个独立进程各160控制步/1600物理步：inference-only 与 timeline。
固定环境种子2026102407、外力种子2026102500；各3.2仿真秒，包含2.0–2.1秒
一次完整10%自重水平扰动。总计最多320控制步/3200物理步，无新PPO。

两组使用同一已保留运行组合。第二组仅在控制步101–108开启 PyTorch 2.9
CPU/CUDA profiler；不启用cProfile、shape、stack或内存分析。
依据官方接口：https://docs.pytorch.org/docs/2.9/profiler.html

详细时间线标记原环境步、物理积分、审计读回与Python检查、场景写入、动作延迟、
BAM计算、观测等。汇总裁剪到8个环境步窗口；原始文件可能含步间策略推理和外层
对照数据回读，但这些步间工作不进入汇总。环境步保留原有CUDA完成等待。

报告分别给出kernel、memcpy、memset、CUDA runtime、CPU算子和嵌套范围。
重叠GPU事件按区间并集计算，不能直接把所有核函数时长相加当作耗时。
“窗口内未记录到CUDA活动的时间”不能直接当纯Python时间，也不能证明GPU空闲。
分析器仍有开销，timeline组耗时不用于收益判断；无分析器基准组提供正常耗时参考。

采用判定始终 KEEP_INFERENCE_ONLY；本轮是定位工作，不自动启用新方案。
逐控制步对照22类共同状态，要求全部数值差为0，并核对完整Audit/Trial/扰动。
全部500Hz外力、电机和关节检查照常运行，不降采样、不延迟审计。
如果CUPTI/CUDA活动缺失、窗口不足、检查计数不足或出现异常，保留报告并判失败；
不会安装依赖、请求sudo、连接云端或自动重试来掩盖问题。

## 6. 现在的运行方法（用户电脑）

下载 microduck-stage12-pc-timeline-v1-20261024.py 到 /home/lx/下载。
按顺序执行，上一条成功后再执行下一条：

```bash
python3 "/home/lx/下载/microduck-stage12-pc-timeline-v1-20261024.py" --install-code
git -C /home/lx/microduck-double-balance/workspace push origin double-balance
python3 "/home/lx/下载/microduck-stage12-pc-timeline-v1-20261024.py" --run
```

本轮只需电脑RTX5060与现有项目环境；无需连接Orin，5090保持关闭。
预计数分钟；每个worker墙钟上限230秒，控制边界上限180秒。
打印步数、百分比及每5秒心跳。主运行终端保持开启；不要把它当后台服务。
本轮没有下载/上传模型，所以百分比是运行进度，不是网络传输速度。

另开终端实时查看（每2秒刷新，任务结束后仍刷新，Ctrl+C只退出观察）：

```bash
python3 "/home/lx/下载/microduck-stage12-pc-timeline-v1-20261024.py" --watch
```

--status 单次查看；--stop 请求停止本轮所属任务并保留结果。
--run 遇到已有报告不会自动重跑；--retry-run 是用户显式重跑入口，不用于无证据重试。
主运行终端Ctrl+C会中断本轮任务；断开后先用--status检查，不凭旧RUNNING判断存活。

完成后优先发回：
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-pc-timeline-v1/report.json

成功诊断状态为 PC_TIMELINE_MEASURED，表示两组对照和时间线完成，不表示项目验收通过。
完整trace保留在报告directory对应的 timeline/cuda-trace.json；摘要和哈希已嵌入report.json。
若报告缺失，发回 PC_TIMELINE_REPORT 指向的报告及终端最后的错误，不重新训练。

## 7. 固定环境与模型

PC仓库：/home/lx/microduck-double-balance/workspace
下载目录：/home/lx/下载
代码分支：double-balance
Git SSH远端：git@github.com:liuxue-lab/microduck-double-balance.git
本轮父提交：5d80475823dfc41e662f62752e5919d5523ac5bc。
本轮生成的完整提交、包身份与文件哈希见 review 包中的 build-receipt.json。
本轮用户安装、推送、RTX5060实测尚待执行；未创建 stage12-complete 标签。

PC项目Python3.12，系统ROS Python3.10；ROS2 Humble、Domain62、Fast DDS。
PC项目Torch2.9.1、mjlab1.3.0、mujoco3.10.0、mujoco-warp3.8.1、warp1.12.0。
Orin Nano Super：Ubuntu22.04.5、L4T36.4.3、MAXN_SUPER、Python3.10.12、ORT1.22.0 CPU。

网线直连：PC enp4s0 / jetson-direct / 192.168.50.1/24；
Orin enP8p1s0 / jetson-wired / 192.168.50.2/24。
SSH：jetson@192.168.50.2:22。无有线网关/DNS，WiFi用于上网。
ROS2可靠传输、depth1、volatile、有线接口限定；后续Orin操作沿用既有共享SSH会话。

冻结导出目录：
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-ros2-humble-v1/frozen-export

| 文件/证据 | 大小 | SHA256 |
|---|---:|---|
| final6h_005304.pt | 10041821 | cf7ce9297672943687019afe071d94918d3eb2620c18a1e8fe689512f8c08111 |
| final6h_005304.onnx | 2501039 | 8c1c1250b9b88d9a91e3eb095fea8bce21daa73c251b43f3cd8adc1327f03c4e |
| contract.json | 4041 | e0283bf7f00c8e2c7b57128b91ca3ced3abb4698e8f4c4aa11e74a745ad84929 |
| golden.json | 1606400 | b7ebe914c3933477d35bc0cc70c6b1bb4c1c83e67bd61dc22cd8f0775b1609e7 |
| 最终保留验证报告 | — | d052f3cf8031be5ec5685fc1aa02c906301932d2a4849ac34fd450206b359228 |
| ROS2功能闭环报告 | — | 615d9b29e1c85433eebec2a988031ba01f14fc53ffeebf9906ae9355e6ac28e5 |
| Orin与通信优化报告 | — | 7186dcaf36345244e1d01792bd4af769538fc2030669db69b9c5d95d2da8960a |
| 已保留推理运行报告 | — | 3b7a74f36e2c9f68b8c98aa42489519ca5ff352eadc09ea02861839d63c60f14 |

固定关节顺序、BAM模型、延迟/RNG、归一化与递归状态约定全部保留。
不添加动作滤波或裁剪。无论后续采用何种优化，都须核对相同状态与完整审计。

## 8. 云端、回传和长期协议

剩余云端任务：NONE。云端电源：OFF_USER_CONFIRMED。
证据：用户在2026-10-24明确说“5090已关机”；精确关机时间未知，未向平台核实。
历史提醒时间与实际关机确认不是同一件事，不补造提醒或平台回执时间。
历史训练实例 ssh -p 32185 root@connect.bjb2.seetacloud.com 仅用于追溯；不得探测或启动。
本轮不产生电脑↔云端或电脑↔Orin传输，也不删除旧数据。

已核验回传分支：stage12-push10-final6h-return-20261023t181130z-1b79efbd-s002
回传提交：0cbee30e5a83904371076ca09a8cfb4afd1fa3f4
范围：已核验400文件/5归档；完整范围以既有索引为准，不声称云端所有原始轨迹均已备份。

两份长期规范完整附在本交接所在目录，后续交接须继续引用：

- docs/handoffs/microduck-local-ssh-push-protocol.md
- docs/handoffs/microduck-cloud-gpu-and-transfer-protocol.md

源码经增量Git bundle导入固定仓库，再由用户按既有Git SSH身份推送。
模型、日志和完整轨迹放仓库外；获批的大文件回传使用独立传输分支与固定SHA核验，
不合并进double-balance，不改HTTPS/PAT，不force push。

## 9. 常见错误与保留原则

| 现象 | 处理 |
|---|---|
| 新增目录与未跟踪/忽略文件或符号链接冲突 | 停止导入，保留原文件和install-report；不覆盖、reset、stash或clean |
| 本地分支有新提交或non-fast-forward | 保留双方提交，先审计分歧，不force push |
| GitHub要求密码/PAT | 核对固定SSH远端；沿用现有SSH key与agent，不改网页登录 |
| 下载文件带(1)等后缀 | 核对实际文件名，使用正确完整路径；不要运行同名旧脚本 |
| Unexpected captured input type: qpos | 历史ReadGraph v1把TorchArray误当Tensor；已在v2修复，但v2提速不足未采用 |
| Unexpected joint address type | 历史CommandViews v1误拒int32地址；v2保留int32/int64原类型修复，但速度不足未采用 |
| CUDA profiler activity unavailable / 时间线没有GPU事件 | 保留报告和日志；不当作GPU耗时为0，不自动装依赖或重跑；先审阅实际环境/CUPTI错误 |
| TIMELINE_SAMPLE_COMPLETE / EPISODE_INTERRUPTED | 160步有界诊断的计划停止；不是60秒验收失败，也不是完整任务成功 |
| --watch 返回终端 | 确认运行本轮脚本并带--watch；本轮监控即使完成也持续每2秒刷新 |
| No route to host（后续Orin步骤） | 检查网线和192.168.50.1/24、192.168.50.2/24连接；不要为本轮PC诊断重装ROS |

安装器保留暂存、未暂存、未跟踪和忽略文件；运行包从固定Git对象提取，不覆盖用户工作树改动。
一次正常push成功输出足够，无异常不重复全套远端验证。
测试证据见review包中的validation-summary.json、cpu-tests.log和release-validation.json。
本工作区没有CUDA，CPU测试不能冒充RTX5060或Orin实测；本轮实际性能结论等待用户报告。
