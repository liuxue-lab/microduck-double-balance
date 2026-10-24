Microduck Stage12 — PC scalar Audit + packed reads v1 / 2026-10-24 (Asia/Shanghai)

最新性能分析报告
report(20261024-113718).json：PC_PROFILE_MEASURED，522072 bytes。
SHA256 e2feba442176d65e67b321925b41dd622296fdc79e80ca45fafadc8f4eaae809。
已运行提交ff5a687ff6e6fd80a99479baec50a2caedc23147；包身份
76c1297484d861f18079d140b377fe2e61289bc6fcfab91ce10266e7e8f4e8c3。
两组各160控制步/1600物理步；73440个共同状态标量的差异全部为0，
完整Audit、Trial、扰动内容相等。所有绑定恢复，worker退出码0，无清理错误，
原模型、归一化、冻结文件、训练记录未变。
两份状态轨迹均1205288 bytes，SHA256
2b7bdef6f9681dbf84d07c68d0cd7528e5580d09093a94a5c4b6b7e15b54f2b9。
每组1600次电机检查、1601次关节检查、1600次外力核对、50个施力物理步/一次完整脉冲。

分析结果与限制
正常基准平均24.707238ms、p95 25.932201ms、p99 26.525860ms；
详细分析平均118.513867ms、p95 120.623145ms，含cProfile巨大开销，不能当真实速度。
150个分析窗口中，原Audit.tolerance调用231000次、excursion调用84000次。
这些调用次数可用于选择优化点；被分析器放大的耗时不能直接推算正常瓶颈占比。
正常基准物理步连同外力和边界检查10.223231ms/控制步，scene_write 4.050898ms，
仍是含异步等待的宿主区间，不是纯GPU核函数时间。150/150正常计时步超过20ms。

当前保留与本轮候选
保留InferenceOnly + PhaseCache + BAM Graph v2 + BatchedAuditReads。
CommandViews已修复并数值通过，但平均慢1.12%、p95慢0.75%，不采用。
ReadGraph数值通过但提速不到5%，不采用；HeadGraph也未采用。
此前PackedChecks把完整外力核对与审计数据合并读取，单独提速约3.18%，未达5%门槛。
本轮只把这条已经通过数值/故障检查的读取路径与新的标量辅助函数优化组合测试，
不把以前未采用的单项写成已采用，也不宣称组合一定会达到20ms。

scalar_audit.py的精确范围
原model_limits.py SHA256
d318cdb3b14b078797e69dc89da15f4106f79779fbb764eb59d137708cfdd275。
原packed_checks.py SHA256
dccbfd6805d848f8b4ba72514af7d6a24792c63155e2c063d9edff98b860d4bd。
这两份源码均未修改。为当前Audit实例复制observe_joints、before_integration、
after_integration的原Python代码对象，只给它们提供私有tolerance/excursion辅助函数。
用简单标量比较替代每次创建生成器与展开参数，保留同一IEEE double运算、原数值容差、
原阈值、所有统计字段、事件优先级、计数和停止顺序；不改模块或类的全局方法。
原tolerance在没有有限参数时会报TypeError，该失败行为也保留，不静默返回一个容差。
实例已有覆盖方法、源码或常量不符则停止并保留；不升级依赖绕过检查。

前40次积分前、40次积分后CPU审计，用相同已读回的数据再执行一个独立原Audit，
比较完整实例状态、返回值、异常类型/事件，浮点按float64位模式精确比较。
这些比较在前10控制步的排除计时区间内结束，不增加物理积分、GPU读取或随机数调用。
全部1600物理步继续执行原500Hz电机、关节、外力检查；无降采样、无异步延后。
PackedChecks保持先检查已有外力，再写自身外力，积分前/后完整核对，最后只减去自身外力。
全量外力使用精确相等，无epsilon；NaN或额外力不会被吞掉。
原BAM、命令延迟、独立几何探针、动作顺序、模型与归一化均保留。

本工作区CPU证据
101项CPU测试；其中新增随机float64位模式、正负零/NaN/无穷/无界范围、空参数，
数值容差两侧相邻浮点值、饱和/越界/再入统计、18类审计异常、源码/实例覆盖拒绝，
同输入完整状态损坏检测与恢复。1600子步组合测试比对原完整Audit和一次外力脉冲；
7类物理/取消故障比对停止阶段、计数、外力清理和原事件；两组worker各160步完整对照。
CPU中的CUDA图为固定地址替身，本工作区没有CUDA，不能把这些写成RTX5060实测。
tests/benchmark_scalar_audit.py用报告中的真实14关节/电机参数和确定性合成宿主输入，
不带cProfile，7轮交替顺序，只测CPU Audit。每轮排除100对预热审计，计时1500对；
按每10对折算，中位数原版1.375478ms、标量版0.921588ms，减少33.00%。
每轮完整Audit状态位模式相等。此数值不含GPU、读取打包、物理、策略或通信，
不能当用户电脑整步提升33%或50Hz通过；详细样本见cpu-benchmark.json。

用户电脑短测与采用条件
inference-only基准、audit-fast组合各160控制步/1600物理步，总320/3200。
固定环境种子2026102407、外力种子2026102500；每组3.2仿真秒，2.0–2.1秒
施加一次10%自重随机水平力。不是新的60秒抗扰验收，也不追加PPO。
两组均关闭cProfile，用相同轻量计时包装记录完整环境步、原积分入口、Python Audit、
积分前后读取与审计等区间。嵌套时间不直接相加，仍不等于纯GPU时间。
完整步包含原有CUDA完成等待，排除模型推理和外层对照轨迹记录，保留原步内审计日志。
前10步排除计时；逐步比较22类共同状态字段与完整Audit/Trial/扰动。
只有对照通过，且平均耗时与p95都至少降低5%，才标记AUDIT_FAST_CANDIDATE；
否则KEEP_INFERENCE_ONLY。两者都不自动激活生产，也不等于完整ROS闭环实时验收。

现在需要用户运行
下载microduck-stage12-pc-auditfast-v1-20261024.py到/home/lx/下载。
按顺序执行，上一条成功后再执行下一条：
python3 "/home/lx/下载/microduck-stage12-pc-auditfast-v1-20261024.py" --install-code
git -C /home/lx/microduck-double-balance/workspace push origin double-balance
python3 "/home/lx/下载/microduck-stage12-pc-auditfast-v1-20261024.py" --run

只需电脑RTX5060和现有环境；5090保持关闭，无需连接Orin，主运行终端保持开启。
预计1–3分钟，每组worker上限230秒，打印步数、百分比和5秒心跳。
报告：/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-pc-auditfast-v1/report.json
发回PC_AUDITFAST_REPORT指向的report.json；PC_AUDITFAST_MEASURED只表示短测和对照完成。
AUDITFAST_SAMPLE_COMPLETE/EPISODE_INTERRUPTED为160步计划停止，内嵌60秒验收false属预期。
PC_AUDITFAST_FAILED_PRESERVE/PC_AUDITFAST_PARITY_FAILED_PRESERVE保留失败，不自动补跑。
已有报告时--run不重做；--retry-run才显式重跑。主运行终端Ctrl+C停止并保留结果。
另开终端实时查看（每2秒刷新，结束后继续；Ctrl+C仅退出观察终端）：
python3 "/home/lx/下载/microduck-stage12-pc-auditfast-v1-20261024.py" --watch
--status单次查看；--stop请求停止本轮任务。

Git交接与固定模型
父提交ff5a687ff6e6fd80a99479baec50a2caedc23147；新提交与包身份见build-receipt.json。
仅新增experiments/stage12/pc-auditfast-v1；保留旧程序、模型、全部历史记录。
保留暂存/未暂存/未跟踪/忽略文件；新增路径冲突、符号链接、分支分歧时停止。
不reset/stash/clean/force push。运行包从固定Git对象提取，保留工作树中的用户改动。
仍由用户使用既有SSH身份推送，本工作区无远端写入。

阶段仍为脚踩篮球、球直接接触原头壳；已取消Stage13/14地面轨迹行走任务。
final6h_005304.pt：10041821 bytes；5304次新增已完成更新，累计11314。
SHA256 cf7ce9297672943687019afe071d94918d3eb2620c18a1e8fe689512f8c08111
final6h_005304.onnx：2501039 bytes；SHA256
8c1c1250b9b88d9a91e3eb095fea8bce21daa73c251b43f3cd8adc1327f03c4e
contract.json e0283bf7f00c8e2c7b57128b91ca3ced3abb4698e8f4c4aa11e74a745ad84929
golden.json b7ebe914c3933477d35bc0cc70c6b1bb4c1c83e67bd61dc22cd8f0775b1609e7
Final validation d052f3cf8031be5ec5685fc1aa02c906301932d2a4849ac34fd450206b359228
此前真实ROS功能报告615d9b29e1c85433eebec2a988031ba01f14fc53ffeebf9906ae9355e6ac28e5：
两种固定场景各60仿真秒通过，但完整周期约39.9ms，未通过50Hz实时要求。
128导出金标准、actor61/critic85/action14、h/c=[1,1,256]、原关节顺序与归一化保留。
无新滤波/裁剪，不重新导出模型，不新增量化模型，不追加PPO。

连接、协议和剩余事项
ROS2 Humble、Domain62、Fast DDS；PC192.168.50.1/enp4s0/jetson-direct；
Orin Nano Super192.168.50.2/enP8p1s0/jetson-wired；ssh jetson@192.168.50.2。
网线直连无网关/DNS，WiFi上网。Orin Python3.10.12/ORT1.22.0 CPU；
PC项目Python3.12、Torch2.9.1、mjlab1.3.0；运行前核验，不安装升级依赖。
真实电机适配器/端口、伺服零位/方向、IMU安装/时间戳、头球感知、硬件watchdog
仍缺实物证据，不能编造；本程序没有电机I/O。
两份长期规范完整保留：
docs/handoffs/microduck-local-ssh-push-protocol.md
docs/handoffs/microduck-cloud-gpu-and-transfer-protocol.md
本地仓库/home/lx/microduck-double-balance/workspace；下载/home/lx/下载。
分支double-balance；远端git@github.com:liuxue-lab/microduck-double-balance.git。
回传分支stage12-push10-final6h-return-20261023t181130z-1b79efbd-s002；
回传提交0cbee30e5a83904371076ca09a8cfb4afd1fa3f4，400 files/5 archives已核验。
不能把已回传摘要写成云端全部原始轨迹已备份，完整范围以既有索引为准。
剩余云端任务NONE；云端电源OFF_USER_CONFIRMED，2026-10-24用户确认，精确时间未知，
未向平台核实。历史ssh -p 32185 root@connect.bjb2.seetacloud.com仅作追溯，不连接。
常见Git冲突/认证/下载文件改名按两份规范处理；不改HTTPS/PAT，不强制推送。
本次无电脑↔云端或电脑↔Orin传输；只由用户通过Git SSH推送本轮源码。
下一步审查组合加速的实测结果，通过采用门槛后再安排真实ROS完整周期验收；当前没有硬件验收结论。
