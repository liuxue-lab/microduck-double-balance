Microduck Stage12 — PC packed audit reads v1 / 2026-10-24 (Asia/Shanghai)

这次报告的结论
report(20261024-092151).json：PC_GRAPH_MEASURED；v2 缓冲区修复已在电脑 GPU 上通过。
报告 SHA256 430e9c17573ae07f670d3b8975b21fe537eec3575bf56352533941e3dece50ca
已验证提交 f7a95ee1b89f5c123c3e3281bc23f69660213669。
两组各160控制步/1600物理步，87200项标量对照全部误差0；两份轨迹 SHA256 同为
b2670b65d14c724a538487b837a32c0c926b52148c283dff8ef67ad59f2d9c89。
全部边界/外力检查通过；每组完整施加一次10%自重、0.1秒水平随机方向外力。
150个稳定计时样本（排除前10步计时，未跳过物理或检查）：
phase-cache 平均32.746659ms，p95 34.001211ms，p99 34.265983ms；
bam-graph   平均28.426304ms，p95 29.737554ms，p99 30.112746ms。
平均改善13.19%，p95改善12.54%；候选150/150步仍超过20ms，未通过50Hz完整闭环实时要求。
包括启动在内的160步候选最大67.609420ms；不能把排除预热后的结果当作启动耗时。
BAM图1600次回放；前40个子步同输入比对4600标量、最大误差0，递归缓冲区独立。
原模型/归一化/训练记录未改；5090关闭；本次无Orin/云端连接或电机I/O。

本轮改动
保持已验证的阶段内缓存和 BAM CUDA Graph 原文件完全不变。
增加 packed_checks.py：每个物理子步把完整 xfrc_applied 与积分前电机诊断一起读取，
积分后把完整 xfrc_applied 与直接 qpos/qvel、实际 actuator_force 一起读取。
完整外力的比较改为读取后的逐值精确相等，不设epsilon，不漏掉其他物体或力矩。
不再逐子步创建全零 expected GPU 张量；固定 CPU 计划来自本回合实际 FP32 外力向量。

原零外力检查仍在写入前执行；如果已有不明外力则保留停止，不能覆盖抹除。
原 CPU Audit 完整保留，每2ms积分前查电压/速度/电流/PWM/力矩映射，积分后查实际
关节/力矩；关节越界仍永久记录，电气/非有限值仍立即停止；不延后到控制步末检查。
动作、奖励、actor/critic、终止条件、物理参数、独立几何探针和随机序列均保留。
每个子步外力与边界检查的显式读取点，从单电机组下的5处合为3处；其他读取不计在内。
此为实现层面的读取次数减少，实际耗时收益仍需本机 GPU 测量，不能预先宣称达标。

保留原 EvaluationSession 的施力/脉冲计数/积分/撤销顺序；原源码未替换。
新适配器仅在本轮候选 worker 内使用。异常时仅撤销自身施加外力，不清掉未知外力。
原500Hz Audit、BAM诊断新鲜度、源文件/模型哈希核验及完整状态对照全部保留。
此次没有新量化/剪枝或重新训练。此前 INT8 超差、binary 通信平均耗时变差，仍保留
已接受的 FP32 + JSON；小模型推理与网络不是这次PC环境步超时的主要开销。

固定运行范围
顺序运行 bam-graph 基准与 packed-audit 候选，各160控制步/1600物理步，总320/3200。
两组都用阶段内缓存+BAM Graph；基准仍使用v2的积分后批读取，候选增加本轮打包读取。
环境种子2026102407、外力种子2026102500；每组仿真3.2秒，其中2.0–2.1秒有一次
完整10%自重水平随机方向外力。每组独立创建环境，不自动重试，不运行额外60秒回合。
前10步只从稳定耗时统计排除；报告同时保留全部160步计时、图捕获/同输入检查开销。
全状态轨迹涵盖actor/critic/action、奖励、done、qpos/qvel、ctrl、实际力矩、完整外力、
指标/计数器、摩擦/阻尼、BAM全部诊断及上一时刻力矩。atol=2e-5、rtol=2e-4不变。
只在数值通过且均值和p95都改善至少5%时建议候选；否则保留已验证BAM Graph。
不自动激活生产控制。环境步低于20ms仍不等于包含ROS、推理、日志和调度的全环达标。

本工作区验证
33项 CPU 测试通过：包含原 BAM 数学/缓冲区回归、原完整 Audit 对照、1600子步一次
完整脉冲、非顺序索引/多个电机组、异常外力保留、积分前后失败、越界后重入仍留记录、
取消、精度/微小外力拒绝，以及两组worker轨迹/计数/生命周期对照。
真实Git安装保留检查结果见 release-validation.json。
CPU测试中积分器和CUDA捕获使用替身，只证明适配器的数据和生命周期行为。
这里没有执行新候选真实GPU物理或测速；下一条本机运行才生成实际证据。

运行（现在需要用户在电脑执行）
下载 microduck-stage12-pc-audit-v1-20261024.py 到 /home/lx/下载，依次执行：
python3 "/home/lx/下载/microduck-stage12-pc-audit-v1-20261024.py" --install-code
git -C /home/lx/microduck-double-balance/workspace push origin double-balance
python3 "/home/lx/下载/microduck-stage12-pc-audit-v1-20261024.py" --run

仅使用电脑RTX5060及既有项目环境；无需连接Orin，5090保持关闭。主运行终端保持开启。
预计约1–3分钟，每组worker硬上限230秒；每5秒心跳/进度百分比，不安装或升级依赖。
至少256MiB空闲磁盘；不删除旧报告或文件。已有本轮报告时 --run 不自动重做。
--retry-run 才显式新建目录重做固定两组短测试；失败不会自动换算法或隐瞒错误。
另开终端实时显示：
python3 "/home/lx/下载/microduck-stage12-pc-audit-v1-20261024.py" --watch
--watch每2秒刷新，完成后仍刷新；观察终端Ctrl+C仅退出观察。
--status查看一次，--stop请求停止并保留结果；主运行终端Ctrl+C会中断本轮。

结果路径
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-pc-audit-v1/report.json
把 PC_AUDIT_REPORT 指向的 report.json 发回即可。
PC_AUDIT_MEASURED：短测与数值通过；还要看recommendation是否建议保留候选。
PC_AUDIT_PARITY_FAILED_PRESERVE / PC_AUDIT_FAILED_PRESERVE：保存失败原因和原始日志。
AUDIT_SAMPLE_COMPLETE / EPISODE_INTERRUPTED 是160步主动停止；内嵌60秒验收false是预期。
新记录使用native-pc-audit-v1，原graph-v2报告/轨迹不改动。

Git交接
父提交f7a95ee1b89f5c123c3e3281bc23f69660213669，新提交/包身份见build-receipt.json。
仅新增experiments/stage12/pc-audit-v1，保留旧源码和全部冻结记录。
已暂存/未暂存/未跟踪改动保留；忽略文件/符号链接冲突或分支分歧时保留停止。
不reset/stash/clean/force push；实际执行包来自固定Git对象，不执行已编辑的工作树文件。
本工作区不向远端写入。用户继续用现有Git SSH身份推送，不改HTTPS/PAT认证。

固定模型与项目
阶段仍为脚踩篮球、球直接接触原头壳；不恢复已取消 Stage13/14 地面轨迹行走任务。
Checkpoint final6h_005304.pt，10041821 bytes，5304 次新增已完成更新，累计 11314。
SHA256 cf7ce9297672943687019afe071d94918d3eb2620c18a1e8fe689512f8c08111
ONNX final6h_005304.onnx，2501039 bytes。
SHA256 8c1c1250b9b88d9a91e3eb095fea8bce21daa73c251b43f3cd8adc1327f03c4e
contract.json SHA256 e0283bf7f00c8e2c7b57128b91ca3ced3abb4698e8f4c4aa11e74a745ad84929
golden.json SHA256 b7ebe914c3933477d35bc0cc70c6b1bb4c1c83e67bd61dc22cd8f0775b1609e7
Final validation SHA256 d052f3cf8031be5ec5685fc1aa02c906301932d2a4849ac34fd450206b359228
先前真实 ROS 功能闭环报告 SHA256 615d9b29e1c85433eebec2a988031ba01f14fc53ffeebf9906ae9355e6ac28e5
128 导出金标准、原 61D/85D/14D 与 h/c=[1,1,256]、BAM XL330、原动作顺序与归一化保留。
仍无动作滤波/裁剪，原模型从 frozen-export 保留；本次不重新导出，不生成新量化模型。

既有连接/环境仅留作后续使用：ROS 2 Humble，Domain62；PC192.168.50.1，
Orin Nano Super192.168.50.2，SSH jetson@192.168.50.2:22。
PC enp4s0/jetson-direct；Orin enP8p1s0/jetson-wired；直连无网关/DNS。
Orin 已测 Python3.10.12/ORT1.22.0 CPU；本次 PC 项目环境此前为 Python3.12、
Torch2.9.1、mjlab1.3.0；运行前核验，不为此安装升级。
真实电机适配器与端口、伺服零位/方向、IMU安装与时间戳、头球感知和硬件 watchdog
仍未获得足够实物证据，不能编造。此程序没有电机 I/O。

两份长期交接规范保留在包中：
docs/handoffs/microduck-local-ssh-push-protocol.md
docs/handoffs/microduck-cloud-gpu-and-transfer-protocol.md
Local repository /home/lx/microduck-double-balance/workspace；downloads /home/lx/下载。
Branch double-balance；origin git@github.com:liuxue-lab/microduck-double-balance.git。
回传分支 stage12-push10-final6h-return-20261023t181130z-1b79efbd-s002；
回传提交 0cbee30e5a83904371076ca09a8cfb4afd1fa3f4，400 files / 5 archives 已核验。
是否包含所有云端原始轨迹以既有索引为准，不把已回传摘要写成全部原始数据已备份。
本阶段剩余云端任务 NONE；云端电源 OFF_USER_CONFIRMED。历史 SSH
ssh -p 32185 root@connect.bjb2.seetacloud.com 只作追溯，不连接、不复核当前实例。
常见 Git 冲突/认证/下载文件改名问题仍按两份规范处理；不改 HTTPS/PAT，不强制推送。
本次无电脑↔云端或电脑↔Orin数据传送；只通过既有 Git SSH 推送源码。
