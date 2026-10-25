Microduck Stage12 — PC contiguous command views v2 / 2026-10-24 (Asia/Shanghai)

最新失败报告与修复
report(20261024-110637).json：PC_COMMANDVIEW_FAILED_PRESERVE。
SHA256 f813c60aaccdd991285e24970d77282847a34634a15c5be695302c410b2020dc。
已运行提交f6fc8323f9168f0462d27339cb5857ab591f8107；package identity：
92025f2837aa998cb5a9f544c5d643b8787b2747fa02a454b2e4bd86445166df。
基准完成160控制步/1600物理步；平均24.785181ms、p95 25.998691ms。
command-views候选初始化时报Unexpected joint address type；0控制步/0物理步，未施加外力。
所有绑定恢复，无清理错误；原检查点、归一化、冻结文件和训练记录未改变。

原源码Entity._compute_indexing把joint_q_adr/joint_v_adr建成torch.int(int32)，
Actuator.initialize把target_ids建成torch.long(int64)。上版错误地要求所有字段为int64；
上版测试人工构造全int64索引，未覆盖原始初始化产生的混合类型。这是适配器兼容性遗漏。
v2分别检查每个字段，接受一维int32/int64索引，保留原dtype、地址和数值，不转换原索引。
不接受浮点、布尔、int16、非一维或非Tensor索引；失败信息注明字段、dtype、shape和device。
正常报告也新增各索引字段的类型元数据。切片映射、命令延迟、BAM和审计语义未改。

新增回归测试执行原Entity._compute_indexing与Actuator.initialize；仅用CPU元数据代替
编译后的MuJoCo模型，保留原初始化方法中的所有Tensor类型声明及映射构造。
原v1代码在这种混合索引下复现同一错误，v2通过；还覆盖三类索引的8种int32/int64组合，
确认索引对象、存储、类型和值没有被修改。原1600子步及两组完整worker测试也改用原始
初始化方法产生的混合类型，不再用手工全int64夹具掩盖问题。
启动前核验本次失败报告及此前成功报告，结果写到新的native-pc-commandview-v2目录。
实际GPU运行和性能仍待用户实测，不把CPU测试写成实测通过。

此前通过的短测与保留决定
report(20261024-105145).json：PC_READGRAPH_MEASURED。
SHA256 9a49846bd3aaf8724b8082c9cca51a59c402ca583d6cc7c450d6933ab99347c5。
已运行提交0079501811f03873729bc1f1f65169a54fcbb2a5；package identity：
ca031a1783d107eb3cab318fc77045283b07e5d542e8ee3f5da83fa0361a400c。
TorchArray类型错误已修复。两组各160控制步/1600物理步完成，73440个共同字段标量差异全部为0；
完整Audit、Trial和扰动内容完全相等。两份轨迹均1205288 bytes，SHA256：
2b7bdef6f9681dbf84d07c68d0cd7528e5580d09093a94a5c4b6b7e15b54f2b9。
基准平均24.800561ms、p95 26.050933ms、p99 26.291958ms；
只读图平均24.223366ms、p95 25.321920ms、p99 25.973757ms。
平均改善2.327%、p95改善2.798%，未达到固定5%门槛，因此KEEP_INFERENCE_ONLY。
两组150/150稳定计时步仍超过20ms；完整ROS闭环50Hz实时验收尚未完成。
每组1600次电机检查、1601次关节检查、1600次外力核对、50个施力物理步/一次完整脉冲。
所有边界通过，绑定恢复，原记录/模型/归一化/冻结文件未改变。新PPO=0，5090已关闭。

本轮候选
基准仍为已通过的InferenceOnly + PhaseCache + BAM Graph v2 + BatchedAuditReads。
未启用未达采用门槛的HeadGraph、PackedChecks或ReadGraph组合。
最新基准scene_write宿主时间约4.085ms/控制步，含同步，不等于纯GPU核函数时间。
核对mjlab原源码后发现每次get_command用多次高级索引提取五类命令字段：
position_target、velocity_target、effort_target、pos、vel。
新增command_views.py：仅在目标关节、组合qpos地址、组合qvel地址都连续时，使用当前
原始数组的基本切片视图，省去这些索引产生的临时Tensor和GPU索引工作。
索引由实际target_ids、joint_q_adr、joint_v_adr解析，支持非零起点，不硬编码14关节地址；
非连续/重排索引使用原get_command，不为了加速改变顺序。每次调用重新读取当前源数组。

语义与限制
返回的是当次调用使用的共享视图，不是可供跨步保存的历史快照。原Entity调用顺序仍为
get_command → apply_delay → compute → write_ctrl。
保留原命令延迟缓冲append/compute及全部随机数调用；原CircularBuffer复制当前值进历史。
保留已通过的BAM Graph输入暂存copy_、上次转矩独立快照、摩擦写入和新鲜诊断字典。
没有替换BAM公式，没有新CUDA图，没有物理预热、策略改动、奖励改动或训练。
原500Hz电机/关节边界、全量外力检查、异常停止、仅清除自身外力的规则全部保留。
每组前40次视图读取分别与原get_command五类输出进行FP32位模式精确比较。
只读比较不推进延迟历史、不消耗随机数，不增加物理积分。相同NaN交给原Audit处理。
每次核验索引存储和命令源形状/精度/设备，以及原delay和已接受BAM消费入口。
索引内容在初始化和关闭时严格比对；这些映射在已核对源码的单回合中固定，禁止重置。
源数据的存储可刷新，但每次重读当前值；不把前一个物理步的位置当作新状态。
源文件不符、同输入不一致或映射改变则保留失败；不安装升级依赖来绕过检查。

短测与采用条件
inference-only基准、command-views候选各160控制步/1600物理步，总320/3200。
固定环境种子2026102407、外力种子2026102500；每组3.2仿真秒，在2.0–2.1秒施加一次
10%自重随机水平力。这里只是优化短测，不是完整60秒抗扰验收，也不追加训练。
比较22类共同字段，原尺寸共73440个标量；另要求完整Audit、Trial和扰动内容完全一致。
共同字段不含已在两组中省去的critic和训练奖励。零奖励Tensor明确标为wrapper占位。
前10控制步仅排除在稳定计时之外，仍记录所有物理、轨迹及检查，并另报全部启动耗时。
必须数值与审计对照通过，且环境步平均耗时、p95都至少改善5%，才建议COMMAND_VIEW_CANDIDATE；
否则继续KEEP_INFERENCE_ONLY。不会自动启用生产控制。
环境步计时含CUDA完成，不含策略推理、网络和对照轨迹写盘；小于20ms也不等于整个ROS闭环达标。

离线验证
80项CPU测试通过，含原始TorchArray/WarpBridge、原Entity调用方法、原ActuatorCmd/get_command、
原apply_delay、完整原CircularBuffer/DelayBuffer、BAM算术及CPU Audit。
新增检查覆盖五类共享命令值、当前状态存储刷新、非零索引起点、非连续索引原路径、
命令延迟历史和随机数状态、BAM暂存快照、源/schema/消费入口改变、同输入损坏和清理。
完整1600子步比较原Entity→原延迟→BAM→原物理生命周期、全部Audit与扰动记录；
并测试两组worker各160控制步的完整生命周期和逐步状态相等。
此工作区没有CUDA；BAM图只做固定地址CPU替身测试，不把CPU结果当作新的GPU性能证据。
真实Git安装保全检查结果见release-validation.json。

运行：现在需要用户在电脑执行
下载microduck-stage12-pc-commandview-v2-20261024.py到/home/lx/下载，上一条成功后再执行下一条：
python3 "/home/lx/下载/microduck-stage12-pc-commandview-v2-20261024.py" --install-code
git -C /home/lx/microduck-double-balance/workspace push origin double-balance
python3 "/home/lx/下载/microduck-stage12-pc-commandview-v2-20261024.py" --run

只需电脑RTX5060和既有项目环境，5090保持关闭，无需连接Orin；主终端保持开启。
预计1–3分钟，每组worker上限230秒；每5秒心跳，打印步数和百分比。
报告：/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-pc-commandview-v2/report.json
发回PC_COMMANDVIEW_REPORT指向的report.json。
PC_COMMANDVIEW_MEASURED表示本轮短测和对照完成，仍需看recommendation。
COMMANDVIEW_SAMPLE_COMPLETE/EPISODE_INTERRUPTED是160步计划停止，内嵌完整60秒验收false属预期。
PC_COMMANDVIEW_PARITY_FAILED_PRESERVE/PC_COMMANDVIEW_FAILED_PRESERVE保留失败，不自动补跑。
已有本轮报告时--run不重做；--retry-run才显式重跑。主运行终端Ctrl+C中断并保留结果。
另开终端实时查看：
python3 "/home/lx/下载/microduck-stage12-pc-commandview-v2-20261024.py" --watch
每2秒刷新，结束后也继续；观察终端Ctrl+C仅退出观察。--status一次查看，--stop请求停止。

Git与文件保全
父提交f6fc8323f9168f0462d27339cb5857ab591f8107，新提交及包身份见build-receipt.json。
只新增experiments/stage12/pc-commandview-v2；旧程序、模型、历史报告全部保留。
保留已暂存、未暂存、未跟踪、忽略文件；新增路径冲突/符号链接/分支分歧时停止。
不reset/stash/clean/force push。运行包从固定Git对象提取，不误用被修改的工作树代码。
用户用已有Git SSH身份推送；本工作区没有远端写入，两个长期协议随包保留。

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
