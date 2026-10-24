Microduck Stage12 — PC head kinematics graph v1 / 2026-10-24 (Asia/Shanghai)

最新报告审查
report(20261024-093949).json：PC_AUDIT_MEASURED。
SHA256 0a7e6af238c531b3c176c436b9223f4d41c01769a71873dbd2ad052305253485
已运行提交843d9e5719e3b57e64072b4c52b7b421197fbae7，原记录未改变。
两组各160控制步/1600物理步，87200项数值完全一致；完整Audit、Trial和脉冲记录也一致。
两份轨迹SHA256：b2670b65d14c724a538487b837a32c0c926b52148c283dff8ef67ad59f2d9c89。
原BAM Graph：平均28.323174ms，p95 29.520274ms，p99 29.977392ms。
合并检查读取：平均27.423385ms，p95 28.621475ms，p99 28.996667ms。
平均改善3.18%、p95改善3.04%，低于原先均值和p95各至少5%的采用门槛。
所以KEEP_BAM_GRAPH：新一轮基准保留v2已验证BAM Graph，未加入PackedChecks。
候选150/150稳定计时步仍超过20ms；不把这次正确短测说成50Hz实时验收完成。
包括启动的160步最大值：基准63.558292ms，合并读取61.923355ms。
500Hz电机/关节检查和外力检查完整；每组仅施加一次完整0.1秒、10%自重水平外力。
5090关闭，新增PPO=0；原模型、归一化、ROS配置和训练记录保留。

本轮候选
保持原阶段内缓存、BAM Graph、原积分后批量读取不变。
新增head_graph.py，将原 _stage11_head_kinematics 的算术录入CUDA Graph，降低
重复提交小型GPU运算的开销。仍调用原 stage11_head_material_values 和原
matrix_from_quat，没有改写公式、混合精度、算子融合、动作裁剪或滤波。

每次调用从当前场景读取6组数据：头部参考位姿、上球位置/线速度/角速度、参考点线速度/
角速度，共22个FP32值；一次复制到独立固定输入。即使实体属性的存储地址变化，仍读取
当前值，不让图捕获冻结旧的属性对象。输入形状、精度、设备和图缓冲区地址逐次校验。
图在只含这些输入的隔离场景副本上预热/捕获；不推进物理、不改当前场景、随机数或策略。
9个图输出每次复制成新快照后返回，避免后续回放覆盖此前的缓存或调用者保留的结果。
图只用于已有manager阶段内缓存。阶段之外的独立几何探针仍重新调用原始实时场景计算，
不会用图输出验证自身；阶段之间不复用缓存，物理/forward/reset/scene-write仍使缓存失效。

源码核验
原MDP SHA256 9f970ae07c6912873fc44ee1850b20629fabc82cb20d3a95cf08f5cbd3f81e76。
mjlab1.3.0 math.py SHA256 47757748386173cc4ded5fb5dc5619336393ee6296769fa8b985738ebe50d436。
核验原模块和四元数函数绑定；不相符则保留停止，不为此更新依赖。
来源：https://github.com/mujocolab/mjlab/blob/v1.3.0/src/mjlab/utils/lab_api/math.py
PyTorch CUDA Graph说明：https://docs.pytorch.org/docs/2.9/notes/cuda.html#cuda-graphs
候选首次40次头部计算用同一份暂存输入与原函数对照（360标量）；原atol=2e-5、rtol=2e-4。
每组BAM最初40子步的独立同输入对照继续保留。所有图预热都没有额外物理积分。

固定运行范围与判断
顺序执行bam-graph基准、head-graph候选各160控制步/1600物理步，总320/3200。
环境种子2026102407、外力种子2026102500，每组3.2仿真秒，在2.0–2.1秒施加一次完整
10%机器人自重水平随机方向外力。不是完整60秒验收，没有自动重试或额外训练回合。
两组保留全部原奖励、actor/critic、统计、终止条件、500Hz边界检查和外力核对。
逐控制步比较actor/critic/action/reward/done、qpos/qvel、力矩/外力、指标/计数器、摩擦/
阻尼及全部BAM诊断。新图应有640次回放、640份输出快照、160次独立原始探针计算。
前10个控制步仅从稳定耗时统计排除，仍保留物理/检查/轨迹；报告另含全部160步启动耗时。
只有数值通过且平均值、p95均改善至少5%时才建议HEAD_GRAPH_CANDIDATE，否则KEEP_BAM_GRAPH。
不自动启用生产控制；环境步计时不包括策略推理/网络和对照轨迹写盘，低于20ms也不能
代替后续完整ROS/物理闭环实时测试。实际提速未测，不预先承诺达标。

离线验证
41项CPU测试通过，包含原源码数学160组动态输入、live数据不变、独立探针、旧结果保留、
属性存储更新、形状/精度/缓冲区变动拒绝、故障输出拒绝、原BAM回归、500Hz检查/施力
原流程对照和两组worker生命周期。CPU使用固定地址覆盖替身，不能代替实际CUDA捕获。
真实Git安装保留检查结果见release-validation.json；本工作区没有运行新候选GPU物理。

运行（现在需要用户在电脑执行）
下载microduck-stage12-pc-kinematics-v1-20261024.py到/home/lx/下载，依次执行：
python3 "/home/lx/下载/microduck-stage12-pc-kinematics-v1-20261024.py" --install-code
git -C /home/lx/microduck-double-balance/workspace push origin double-balance
python3 "/home/lx/下载/microduck-stage12-pc-kinematics-v1-20261024.py" --run

仅需电脑RTX5060和既有项目环境；无需连接Orin，5090保持关闭。保持主运行终端开启。
预计1–3分钟，每组worker上限230秒；每5秒心跳/进度百分比。不安装升级依赖、不清理旧文件。
已有本轮报告时--run不重做；--retry-run才显式创建新目录再做固定两组，不自动重试。
另开终端实时显示：
python3 "/home/lx/下载/microduck-stage12-pc-kinematics-v1-20261024.py" --watch
--watch每2秒刷新，结束后也保持刷新；观察终端Ctrl+C仅退出观察。
--status查看一次，--stop请求停止并保留；主运行终端Ctrl+C中断本轮。

结果
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-pc-kinematics-v1/report.json
把PC_KINEMATICS_REPORT指向的report.json发回即可。
PC_KINEMATICS_MEASURED表示短测和轨迹对照通过，还需看recommendation是否建议保留候选。
PC_KINEMATICS_PARITY_FAILED_PRESERVE / PC_KINEMATICS_FAILED_PRESERVE保留原因和日志。
KINEMATICS_SAMPLE_COMPLETE / EPISODE_INTERRUPTED表示160步主动停止，内嵌60秒验收false是预期。

Git交接
父提交843d9e5719e3b57e64072b4c52b7b421197fbae7；新提交/包身份见build-receipt.json。
只新增experiments/stage12/pc-kinematics-v1；保留旧源码、图修复和全部冻结报告。
暂存/未暂存/未跟踪改动均保留，忽略文件/符号链接路径冲突或分支分歧时停止。
不reset/stash/clean/force push。运行包取自固定Git对象，不运行已编辑的工作树内容。
用户使用现有Git SSH身份推送；本工作区无远端写入。两份长期协议随包保留。

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
