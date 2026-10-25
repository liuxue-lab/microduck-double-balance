Microduck Stage12 — MuJoCo / Orin / ROS2 Humble 闭环 v1
2026-10-24；用户时区 Asia/Shanghai；运行目录时间戳采用 UTC。

本轮目的
电脑 RTX5060 运行原 MuJoCo 被控对象，Orin Nano Super 运行最终 ONNX 策略。
通过现有 ROS2 Humble / Fast DDS / 原网线完成两个固定回合。
仅仿真和冻结模型推理，不训练，不连接实体电机，不重新导出模型，不安装软件。

已核对的用户结果
reference/ros-link-passed.json 来自 report(20261024-055410).json。
SHA256: 1a7a0a02596bdc4a797a509c37cc5ed9c9e12cc986d37d1b7d00c2b047ccbecb
ROS_LINK_PASSED，128/128 动作和 LSTM 状态对比通过。
RTT 平均3.3517ms、p95 3.7460ms、p99 3.9989ms、最大13.0201ms，超过20ms为0次。
Orin推理p95 0.8111ms、最大9.3939ms；这些是固定输入链路检查的实测值。
最大动作误差1.430511474609375e-6，最大状态误差1.6689300537109375e-6。
PC系统Python3.10.12已能使用Humble，但没有ORT；本方案不需要PC ROS进程装ORT。
Orin实测使用/usr/bin/python3、Python3.10.12、ORT1.22.0、NumPy1.26.4、CPUExecutionProvider。
Orin Ubuntu22.04.5，L4T36.4.3，MAXN_SUPER；复用这个已经通过验证的环境。
本轮准备完成不表示已经在用户设备上完成物理闭环。

执行顺序（上一条成功后执行下一条）
下载 microduck-stage12-ros2-humble-loop-v1-20261024.py 到 /home/lx/下载。

python3 "/home/lx/下载/microduck-stage12-ros2-humble-loop-v1-20261024.py" --install-code

git -C /home/lx/microduck-double-balance/workspace push origin double-balance

python3 "/home/lx/下载/microduck-stage12-ros2-humble-loop-v1-20261024.py" --run

电脑与Orin保持开机和原网线连接；这是前台测试，保留运行终端。
5090不需要开启；运行闭环本身不需要互联网下载。Git push仍使用原SSH远端。
若SSH会话失效，会按普通SSH提示输入Orin上jetson用户密码一次，不保存密码。
本程序不自动推送GitHub，也不会改变网络、代理、SSH或系统依赖配置。

回合与预算
1. quiet-2407：原环境seed=2026102407，无外力，最多60仿真秒。
2. regression-2500：同一环境seed，扰动seed=2026102500，最多60仿真秒。
第二回合按当前机器人质量（不包含两个球）乘9.81乘0.1计算水平力。
在2、4、…、58秒各施力0.1秒，共29次；方向由固定私有随机序列产生。
作用于trunk_base惯性质心，世界水平坐标系，不额外施加扭矩。
两回合都是原最终模型已通过的回归案例，用于验证更换计算/通信路径后的行为。
不据此宣称新增随机序列泛化，不重跑原27回合验证，不重新选择模型。
每回合最多3000控制步/30000物理子步；首回合失败后不继续第二回合。
每回合墙钟上限900秒，仿真worker上限2000秒；控制器留60秒清理窗口。
远端Python服务器上限2200秒，独立timeout上限2230秒，随后最多5秒强制清理。
上限是保护边界，不是预计用时。120秒是总仿真时间，墙钟可能更长。
控制器会显示初始化心跳、已完成步数和真实进度百分比，不用时间伪造进度。

实时查看（另开一个电脑终端）
python3 "/home/lx/下载/microduck-stage12-ros2-humble-loop-v1-20261024.py" --watch

每2秒刷新，即使任务已结束也继续显示，Ctrl+C仅退出这个监控终端。
在运行主终端按Ctrl+C会停止本轮闭环并保存结果。

一次性查看：
python3 "/home/lx/下载/microduck-stage12-ros2-humble-loop-v1-20261024.py" --status

请求停止本轮：
python3 "/home/lx/下载/microduck-stage12-ros2-humble-loop-v1-20261024.py" --stop

普通--run只执行首个尝试。已有记录时显示结果，不自动重复测试。
仅在检查并修复具体原因、明确需要重新测试时：
python3 "/home/lx/下载/microduck-stage12-ros2-humble-loop-v1-20261024.py" --retry-run
重试建立新的独立运行目录，保留旧报告和轨迹；这仍不是训练。

数据流与模型契约
PC项目.venv/Python3.12保留原MuJoCo环境；PC ROS使用系统Python3.10。
二者用私有/tmp目录内的Unix socket逐请求交换，验证同一用户的连接凭据。
不混装Python3.10/3.12，不借用机械臂端口8765。
PC ROS与Orin继续使用Domain62、Fast DDS、可靠KEEP_LAST1、volatile、UDPv4。
Fast DDS仅使用PC 192.168.50.1/enp4s0 与Orin 192.168.50.2/enP8p1s0。
每次运行独立ROS话题名；同一时刻最多一个待返回动作。
消息绑定会话、模型SHA、回合、步号、reset和完整观测摘要。
拒绝非有限数、错误长度、跳号、重复、乱序、旧回复、非预期reset及超时动作。
客户端、ROS桥、Orin分别累计完整传输记录哈希；结束时三方必须完全一致。
PC逐步计算原冻结PyTorch CPU参考动作作数值核对；比较容差保持atol2e-5/rtol2e-4。
只有Orin返回且通过核对的动作进入仿真；没有本地动作回退。
策略LSTM的h/c逐步沿用，仅在两回合起点（全局序号0和3000）清零。
Orin每次启动有一次不作用于仿真的预热，之后清零状态；预热不计入控制步。

保留61D actor、85D critic、14D动作及原关节顺序/缩放/偏置。
观测归一化已经固化到ONNX，外部不再归一化；动作不额外裁剪或滤波。
actor[55:61]保持原头顶球传感观测，不补零：
stage11_head_material_frame中位置[20*x,25*y,50*(z-0.020)]、相对速度2*[vx,vy,vz]。
原500Hz模型边界/电流估计/PWM/扭矩检查、无外力清零检查和29次脉冲核对全部沿用。
原任务物理积分/接触/几何指标、auto_reset=False、仅time_out才有完整60秒验收保持不变。

计时结果如何理解
本轮是逐步等待返回动作的锁步仿真：仿真收到有效动作才前进一步。
单次请求1秒是功能测试的失败上限，不能解释为实体控制允许1秒延迟。
单次超过20ms会计数；达到1秒的过期动作被拒绝，不用旧动作填补，不自动重试。
报告分列ROS往返、含本机IPC的往返、Orin推理及含参考推理/500Hz审计的诊断周期耗时。
功能验收不以“多数包快于20ms”替代；hardware_acceptance和realtime_acceptance始终false。
即使每个ROS请求都小于20ms，也不能证明50Hz完整控制链硬实时或真实机械安全。
本轮不模拟真实电机在网络等待时继续运动的延迟影响；下一阶段需独立做时序/失联控制与实物传感接入。

结果与导出
主报告：
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-ros2-humble-loop-v1/report.json
完整历史：上述目录/runs/<UTC-编号>/。
每次目录包含simulator-report.json、bridge-report.json、orin-server-report.json、各进程日志、
loop-transcript.jsonl和case-01/02下的物理边界/扰动/几何轨迹与result.json。
主报告内嵌各节点结果和日志末尾；失败时优先提供这个report.json即可。
CLOSED_LOOP_FUNCTIONAL_PASSED：两回合完整通过，实时和实物验收仍待后续。
CLOSED_LOOP_FAILED_PRESERVE：首个失败已保留；按报告原因处理，不自动训练。
CONTROLLER_EXITED_PRESERVE：只读状态发现控制器已经退出，不自动重启或假定仍在运行。
本轮不删除任何旧模型、环境、报告或资料。

冻结身份
最终checkpoint final6h_005304.pt，10,041,821字节：
cf7ce9297672943687019afe071d94918d3eb2620c18a1e8fe689512f8c08111
ONNX final6h_005304.onnx，2,501,039字节：
8c1c1250b9b88d9a91e3eb095fea8bce21daa73c251b43f3cd8adc1327f03c4e
contract.json：e0283bf7f00c8e2c7b57128b91ca3ced3abb4698e8f4c4aa11e74a745ad84929
golden.json：b7ebe914c3933477d35bc0cc70c6b1bb4c1c83e67bd61dc22cd8f0775b1609e7
此前最终独立仿真报告SHA：d052f3cf8031be5ec5685fc1aa02c906301932d2a4849ac34fd450206b359228
最终模型heldout18/18；整个含早期候选的27回合是24通过，不能写27/27。
模型回传分支stage12-push10-final6h-return-20261023t181130z-1b79efbd-s002。
回传提交0cbee30e5a83904371076ca09a8cfb4afd1fa3f4；400文件/5归档已在电脑验证。
原训练完成5304次/消耗5305次，累计策略更新11314；原账本和365个源文件逐一校验。
全部云端原始大型轨迹并非均已回传；保留原回传清单的留云范围，不称全量备份。

Git与长期协议
固定仓库/home/lx/microduck-double-balance/workspace，下载目录/home/lx/下载。
分支double-balance；origin=git@github.com:liuxue-lab/microduck-double-balance.git。
本增量基于已运行v4提交c97435ac387e3886a17d0f737e9e79075589f605。
只新增experiments/stage12/ros2-humble-loop-v1；新提交与包身份见build-receipt.json。
保留已暂存/未暂存/未跟踪的修改；遇到忽略文件/符号链接/新提交冲突即停止，不覆盖。
不修改历史完成/硬件验收标签，模型归档不进入源码分支。
每次交接保留：
docs/handoffs/microduck-local-ssh-push-protocol.md
docs/handoffs/microduck-cloud-gpu-and-transfer-protocol.md
本轮无需云GPU；云端剩余必要计算/打包任务为无。
历史最后训练SSH为ssh -p 32185 root@connect.bjb2.seetacloud.com，仅供追溯，本程序不连接。
此前已提醒手动关闭5090；实际关机确认仍为UNCONFIRMED，不执行任何实例电源操作。

后续实体硬件仍待用户确认
14个XL330的ID/线序、串口转换器、供电和急停方式；IMU接口、安装朝向与标定。
头顶球相对位置/速度的真实传感来源仍未知；不会擅自使用零值替代。
已确认的电脑—Orin原网线连接可继续复用，以上未知项不阻挡本轮仿真。

本地构建验证边界
CPU测试包含合成61D归一化LSTM/14D动作的6000次连续请求，通过真实socketpair字节流、
内存ROS API替身和完整桥接处理链，每步对照PyTorch，验证两次reset与三方记录一致。
构建环境禁止创建新的AF_UNIX socket，测试的路径bind/listen/connect使用适配夹具；
socketpair、消息分片、EOF和SO_PEERCRED使用真实系统调用。
这些不是实际DDS/LAN/Jetson/MuJoCo运行证据；没有运行用户最终模型的真实物理闭环。
测试环境Torch2.9.1+cpu、ORT1.23.2、NumPy2.5.4；Orin复用已实测通过的ORT1.22.0。
原仿真审计生命周期/施力次数测试也保留。Python3.10语法和真实Git导入另外检查。
详细结果见validation-summary.json、cpu-tests.log、release-validation.json。
