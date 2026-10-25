Microduck Stage12 — Humble APT 修复 v3
2026-10-24，用户时区 Asia/Shanghai

已经确认完成的部分
用户报告：report(20261024-042649).json
报告 SHA256: 0c06c8a8ee83da9c153f7575cf93f1d6fdbcacf0e5fa0ddc173eb0835049e609
EXPORT_PARITY_PASSED：128 次连续动作/LSTM h/c 对比，2 次 reset。
最大动作绝对误差 1.430511474609375e-6，最大状态误差 9.5367431640625e-7。
actor、归一化、原始训练记录未变；新增 PPO=0。
导出的 final6h_005304.onnx 为 2,501,039 字节，SHA256：
8c1c1250b9b88d9a91e3eb095fea8bce21daa73c251b43f3cd8adc1327f03c4e
原 checkpoint SHA256：
cf7ce9297672943687019afe071d94918d3eb2620c18a1e8fe689512f8c08111
本轮直接复用这个 ONNX、contract.json 和 golden.json；不重新导出或训练。

本次已定位的失败
用户报告：report(20261024-045746).json，随后提供 command-003.log。
电脑的 apt-get update 退出100，唯一报错源为：
https://packages.microsoft.com/repos/code
HTTPS握手失败，报错IP为IPv6 2620:1ec:46::74；其他已列出的源取得了索引。
不能据此断言IPv6本身或证书错误是根因。Humble安装尚未开始。
本次PC有线carrier=1，ping Orin 2/2成功，TCP22可达；SSH身份验证和DDS尚未完成。
可用RAM约12.1GB，磁盘余量约143.5GB，本次不需要删除文件。

v3的针对性处理
每次APT更新读取系统实际配置的sources.list及.list/.sources，创建本次运行专用副本。
只从副本排除主机packages.microsoft.com、路径/repos/code的启用条目；
支持混合文件、多URI、多行deb822和内嵌Signed-By，其他条目保留。
不删除、改名或禁用系统原VS Code源，不更换Ubuntu镜像，不改全局IPv6/代理/DNS。
APT更新、候选查询、模拟、安装都使用同一份副本和命令级Acquire::ForceIPv4=true。
安装官方ROS源配置包或启用Universe后，会重新快照并更新，纳入新增源。
保留系统APT列表缓存，不清除被临时排除源的已有索引；关闭这几条命令的二进制缓存写入。
继续严格检查所有参与源的索引更新结果，不使用trusted=yes或跳过TLS/GPG校验。
这解决本次Humble流程被无关源阻塞的问题，不表示VS Code源自身的TLS问题已经修好。

安装输出会实时显示并保存到command-NNN.log。总report.json同时嵌入每条命令的
退出码、日志路径、SHA256、输出末尾最多64KiB及错误上下文，超长输出明确标注截断。
APT或pip失败、命令超时、Orin端安装失败也能留下具体错误。sudo密码仍仅在终端输入。
未在用户电脑执行安装，也未实际连到Orin；需用户运行后才能判断环境和ROS链路通过。

保持的连接方式（已对照用户补充的连接指南）
PC: lxlab / enp4s0 / jetson-direct / 192.168.50.1/24
Orin: yahboom / enP8p1s0 / jetson-wired / 192.168.50.2/24
SSH: jetson@192.168.50.2，端口22
ROS 2 Humble / ROS_DOMAIN_ID=62
网线直连，Wi-Fi 继续提供互联网，直连网口无默认网关和 DNS。
原指南原文保存在 reference/verified-wired-connection.md。
仅在存储属性与指南一致且连接未就绪时启用已有 jetson-direct。
不创建或修改网络连接，不执行 connection down，不改 Wi-Fi、DNS、防火墙或 SSH 配置。
若仍不通，脚本完成电脑环境安装后保留报告，给出 Orin 本机检查提示。

本轮环境处理
电脑已有可用 Humble 时直接复用；缺少时按官方方法安装 ROS-Base、rclpy、
std_msgs 与 rmw_fastrtps_cpp。需要时启用 Ubuntu Universe，并安装官方
ros2-apt-source 1.3.0~jammy 小型源配置包；包已经下载并校验 SHA256：
110b9a462d55252decb8b7c816f61c2ba0d9890ce5fb93ac504e97cae5860d76
未打包完整 ROS 安装包；APT/PyPI 下载需要两端保留可用互联网连接。
安装前运行 APT 模拟，拒绝包删除、降级以及 NVIDIA/CUDA/内核/其他 Python
版本变动。实际安装使用 --no-remove；不运行全系统 upgrade 或 autoremove。
若系统原本有未完成 DPKG 操作，或参与安装的其他源签名/网络错误，保留具体日志停止。

Orin 优先复用已有兼容环境。需要新建时使用：
/home/jetson/venvs/microduck-stage12-humble-v2
解释器 /usr/bin/python3（既有 Python3.10），包含系统站点包以读取 ROS 的依赖。
仅在这个新环境中安装 onnxruntime==1.23.2、numpy==1.26.4。
1.24.4 的官方 wheel 要求 Python>=3.11，不能直接用于 Humble 的 Python3.10。
1.23.2 的 Python3.10/aarch64 wheel 已核实存在并下载验证。
旧项目 .venv 的 Python3.12、PyTorch、CUDA、JetPack 和机械臂环境不升级、不删除。
双机继续使用已有冻结图和逐步 golden 数据做数值一致性检查，不能只凭版本号
宣布部署通过。CPUExecutionProvider、FP32；本轮不切换 TensorRT/GPU 图优化。

磁盘与 RAM
只读记录 MemAvailable、swap、根目录/用户目录磁盘余量和可用 inode。
常规检查至少预留 128MiB 磁盘、512MiB 可用 RAM。
需安装时保守预留 PC 2GiB / Orin 1GiB 磁盘。
磁盘/inode 不足时，只自动执行 apt-get clean，删除可重新下载的 APT 缓存。
RAM 不足会要求关闭不用的应用，不删磁盘文件来冒充解决 RAM。
不删除训练 checkpoint、返回归档、报告、视频、Git 数据、源码、旧运行环境。
这些文件部分仍被固定 SHA 校验引用；如果缓存清理后仍不足，先保留清单和报告，
根据实际占用再选择可以去掉的冗余副本，不盲删整个 artifacts。

电脑执行（按顺序，上一条成功再执行下一条）
下载 microduck-stage12-ros2-humble-v3-20261024.py 到 /home/lx/下载。

python3 "/home/lx/下载/microduck-stage12-ros2-humble-v3-20261024.py" --install-code

git -C /home/lx/microduck-double-balance/workspace push origin double-balance

给 Orin 上电、等待启动，接原网线，两边 Wi-Fi 保持联网：

python3 "/home/lx/下载/microduck-stage12-ros2-humble-v3-20261024.py" --repair

只读诊断，不启用网络、不安装软件、不清理缓存：
python3 "/home/lx/下载/microduck-stage12-ros2-humble-v3-20261024.py" --diagnose

查看最近报告：
python3 "/home/lx/下载/microduck-stage12-ros2-humble-v3-20261024.py" --status

安装时按终端提示输入当前机器的 sudo 密码。SSH 继续复用已建立的会话；
若需要 Orin 系统依赖，会在 Orin 的同一终端会话中提示其 sudo 密码。
密码不会保存在脚本、日志或报告。安装期间保持终端和设备开启，不要中断 APT。
APT更新和安装输出实时显示；每5秒附带经过时间和日志路径，安装输出同时纳入报告。

报告与状态
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-ros2-humble-v3/report.json
每次历史报告另存 runs/<UTC时间-随机编号>/，不会覆盖 v1 的导出结果。
ROS_LINK_PASSED：128 组动作/状态对比通过，报告含 RTT/推理延迟/20ms 超时数。
PC_READY_ORIN_CONNECTION_PENDING：电脑环境已处理，需按报告提示检查 Orin 的物理
连接、jetson-wired 或 SSH 服务。恢复后再次 --repair 会复用已完成部分。
ENVIRONMENT_PENDING / REPAIR_FAILED_PRESERVE：报告给出具体未通过项和安装记录。
DIAGNOSIS_COMPLETE：仅只读诊断完成，不表示 ROS 链路通过。
新报告仍保留 hardware_acceptance=false、realtime_acceptance=false。
128组固定输入链路检查不是 MuJoCo 闭环、更不是实物机器人验收。

Orin 无法远程到达时，按原指南在 Orin 本机终端检查：
sudo nmcli connection up "jetson-wired"
ip -4 -br addr show enP8p1s0
sudo systemctl status ssh --no-pager
目标地址应为 192.168.50.2/24；不要把 192.168.8.88 当成当前地址。
只有确认 SSH 服务未启动时，才在 Orin 本机执行 sudo systemctl start ssh。
本轮脚本不会在不可达的 Orin 上假装已执行任何操作。

后续还需确认的实物信息
Microduck 14个XL330实际ID/线序/串口转换器/供电/急停，IMU接口与安装标定。
当前61维观测的[55:61]是头顶球相对位置/速度，实物测量来源仍未确认。
本轮不驱动电机；硬件信息不阻挡冻结模型和双机推理链路检查。
链路通过后再接 PC MuJoCo 被控对象、Orin 策略的闭环，检查50Hz时序与超时行为。

代码与云端交接
固定源目录：/home/lx/microduck-double-balance/workspace
固定下载目录：/home/lx/下载
分支 double-balance，origin git@github.com:liuxue-lab/microduck-double-balance.git
增量基于 v2 提交 a9f7e33a24147d64e5c4d4ab9134b86385295a19。
新提交见 build-receipt.json；只增加 experiments/stage12/ros2-humble-v3。
保留原暂存/未暂存/未跟踪文件，拒绝覆盖忽略文件、符号链接路径或分歧提交。
由用户按现有 SSH 身份 push，本脚本不远端写入 GitHub。
继续保留两份协议：
docs/handoffs/microduck-local-ssh-push-protocol.md
docs/handoffs/microduck-cloud-gpu-and-transfer-protocol.md
模型回传提交 0cbee30e5a83904371076ca09a8cfb4afd1fa3f4，400文件已验证返回。
训练最终完成5304次/消耗5305次；最终模型独立仿真18/18通过。
不重开PPO预算、不修改旧账本、不重跑27回合验证。
云端剩余任务：无；之前已提醒手动关机，实际电源状态仍未获确认。
不需要开启5090，不执行任何实例启停或自动关机。

验证边界
36项CPU测试通过：ORT1.23.2上的合成LSTM连续推理、内存ROS API双端流程、
网络分类/原连接激活、APT模拟保护、资源不足与缓存清理条件、冻结产物身份。
新增APT源过滤、原文件保护/刷新、真实APT URI规划（不下载不安装）、错误嵌入及远端失败报告测试。
内存替身测试中的RTT不是Jetson实测。Git导入验证结果见 release-validation.json。
本构建环境没有安装用户机器的系统软件、执行真实LAN/DDS、清理用户文件或训练。

官方核对来源
https://github.com/ros2/ros2_documentation/blob/humble/source/Installation/Ubuntu-Install-Debs.rst
https://github.com/ros2/ros2_documentation/blob/humble/source/Installation/_Apt-Repositories.rst
https://github.com/ros-infrastructure/ros-apt-source/releases/tag/1.3.0
https://pypi.org/project/onnxruntime/1.23.2/
https://pypi.org/project/onnxruntime/1.24.4/

APT配置官方依据
https://manpages.ubuntu.com/manpages/jammy/man5/apt.conf.5.html
https://manpages.ubuntu.com/manpages/jammy/man5/sources.list.5.html
https://ubuntu.com/pro-client/docs/en/docs/howtoguides/use_custom_apt_config/
