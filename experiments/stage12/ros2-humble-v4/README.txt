Microduck Stage12 — 下载进度与 SSH 诊断 v4
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

最新用户运行结果
PC已完成ROS Humble安装：HOST_SETUP_STATUS=HOST_SETUP_COMPLETE。
后续jetson@192.168.50.2提示密码后显示Connection closed by 192.168.50.2 port 22，SSH退出255。
目前不能断言是密码错误、服务器拒绝、认证后关闭还是连接模式问题。
Orin身份探针、依赖安装、ROS双机链路都尚未完成。
新版再次检查已有环境，可用即复用，PC无需重新下载已安装的Humble。
本次下载进度需求不改变最终模型、训练预算、观测或动作契约。

v4的下载与安装显示
APT通过专用终端输出通道保留原生进度，并把完整输出写入日志。
每批软件包先 --download-only 下载并校验，再用 --no-download 安装缓存。
下载前后的安装计划必须一致，仍拒绝删除、降级或GPU/内核环境变动。
屏幕显示示例（数字仅用于解释格式）：
  42% [100 ros-xxx 3100 kB/8200 kB 38%] 250 kB/s 2min 30s
开头是APT对当前批次的综合下载进度；中间是当前包字节量与百分比；
右侧B/s是当前APT下载任务的速率，随后是预计剩余下载时间。
该百分比由APT综合字节与包数量计算，不是整台电脑网卡流量，也不是全项目完成率。
首次几秒或等待网络时，APT可能暂不显示速率或ETA；不会虚构速度。
只有下载命令成功才显示“本批下载完成100%”，随后单独显示安装进度。
刷新软件索引的总量会动态变化，明确标注百分比仅为APT估计。
Orin上的pip同样显示文件量、速度、ETA，并按当前文件字节量补上百分比。
Pip完整缓存保存于 ~/.cache/microduck-stage12-pip；APT复用已有完整包，
未完成包由APT按服务器支持情况尝试续传，不保证所有服务器都支持断点续传。
保留原有源配置、VS Code源临时排除及命令级IPv4；不自动切换VPN或代理。
进度显示本身不能保证提高原软件源的下载速度。

已有任务与时间限制
若v2/v3仍在运行，新版先等它结束再继续；不会杀旧进程或删除锁文件。
执行系统包管理前检查APT/DPKG持锁情况，已有任务结束后自动继续。
下载步骤最长2小时，索引刷新最长30分钟；仍保留网络连接/数据超时和重试。
DPKG安装配置阶段不再使用20分钟墙钟强杀；此时Ctrl+C会请求在当前配置结束后停止。
安装期间继续保持终端和设备开启。电脑/Orin的每个下载批次单独显示进度。

SSH诊断
固定从192.168.50.1直连192.168.50.2:22，不通过SSH ProxyJump/ProxyCommand。
首先以jetson身份执行/usr/bin/true，再由ControlPersist保留共享会话；
后续命令和SCP复用该会话，保持每次新会话最多一次密码输入。
请输入Orin上jetson用户的登录密码，不能用电脑lx用户的sudo密码代替。
使用OpenSSH -vv -E保存连接和认证阶段日志，并嵌入最终report.json。
密码不由Python读取或保存；不改变SSH服务、用户密码、密钥或主机密钥检查。
认证前/认证后断开、权限拒绝、主机密钥失败分别记录，不将Connection closed直接归因于密码。
如仍失败，提供本版报告即可；真实服务端原因仍可能需要Orin本机日志。
可单独运行 --ssh-check：只做普通、非复用的SSH只读命令，生成报告，不安装软件。
正常继续任务使用 --repair，不必先做一次额外认证。

继承的APT修复
系统原sources.list及.list/.sources只读快照；仅在当前命令的副本中排除
packages.microsoft.com/repos/code。保留其他源、系统IPv6、代理、DNS及安全校验。
添加官方ROS源或Universe后刷新副本。保留现有索引缓存和错误报告。
所有新的安装仍以当前Ubuntu22.04与Python3.10为前提。

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
下载 microduck-stage12-ros2-humble-v4-20261024.py 到 /home/lx/下载。

python3 "/home/lx/下载/microduck-stage12-ros2-humble-v4-20261024.py" --install-code

git -C /home/lx/microduck-double-balance/workspace push origin double-balance

给 Orin 上电、等待启动，接原网线，两边 Wi-Fi 保持联网：

python3 "/home/lx/下载/microduck-stage12-ros2-humble-v4-20261024.py" --repair

只读诊断，不启用网络、不安装软件、不清理缓存：
python3 "/home/lx/下载/microduck-stage12-ros2-humble-v4-20261024.py" --diagnose

查看最近报告：
python3 "/home/lx/下载/microduck-stage12-ros2-humble-v4-20261024.py" --status

安装时按终端提示输入当前机器的 sudo 密码。SSH 继续复用已建立的会话；
若需要 Orin 系统依赖，会在 Orin 的同一终端会话中提示其 sudo 密码。
密码不会保存在脚本、日志或报告。安装期间保持终端和设备开启，不要中断 APT。
APT原生百分比与速度持续刷新；超过5秒没有输出时显示等待提示，所有输出同时保存到报告所指日志。

报告与状态
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-ros2-humble-v4/report.json
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
增量基于 v3 提交 772d07ffbb97c3216f5f71f15cb7461aff24248e。
新提交见 build-receipt.json；只增加 experiments/stage12/ros2-humble-v4。
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
43项CPU测试通过：ORT1.23.2上的合成LSTM连续推理、内存ROS API双端流程、
网络分类/原连接激活、APT模拟保护、资源不足与缓存清理条件、冻结产物身份。
新增APT源过滤、原文件保护/刷新、真实APT URI规划（不下载不安装）、错误嵌入及远端失败报告测试。
新增PTY、pip百分比、APT锁检测、SSH诊断及真实APT下载/缓存复用测试。
真实APT测试只访问临时本地HTTP仓库，不安装任何包；构建容器无_apt UID映射，测试夹具单独使用当前UID，生产代码保留系统APT沙箱。
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

单独SSH诊断（仅需要时执行）
python3 "/home/lx/下载/microduck-stage12-ros2-humble-v4-20261024.py" --ssh-check

进度与SSH官方依据
https://manpages.ubuntu.com/manpages/jammy/man8/apt-get.8.html
https://github.com/Debian/apt/blob/2.4.5/apt-private/acqprogress.cc
https://man.openbsd.org/ssh
https://man.openbsd.org/ssh_config
