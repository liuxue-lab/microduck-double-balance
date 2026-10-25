Microduck Stage12 — ROS 2 Humble / Orin Nano Super 下一步准备
日期：2026-10-24（用户时区 Asia/Shanghai）

本轮目标
冻结已通过本地仿真验证的 final6h_005304.pt，使用原始 mjlab 导出路径生成
包含观测归一化的 ONNX；比较连续推理动作与 LSTM 状态；检查原有网线连接上
两台机器的 Humble 环境。条件齐全后执行 128 组双机推理对比并测量延迟。
这是推理链路检查，不是完整的 MuJoCo 闭环/HIL 验收，也不驱动真实电机。
没有新增 PPO、没有重新随机扰动选模，没有操作 5090。

已经完成的依据
最终候选：final6h_005304.pt，10,041,821 字节。
SHA256: cf7ce9297672943687019afe071d94918d3eb2620c18a1e8fe689512f8c08111
累计策略更新 11,314；本轮完成 5,304，消耗 5,305 次。
本地最终模型选模 3/3；独立验证 18/18（2 个无外力，16 个随机扰动序列）。
整个选模过程共 27 回合、24 回合通过，不能说所有候选都通过。
每回合 60 秒，10% 机器人自重水平外力，每 2 秒持续 0.1 秒，29 次脉冲。
只代表这组仿真验证通过，不代表所有随机序列、严格接触验收或硬件验收通过。
已核对报告 SHA256:
d052f3cf8031be5ec5685fc1aa02c906301932d2a4849ac34fd450206b359228

固定连接（恢复自之前的《Jetson-Orin-Nano-有线直连与SSH连接指南》）
电脑 lxlab: enp4s0 / jetson-direct / 192.168.50.1/24
Orin yahboom: enP8p1s0 / jetson-wired / 192.168.50.2/24
SSH: jetson@192.168.50.2，端口 22
ROS 2: Humble，ROS_DOMAIN_ID=62
网线直连；两边 Wi-Fi 继续负责互联网。直连网口不设置默认网关或 DNS。
脚本不修改 NetworkManager、路由、DNS、SSH 配置、默认 Python 或 .bashrc。
DDS 仅本次进程使用 Fast DDS UDPv4 + 对应有线 IP 白名单，不改系统配置。
不占用另一个机械臂项目的 TCP 8765 端口，不使用其 7 维动作协议。

环境隔离
原 Microduck 项目 .venv 为 Python 3.12，继续用于 mjlab 和 ONNX 导出。
Humble 节点使用已有 Python 3.10 + /opt/ros/humble/setup.bash。
探针读取 /usr/bin/python3；Orin 还检查历史已有
/home/jetson/venvs/robomimic-hil/bin/python，但不改动该环境。
本步使用 ONNX Runtime CPUExecutionProvider、FP32、单推理线程，测量实际耗时。
未假定 CUDA/TensorRT 版本兼容，不自动安装/升级 JetPack、PyTorch 或全局软件。
若现有环境缺包，报告明确列出缺项；读取报告后再准备针对当前机器的安装步骤。
不会为了通过检查而改用其他 ROS 发行版或将 rclpy 混装到 Python 3.12。

在电脑运行（无需启动云端 5090）
1. 下载 microduck-stage12-ros2-humble-v1-20261024.py 到 /home/lx/下载。
2. 导入本轮增量代码（只增加 experiments/stage12/ros2-humble-v1）：
   python3 "/home/lx/下载/microduck-stage12-ros2-humble-v1-20261024.py" --install-code
3. 用户按原协议推送代码：
   git -C /home/lx/microduck-double-balance/workspace push origin double-balance
4. 打开 Orin，接好原来的网线，电脑和 Orin 保持唤醒；运行：
   python3 "/home/lx/下载/microduck-stage12-ros2-humble-v1-20261024.py" --prepare

首次 SSH 可能要求正常的主机指纹确认及一次密码认证，后续命令复用 SSH 会话。
不保存密码、不新增密钥、不关闭主机密钥验证。仅通过这条既有 LAN SSH/SCP
传送 ONNX、固定输入和检查代码，不连接云 GPU，也不向任何人发送消息。
本次运行是前台任务，请保留终端。Ctrl+C/SIGHUP/SIGTERM 会中止并保留报告。
导出最多 20 分钟；Orin 推理服务有独立 90 秒上限，无自动重试。
进度每 5 秒刷新一次，详细输出存入报告指向的 export.log / ros-client.log。

只做双机只读探针：
   python3 "/home/lx/下载/microduck-stage12-ros2-humble-v1-20261024.py" --probe
查看最近结果：
   python3 "/home/lx/下载/microduck-stage12-ros2-humble-v1-20261024.py" --status

结果位置
根目录：
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-ros2-humble-v1
report.json: 最近一次总报告；每次报告/日志另保存在 runs/<时间-随机编号>/。
frozen-export/: 仅在导出和一致性比较全部通过后建立，含：
  final6h_005304.onnx — 烘焙归一化的原始 LSTM 推理图
  contract.json — 实际观测排列、动作缩放/偏置、关节顺序、输入输出及状态约定
  golden.json — 128 组连续数值对比输入及原 PyTorch 的动作/h/c 结果
  export-report.json — SHA256 清单、容差、误差和冻结状态证明
Orin 新增路径：/home/jetson/microduck-stage12/ros2-humble-v1/runs/<随机会话>/

状态解释
EXPORT_AND_ROS_LINK_PASSED:
  最终模型导出及双机数值一致性通过。报告含 RTT、推理耗时、20ms 超时次数。
  即使 128 个样本均低于 20ms，也不宣布实时 HIL/真实机器人已经验收。
EXPORT_PASSED_ENVIRONMENT_PENDING:
  已完成导出；Humble/ONNX Runtime/直连条件尚有缺项。请提供 report.json。
PREFLIGHT_BLOCKED / PREFLIGHT_READY:
  仅 --probe 的环境结果，尚未导出或执行双机推理。
PREPARATION_FAILED_PRESERVE:
  停止并保留现场。提供 report.json 和相应日志，不重训、不覆盖原记录。

原始数据保护与恢复
保留 365 个原源码 SHA256、训练账本、原候选、原验证报告。导出前后复核。
重复 --prepare 会复核并复用已通过的 frozen-export，不重复导出已完成模型。
失败的 export-staging 留在该次 runs 目录，后续不会将它当成合格模型。
若安装提示分支分歧或路径冲突，不 reset/stash/clean/覆盖；保留安装报告。
若 SSH 不通，检查 Orin 已开机、原网线已连接；不改写既有静态地址配置。
若 ROS 发现失败，报告和两端日志保留；不自动关闭防火墙或切换 Wi-Fi 通道。

明确尚未掌握的 Microduck 实物信息
14 个 XL330 的实际 ID/线序、串口转换器和设备路径、供电和急停线路；IMU
型号/接口/安装坐标与标定。本轮不猜测，也不根据其他机器人项目替换。
此外，当前 Stage12 模型不是只靠 IMU 和关节量即可运行的盲策略：
actor 的 [55:61] body_command 槽被替换为头顶小球相对头壳参考系的位置/速度。
位置特征 [20*x,25*y,50*(z-0.020)]；相对速度特征 2*[vx,vy,vz]。
PC 仿真可以提供这 6 维；实物传感来源尚未确认，不能省略或偷偷填零。
网络通信的已知连接方式已经恢复，以上未知硬件细节不影响本步推理链路检查。

后续边界
下一环节为 PC MuJoCo 模拟被控对象、Orin 经 Humble 执行同一个冻结策略的闭环。
需要逐步核查实际观测时序、LSTM reset、丢包/超时、50Hz 截止时间和回放。
本轮 128 组固定输入是数值/链路测试，不是随机扰动复测，也不是新的训练。
真实电机联调必须先补齐上面的接线、标定和小球测量信息。

Git 和云端交接（继续遵守）
代码仓库：/home/lx/microduck-double-balance/workspace
下载目录：/home/lx/下载
分支：double-balance
SSH origin：git@github.com:liuxue-lab/microduck-double-balance.git
本包基于 19ebce52ab704a0e36c2d956d88f3fc36e082bc6 的增量 Git bundle。
最终新提交和包身份见 build-receipt.json；用户自己 push，无自动远端写入。
必须保留并阅读：
  docs/handoffs/microduck-local-ssh-push-protocol.md
  docs/handoffs/microduck-cloud-gpu-and-transfer-protocol.md
模型回传提交：0cbee30e5a83904371076ca09a8cfb4afd1fa3f4
原返回目录：
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-push10-final6h-v1-client/returned/0cbee30e5a83904371076ca09a8cfb4afd1fa3f4
回传 400 文件已在用户电脑校验通过，归档不合并进代码分支。
剩余云任务：无。已于之前回传/验收后提醒用户手动关机；实际断电状态
UNCONFIRMED，本轮不启动、不关停实例。5090 可以继续保持关闭。
新的冻结 ONNX 留在本地和此次 LAN 目标目录，没有自动上传模型或改动 Git 远端。

交付前验证与尚未执行项
CPU 合成 LSTM 使用固定 mjlab exporter，检查连续动作和 h/c 数值一致性、reset、
NaN/身份/乱序拒绝、环境隔离、网络就绪条件与产物 SHA256。
双端节点的 128 请求交互另经内存 ROS API 替身测试；这不等同于真实 DDS
通信测试。共 13 项 CPU 测试通过，日志中的替身 RTT 不是 Orin 实测延迟。
真实 Git 测试检查增量导入、重复导入、暂存/未暂存/未跟踪内容、忽略文件冲突、
目录符号链接冲突和分歧提交保护。结果见 cpu-tests.log / release-validation.json。
当前协作环境没有用户的最终 checkpoint 文件、RTX5060 或 LAN Orin，实际最终
导出与 DDS/ARM 测试由以上 --prepare 在电脑执行，不能声称已经运行通过。

实现核对的官方来源
https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html
https://docs.ros.org/en/humble/How-To-Guides/Using-Python-Packages.html
https://github.com/ros2/rclpy/tree/humble
https://fast-dds.docs.eprosima.com/en/v2.6.11/fastdds/transport/whitelist.html
https://onnxruntime.ai/docs/install/
https://github.com/mujocolab/mjlab/blob/v1.3.0/src/mjlab/rl/runner.py
https://github.com/leggedrobotics/rsl_rl/blob/v5.0.1/rsl_rl/models/rnn_model.py
