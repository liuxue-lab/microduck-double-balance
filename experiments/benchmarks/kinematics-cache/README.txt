Microduck Stage12 — PC phase-cache candidate v1 / 2026-10-24 (Asia/Shanghai)

新报告已审查，5090 已由用户确认关机
报告 report(20261024-071856).json，状态 OPTIMIZATION_MEASURED。
SHA256 7186dcaf36345244e1d01792bd4af769538fc2030669db69b9c5d95d2da8960a
运行包 d9242b487255bc5fb1dbdd1544115c01aa1688fb4c21c72d2b2b6bdb2f78e1c1。
原模型/记录未变，本机及远端工作进程已退出；没有新增 PPO、没有电机动作。
5090 电源状态更新为 OFF_USER_CONFIRMED，依据本次用户原话“5090已关机”。
确认日期 2026-10-24；没有平台遥测或实际关机时刻，不编造具体时间，不发送电源命令。
此记录更新当前交接状态，不修改历史冻结报告中的 UNCONFIRMED 字段。

优化实测结论
原 FP32 离线推理均值 0.352815 ms，p95 0.388907 ms。
缓冲区 FP32 均值 0.339043 ms，p95 0.374763 ms；数值完全一致，但 p95 仅改善约
3.64%，未达到事先规定的 5% 门槛，暂不选用。
MLP INT8 均值 0.293598 ms，p95 0.328585 ms；均值改善约 16.78%，但最大动作绝对
误差 0.04030265，84000 个动作分量中 83141 个超出原严格容差。没有启用量化候选。
这不是 INT8 闭环失败的证据；本次没有做量化闭环，不能据此判定会跌落或证明安全。
全连接层现有零权重比例均为 0，不满足现有 2:4 稀疏模式；没有剪枝/微调。

50 Hz 记录回放（计划时刻至回复验证完成，不是完整物理闭环）：
  JSON+FP32：6000/6000；mean 2.593678 ms，p95 3.052529 ms，max 3.722953 ms。
  binary+FP32：6000/6000；mean 2.762698 ms，p95 2.952425 ms，max 10.540997 ms。
两组均无 20 ms 超期、动作误差为 0。二进制 p95 略低但均值更高，故保留 JSON。
100 ms 延迟、无回复、重复回复三类检查全部通过；故障后不再发布/不采用旧动作。
离线推理与定时回放的工作负载不同，不能拿 INT8 离线均值对比 FP32 回放均值来夸大加速。

电脑端 100 步剖析：环境控制步 mean 34.897362 ms、p95 36.087328 ms，100/100 超过
20 ms。此前完整功能闭环均值约 39.9 ms；完整 50 Hz 尚未通过。新剖析有额外 CUDA
同步，不能用两个均值直接宣称加速。100 步主动结束导致其内嵌 60 秒 acceptance=false，
这是短剖析的预期状态，不是新一次跌落。正式模型此前 heldout 为 18/18；所有候选选优
合计 24/27，不写成 27/27。ROS/量化检测不需要重跑。

本次新增候选和范围
已阅读固定版本源码并核对项目文件 SHA256，证据见 reference/source-inspection.json。
同一 manager.compute 内会重复计算 _stage11_head_kinematics。候选只复用该纯函数的
结果；不同参考坐标系分别缓存，不同环境不复用；每次 compute 入口/出口清空。
physics step、forward、reset、scene write 入口/出口清空，并在状态写入期间禁用缓存。
奖励、终止、完整 actor/critic 观测及原有统计照常计算；不删奖励、不跳 critic、不降低
物理频率、不减少模型边界检查、不改控制频率/求解器/动作/模型/归一化。
独立几何探针仍重新计算，不复用统计 manager 的缓存。所有临时函数绑定会恢复。
这只是局部计算复用候选，实际速度和数值一致性尚待电脑运行，不自动投入正式控制。

--run 固定执行两组，各 160 个控制步、1600 个物理步：
1. 原版 baseline，seed=2026102407、force_seed=2026102500。
2. phase-cache，同样种子/模型/初始配置，各重新创建一个环境，不并行、不自动重试。
每组仿真时长 3.2 秒，包含在 2.0–2.1 秒的一次完整 10% 自重随机方向水平扰动。
总上限 320 控制步 / 3200 物理步；不扩为完整回合，不做新的随机种子泛化验收。
保留原 500 Hz 的关节/PWM/模型电流/力矩检查，核对每组 1600 次和 50 个施力物理步。
逐控制步对照 61D actor、85D critic、14D action、qpos/qvel、ctrl、作用力、外力、
奖励、done、统计和计数器；使用原绝对 2e-5、相对 2e-4 容差，不放宽门槛。
计时剔除最初 10 步，包含环境步完成前的 CUDA 同步，状态轨迹复制在计时范围外。
同时输出 action、scene、物理及检查、奖励、统计、观测等阶段的主机耗时。它们可能
包含等待且相互嵌套，不能当成可直接相加的纯 GPU kernel 时长。
只有数值对照通过、均值和 p95 均改善至少 5% 时，才建议后续完整 ROS 闭环评测。
即使环境步低于 20 ms，也没有包含策略推理、通信、调度等开销，不宣称完整 50 Hz。
一次固定顺序的短对照仅用于选择候选，不能当作跨负载性能保证或 60 秒稳定性验收。

运行方式
把 microduck-stage12-pc-cache-v1-20261024.py 下载到 /home/lx/下载，然后依次执行：
python3 "/home/lx/下载/microduck-stage12-pc-cache-v1-20261024.py" --install-code
git -C /home/lx/microduck-double-balance/workspace push origin double-balance
python3 "/home/lx/下载/microduck-stage12-pc-cache-v1-20261024.py" --run

本次仅使用电脑 RTX5060、既有项目 Python 环境，无需连接 Orin；5090 保持关闭。
准备与两组短对照通常约 1–3 分钟，以实际输出为准；每个工作进程上限 230 秒。
主运行终端保持开启。程序打印进度百分比和每 5 秒心跳，不运行 apt/pip。
电脑 GPU 被其他计算占用或版本/冻结源码不符时，保留并停止，不改本机环境。
需要至少 256 MiB 可用磁盘；不删除旧数据。

第二个终端实时查看：
python3 "/home/lx/下载/microduck-stage12-pc-cache-v1-20261024.py" --watch
每 2 秒刷新，完成后也继续；该观察窗口 Ctrl+C 只退出观察。
--status 查看一次；--stop 请求停止本轮并保留数据。主运行窗口 Ctrl+C 会中断本轮。
已有结果时 --run 不自动重跑；--retry-run 显式新建目录重做这两组短对照，不训练。

结果
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-pc-cache-v1/report.json
PC_CACHE_MEASURED 表示短对照测量完成且数值通过，仍需看 recommendation 是否保留原版。
PARITY_FAILED/FAILED_PRESERVE 时查看 errors、arms、comparison 和 log_tails。
各组 EPISODE_INTERRUPTED / CACHE_SAMPLE_COMPLETE 是到 160 步主动结束，不是 60 秒验收。
模型边界事件、缺少物理检查、未完成施力、数据改变或记录不完整均不接受。
运行后把 PC_CACHE_REPORT 指向的 report.json 发回。完整逐步轨迹留在 run 目录，可按需
进一步审查；主报告含摘要、轨迹哈希与日志尾部，不重新传输模型。

Git 与保护
增量 bundle 父提交 23b00b7b5a77135467b2124823c1fab54bd1eed9；新提交见 build-receipt.json。
只新增 experiments/stage12/pc-cache-v1，不修改旧源码、模型和历史记录。冲突/分歧/符号
链接路径保留停止；不 reset/stash/clean/force push。已有暂存、未暂存及未跟踪改动保留。
运行包读固定 Git 对象，包文件修改会被拒绝。没有在本工作区执行远端 push。
本次不是硬件验收完成，不新建虚假的全阶段完成标签；正常用户 SSH push 输出即足够。

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
