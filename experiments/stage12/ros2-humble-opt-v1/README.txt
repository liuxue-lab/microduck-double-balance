Microduck Stage12 — ROS 2 Humble / Orin Nano Super optimization measurements v1
Prepared 2026-10-24 (Asia/Shanghai). No new training. Not a hardware acceptance release.

已确认的上一步
用户回传报告 CLOSED_LOOP_FUNCTIONAL_PASSED，SHA256：
615d9b29e1c85433eebec2a988031ba01f14fc53ffeebf9906ae9355e6ac28e5
原生 MuJoCo↔Orin 完成 quiet-2407 与 regression-2500 两个 60 秒回合；6000 次
原模型动作比对通过，扰动回合实际 29 个脉冲。ROS RTT p99=3.276996 ms；
IPC+ROS RTT p99=3.642689 ms，Orin 推理均值约 0.756168 ms。
完整诊断周期均值约 39.9 ms，6000 次均超过 20 ms。因此功能通过，尚不能宣称
完整闭环达到 50 Hz。新报告引用的原始记录不修改，也不重新做 27 回合选优。
原选优过程 24/27，最终模型 heldout 18/18；不能写成所有候选全部通过。

本次 --run 会按顺序做什么
1. 核对原模型、128 条导出金标准、365 个冻结源码和历史回传/验收记录。
2. PC RTX5060 原模型进行 100 个无扰动控制步（最多 1000 物理步），拆分物理
   积分、CUDA 等待、逐物理步边界检查的耗时。不修改奖励、参数或模型边界；
   不进行完整任务验收。额外 CUDA synchronize 本身会改变耗时，只用于定位。
3. 检查全连接层零权重比例和现有 2:4 稀疏模式。剪枝只分析，不删除权重、不微调。
   对受支持的 Gemm 构造隔离的 MatMul+Add 等价图，并核验 128 条金标准动作和
   LSTM 状态。仅量化 LSTM 下游、通往动作且不影响 h/c 的常量权重 MatMul。
   产生可选的动态、逐通道 INT8 候选；LSTM 节点及权重保持原样。
   缺包、图不支持或转换失败时记录 UNAVAILABLE，仍可继续测量原 FP32。
4. 读取已经通过的 6000 条真实闭环观测/动作记录，逐条核验请求、回复及完整哈希链。
   通过 192.168.50.1↔192.168.50.2 网线和既有 SSH 会话传输，打印百分比和平均
   MiB/s；每个文件在 Orin 验 SHA256。不从互联网下载或升级软件。
5. Orin CPU 上交错测量原 FP32、固定缓冲区 FP32、可用的 MLP INT8，每种 6000 步。
   每种预热 32 步并清空状态，测量时只在序号 0/3000 清空 h/c。每步轮换执行顺序；
   保留动作误差、状态误差、均值、p95/p99/max，记录运行环境，不改时钟或功耗模式。
6. 进行两个固定 50 Hz 回放：JSON+FP32、二进制+FP32，各 6000 次、约 120 秒。
   若相同模型的缓冲区版本达到严格数值一致且推理 p95 降低至少 5%、均值也降低，
   再加一次二进制+缓冲区 FP32。不同命名变体各一次，不自动重试失败变体。
7. 对建议组合分别注入 100 ms 延迟、不回复、重复上一条回复。每组只发 2 条请求；
   第一条正常，第二条触发闭锁。超期/旧回复丢弃，后续发布被拒绝，不使用缓存动作，
   无自动重新激活。协议截止为计划时刻后 20 ms；故障检测诊断容差另列为 5 ms。
8. 输出建议和报告。任何方法都不自动替换现有控制程序或原始模型。INT8 即使通过
   数值检查也只保留为候选，部署前仍需新的物理仿真闭环检查。

优化与测量边界
* FP32 缓冲区复用使用 ORT CPU I/O binding；h/c 双缓冲，避免输入输出内存重叠。
  原图与归一化保持不变，单线程会话配置与已通过版本一致。
* 二进制观测帧 308 字节、回复 160 字节（不含 DDS/UDP/以太网开销）；包含版本、
  完整 session/model 标识、序号、reset、观测摘要和有限值检查。61 维观测必须可被
  float32 无损表示，否则停止。语义哈希校验仍保留内部 JSON 编码。
* 维持可靠、KEEP_LAST depth=1、VOLATILE QoS，Domain 62，仅绑定已核对有线地址。
  没有通过减弱身份/超时检查来得到更低的耗时。
* 回放消息预先构造，计划时刻至“收到且验证完成”不得超过 20 ms，超时立即闭锁，
  不通过补发追赶时钟。回放没有接入实时物理环境，没有采集真实传感器或驱动电机。
  成功仅证明这批记录、这次负载下推理/传输可按期完成，不证明完整 HIL 或硬实时。
* UInt8MultiArray 只用作免安装消息包的测量载体；后续正式机器人接口应定义有语义、
  有版本的专用 ROS 消息。此包不宣称实现 ROS 跨机器零拷贝。
* 动态量化有额外开销，剪枝需要匹配稀疏算子；小型循环网络不保证加速。当前先不启用
  TensorRT/FP16、GPU 推理、CPU 实时调度或强制稀疏化。用测量结果决定下一步。
* 优化建议要求回放 p95 至少改善 5% 且均值降低，再检查对应组合的三类故障。
  所有变体结果都会保留；某变体失败时总状态为 CHECK_FAILED_PRESERVE，不掩盖失败。
* new_PPO_updates=0；cloud_connections=0；motor_io=false；hardware_acceptance=false；
  realtime_acceptance=false；无自动重训、无电机口、无自动开关机、无删除旧资料。

如何执行（先把 .py 下载到 /home/lx/下载）
python3 "/home/lx/下载/microduck-stage12-ros2-humble-opt-v1-20261024.py" --install-code
git -C /home/lx/microduck-double-balance/workspace push origin double-balance
python3 "/home/lx/下载/microduck-stage12-ros2-humble-opt-v1-20261024.py" --run

电脑、Orin 保持开机并按既有方式直连；运行主终端保持开启。这是前台测量，通常包括
4–6 分钟的固定频率回放，另加准备、剖析和基准时间。无需开启 5090。
第一条只导入代码，不连接 Orin；第二条由用户使用已有 Git SSH 身份推送。
安装仅添加 experiments/stage12/ros2-humble-opt-v1/，基于
f145cbe9185a17fc74e083955bfc3840e1e2f732；新提交与包身份见 build-receipt.json。
若分支已有不同新提交、同名文件、被忽略路径或符号链接冲突，保留并停止。
已暂存/未暂存/未跟踪改动会快照并原样保留，不 reset/stash/clean/force push。
运行源从固定 Git 对象读取；包被修改时停止，不用未提交修改替换验证代码。
本次不是 Stage12 全阶段完成，不创建虚假的硬件完成标签。

实时查看（第二个终端）
python3 "/home/lx/下载/microduck-stage12-ros2-humble-opt-v1-20261024.py" --watch
每 2 秒刷新；即使完成也继续显示，Ctrl+C 只退出这个观察窗口。
--status 只显示一次。--stop 请求停止本轮并保留结果，不关闭电脑或 Orin。
Ctrl+C 主运行窗口会停止本轮。已有结果时 --run 不自动重跑；--retry-run 明确创建
新的独立测量目录，重复本轮测量，并保留旧目录，不恢复训练。

最终报告
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-ros2-humble-opt-v1/report.json
状态 OPTIMIZATION_MEASURED 表示预定检查完成；具体方法是否被拒绝，查看 recommendation
和 policy_benchmark。CHECK_FAILED_PRESERVE 时看 errors、replay_variants、fault_cases
以及 log_tails。报告包含远端确认情况；SSH 断开不能视为远端已经退出。
每个远端任务有独立限时（最长 220+5 秒）与取消文件；仅清理本轮拥有的进程组，
不关闭共享 SSH 主会话，不终止其他用户程序。

故障处理
* No route to host：先恢复既有直连地址/路由；不要改回旧 192.168.8.88。
* SSH Connection closed：保留报告，确认 Orin 用户/连接；不关闭主机密钥校验。
* 找不到 accepted loop transcript：保留原 native-ros2-humble-loop-v1/runs 下记录，
  不用新造数据替代，不自动重做两回合来产生另一套证据。
* QUANTIZATION_CANDIDATE_UNAVAILABLE：这不等于原模型失败，查看具体原因与 FP32 测量。
* DISPATCH_DEADLINE/RESPONSE_DEADLINE：记录本次超时，不降低频率后仍称 50 Hz 通过。
* 软件版本不符/磁盘少于 512 MiB：停止并报告；此包不运行 apt/pip 或删除旧文件。
* Git non-fast-forward/代码路径冲突：按固定交接规范审计，禁止 force push。

固定模型/回传与环境
Checkpoint final6h_005304.pt (10041821 bytes)
SHA256 cf7ce9297672943687019afe071d94918d3eb2620c18a1e8fe689512f8c08111
ONNX final6h_005304.onnx (2501039 bytes)
SHA256 8c1c1250b9b88d9a91e3eb095fea8bce21daa73c251b43f3cd8adc1327f03c4e
contract.json SHA256 e0283bf7f00c8e2c7b57128b91ca3ced3abb4698e8f4c4aa11e74a745ad84929
golden.json SHA256 b7ebe914c3933477d35bc0cc70c6b1bb4c1c83e67bd61dc22cd8f0775b1609e7
Final simulation validation report SHA256 d052f3cf8031be5ec5685fc1aa02c906301932d2a4849ac34fd450206b359228
Accepted transcript chain d1a37f72ab7502654004940edfe43ed675ade75ce828edeac22738108315b028
Frozen export: /home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-ros2-humble-v1/frozen-export
Actor=61, critic=85, actions=14, h/c=[1,1,256], original 14-joint order, BAM XL330.
No action clipping/filtering. Actor 55:61 remains the accepted head-ball relative state encoding.
PC system Python3.10 for Humble; project .venv Python3.12 for existing simulation/export tools.
Orin observed Python3.10.12, ORT1.22.0 CPU, NumPy1.26.4, L4T36.4.3, MAXN_SUPER.
Read-only probe checks current readiness; CPU tests here used ORT1.23.2 on x86_64, not Orin1.22.
No Jetson timing, TensorRT availability, or quantized final-model accuracy is claimed before --run.

Project scope / unresolved hardware
Keep Stage12: feet on basketball, top ball on actual head shell. No Stage13/14 walking task.
Hardware still Orin Nano Super, SSH jetson@192.168.50.2:22, PC192.168.50.1; Domain62.
PC enp4s0 / jetson-direct; Orin enP8p1s0 / jetson-wired; direct link no gateway/DNS.
Physical Dynamixel adapter/port, actual servo calibration, IMU mounting/time stamping, head-ball
sensing and motor-side watchdog/enable remain to be supplied/confirmed before physical actuation.
Do not invent those connections. This package cannot enable real motors.

Both standing handoff protocols are retained verbatim:
docs/handoffs/microduck-local-ssh-push-protocol.md
docs/handoffs/microduck-cloud-gpu-and-transfer-protocol.md
Repository /home/lx/microduck-double-balance/workspace; downloads /home/lx/下载.
Branch double-balance; origin git@github.com:liuxue-lab/microduck-double-balance.git.
Return branch stage12-push10-final6h-return-20261023t181130z-1b79efbd-s002;
return commit 0cbee30e5a83904371076ca09a8cfb4afd1fa3f4, 400 files / 5 archives verified.
Return inventory is authoritative; no claim that every raw cloud trajectory was retrieved.
Cloud remaining tasks NONE for this stage. Historical latest SSH was
ssh -p 32185 root@connect.bjb2.seetacloud.com (not connected or revalidated here).
Cloud power state UNCONFIRMED. 2026-10-24 handoff reminder: 5090 is not required; if it is
still running, shut it down manually on the platform after preserving needed data. No power API used.
Remote Git push has not been performed in this workspace; a normal successful user SSH push suffices.

Primary technical references consulted 2026-10-24
https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
https://onnxruntime.ai/docs/api/python/api_summary.html
https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html
https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/data-formats-tensors.html
https://docs.ros.org/en/humble/p/std_msgs/

Review archive includes source, tests, accepted report, both protocols, build receipt, Git bundle,
test logs and installer preservation checks. It does not contain the user's final model or trace.
Local CPU synthetic-model tests are implementation checks only, not final-model deployment evidence.
