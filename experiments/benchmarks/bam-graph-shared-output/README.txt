Microduck Stage12 — PC BAM CUDA Graph candidate v1 / 2026-10-24 (Asia/Shanghai)

已完成的新报告审查
report(20261024-085220).json，PC_CACHE_MEASURED。
SHA256 8e48c09c82c7d8b17dbb0823f6e3f3f005c18c3052d77afacbe0f43e1e028002
包身份 70e196cd67be34dc8d4ae846ccdf5570037e1706f03ad7aef68f5dcc4858ece8。
两组各 160 控制步 / 1600 物理步 / 3.2 秒仿真，各含一次完整 10% 自重水平扰动。
全部 61120 个对照数值完全一致，两个逐步轨迹文件 SHA256 也相同；两组均无模型
边界违规，保留全部检查并恢复临时绑定。EPISODE_INTERRUPTED / CACHE_SAMPLE_COMPLETE
表示短测试按预算主动结束，不是跌落；也不是新一次 60 秒验收通过。
剔除前 10 步后，各 150 个环境步：
  原版：mean 35.109492 ms，p95 36.388906 ms，max 36.876894 ms。
  缓存：mean 32.813133 ms，p95 34.027331 ms，max 34.540252 ms。
均值减少 6.54%，p95 减少 6.49%。保留该缓存作为本轮对照基准，尚未部署到正式闭环。
本次单次固定顺序短测不是跨负载性能保证。两组所有环境步仍超过 20 ms，因此完整
PC→ROS→Orin→PC 50 Hz 还未达标。环境步计时不包含外部策略推理/通信/对照轨迹写入。
缓存组每控制步主机计时：scene write 7.619 ms、physics+force+bounds 11.064 ms。
这些是包含等待的主机区间，不能当作可直接相加的纯 GPU 运算时间。
原 FP32 + JSON 继续保留；此前 INT8 超严格数值容差、二进制通信未达到选用门槛，
本轮不重复量化/网络测试、不剪枝、不修改模型、不再训练。
5090 已由用户确认关闭（OFF_USER_CONFIRMED）；不连接云端，也不发送电源命令。

本轮准备好的候选
1. 两组都使用已通过短轨迹一致性检查的阶段内头部运动学缓存；独立几何探针仍重新计算。
2. 新候选把现有 FrictionDRBamActuator.compute 的原 GPU 运算录入 CUDA Graph，
   降低重复从 Python 提交 GPU 运算的开销。未重写电机方程、未做算子融合或精度转换。
   项目 actuator 及三个 BAM 模块逐一匹配原评估器 SHA256；源码审查证据在 reference。
3. 预热/捕获使用隔离 actuator 副本，独立 BAM model/actuator Python 状态，读当前
   模拟数据；其唯一模型写入 _write_frictions 被导向输出张量。预热/捕获不推进物理、
   不写真实模型、控制延迟、随机数或真实上一时刻力矩。正式回放后仍调用原模型写入函数。
4. 每次回放验证所有被捕获输入的地址、形状、步长、类型、设备和标量配置。地址或
   配置改变就保留停止，不静默重捕获。保留所有输入/输出的强引用，避免地址失效。
   上一时刻力矩和本步命令显式复制到独立输入；每步产生新诊断字典，原新鲜度检查保留。
5. 每个电机组最初 40 个物理子步，在写入候选输出之前，用同一输入执行原代码对照
   力矩、摩擦、供电、增益以及全部诊断。原 atol=2e-5、rtol=2e-4，非有限值直接停止。
   这些检查在最初 10 个控制步的计时预热范围内，不把捕获开销混入稳定运行均值。
6. 将积分后的 qpos、qvel、实际 actuator_force 合并成一次 GPU→CPU 读取；使用固定
   关节索引张量。原 Audit 仍每个物理子步执行，关节顺序/检查阈值/计数/异常行为不变。
   积分前检查、外力前后核对、独立稳定性检查等均保留，物理频率仍 500 Hz。

固定执行范围
--run 顺序执行 phase-cache 基准与 bam-graph 候选各 160 步，总上限 320 控制步 /
3200 物理步。每组重新创建环境，环境种子 2026102407、外力种子 2026102500，
仿真 3.2 秒，其中 2.0–2.1 秒施加一次完整水平随机方向、10% 机器人自重外力。
无额外环境重试或完整回合。CPU 同输入测试的矩阵不是实际物理回合。
每组前 10 步仅排除计时，仍保留所有物理、检查和轨迹；采集其余 150 步完整环境耗时。
逐步对照原 actor/critic/action、qpos/qvel、ctrl、力矩、外力、奖励、done、统计、计数器，
本版额外对照 BAM 写入的摩擦/阻尼、上一时刻力矩及所有诊断字段。
只有短轨迹数值通过、均值和 p95 都改善至少 5%，才建议后续验证候选；否则保留
phase-cache。单个候选组合的收益不拆写成“CUDA Graph 单独收益”或“批读取单独收益”。
本程序不自动启用正式控制；即使环境步低于 20 ms，也仍需完整 ROS/物理闭环的实际验证。

本工作区验证与限制
22 项 CPU 测试已通过：缓存边界、原 BAM 数学 160 组动态输入（有/无电压下跌）、
隔离状态、约束有效行、数值异常、地址变化、回放状态提交、批量读取的顺序/新鲜度/
异常传播、两组 worker 轨迹一致性和生命周期等；真实 Git 安装/改动保留检查结果见 release-validation。
CPU 回放使用显式仿真替身，只核查 Python 状态衔接；这里没有 CUDA GPU，未实际捕获
GPU 图、未测新候选速度、未跑候选真实物理。以上不冒充真实 CUDA/RTX5060 验证。
GPU 捕获不兼容、同输入输出不符、轨迹不符、保护检查不完整或输入文件改变时，保留
失败报告，不自动安装依赖、不切换算法、不悄悄用原版把候选标成成功。
PyTorch 2.9 官方 CUDA Graph 说明：
https://docs.pytorch.org/docs/2.9/notes/cuda.html#cuda-graphs

运行（到了需要用户在电脑执行的步骤）
下载 microduck-stage12-pc-graph-v1-20261024.py 到 /home/lx/下载，依次执行：
python3 "/home/lx/下载/microduck-stage12-pc-graph-v1-20261024.py" --install-code
git -C /home/lx/microduck-double-balance/workspace push origin double-balance
python3 "/home/lx/下载/microduck-stage12-pc-graph-v1-20261024.py" --run

只需电脑 RTX5060 及既有项目环境；无需连接 Orin，5090 保持关闭。主终端保持开启。
一般约 1–3 分钟，首次 GPU 捕获时间以实际为准，每个 worker 上限 230 秒；不运行 apt/pip。
每 5 秒心跳并输出进度百分比；需要至少 256 MiB 磁盘，无自动清理旧文件。
第二个终端实时查看：
python3 "/home/lx/下载/microduck-stage12-pc-graph-v1-20261024.py" --watch
--watch 每 2 秒刷新，完成后也刷新；该观察终端 Ctrl+C 仅结束观察。
--status 查看一次；--stop 请求停止并保存。主运行终端 Ctrl+C 会中断本轮。
已有报告时 --run 不自动重做；--retry-run 才显式新建目录重做固定两组短测试。

结果文件
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-pc-graph-v1/report.json
把程序打印的 PC_GRAPH_REPORT 路径对应 report.json 发回即可。
PC_GRAPH_MEASURED：短测结束且轨迹对照通过，仍看 recommendation 是否建议保留新候选。
PC_GRAPH_PARITY_FAILED_PRESERVE / PC_GRAPH_FAILED_PRESERVE：保留并报告原因和日志。
GRAPH_SAMPLE_COMPLETE / EPISODE_INTERRUPTED：160 步主动停止，内嵌 60 秒验收为 false 是预期。
逐步原始轨迹保留在本机；主报告包含哈希、保护检查、阶段计时、图回放计数和同输入误差。

Git 交接
父提交 daa8ae218fa254efdffcef2ae72a43df4518ac8a，新提交和包身份见 build-receipt.json。
只新增 experiments/stage12/pc-graph-v1；未修改旧源码或冻结报告。
保留暂存、未暂存、未跟踪改动，忽略文件/符号链接路径冲突或分支分歧时保留停止。
不 reset/stash/clean/force push；运行包读取固定 Git 对象，不从用户已编辑的工作树运行。
本工作区不执行远端 push；用户仍用既有 Git SSH 身份推送。两份长期协议随包保留。

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
