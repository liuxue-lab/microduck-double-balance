Microduck Stage12 — PC audit read graphs v1 / 2026-10-24 (Asia/Shanghai)

最新报告审查
report(20261024-101746).json：PC_RUNTIME_MEASURED。
SHA256 3b7a74f36e2c9f68b8c98aa42489519ca5ff352eadc09ea02861839d63c60f14
已运行提交 a223d4d18e520c322378c48c48816c26c5006a4a，package identity：
019d160a44ca51a5294ff15a09ae2ecf2529d30af7af01d41a9b9cd69a0cc8b4。
两组各160控制步/1600物理步，共同字段73440个标量的差异全部为0。
完整Audit、Trial和扰动内容一致。两份轨迹均1205288 bytes，SHA256：
2b7bdef6f9681dbf84d07c68d0cd7528e5580d09093a94a5c4b6b7e15b54f2b9。
基准平均28.219125ms、p95 29.306557ms、p99 29.750614ms；
精简推理平均24.778366ms、p95 26.075866ms、p99 26.338264ms。
平均下降12.193%、p95下降11.024%，超过固定5%采用门槛；保留InferenceOnly为下一轮基准。
候选150/150稳定计时步仍超过20ms。完整ROS闭环50Hz实时验收尚未完成。
每组1600次电机检查、1601次关节检查、1600次外力核对；一次完整10%自重0.1秒外力。
原模型、归一化、冻结源码和训练记录保留；5090关闭，新PPO=0，无Orin或电机I/O。

本轮候选
已测剩余physics+force+bounds宿主时间约10.23ms/控制步（含同步，并非纯GPU积分时间）。
新增read_graph.py，在已有PackedChecks完整检查流程中捕获两组只读操作：
积分前：完整外力/外力矩，加原BAM诊断、电机命令和原关节位置索引。
积分后：完整外力/外力矩，加原qpos/qvel/actuator_force索引。
图仅执行原索引、expand、stack、cat；不改公式，不推进物理，不写机器人或模型数据。
原EntityData.joint_pos属性继续使用，按当前qpos和原索引读取；不是提前保存一份旧关节状态。
没有修改BAM Graph v2，没有启用上一轮未提速的头部HeadGraph。
此前PackedChecks单独约3%收益未采用；本轮测的是加入图捕获后的组合，不把旧结果当作新收益。

检查及故障语义
保留每子步施力前的零外力检查，积分前后的全量外力严格相等检查和原CPU Audit。
只删除本适配器自己施加的外力；不清除未知外力。不降低500Hz检查频率，不推迟异常停止。
首40次积分前读取和首40次积分后读取分别与原同输入即时计算逐位比较。
采用FP32位模式相等，包括NaN位模式，以便相同非有限值仍由原Audit给出原故障事件。
图输入保留强引用，逐次核验底层地址、形状、步幅、精度和设备；变化则停止，不自动重捕获。
每次回放结果复制为新的主机列表，避免后续回放覆盖旧审计样本或第一次越界记录。
图预热/捕获只有只读数据操作，没有额外物理积分或PPO。
原EntityData源码SHA256：fd1116692f82278bda721d44c015d27e253d2e866608c8e12824df039dfb4657。
来源：https://github.com/mujocolab/mjlab/blob/v1.3.0/src/mjlab/entity/data.py

继承的推理精简范围
两组都保留已通过的InferenceOnly：全量actor/critic初始化与严格检查点加载后，省去训练
奖励及critic观测计算；原actor61D噪声、缩放、延迟、历史、归一化和策略保持不变。
奖励零Tensor仅为wrapper占位，明确标为reward_not_computed及zero_reward_is_wrapper_placeholder；
不作为真实奖励，不用于训练，不生成伪造critic。两组共同字段比较继续排除critic/reward。
命令、传感器、全部7项任务指标、原终止条件、独立几何探针和物理积分保留。
动作顺序、BAM参数和500Hz模型边界没有改变。

运行范围与采用门槛
顺序运行inference-only基准、graph-reads候选各160控制步/1600物理步，总320/3200。
环境种子2026102407，外力种子2026102500；每组3.2仿真秒，2.0–2.1秒施加一次完整
10%自重随机水平外力。不是完整60秒验收；没有自动重试，不追加训练回合。
逐步比较22类共同字段、全部BAM诊断和计数器；同模型尺寸预期73440个标量。
另比较完整Audit、Trial和扰动内容，仅允许输出文件路径不同。
候选应有两张只读图各1600次回放、各40次同输入对照，全部500Hz审计计数完整。
前10步只从稳定计时排除，仍保留全部轨迹和检查，另报全部160步启动耗时。
只有对照通过且平均耗时与p95都减少至少5%，才建议READ_GRAPH_CANDIDATE；
否则KEEP_INFERENCE_ONLY。运行不会自动启用生产控制。
环境步计时含CUDA完成，不含策略推理、网络与对照轨迹写盘；即使小于20ms，也不能
替代完整ROS/物理闭环验收。实际新候选速度与GPU捕获行为仍待用户电脑实测。

离线验证
60项CPU测试通过，包括原EntityData关节属性、原Audit与原BAM算术，1600子步完整
轨迹/审计/扰动对照、多组电机和非单位索引顺序、固定输出地址重放、旧样本保留、
输入/输出存储改变拒绝、捕获失败、NaN/未知外力/积分器失败/陈旧诊断故障停止和清理。
继承actor随机数/延迟/历史一致性测试及两组worker生命周期测试。
CPU图采用固定地址覆盖替身；本工作区没有CUDA，未冒充实际GPU捕获或提速验证。
安装保留检查使用真实Git，结果见release-validation.json。

运行（现在需要用户在电脑执行）
下载microduck-stage12-pc-readgraph-v1-20261024.py到/home/lx/下载，依次执行：
python3 "/home/lx/下载/microduck-stage12-pc-readgraph-v1-20261024.py" --install-code
git -C /home/lx/microduck-double-balance/workspace push origin double-balance
python3 "/home/lx/下载/microduck-stage12-pc-readgraph-v1-20261024.py" --run

只需电脑RTX5060和既有项目环境；Orin暂时不用连接，5090保持关闭，主终端保持开启。
预计1–3分钟；每组worker上限230秒；每5秒心跳，进度显示已完成步数和百分比。
不安装升级依赖，不删除旧文件。已有本轮报告时--run不会重做；--retry-run才显式重跑。
另开终端实时查看：
python3 "/home/lx/下载/microduck-stage12-pc-readgraph-v1-20261024.py" --watch
--watch每2秒刷新，结束后也继续；观察终端Ctrl+C仅退出观察。
--status查看一次；--stop请求停止并保留；主运行终端Ctrl+C中断本轮。

结果与恢复
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-pc-readgraph-v1/report.json
发回PC_READGRAPH_REPORT指向的report.json即可。
PC_READGRAPH_MEASURED表示本轮短测/对照完成，还需看recommendation是否建议保留。
PC_READGRAPH_PARITY_FAILED_PRESERVE / PC_READGRAPH_FAILED_PRESERVE保留日志和原因，不自动补跑。
READGRAPH_SAMPLE_COMPLETE / EPISODE_INTERRUPTED表示160步主动停止；内嵌60秒验收false是预期。
如源文件、储存地址或数值不同，保留失败记录，不为了通过而放宽容差或升级依赖。

Git交接
父提交a223d4d18e520c322378c48c48816c26c5006a4a；新提交和包身份见build-receipt.json。
只新增experiments/stage12/pc-readgraph-v1，不覆盖旧源码和模型、历史报告。
保留已暂存、未暂存、未跟踪、忽略文件；路径冲突、符号链接或分支分歧时停止。
不reset/stash/clean/force push；运行包从固定Git对象提取，不使用被编辑的工作树代码。
用户通过现有SSH身份推送；本工作区无远端写入，长期协议随包保留。

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
