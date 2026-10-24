Microduck Stage12 — PC inference-only runtime v1 / 2026-10-24 (Asia/Shanghai)

最新报告审查
report(20261024-095630).json：PC_KINEMATICS_MEASURED。
SHA256 a998aa617de7f638c10151a74ddf1b0086fe4241e59a5fc957064d3889170734
已运行提交 ea03ceed3cd99da361d4f7f7b73685a6cbb8bd3e，原记录未改变。
两组各160控制步/1600物理步，87200项数值完全一致；完整Audit、Trial和脉冲记录一致。
基准平均28.263570ms、p95 29.471595ms；头部图平均28.333709ms、p95 29.685311ms。
本次没有提速，KEEP_BAM_GRAPH，不启用HeadGraph；不把单次对照说成统计意义的退化。
两组150/150稳定计时步均超过20ms，完整50Hz实时验收仍未通过。
已通过功能闭环与最终模型验证仍保留；本次不重新训练、不量化、不剪枝。

本轮候选及适用边界
保留原PhaseCache、BAM Graph v2、积分后批量审计读取。
在严格加载actor+critic检查点、完成原始全量reset之后，仅在冻结策略推理期间：
1. 不再执行16项训练奖励计算。
2. 原ObservationManager.compute只遍历actor组。原compute_group完整保留，继续执行
   actor的传感器读取、噪声、裁剪/缩放、延迟、历史及NaN检查，61D布局不变。
3. 保留全部物理积分、命令更新、传感器、7项任务指标、独立几何探针、终止条件、
   500Hz关节/电机边界检查和每子步外力核对。
不启用上一轮未达采用门槛的HeadGraph或PackedChecks。

奖励缺省的明确含义
RslRlVecEnvWrapper需要返回Tensor奖励，候选返回预先分配的零占位Tensor。
该零值不是实际奖励，不能用于PPO、奖励评价或训练日志。
报告含reward_not_computed=true和zero_reward_is_wrapper_placeholder=true；删除候选日志中
未再计算的Episode_Reward/*与Metrics/angular_momentum_mean，保留其他任务指标。
候选不生成伪造的critic向量。初始reset仍有actor61D/critic85D，后续只返回actor。
两组比较明确排除critic和reward；不声称本轮还比较了87200项全量训练字段。
原actor推理只读取actor组；新增验证policy.obs_groups==['actor']，不符合则停止。

启动资格与恢复
逐文件SHA256核验mjlab观察/奖励manager、基础/velocity观察与奖励、项目MDP和native_task。
逐项核验14个critic函数与16个奖励函数的原始身份；不接受未知有状态函数。
critic必须无噪声、无延迟、无历史、无类状态缓存，避免省略计算改变actor的随机数或状态。
仅允许单环境、torch.inference_mode、无自动reset/课程、50Hz/500Hz/decimation10。
作用域内请求critic或reset会停止；退出时还原所有绑定并使观测缓存失效，随后关闭环境。
不会修改安装的依赖、冻结源码、原配置奖励权重、训练模型或归一化参数。

原始源码核查依据
https://github.com/mujocolab/mjlab/tree/v1.3.0/src/mjlab/managers
https://github.com/mujocolab/mjlab/tree/v1.3.0/src/mjlab/envs/mdp
https://github.com/mujocolab/mjlab/tree/v1.3.0/src/mjlab/tasks/velocity/mdp
https://github.com/leggedrobotics/rsl_rl/blob/v5.0.1/rsl_rl/models/mlp_model.py
https://github.com/leggedrobotics/rsl_rl/blob/v5.0.1/rsl_rl/runners/on_policy_runner.py
逐文件URL、Git blob身份与SHA256见reference/runtime-primary-sources.json及package/runtime_sources.json。

固定运行范围与判断
顺序执行bam-graph基准、inference-only候选各160控制步/1600物理步，总320/3200。
种子2026102407，外力种子2026102500；每组3.2仿真秒，2.0–2.1秒一次10%自重水平随机外力。
这是部署计算的短对照，不是完整60秒验收；无自动重试或新训练回合。
逐步比较22类共同字段：actor、actions、dones、物理位置/速度/控制/力矩/外力、指标、
稳定计数器、成功状态、摩擦/阻尼和全部BAM诊断。与上一轮同模型尺寸时共73440个标量。
另逐项比较完整Audit、Trial与扰动内容（仅排除输出文件路径）；任何差异都不选用候选。
前10步仅从稳定计时统计排除；物理/检查/轨迹不排除，报告同时包含全部160步耗时。
两组均在timer挂载前完成所有适配器安装，确保计时和恢复顺序一致。
只有状态对照通过且平均值、p95均改善至少5%才建议INFERENCE_ONLY_CANDIDATE。
仍不自动启用生产控制。环境步计时含CUDA完成，不含策略推理/网络及对照轨迹写盘。
即使环境p95小于20ms，也需后续完整ROS/物理闭环测试，不能据此宣称50Hz达标。

离线验证
51项CPU测试通过，包含原ObservationManager.compute/compute_group的实际源码方法，
160步actor随机数、延迟/历史、初始观测字典和清理恢复对照；原RSL观察选择/归一化对照；
未知函数/critic状态拒绝、异常与中途安装失败恢复、原BAM算术与500Hz检查回归。
继承头部图/合并检查测试留作回归，但本轮不启用这些候选。
真实Git安装保留检查见release-validation.json；本工作区未执行新候选GPU物理测试。

运行（现在需要用户在电脑执行）
下载microduck-stage12-pc-runtime-v1-20261024.py到/home/lx/下载，依次执行：
python3 "/home/lx/下载/microduck-stage12-pc-runtime-v1-20261024.py" --install-code
git -C /home/lx/microduck-double-balance/workspace push origin double-balance
python3 "/home/lx/下载/microduck-stage12-pc-runtime-v1-20261024.py" --run

仅需电脑RTX5060及既有环境；无需连接Orin，5090保持关闭，主运行终端保持开启。
预计1–3分钟，每组worker上限230秒；每5秒心跳，并显示进度百分比。
不安装升级依赖、不清理旧文件。已有报告时--run不重做；--retry-run才显式重跑固定两组。
另开终端实时显示：
python3 "/home/lx/下载/microduck-stage12-pc-runtime-v1-20261024.py" --watch
--watch每2秒刷新，完成后也继续刷新；观察终端Ctrl+C只退出观察。
--status查看一次；--stop停止当前对照并保留；主运行终端Ctrl+C中断本轮。

结果
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-pc-runtime-v1/report.json
把PC_RUNTIME_REPORT指向的report.json发回。
PC_RUNTIME_MEASURED表示短测和对照完成；仍需看recommendation是否建议保留候选。
PC_RUNTIME_PARITY_FAILED_PRESERVE / PC_RUNTIME_FAILED_PRESERVE保留失败原因和日志。
RUNTIME_SAMPLE_COMPLETE / EPISODE_INTERRUPTED是160步主动停止；内嵌60秒验收false是预期。

Git交接
父提交ea03ceed3cd99da361d4f7f7b73685a6cbb8bd3e；新提交与包身份见build-receipt.json。
只新增experiments/stage12/pc-runtime-v1，保留旧文件、源码、冻结报告和原模型。
保留暂存/未暂存/未跟踪/忽略文件；路径符号链接、冲突、分支分歧时停止。
不reset/stash/clean/force push。运行包从固定Git对象提取，用户使用现有SSH身份推送。
本工作区没有远端写入。长期协议与故障恢复规范随包保留。

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
