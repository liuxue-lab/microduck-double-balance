Microduck Stage12 — PC accepted-runtime profile v1 / 2026-10-24 (Asia/Shanghai)

最新报告已完整跑通
report(20261024-111540).json：PC_COMMANDVIEW_MEASURED。
SHA256 718f8f83319b8fcd22158b51b045508b3c220f55b7505119ee22abeb2497a0cf。
提交3168d7d0ec94b9495c9ef61d5a044c3beae15390；包身份
e1fb1455ec38ede197349bfb983cb2c94b7de7f5879b074e93135219006cbc43。
两组各160控制步/1600物理步；73440个对照标量的差异全为0。
完整Audit、Trial、扰动内容相等。每组1600次电机检查、1601次关节检查、
1600次外力核对、50个施力物理步、1次完整10%自重脉冲。
两份轨迹均1205288 bytes，SHA256
2b7bdef6f9681dbf84d07c68d0cd7528e5580d09093a94a5c4b6b7e15b54f2b9。
索引兼容性已修复：target_ids为int64，joint_q_adr/joint_v_adr为int32。
命令视图1600次走连续切片，40次同输入2800个标量位模式比较通过，无fallback。
所有绑定恢复、worker退出码0、无清理错误，检查点、归一化、冻结文件及原记录未变。

采用决定
基准平均24.599920ms、p95 25.839590ms、p99 26.231405ms；
命令视图平均24.874879ms、p95 26.033123ms、p99 26.485491ms。
平均慢1.12%、p95慢0.75%，所以KEEP_INFERENCE_ONLY，不采用命令视图。
两组150/150稳定计时步均超过20ms；50Hz完整闭环实时验收仍未完成。
此前只读图平均改善2.327%、p95改善2.798%，也未到固定5%门槛。
当前保留InferenceOnly + PhaseCache + BAM Graph v2 + BatchedAuditReads。
HeadGraph、PackedChecks、ReadGraph、CommandViews均不在本轮执行路径启用。
Orin仍保留已验证的FP32与JSON链路；INT8误差超限未采用，未做剪枝或追加训练。

为什么本轮做详细分析
最新基准物理步连同外力/边界检查10.195558ms/控制步，scene_write 4.030634ms。
这些是宿主墙钟区间，包含可能的GPU等待，不能称为纯物理GPU核函数耗时。
本轮两组都运行同一已保留方案：inference-only普通计时；profile详细分析。
每组160控制步/1600物理步，总320/3200，前10控制步不参与计时分析。
固定环境种子2026102407、外力种子2026102500，各3.2仿真秒；2.0–2.1秒
施加一次10%自重随机水平力。不是新60秒抗扰验收，也不是训练。
仍逐步对照全部22类共同状态字段，完整Audit/Trial/扰动必须相等。

记录内容与解读
host_profile.py仅用标准库cProfile、pstats及perf_counter_ns，无新增依赖。
细分原sim.step入口、积分前后读取与审计、Python Audit、零外力核对、自有外力清除，
以及scene.write中的原get_command、apply_delay、BAM compute。
每个窗口只覆盖原EvaluationSession.step和原有末尾CUDA完成等待；模型推理和
外层对照轨迹序列化不在窗口内。原step自身的Audit和push10-trace写入仍计入。
没有增加Tensor读取、同步、CUDA事件、warmup积分或随机数调用；不改全局Torch方法。
保留原延迟缓冲、全部BAM算术与检查、物理积分、独立几何探针和500Hz边界检查。

inclusive_ms包含嵌套区间；exclusive_ms只扣除直接嵌套且被本轮包装的区间，
仍包含未包装的工作和采集开销。父子区间、cProfile与区间计时不能直接相加。
异步GPU工作可能在后续cpu()/equal()/item()等同步处记账；本报告不能分离纯GPU时间。
原physics_step中内联的torch.equal由cProfile按调用者记录，不替换它的实现。
详细分析会带来额外开销，profile组的快慢不得当成优化收益。recommendation始终
KEEP_INFERENCE_ONLY、diagnostic_only=true，不通过时保留失败，不激活任何候选。
report.json内嵌完整cProfile函数表、调用者、调用次数、自身/累计时间和嵌套区间。
本机另存host-profile.pstats供可选分析；本轮只需发回report.json即可。
成功须有150个分析窗口、各1500次原积分和审计、每组1500次原命令/延迟/BAM调用。
包装按逆序移除；异常仍执行原清理。已有Python profiling hook时停止并保留它。

离线验证
本工作区仅有CPU，不能代替RTX5060实测。
89项CPU测试，包括原始混合索引、Tensor/Warp桥、延迟历史和随机数、BAM与完整Audit。
新增两组worker各160步生命周期对照；1500次分析调用计数、外层轨迹排除、pstats输出；
嵌套计时、递归调用计数、已有profile hook保留、部分安装与异常退出恢复。
6类故障使用原EvaluationSession/Audit比较分析开关前后：电机NaN、积分后异常外力、
关节NaN、积分异常、过时诊断、已有外力；原停止阶段、计数、清理与失败记录相等。
CPU中的CUDA图是固定地址替身；实际新GPU计时和轨迹一致性均待用户运行。
真实Git保全检查另见release-validation.json。

现在需要用户运行
下载microduck-stage12-pc-profile-v1-20261024.py到/home/lx/下载。
按顺序执行，上一条成功后执行下一条：
python3 "/home/lx/下载/microduck-stage12-pc-profile-v1-20261024.py" --install-code
git -C /home/lx/microduck-double-balance/workspace push origin double-balance
python3 "/home/lx/下载/microduck-stage12-pc-profile-v1-20261024.py" --run

只需电脑RTX5060及现有项目环境；5090保持关闭，无需连接Orin，主运行终端保持开启。
预计1–3分钟；每组worker上限230秒，持续打印步数、百分比与5秒心跳。
报告：/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-pc-profile-v1/report.json
发回PC_PROFILE_REPORT指向的report.json。PC_PROFILE_MEASURED只表示本轮采集与对照完成。
PROFILE_SAMPLE_COMPLETE/EPISODE_INTERRUPTED是160步计划停止；完整60秒验收false属预期。
PC_PROFILE_PARITY_FAILED_PRESERVE或PC_PROFILE_FAILED_PRESERVE会保留失败，不自动补跑。
已有报告时--run不重做；--retry-run才显式重跑。主运行终端Ctrl+C中断并保留结果。
另开终端实时查看（每2秒刷新，完成后仍刷新；Ctrl+C只退出该观察终端）：
python3 "/home/lx/下载/microduck-stage12-pc-profile-v1-20261024.py" --watch
--status单次查看；--stop请求停止当前本轮任务。

Git交接与固定模型
父提交3168d7d0ec94b9495c9ef61d5a044c3beae15390；新提交与包身份见build-receipt.json。
仅新增experiments/stage12/pc-profile-v1；保留旧程序、模型、全部历史记录。
保留暂存/未暂存/未跟踪/忽略文件；新增路径冲突、符号链接、分支分歧时停止。
不reset/stash/clean/force push。运行包从固定Git对象提取，保留工作树中的用户改动。
仍由用户使用既有SSH身份推送，本工作区无远端写入。

阶段仍为脚踩篮球、球直接接触原头壳；已取消Stage13/14地面轨迹行走任务。
final6h_005304.pt：10041821 bytes；5304次新增已完成更新，累计11314。
SHA256 cf7ce9297672943687019afe071d94918d3eb2620c18a1e8fe689512f8c08111
final6h_005304.onnx：2501039 bytes；SHA256
8c1c1250b9b88d9a91e3eb095fea8bce21daa73c251b43f3cd8adc1327f03c4e
contract.json e0283bf7f00c8e2c7b57128b91ca3ced3abb4698e8f4c4aa11e74a745ad84929
golden.json b7ebe914c3933477d35bc0cc70c6b1bb4c1c83e67bd61dc22cd8f0775b1609e7
Final validation d052f3cf8031be5ec5685fc1aa02c906301932d2a4849ac34fd450206b359228
此前真实ROS功能报告615d9b29e1c85433eebec2a988031ba01f14fc53ffeebf9906ae9355e6ac28e5：
两种固定场景各60仿真秒通过，但完整周期约39.9ms，未通过50Hz实时要求。
128导出金标准、actor61/critic85/action14、h/c=[1,1,256]、原关节顺序与归一化保留。
无新滤波/裁剪，不重新导出模型，不新增量化模型，不追加PPO。

连接、协议和剩余事项
ROS2 Humble、Domain62、Fast DDS；PC192.168.50.1/enp4s0/jetson-direct；
Orin Nano Super192.168.50.2/enP8p1s0/jetson-wired；ssh jetson@192.168.50.2。
网线直连无网关/DNS，WiFi上网。Orin Python3.10.12/ORT1.22.0 CPU；
PC项目Python3.12、Torch2.9.1、mjlab1.3.0；运行前核验，不安装升级依赖。
真实电机适配器/端口、伺服零位/方向、IMU安装/时间戳、头球感知、硬件watchdog
仍缺实物证据，不能编造；本程序没有电机I/O。
两份长期规范完整保留：
docs/handoffs/microduck-local-ssh-push-protocol.md
docs/handoffs/microduck-cloud-gpu-and-transfer-protocol.md
本地仓库/home/lx/microduck-double-balance/workspace；下载/home/lx/下载。
分支double-balance；远端git@github.com:liuxue-lab/microduck-double-balance.git。
回传分支stage12-push10-final6h-return-20261023t181130z-1b79efbd-s002；
回传提交0cbee30e5a83904371076ca09a8cfb4afd1fa3f4，400 files/5 archives已核验。
不能把已回传摘要写成云端全部原始轨迹已备份，完整范围以既有索引为准。
剩余云端任务NONE；云端电源OFF_USER_CONFIRMED，2026-10-24用户确认，精确时间未知，
未向平台核实。历史ssh -p 32185 root@connect.bjb2.seetacloud.com仅作追溯，不连接。
常见Git冲突/认证/下载文件改名按两份规范处理；不改HTTPS/PAT，不强制推送。
本次无电脑↔云端或电脑↔Orin传输；只由用户通过Git SSH推送本轮源码。
下一步根据采集结果选择优化点，再做真实ROS完整周期验收；当前没有硬件验收结论。
