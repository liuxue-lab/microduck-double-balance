Microduck Stage12 PC 同步开销优化 v1 — 2026-10-24

当前状态与目标
用户截图已确认 TIMELINE_FIX_INSTALLED_VERIFIED，并成功推送
02d34b4..f33210a double-balance -> double-balance。
当前父提交 f33210ad7c474161b85e9f4cb5dc6db65f877db7。CUPTI采集故障已经关闭，
本轮不再诊断/升级CUPTI，不重新训练、导出模型或连接云端/Orin。
直接目标：降低电脑环境步耗时，同时保持原物理、动作、延迟、RNG和审计结果。
此前无分析器均值24.9927876ms、p95 26.278897ms，150/150步超过20ms。
完整ROS2/Orin 50Hz实时验收仍未通过；PC环境步不等于完整闭环周期。

证据与改动依据
report(20261024-133007).json SHA256：
1c12ea0d8602dfc3aeb83b9662936b5145a1ef659f936ae1e52e8d1fc2f8e114
采集状态PC_CUPTI_REPAIR_TIMELINE_MEASURED，8步35522个CUDA核函数；73440个共同状态
标量差异全零。8步含1192次cudaStreamSynchronize、184次aten::nonzero，以及40次
quat_from_matrix。正常宿主scene_write约4.07ms/控制步；含分析器的apply_delay
约21.69ms/8步，quat_from_matrix约19.43ms/8步。这些分析器时长不能当正常收益预测。

源码确认两个同步来源：
1. 原CircularBuffer每次append中的torch.any及读取中的torch.all会让Python等待GPU。
   本轮保留原DelayBuffer._update_lags/_sample_lags，使用原次数、原顺序的随机采样。
   只将历史回填和选帧改成固定形状GPU选择。保持初始回填、回绕、lag裁界和重置内容。
   输出是本次独立张量，由原BAM暂存立即消费，不复用可被下次写覆盖的历史视图。
2. 原quat_from_matrix的正数开方和最终one-hot布尔索引具有动态输出长度。
   本轮使用where和gather固定输出形状；保留四个候选四元数、原浮点运算顺序、
   0.1分母下限、argmax并列选择和wxyz约定。仍为PyTorch原TorchScript/FP32；
   不增加torch.compile、fast-math、精度转换或新的运动学缓存。

官方说明与接口：
https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/sync-free-code.html
https://docs.pytorch.org/docs/2.9/generated/torch.gather.html
https://docs.pytorch.org/docs/2.9/generated/torch.where.html
这里根据已采集时间线和固定源码作出优化选择，不声称已实测出具体加速比例。

一次运行的内容
--apply 自动导入内嵌增量Git bundle，再顺序运行四个独立进程：
baseline（原保留方案）、quat（只改四元数选择）、delay（只改延迟历史选择）、combined。
每组固定160控制步/1600物理步、3.2仿真秒，环境种子2026102407，外力种子2026102500；
包含2.0–2.1秒一次完整10%自重水平扰动。各组排除前10步，统计剩余150步。
四组均不开启CPU/CUDA profiler或cProfile，以完成CUDA计算后的环境步墙钟时间计时。
动作策略与对照轨迹保存位于计时范围外，不能将该时间当完整ROS闭环时间。

每个候选与baseline比较原22类共同状态（73440标量），要求数值差异全零，并核对
完整Audit、Trial和扰动。新增初始及每步结束的延迟历史/RNG字节快照共161行，
比较全部历史值、lag、步计数、相位、环形指针及CPU/CUDA随机数状态。
暖机内四元数前40次同输入FP32位比较；延迟前40个物理步用独立原实现对照，
包括RNG消耗。影子对照的随机数状态在finally中恢复；不增加物理积分。
若暖机结束仍未完成同输入对照，则停止，不把额外核验开销混入正式计时。

选择规则保持既定5%门槛：均值和p95都需改善至少5%。从合格者选择均值最快者，
自动在新进程追加candidate-confirm及baseline-confirm，反转对照顺序。
两次候选的均值及p95都必须比两次baseline中对应的较快值改善至少5%，才给出
VERIFIED_QUAT_CANDIDATE / VERIFIED_DELAY_CANDIDATE / VERIFIED_COMBINED_CANDIDATE。
否则KEEP_INFERENCE_ONLY。无合格者只跑四组；最多六组960控制步/9600物理步。
数值、RNG、边界、源文件或进程异常立即保留停止，不通过调门槛、重采种子制造通过。
不自动启用生产方案。短程性能选择不能替代60秒功能与完整周期实时验收。

运行命令（一次粘贴）
下载 microduck-stage12-pc-sync-v1-20261024.py 至 /home/lx/下载：

python3 "/home/lx/下载/microduck-stage12-pc-sync-v1-20261024.py" --apply &&
git -C /home/lx/microduck-double-balance/workspace push origin double-balance

脚本本身无网络需求；第二条git push沿用用户已有Git SSH身份。
本轮只使用电脑RTX5060，5090保持关闭，无需连接Orin。保持主终端开启。
每个worker墙钟上限230秒，物理控制边界上限180秒；正常预计几分钟。
进度为每组控制步百分比与已完成组数，不是网络下载进度。

结果只需回传一份：
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-pc-sync-v1/report.json
成功测量状态PC_SYNC_MEASURED不等于达到50Hz；候选选择见recommendation。
所有完整日志、共同状态与RNG/延迟对照快照保留在报告directory所指目录。
失败状态PC_SYNC_FAILED_PRESERVE也上传同一报告，不必分别上传各组日志。

固定入口与恢复
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-pc-sync-v1/microduck-stage12-pc-sync-v1.py
--status 单次查看；--watch 每2秒刷新，完成后继续显示，Ctrl+C只退出观察。
--stop 请求停止本入口的活动对照并保留结果。
--run 遇到已有结果不重复；--retry-run是明确复测入口，不在失败后自行重试。
前台主终端Ctrl+C/断开会中止；先--status检查，不以旧RUNNING推断仍运行。
导入只增加experiments/stage12/pc-sync-v1目录，不覆盖365个冻结原源码或依赖包。
暂存、未暂存、未跟踪与忽略文件冲突均保留；不reset、stash、clean或force push。
增量bundle父提交固定f33210ad7c474161b85e9f4cb5dc6db65f877db7，目标提交见build-receipt。
新运行包从固定Git对象导出，已安装源文件变动会被拒绝。
不创建stage12-complete标签；正常push成功输出即可，无异常不重复远端核验。

冻结项目约定
范围：电脑仿真↔有线ROS2 Humble↔Orin Nano Super FP32 ONNX，无真机电机接入或真机演示。
原保留组合：InferenceOnly + PhaseCache + BAM Graph v2 + BatchedAuditReads。
JSON + FP32；INT8此前最大动作误差约0.04030而拒绝，未启用剪枝或未经验证的FP16。
actor61/critic85/action14、LSTM h/c=[1,1,256]，归一化烘焙，关节顺序不变，不添加动作滤波。
控制50Hz、物理500Hz、电机/关节/外力检查500Hz；不改积分器或碰撞、BAM参数、物理频率。
final6h_005304.pt SHA256 cf7ce9297672943687019afe071d94918d3eb2620c18a1e8fe689512f8c08111
final6h_005304.onnx SHA256 8c1c1250b9b88d9a91e3eb095fea8bce21daa73c251b43f3cd8adc1327f03c4e
导出目录 /home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-ros2-humble-v1/frozen-export
heldout18/18、campaign24/27；6000步功能闭环通过，不等于50Hz实时验收。
允许位置漂移，strict_contact_acceptance=false，不将有限几何保持写成严格持续接触。

仍未完成：PC性能优化 → 最终60秒ROS2/Orin完整周期验收 → 部署定版 → 材料与归档。
完整周期还含通信及Orin推理；需报告均值/p95/p99/最大值/超过20ms次数，按既定截止门槛判定。
后续Orin操作沿用既有共享SSH会话，jetson@192.168.50.2，PC192.168.50.1，ROS Domain62。

云端与长期协议
云任务NONE，5090为OFF_USER_CONFIRMED，用户2026-10-24确认；精确关机时刻未知。
不探测历史实例、不启动云GPU、不增加PPO、不重新做Stage06 smoke。
已核验回传400文件/5归档，非所有云端原始轨迹。
回传分支stage12-push10-final6h-return-20261023t181130z-1b79efbd-s002，
回传提交0cbee30e5a83904371076ca09a8cfb4afd1fa3f4，已完成回传不重做。
固定仓库/home/lx/microduck-double-balance/workspace，分支double-balance；
固定远端git@github.com:liuxue-lab/microduck-double-balance.git。
完整长期规范附在本目录，后续交接继续保留：
docs/handoffs/microduck-local-ssh-push-protocol.md
docs/handoffs/microduck-cloud-gpu-and-transfer-protocol.md
模型、日志和完整轨迹放仓库外；源代码增量bundle导入后由用户SSH push。
下载带(1)后缀时使用准确实际文件名；SHA不符或Git分歧立即保留停止，不改HTTPS/PAT。

开发验证范围
使用PyTorch2.9.1 CPU与固定原始源码进行等价测试：16384组随机姿态、极值/非有限值、
四元数并列分支、延迟回填与回绕、部分重置、lag裁界、RNG保持、失败恢复与绑定清理。
完整四模式流程采用原BAM/延迟/审计CPU夹具及模拟积分，不能冒充RTX5060实际物理测试。
另测选择门槛、确认基准、缺失快照拒绝与真实Git增量导入的工作树保护。
此开发环境无CUDA，具体性能和最终完整模型数值对照由--apply在用户电脑上测量。
