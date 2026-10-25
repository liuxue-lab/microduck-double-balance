# Microduck Stage12：累计优化完整闭环验证交接

更新：2026-10-25，Asia/Shanghai。范围仍为 PC 仿真↔有线 ROS2 Humble↔Orin Nano Super FP32 推理；不接电机或实物机器人，不做真机演示。Stage13/14 行走取消。项目尚未最终验收。

持续遵循[本地 SSH 推送规范](microduck-local-ssh-push-protocol.md)及[云 GPU 与传输规范](microduck-cloud-gpu-and-transfer-protocol.md)，两份全文保留。历史命令只供追溯，不授权重做已完成工作。

## 已核实状态

原 `TIMELINE_ARM_FAILED_PRESERVE` 已解决：旧 CUPTI12.8.90 在 RTX5060 上产生 `CUPTI_ERROR_INVALID_DEVICE`，隔离使用12.9.79后完整 GPU 时间线成功。固定入口 `native-pc-timeline-fixed-v2` 已验证，用户截图确认提交 `f33210ad7c474161b85e9f4cb5dc6db65f877db7` 正常 SSH 推送。无需重装修复、修改驱动或重复远端核验。

最新 `report(20261024-160716).json`：`PC_ACTIONGRAPH_MEASURED`，errors=[]，5组、800控制步、8000物理步。五组交叉对照的73440公共状态标量差异均为0，完整 Audit、Trial、扰动一致，161次动作历史/目标/辅助力/延迟/RNG快照完全一致。

| 方案 | 均值ms | p95ms |
|---|---:|---:|
| baseline |25.091286|26.285986|
| action-graph |24.511496|25.891798|
| cumulative |22.452714|23.670329|
| candidate-confirm |22.917374|24.269924|
| baseline-confirm |25.283663|26.687299|

结论 `VERIFIED_CUMULATIVE_CANDIDATE`，confirmation_passed=true。两次累计测量的均值/p95均比两次基准中更快者改善至少5%；所有正式计时步仍超过20ms。该报告证明短程一致性和可复现收益，不能证明完整60秒或50Hz实时HIL通过。

成功报告：`/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-pc-actiongraph-v1/report.json`。
SHA256：`a48093e63cce12fa50052f00b425e8eb28542e16aed78210c374e7655505a3c7`。
已安装源提交：`25378e7fccbdba8b127b593eeaa47403c56ca866`，包身份 `5c7513b74ecfef83be357883e921508a5c61fb402d3dde45fdde2d996c108cca`。最新报告不证明该提交已经push。

此前pc-sync报告SHA256 `afe6ba189a35733df2470d8e95f4f7600070e5752eab99669e851cccdd8a167a`，未满足收益门槛，原历史结论保留。修复成功报告SHA256 `1c12ea0d8602dfc3aeb83b9662936b5145a1ef659f936ae1e52e8d1fc2f8e114`；固定CUPTI库SHA256 `2fdab19dc3fccdd4b2f5eba137aabefc68e685f6497bb38eea649866c5672a2a`。

## 本轮工作与判定

新入口使用已验证累计方案：InferenceOnly + PhaseCache + BAM Graph v2 + StaticQuaternion + ScalarAudit + PackedChecks + ActionGraph。动作图、四元数适配器与数学函数逐字节复用并校验成功安装包；其余适配器来自固定旧timeline包。仅作用于本轮新建闭环进程，不做全局生产开关。原延迟抽样、随机数、物理步、扰动、500Hz检查阈值和频率保留，不启用FixedDelay、CommandViews、ReadGraph或HeadGraph。

复用此前成功的ROS2 JSON协议、Orin FP32推理路径和SSH连接。只跑两组：quiet-2407（环境seed2026102407，无扰动）与regression-2500（同环境seed，force_seed2026102500）。每组3000控制步/30000物理步/60仿真秒；墙钟时间可以更长。每个动作均来自Orin并与本机冻结CPU参考策略比较，无本地动作回退；h/c仅在请求0、3000重置。三方6000请求的哈希链必须一致。

状态 `CUMULATIVE_LOOP_FUNCTIONAL_PASSED_REALTIME_PENDING` 表示新累计方案完成两组功能闭环；**不能解读成50Hz实时通过**。本轮为lockstep，保持原1秒RPC异常保护，它不是20ms实时截止期。原20ms实时HIL要求保留待办，不以推理均值、部分步骤耗时或排除预热后的统计代替。功能失败时停止首个失败回合，保留证据，不补采或隐式重试。

用户已提出仿真与实际部署的区别。在线HIL需要仿真被控对象根据Orin当前动作生成下一观测；离线录制数据回放不能验证这种闭环。真机运行时由真实物理系统产生状态，不要求同时跑MuJoCo；日志、曲线和显示可独立处理。本项目不扩展真机接入范围。PC仿真超过20ms，不等价于Orin部署推理无法50Hz。旧推理FP32平均0.352815ms/p95 0.388907ms，JSON调度到验证平均2.593678ms/p95 3.052529ms，是历史独立证据，不代替本轮完整数据。

## 数据与时钟

每组保存timing.csv，分别记录观测拷贝、完整请求调用、IPC往返、ROS往返、Orin推理、本机参考比较、动作传输、环境步、状态采集、活动周期及相邻控制起点间隔。嵌套区间不能相加。Orin推理在Orin本地计时，不跨主机相减时钟。环境耗时含CUDA完成和全部原检查；状态采集计入活动周期。相邻起点间隔还包括外层检查、周期性日志；末步没有下一起点，值为空。全部3000步统计保留，并单列去掉初始10步的诊断视图；后者不作为降低实时标准的依据。

每组state-data.npz保存初态及3000个50Hz qpos/qvel/ctrl快照，state-data.json说明字段。原push10-trace.jsonl保留指标与扰动；loop-transcript.jsonl含所有观测和远程动作。用于运行后画曲线或结合相同冻结模型制作状态回放。记录回放不是新闭环仿真，也不是可恢复完整仿真内部状态的checkpoint。本轮不增加在线渲染。最终报告记录导出文件SHA256。

## 下一条命令

连接Orin电源与既有网线，保持5090关闭。下载到 `/home/lx/下载` 后执行：

```bash
python3 "/home/lx/下载/microduck-stage12-cumulative-loop-v1-20261025.py" --apply &&
git -C /home/lx/microduck-double-balance/workspace push origin double-balance
```

主终端保持开启。复用已有SSH共享连接；若已失效，可按原流程正常输入一次Orin登录密码，后续不重复认证，不读取或存储密码。

只需回传：

```text
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-cumulative-loop-v1/report.json
```

报告包含两组结果、适配器计数、通信哈希链、耗时和日志尾部。数据留在同目录runs/唯一运行目录内。通过后在该运行目录写validated-pipeline.json，标记累计方案完成完整功能验证，实时验收仍待完成。

固定入口同目录 `microduck-stage12-cumulative-loop-v1.py` 支持 `--status`、`--watch`（2秒刷新，完成后保持）、`--stop`。`--run`复用已存在结果，不自动重跑；确需重试时 `--retry-run` 建新目录，保留旧记录。共用native-ros2-humble-operation.lock，拒绝并发。

## 不变模型与接线

- 检查点final6h_005304.pt，本轮PPO5304/总11314；heldout18/18，campaign24/27，不能写27/27。
- 检查点SHA256 `cf7ce9297672943687019afe071d94918d3eb2620c18a1e8fe689512f8c08111`。
- 原验证报告SHA256 `d052f3cf8031be5ec5685fc1aa02c906301932d2a4849ac34fd450206b359228`。
- ONNX SHA256 `8c1c1250b9b88d9a91e3eb095fea8bce21daa73c251b43f3cd8adc1327f03c4e`。
- 冻结导出 `/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-ros2-humble-v1/frozen-export`。
- actor61/critic85/action14；h/c各[1,1,256]；归一化已烘焙，无关节重排、滤波、额外归一化或裁剪。保持FP32，不重训、重导出或使用INT8/未经验证的FP16。
- 双球、原头壳、位置漂移允许保持；不宣称严格持续接触。扰动自重10%、水平独立均匀方向，每2秒0.1秒，2至58秒固定起始，共29次。
- PC192.168.50.1/enp4s0/jetson-direct；Orin192.168.50.2/enP8p1s0/jetson-wired；jetson@192.168.50.2；domain62，FastDDS reliable/depth1/volatile/wired-only。
- 原功能闭环已通过2×3000控制步/30000物理步及6000次ONNX比较，旧完整周期约39.9ms；本轮是优化后的新完整验证。

## Git、回传与云GPU

固定仓库 `/home/lx/microduck-double-balance/workspace`，分支double-balance，远端 `git@github.com:liuxue-lab/microduck-double-balance.git`。本增量bundle基于 `25378e7fccbdba8b127b593eeaa47403c56ca866`，仅增加experiments/stage12/cumulative-loop-v1；新提交和包身份由code-installed.json记录。只做可保留已有改动的fast-forward，不reset/stash/clean，不覆盖冲突，不自动push、不force push；用户正常SSH push输出足够，不重复全套核验。不创建新stage12-complete标签。

回传分支stage12-push10-final6h-return-20261023t181130z-1b79efbd-s002，完整提交 `0cbee30e5a83904371076ca09a8cfb4afd1fa3f4`；400文件/5归档已校验，并非全部云端原始轨迹均已回传。云端剩余任务无，5090已由用户确认关闭，OFF_USER_CONFIRMED。先前已提醒；当前证据没有确切提醒与确认时刻，不补造时间。不开机、不探测旧云主机、不运行旧关机守护。本轮只用笔记本与Orin。

## 常见错误、验证与剩余工作

| 现象 | 处理 |
|---|---|
| 功能通过但REALTIME_PENDING | 正常区分功能与实时证据；回传报告，不改成50Hz通过 |
| 源哈希、动作比较、计数、模型边界不符 | 保留并停止；不要跳过检查或修改期望哈希 |
| SSH连接失效或主机密钥变化 | 沿用既有连接规范；密钥变化先核对主机，不关闭校验 |
| 本地分支分歧、路径冲突 | 保留install-report及双方文件，不强制合并 |
| 主终端中断或结果失败 | 先读report/--status，不并发重跑；明确重试才使用--retry-run |
| 下载名带(1) | 使用实际路径，避免运行旧同名文件 |
| Git要求PAT/密码 | 保持SSH远端和既有key/agent，按全文规范处理 |
| 又看到旧timeline失败 | 旧入口仅为历史，fixed-v2已解决；不要重复重装 |

CPU测试验证实际适配器方法在合成积分器中的两组共60000子步、29次扰动、检查计数与恢复；6000真实socket数据包用合成递归策略和内存ROS节点测试编排；另覆盖首组失败后停止、超时/畸形请求拒绝、数据导出与Git安装保护。这些不是实际CUDA、MuJoCo、ONNX或DDS硬件成绩。后者需要本机报告。

剩余：审查本轮完整功能报告→保留独立实时HIL的20ms目标并处理PC仿真瓶颈/调度验证→部署代码冻结→文档、离线仿真回放及归档。若功能验证先完成，可明确记录这一成果，不用PC仿真速度否定已通过的Orin推理链路，也不取消尚未通过的实时目标。
