Microduck Stage12 local-completion-v5 — 2026-10-16

当前结论与进度
已通过GitHub读取最新v4固定提交，重建并完整核验31个分片、527个返回成员和493个程序成员。
Stage11仍为11/11 CLOSED_SCOPED（100%）。Stage12 CPU13/13、配置21/21、410项云输入准备已完成并复用。
v4已成功构造真实RL环境并完成初始reset，执行3次CPU forward、1次CPU kinematics、2次GPU forward。
接触排序图已关闭，最终context的42个资源已正常释放（39 Mesh、1 Bvh、2 Texture2D）。
新阻塞为状态所有者登记把NullRecorderManager当成RecorderManager访问_terms。
公共step=0、macro=0、积分新增0；N1未通过，N2未开始，物理验收仍0/2点通过；没有有效吞吐测量。
累计仍为674456 completed / 674457 attempted；PPO=0；cloud_training=0；不编造Stage12整体百分比。

v5已修复这处已知接口故障，并增加真实管理器的完整状态链验证。
全部可在当前无GPU环境中完成的相关检查通过，不等于未运行的GPU数值验收已经通过。
当前还不需要开云电脑。下一条本机命令连续执行剩余N1/N2、对账、归档和传输准备；通过后生成云端准备命令并停止。
训练预算另行明确，不自动push、PR、merge，不自动开关云电脑，不重跑任何已完成批次。

怎样更快解决问题
本轮v4原生会话约55.7秒即到达Python接口错误；主要问题是正式配置与审计适配器的接口覆盖不足。
加速做法已落实：
(1) 对真正使用的NullRecorderManager和普通RecorderManager分别验证，避免仅靠通用模拟对象通过。
(2) 在构造任何真实环境之前运行5项独立接口检查，一次记录所有独立失败；CPU13项旧报告继续复用。
(3) 检查登记→快照→修改/重绑定→恢复→真实step/reset方法→终止/截断→关闭，而非只验证出错语句。
(4) 本机成功构造后自动记录所有者字段/类型/数组布局；宏失败另留字段清单，后续排查可一次看到同类问题。
(5) 一个命令自动走完N1/N2及回传准备，成功后直接生成带实际commit的云端命令；中间不再拆成多个手工准备批次。

如果执行环境能够直接访问本机workspace和RTX5060，修改、日志读取与真实GPU验收可在同一会话完成，
会减少人工回传的等待。当前聊天通过GitHub读取证据，不具备你本机的执行通道，因此不能宣称已代跑本机GPU。
本包继续使用你确定的手动Git SSH push流程，不要求新增远程访问或现在租云电脑。

1. 本轮固定回包及实际消耗
仓库：liuxue-lab/microduck-double-balance
分支：stage12-local-completion-v4-transfer-20261016t010649z-420d06
提交：589564d97d2242f0af2a1ab6989c9a8355fd7291
运行：20261016T010649Z-420d06
本机根：/home/lx/microduck-double-balance/artifacts/double-balance-stage12/local-completion-v4
返回ZIP：stage12-local-return-20261016T010649Z-420d06.zip
字节：16127778
SHA256：0c47cd47aec9e12c62b1e1cc20f8971206969de8f758b14e78ab51543d132518
PACKAGE SHA256：94179d5f815fc5e59315a915c56a9b62c83777c93879c450dbdcdc44f2c12127
状态：LOCAL_COMPLETION_STOPPED_PRESERVE；preservation=PASS；32份证据冻结在prior-local-completion-v4/。
所有分片文本/原始字节、SHA、偏移、ZIP/CRC、RETURN-MANIFEST均核验；程序与交付v4逐字节一致。

错误：baseline/g11/formal_owners.py roots_for第136行访问manager._terms。
真实正式配置recorders={}，mjlab因此创建NullRecorderManager，原生字段只有active_terms=[]、cfg=None。
这不是GPU显存不足，也不是NVRTC长路径重现；v4短目录编译和完整环境关闭在主机已成功执行。
已构造1个真实环境；初始reset完成；3次CPU forward和2次GPU forward的10行WAL完整且无pending。
CPU kinematics完成1次。所有完整native_counts和资源关闭证据固定保留，不能据“积分0”把这些操作清零。
v3此前的2次CPU forward、1次CPU kinematics及1次失败构造尝试也保留，旧预算不返还。
N1会话退出exitcode=1，未强制中断；进程已经关闭，不能从其内存resume。
v5是新的有界验收尝试，仍需新构造环境；它不重新执行已经完成的CPU13项、源码收集或历史物理批次。

2. 修复及验证
新增integration/owner_roster.py，只在冻结roots_for的记录器分支增加显式类型/schema分派。
NullRecorderManager必须是已知精确类，cfg=None、active_terms为空、确实没有_terms；其active_terms列表照常进入快照。
普通RecorderManager保留全部_terms，每个term照常登记；未知类型、伪造_terms或不一致列表仍拒绝。
没有给真实空管理器伪造_terms，没有跳过全部recorder状态，没有改变冻结Stage11源码或放宽快照验证。
setup、每个macro及rng_policy统一使用同一个活动roots_for，消除只修一条调用路径的风险。

新增integration/manager_contracts.py，在CUDA环境构造之前完成5项真实管理器接口检查，收集所有独立错误。
新增integration/owner_inventory.py，记录真实对象字段、类型和数组布局；不复制数组数据，不调用管理器方法，
不推进物理、不抽取RNG；仅为诊断，不能作为状态快照、恢复依据或验收PASS。
manager-interface-preflight.json、owner-inventory.json和宏失败字段清单都进入回包。

本轮已通过：
- 使用真实NullRecorderManager重现v4的同一个_terms异常，并验证修复后的完整所有者登记。
- 使用9类真实mjlab管理器、固定版本ManagerBasedRlEnv的step/reset/close方法、真实Torch和Warp CPU数组，
  检查空/普通记录器×正常/terminated/truncated/两者同时，共8种组合；覆盖原对象重绑定、tuple缓存key、
  私有RNG容器、active_terms、真实DelayBuffer和tensor状态的精确恢复、终止观测保留及后续连续调用。
  场景/仿真/宏推进是明确标注的CPU fixture；未调用ManagerBasedRlEnv.__init__、MuJoCo forward或物理积分。
- 3个真实spawn进程中分别执行上述8种组合、5项接口检查、原有清理检查及生产bootstrap/IPC/N1/N2编排。
  每进程独占短目录和边界拒绝仍通过；fixture消耗不加入真实累计。
- 60项传输、账本、Git、目录隔离、原数值辅助类AST、时钟跨20宏、预算/控制器/checkpoint门禁单元检查通过。
  控制器更新/checkpoint检查使用合成字节与假算法，不是PPO更新或旧checkpoint加载。
- 18项旧v1/v2/v3/v4证据复用、防篡改、防重复启动检查通过；保留v4已完成构造及10行WAL。
- 10项交付检查通过：真实本地Git分片往返、固定commit云解包/命令、诊断报告回包、篡改/零预算等拒绝。
  交付测试的物理PASS回执为合成fixture，不能用于真实放行。
- 156份baseline文件逐字节不变；活动预算、账本、终止适配、进程通信、控制器等8份关键文件逐字节不变。
  roots_for去除唯一显式空记录器分派后与冻结AST相同；全部数值辅助类AST相同。

上一版真实NVRTC六模块离线编译结果保留；本版编译路径和CUDA数值核未修改，不重复无关编译。
完整报告在validation/results/；历史结果在inherited目录。开发执行真实物理积分0、PPO0、云训练0。
不能承诺尚未执行的GPU数值路径“绝对没有下一处错误”；新错误会保留全部已消耗操作和现场并停止，不自动重试。

3. 一次本地运行的完整命令
下载新ZIP到 /home/lx/下载，保留旧ZIP和旧解压目录。

unzip -n /home/lx/下载/microduck-stage12-local-completion-v5-20261016.zip -d /home/lx/下载 &&
python3 -I -S /home/lx/下载/microduck-stage12-local-completion-v5/microduck-stage12-local.py --verify-package &&
python3 -I -S /home/lx/下载/microduck-stage12-local-completion-v5/microduck-stage12-local.py --run --native-integrations 38720 --wall-seconds 43200

源码：/home/lx/microduck-double-balance/workspace
现有解释器：/home/lx/microduck-double-balance/workspace/.venv/bin/python，不resolve为系统Python。
新证据根：/home/lx/microduck-double-balance/artifacts/double-balance-stage12/local-completion-v5
执行顺序：核验旧证据/身份 → 复用CPU报告及410输入 → N1 → N2 → 对账 → ZIP → 本地传输提交。
中间不用回聊天确认；任何真实失败立即保留并停止，不自动补跑、不扩大上限。
本命令只授权这次有界本地仿真验收，不授权PPO或云计算。既有v5 started存在时只显示状态并退出。

看到LOCAL_TRANSFER_COMMIT_READY_MANUAL_PUSH_PENDING后，手动执行：

git -C /home/lx/microduck-double-balance/artifacts/double-balance-stage12/local-completion-v5/transfer push origin HEAD

粘贴push结果与Stage12TransferCommit即可；不用上传返回ZIP。
助手从固定完整commit读取；正常push后不要求你重复全套远端验证。GitHub显示的PR链接不用点击。
push失败只重试上面的push，不再--run，不删除started、不force push、不合并传输分支。

只读状态：
python3 -I -S /home/lx/下载/microduck-stage12-local-completion-v5/microduck-stage12-local.py --status

只读限额：
python3 -I -S /home/lx/下载/microduck-stage12-local-completion-v5/microduck-stage12-local.py --plan --wall-seconds 43200

只读GPU查看，Ctrl+C只退出此监控：
nvidia-smi -l 2

campaign.log的完整路径每5秒随心跳打印；owner下progress.json记录公共step、macro、计数及WAL pending。
bootstrap-report.json位于owner-0-bootstrap等目录，未放入会被回传排除的runtime-scratch。
仅当return-receipt.json已成功形成、但传输提交未完成时，可只执行：
python3 -I -S /home/lx/下载/microduck-stage12-local-completion-v5/microduck-stage12-local.py --prepare-transfer
此命令不重新进行CPU检查或物理验收。若归档/receipt未形成，保留现状和错误供审查，不伪造回执。

4. 保留/复用与原有限额
v1固定提交623484cd5dbb8182205dc035dd13fc95d831b821的CPU13/13和410项输入继续复用。
cloud-inputs.zip：16181045字节，SHA256=24701df9e56face7980a1a2e0544dfebc41a4cf1b84ec4912be795d1a35a43b2。
先校验旧ZIP、started对应run/PACKAGE、return/transfer回执及所有保留报告，再复制输入，不重采源码或资产。
新校验明确区分：v1原生owner目录从未建立；v2原生owner目录必须保留，WAL必须仍为空、原生报告零消耗，N2仍未开始。
另核验v3的29份报告和归档：必须保留2次CPU forward、1次CPU kinematics、1次失败构造及4行WAL；N2仍未开始。
还核验v4的32份报告、3次CPU forward、2次GPU forward的10行WAL、已完成构造和42资源释放记录。
任一证据变化就停止重新对账，不能套用零消耗结论。新v5是一次新声明的构造/验收尝试，不能从已经关闭的v4环境内存resume。
新批次单独计费/计数，不返还失败尝试预算；不重跑任何已经完成的物理点或CPU检查。

计划SHA256仍为12ef65c6f34c555f923ecf7b11c555402418612e17eeec28a59ff4dbfcbecd77。
N1：1个owner、5个公共step；N2：2个独立owner、各3个公共step。每进程native nworld=1。
每点包含1个warmup、2个计时step，检查缓存观测、显式/空reset、未选中owner不变和关闭回执。
总上限：3个真实环境、110 macro、38720内部积分、155902 GPU forward、144 CPU forward、18次reset尝试。
策略/优化器构造0，torch.load/save=0，PPO=0，cloud_training=0。
预留证据11.375 GiB，另需4 GiB余量；建议artifacts所在文件系统至少16 GiB可用磁盘，低于2 GiB停止并保留。
每个owner另在/tmp检查至少1 GiB可用空间；若/tmp为独立分区，不能用artifacts余量代替。
43200秒为本次计算子进程的12小时上限，不是耗时承诺；身份检查、清理、打包另计。
超时只结束本次创建的进程组，不终止无关进程，不自动重试或开云实例。

保持physics_dt=.002、decimation=10、public_dt=.020、10秒episode、actor61/critic85/action14、
真实头部几何、BAM、hold=0、32/64/128试探和mandatory exact replay。
保持float32累计>=5秒判定，不增加epsilon/rounding，不放宽任何阈值。
短pilot不证明完整10秒episode、持续5秒稳定、长期吞吐或学习效果；计时含IPC、审计、同步，启动/JIT单列。
公共transitions/s与内部integrations/s分开；driver free/total是采样，Torch峰值按owner报告。
不从N1/N2自动扩展到4/8/4096，不默认换更快GPU就能消除CPU审计/IPC成本。

5. 什么时候需要云电脑，什么时候正式训练
若输出LOCAL_NATIVE_PILOT_PASS_CLOUD_BUDGET_PENDING且preservation=PASS：本轮本机验收结束，
下一步才需要明确云GPU型号/时长/费用并进行云端验收。现在不必启动云电脑。
若输出LOCAL_COMPLETION_STOPPED_PRESERVE：保留所有证据，上传固定回包；不把传输成功当物理PASS。
v5的真实积分必须以新WAL汇总更新当前674456/674457基线，不能提前假定仍为零或全部完成。

本地PASS后自动生成（仅生成，不执行）：
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/local-completion-v5/CLOUD-COMMANDS.sh
完整内容已填入本次真实传输分支和40位commit，因此不必再等助手补分支名。
手动push成功且预算另行明确后，在当时提供的云主机执行其中命令；不用ZIP往返。
查看文件：
cat /home/lx/microduck-double-balance/artifacts/double-balance-stage12/local-completion-v5/CLOUD-COMMANDS.sh

云命令要求STAGE12_GPU_HOURS、STAGE12_SPEND_LIMIT、STAGE12_CURRENCY为已同意的正数/币种，默认不代填。
采用Git SSH独立clone → 固定commit/分片/程序/410输入校验 → uv sync --locked --python 3.12.14 →
来源及依赖/CUDA设备清单。入口为cloud-stage12.py，停在CLOUD_ENVIRONMENT_IDENTITY_PASS_NATIVE_AND_PPO_PENDING。
它不会直接调用train、构造PPO、加载旧checkpoint或自动开关机。记录的云时长/费用不是平台计费停止器。

正式开训需完成以下真实条件，不以“再传几次”估计：
(1) 本地N1/N2有界物理验收通过；(2) 已选云设备的源码/资产/依赖、native生命周期及完整episode/reset验收；
(3) 另行明确N/B、GPU小时、费用、云积分/forward、模型/优化器构造、PPO更新和checkpoint读写预算，
在该预算内通过真实更新及保存/恢复验证，然后才放行正式训练。
cloud-budget-template.json仍为PENDING、PPO_updates=0，历史5090/A800预算不代替本轮授权。
native_factory当前限定本地5060；其他云设备需依其实际配置与预算适配，不能直接拿本机命令开训。
本包准备了控制器和checkpoint策略，不宣称已交付可绕过这些条件的正式训练启动器。

6. checkpoint与训练接入策略
旧checkpoint只读保留，不反序列化、不上传、不直接resume：
/home/lx/microduck-double-balance/artifacts/double-balance-stage08/returned/5d998fc322345102/evidence-20260930T093645-8955/extension/E-20260929/segment-001/checkpoints/update_004000.pt
10073500字节；SHA256=86d55c3703c18fcf4817db49c6e4dcf839b3f5195d68ae2c559db7e222bded97。
schema SHA256=e740e6ca4ad74c53a0eaaa9c95dcae9b69fd5b0edf122593743cd015dcb388a7。
任务/特征/时序不同，旧normalizer、Adam和LSTM状态迁移仍BLOCKED。
计划为fresh actor LSTM61→256单层、MLP512/256/128→14；critic85→512/256/128→1，统计与优化器重建。
专用24步更新循环保留终止观测，truncated且非terminated才加一次terminal V bootstrap。
N1/B1，N2/B1或B2；不能继承B4。只读缓存不能偷偷推进环境或RNG。
每次更新先写attempt；checkpoint独占partial、fsync、无覆盖原子发布、SHA回执，最后提交更新账本。
仅恢复最后已提交更新，身份/hash/账本验证后weights_only=True加载，恢复Adam及学习率标量。
更新边界重新reset环境/RNN/rollout并恢复policy RNG，不宣称中途轨迹逐位续训。
24步rollout内部积分上限N1=84480、N2=168960，setup/reset另计；只是预算算术，不是运行授权。

7. 固定身份、协议、错误处理与云状态
工作区：/home/lx/microduck-double-balance/workspace；下载：/home/lx/下载；产物在仓库外artifacts。
正式分支double-balance；SSH origin=git@github.com:liuxue-lab/microduck-double-balance.git。
已审查HEAD=73ee01930a031d76a9d57419afdfd4464a547ab3。
Stage11为11/11 CLOSED_SCOPED，stage-11-complete已推送。
关闭commit=6ba0dc2de381321576f32310e1995a8e2e11e566；tag对象=bd1b219be55c813811d4f7cfa25d245b37166688。
tracked diff SHA256=b90daa0274c14d95cec9511ec4ac93568bd604e93ce3c60df1bb1c4155005d7e，定义仍为decode().strip().encode()。
不改主checkout，不清理旧补丁，不重跑Stage06/Stage11/readiness v1–v7/integration-v1/local-completion-v1/v2/v3/v4已完成工作。
所有started、归档、旧checkpoint保留；原始native快照和缓存留本机，回包包含关键报告、WAL、非缓存证据SHA索引。
单ZIP≤49 MiB，传输分支不能合并到double-balance；本次也不创建正式阶段标签。

必须长期保留并引用：
docs/handoffs/microduck-local-ssh-push-protocol.md
docs/handoffs/microduck-cloud-gpu-and-transfer-protocol.md
本包完整冻结副本在baseline/g11/provenance/。历史文件内容不改写为新事实。

常见错误：
- push失败：仅重试同一transfer目录push；不改HTTPS/PAT/网页登录，不重跑计算。
- non-fast-forward/同名标签冲突：停止审查，不force、不覆盖。
- Permission denied(publickey)：复用现有SSH key/agent，不默认新建密钥。
- 下载文件带(1)：使用真实文件名并加引号；不把旧包当新包、不覆盖旧解压目录。
- SHA/源码/started变化：保留原物，停止对账；不能改期望值跳过校验。
- 已started但无最终回执：先读原进程/日志/WAL，不能凭无进程推断没消耗或直接重启。
- v1 cwd、v2 uname -p、v3 NVRTC长路径/部分关闭、v4 NullRecorder所有者登记，均保留原报告及验证。
- 原始证据未回传：索引/摘要不是完整备份，不删除本机原始文件或云盘。

2026-10-01用户曾确认5090关机，这不是当前电源证据；当前云电源=UNCONFIRMED。
本轮没有启动、连接或训练云主机；本阶段SSH地址/端口尚未提供，不复用旧端口作永久地址。
未来云计算和必要打包/回传结束且本机SHA校验通过，立即提醒用户手动关机；提醒与实际确认分别记录。
GPU利用率0不证明任务结束；作业退出不等于停止计费。不自动开关实例，不重启已禁用的旧关机daemon。

下一次继续语
继续microduck-double-balance Stage12，读取最新local-completion-v5 Git传输固定提交和本交接。
已审查v4提交589564d97d2242f0af2a1ab6989c9a8355fd7291：真实环境构造/初始reset完成，
3次CPU forward、1次CPU kinematics、2次GPU forward，最终42资源及接触图正常释放；
NullRecorderManager没有_terms导致所有者登记停止。积分新增0，累计674456/674457；物理点仍0/2通过。
v5已显式适配空记录器，并通过真实管理器/固定step-reset方法/CPU状态链及3进程开发检查。
已完成CPU13/13与410输入继续复用；后续GPU验收以新回包和WAL对账，不将fixture或push成功视为PASS。
保留全部阈值、started、旧包、checkpoint，不重跑完成批次；PPO=0、云训练=0，预算另行明确，不自动push/PR/merge。
