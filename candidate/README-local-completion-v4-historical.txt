Microduck Stage12 local-completion-v4 — 2026-10-16

当前进度与结论
已完整读取并核验最新local-completion-v3固定提交及31个分片。Git push成功只证明回传成功。
Stage11：11/11 CLOSED_SCOPED（100%），stage-11-complete已推送。
Stage12已确认：CPU检查13/13 PASS、配置preflight 21/21 PASS、410项云端输入准备完成。
Stage12未确认：本机真实物理验收0/2点通过（N1在构造中停止，N2未开始），无有效物理吞吐结果。
正式训练：PPO=0；cloud_training=0。没有统一加权的Stage12总进度分母，不编造整体百分比。
累计积分仍为674456 completed / 674457 attempted；历史1次失败保留。
注意：v3不是“完全零消耗”。它已完成2次CPU forward和1次CPU kinematics，并发生1次失败环境构造尝试。
这些消耗及4行CPU forward WAL已固定保留，不计作积分、不清零，也不抵扣新批次预算。

本包处理本次实际暴露的两处问题：NVRTC临时目录过长，以及半构造环境被当成完整环境关闭。
已完成无GPU条件下能够验证的对应回归、编译、隔离、复用和传输检查。
真实GPU构造/forward/积分尚未通过；不能把离线编译或CPU fixture称为“所有物理问题已通过”。
下一次本机命令顺序完成N1、N2、对账、归档和传输准备；通过后生成固定commit的云端准备命令并停止。
当前还不需要开云电脑。正式训练预算、GPU型号/时长/费用和PPO验收仍需另行明确。

1. 本次读取的固定回包
仓库：liuxue-lab/microduck-double-balance
分支：stage12-local-completion-v3-transfer-20261016t004935z-c472d2
提交：93fc6ff3779093f5b49c28e9883477f1c1d9f85b
运行：20261016T004935Z-c472d2
本机根：/home/lx/microduck-double-balance/artifacts/double-balance-stage12/local-completion-v3
返回ZIP：stage12-local-return-20261016T004935Z-c472d2.zip
字节：15960950
SHA256：2f049175de8ba835fe050d0eb5e34309b654439b884dcdc4b7570303c779e128
程序PACKAGE SHA256：0ec997d2175907fc42a1daf77bcbde87a1c56499a84fef3008c2a70846a72df4
31个分片文本/解码字节、偏移、SHA、ZIP/CRC/465成员及全部程序字节均核验。

实际状态：LOCAL_COMPLETION_STOPPED_PRESERVE，preservation=PASS，所有积分账本精确。
v3已越过v2的Warp CPU型号查询失败，配置21/21通过，BAM和场景构造进入Scene.initialize。
错误路径：SensorContext → create_render_context → render_util.create_warp_texture → NVRTC。
关键错误：NVRTC_ERROR_COMPILATION / name of directory for temporary files is too long。
主机临时目录：
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/local-completion-v3/20261016T004935Z-c472d2/single_owner_lifecycle/owner-0-bootstrap/runtime-scratch
失败CUDA模块：wp_mujoco_warp._src.render_util_94fd13e.cu。
随后env.close还因缺少_offline_renderer发生AttributeError；这是半构造对象的清理错误。
没有real_environment_constructed，没有公共step/macro/积分，N2未开始。
owner正常报错退出exitcode=1，未强制中断。native-ledger不是空文件：保留4行CPU_forward attempt/complete。
29份v3证据和传输清单已纳入prior-local-completion-v3/，与旧v1/v2/readiness/Stage11全部保留。

2. 本版修复及已执行的开发检查
编译路径修复：integration/runtime_paths.py为每个新owner创建独占0700的真实短目录/tmp/md12-*。
TMPDIR/TMP/TEMP、Warp/CUDA/cache/matplotlib/torch临时目录都进入此owner短目录。
实际Warp kernel_cache_dir使用同一映射，避免只改TMPDIR却继续走长缓存路径。
证据、started、WAL和归档仍在原artifacts树；runtime-paths.json和bootstrap-report.json记录映射并进入回包。
不使用符号链接，不共享旧缓存，不放行整个/tmp，不删除旧目录，也不自动清理新目录。
/tmp至少需1 GiB可用空间；短缓存只含可重建的临时/JIT文件，操作系统重启可能清除/tmp。
任一已有started出现时不能借清缓存重启；真实账本、报告、原始native证据仍须保留。

构造失败修复：显式记录env.__init__是否返回。完整环境继续调用固定版本env.close；
半构造对象只关闭确实已建立的renderer/recorder，不访问不存在的属性。
原始构造异常继续保留，真实关闭异常继续报告；部分构造的资源释放不冒充完整witness PASS。
原生分配属于即将退出的独占子进程，不复用、不resume、不自动再试。

已执行并通过：
- 真实Warp 1.12.0 + 原v3 bootstrap，在235字节长TMPDIR复现同一个render_util编译错误。
- v4在更长的证据树下使用20字节临时目录，实际NVRTC离线编译6个sm_120 CUBIN模块：
  mujoco_warp._src.render_util、mujoco_warp._src.ray、rigid_metadata_kernel、order_kernels、
  batch_kernels、solver_order_kernel。源码SHA与冻结来源一致；没有GPU加载/执行或物理构造。
- 实际spawn 3个进程执行生产native_child/bootstrap、30条真实前置import、Warp CPU初始化、
  配置preflight、N1/N2 IPC编排和关闭。只有物理会话为明确标注的CPU fixture。
  3个0700短目录互不相同；每个进程验证受限Git/字体/CPU型号回退和5种边界拒绝。
- 使用固定版本的真实ManagerBasedRlEnv.close方法及__new__对象，覆盖8类未构造/部分构造/完整关闭/
  关闭异常场景；不调用物理构造器。每个spawn进程均通过这些检查。
- 26项传输/账本/Git/短目录单元检查通过；14项旧报告/WAL复用、防篡改、防重复启动检查通过。
- 9项交付检查覆盖真实本地Git分片、云解包、固定commit命令、短目录回执回包、错误ref/分片/
  已有输出/零预算拒绝。测试的物理PASS回执是合成fixture，不用于真实放行。
- 156份baseline文件逐字节不变；10份活动数值/账本/进程/控制器文件不变。
  native_factory只改缓存映射、初始化完成标记和关闭分派；不修改数值模型、原阈值、预算或算法。

全部本轮证据位于validation/results/，旧检查结果位于inherited目录，不算作本轮重跑。
这里没有CUDA设备。开发检查新增物理积分0、PPO0、云训练0；不重跑用户已经完成的CPU13项。
v3旧开发验证只编译4个辅助模块且临时路径较短，没有覆盖render_util的长路径问题；本次补足该缺口。

3. 一次本地运行的完整命令
下载新ZIP到 /home/lx/下载，保留旧ZIP和旧解压目录。

unzip -n /home/lx/下载/microduck-stage12-local-completion-v4-20261016.zip -d /home/lx/下载 &&
python3 -I -S /home/lx/下载/microduck-stage12-local-completion-v4/microduck-stage12-local.py --verify-package &&
python3 -I -S /home/lx/下载/microduck-stage12-local-completion-v4/microduck-stage12-local.py --run --native-integrations 38720 --wall-seconds 43200

源码：/home/lx/microduck-double-balance/workspace
现有解释器：/home/lx/microduck-double-balance/workspace/.venv/bin/python，不resolve为系统Python。
新证据根：/home/lx/microduck-double-balance/artifacts/double-balance-stage12/local-completion-v4
执行顺序：核验旧证据/身份 → 复用CPU报告及410输入 → N1 → N2 → 对账 → ZIP → 本地传输提交。
中间不用回聊天确认；任何真实失败立即保留并停止，不自动补跑、不扩大上限。
本命令只授权这次有界本地仿真验收，不授权PPO或云计算。既有v4 started存在时只显示状态并退出。

看到LOCAL_TRANSFER_COMMIT_READY_MANUAL_PUSH_PENDING后，手动执行：

git -C /home/lx/microduck-double-balance/artifacts/double-balance-stage12/local-completion-v4/transfer push origin HEAD

粘贴push结果与Stage12TransferCommit即可；不用上传返回ZIP。
助手从固定完整commit读取；正常push后不要求你重复全套远端验证。GitHub显示的PR链接不用点击。
push失败只重试上面的push，不再--run，不删除started、不force push、不合并传输分支。

只读状态：
python3 -I -S /home/lx/下载/microduck-stage12-local-completion-v4/microduck-stage12-local.py --status

只读限额：
python3 -I -S /home/lx/下载/microduck-stage12-local-completion-v4/microduck-stage12-local.py --plan --wall-seconds 43200

只读GPU查看，Ctrl+C只退出此监控：
nvidia-smi -l 2

campaign.log的完整路径每5秒随心跳打印；owner下progress.json记录公共step、macro、计数及WAL pending。
bootstrap-report.json位于owner-0-bootstrap等目录，未放入会被回传排除的runtime-scratch。
仅当return-receipt.json已成功形成、但传输提交未完成时，可只执行：
python3 -I -S /home/lx/下载/microduck-stage12-local-completion-v4/microduck-stage12-local.py --prepare-transfer
此命令不重新进行CPU检查或物理验收。若归档/receipt未形成，保留现状和错误供审查，不伪造回执。

4. 保留/复用与原有限额
v1固定提交623484cd5dbb8182205dc035dd13fc95d831b821的CPU13/13和410项输入继续复用。
cloud-inputs.zip：16181045字节，SHA256=24701df9e56face7980a1a2e0544dfebc41a4cf1b84ec4912be795d1a35a43b2。
先校验旧ZIP、started对应run/PACKAGE、return/transfer回执及所有保留报告，再复制输入，不重采源码或资产。
新校验明确区分：v1原生owner目录从未建立；v2原生owner目录必须保留，WAL必须仍为空、原生报告零消耗，N2仍未开始。
另核验v3的29份报告和归档：必须保留2次CPU forward、1次CPU kinematics、1次失败构造及4行WAL；N2仍未开始。
任一证据变化就停止重新对账，不能套用零消耗结论。新v4是一次新声明的构造/验收尝试，不能从v3半构造处resume。
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
v4的真实积分必须以新WAL汇总更新当前674456/674457基线，不能提前假定仍为零或全部完成。

本地PASS后自动生成（仅生成，不执行）：
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/local-completion-v4/CLOUD-COMMANDS.sh
完整内容已填入本次真实传输分支和40位commit，因此不必再等助手补分支名。
手动push成功且预算另行明确后，在当时提供的云主机执行其中命令；不用ZIP往返。
查看文件：
cat /home/lx/microduck-double-balance/artifacts/double-balance-stage12/local-completion-v4/CLOUD-COMMANDS.sh

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
不改主checkout，不清理旧补丁，不重跑Stage06/Stage11/readiness v1–v7/integration-v1/local-completion-v1/v2/v3已完成工作。
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
- v1 cwd、v2 uname -p、v3 NVRTC长临时目录及半构造关闭错误均保留原报告与针对性验证。
- 原始证据未回传：索引/摘要不是完整备份，不删除本机原始文件或云盘。

2026-10-01用户曾确认5090关机，这不是当前电源证据；当前云电源=UNCONFIRMED。
本轮没有启动、连接或训练云主机；本阶段SSH地址/端口尚未提供，不复用旧端口作永久地址。
未来云计算和必要打包/回传结束且本机SHA校验通过，立即提醒用户手动关机；提醒与实际确认分别记录。
GPU利用率0不证明任务结束；作业退出不等于停止计费。不自动开关实例，不重启已禁用的旧关机daemon。

下一次继续语
继续microduck-double-balance Stage12，读取最新local-completion-v4 Git传输固定提交和本交接。
已审查v3提交93fc6ff3779093f5b49c28e9883477f1c1d9f85b：NVRTC临时目录过长，部分构造关闭出错；
积分新增0，CPU forward已完成2、CPU kinematics已完成1、失败构造1；累计674456/674457。
v4已修复两处已知故障，真实长路径NVRTC/3进程bootstrap/清理/账本/交付开发检查通过。
v1 CPU13/13与410输入复用；本机真实N1/N2当前0/2点通过，后续实际结果必须以新回包和WAL对账。
保留全部阈值、started、旧包、checkpoint，不重跑完成批次；PPO=0、云训练=0，预算另行明确，不自动push/PR/merge。
