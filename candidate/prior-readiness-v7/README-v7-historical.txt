Microduck Double Balance — Stage12 readiness v7
交接日期：2026-10-16（北京时间；运行标识使用UTC）

Stage11=11/11 CLOSED_SCOPED；stage-11-complete 已由用户成功推送。
累计674456 completed / 674457 attempted，历史失败1次保留。
PPO=0、cloud_training=0；原生积分/forward、真实环境/模型/优化器构造及torch.load预算全部为0。
训练预算另行明确。不重跑Stage06/11或Stage12 v1至v6，不删started、旧归档和回执，
不改原物理阈值。本包只创建本轮独立证据提交；由用户手动SSH push，不自动PR或合并。

一、v6回执审查与已完成项

TransferBranch=stage12-readiness-v6-transfer-20261015t175051z-dbbbab
TransferCommit=353227e8c073f8547a7f017c4876c33f99c550a4
Run=20261015T175051Z-dbbbab
ReturnArchiveBytes=948489
ReturnArchiveSHA256=14beea62219a418883df0e49ed7df83d2a3edae6e69d17505064d29ac914071a
ProgramManifestSHA256=b838470c628031d183ab7064e11c3f30ef061abf3abd17747cde646b3f9e31ce

已从该固定提交读取2个分片，核验文本/原始分片、ZIP CRC、275个manifest成员和程序身份。
身份及旧数据保护PASS；249个当前来源、248份保留副本、158项资产前后校验通过，复制0份。
GLFW和Matplotlib导入修复继续通过；GitPython导入也已真实通过：
仅执行2次/usr/bin/git version，GIT_OK=True，/dev/null正常打开2次，无仓库/网络操作。
安装版GitPython为官方release_version_stamp形式，3份审查源码和二进制身份均稳定。

此次已实际完成的CPU配置工作：
- 全部相关依赖导入完成，真实环境/模型/PPO等构造保护已安装且guard_attempts为空。
- 原G11配置语义复现通过，config_semantics_reproduced=True。
- Stage12候选任务在本CPU进程内注册、配置深拷贝和拒绝训练runner检查通过。
- prepared-env-config.json、prepared-agent-config.json和runner-contract-plan.json已生成。
这些通过结果在prior-readiness-v6/保留为配置基线；并未持久安装任务或放行默认train入口。

v6整体仍为STAGE12_PREPARATION_BLOCKED。最后的缓存观测包装器检查报错：
TypeError: unhashable type: 'TensorDict'
错误位于stage12/vec_adapter.py的set(obs)。TensorDict.__iter__按第一个批次维迭代，
得到子TensorDict，并非普通dict的键名。包装器step/reset返回TensorDict，因而触发错误。
这是适配器代码错误；旧开发替身按dict方式迭代，未覆盖该真实协议差异。
没有真实环境、原生积分、模型、PPO或云训练消耗；累计计数不变。
旧v6交接原文保留为prior-readiness-v6/README-v6-historical.txt。

二、v7修改与验证范围

snapshot_observations统一使用set(obs.keys())校验actor/critic分组。
actor=(N,61)、critic=(N,85)的形状检查不变，各张量仍clone后构造TensorDict。
不通过转换为list(obs)或直接迭代obs推断键，也不更改奖励、done或timeout输出。
get_observations读取已提交缓存；step/reset仍只委托底层一次，然后生成观测快照。

开发回归使用TensorDict0.10.0官方__iter__的准确函数体驱动最小协议替身：
先复现v6的同一个unhashable错误，再验证新键访问、旧观测隔离、原地缓冲复用、reset快照、
单次委托和N=1/2/4布局，以及缺失/额外/嵌套分组与错误形状拒绝。
这不是完整Torch/TensorDict运行，不能替代主机验证；不将普通字典测试冒充真实TensorDict测试。

主机v7会使用真实Torch、安装版TensorDict与RslRlVecEnvWrapper，以及不含物理的假环境，
分别覆盖dict缓存和TensorDict缓存。两种情况都检查get_observations/step/reset返回的
TensorDict类型、批次维、61/85布局、快照隔离、不重复计算观测和单次底层调用。
同时要求新生成的环境配置、agent配置、runner静态方案与v6已通过结果完全一致。
完整CPU配置与包装器检查通过后才输出STAGE12_PREPARATION_PASS_TRAINING_BLOCKED。

沿用已通过的导入边界，不增设新依赖源码放行条件：
- GLFW精确库路径/身份、Matplotlib原生字体回退保持。
- GitPython只在显式导入窗口允许两次准确只读版本查询；窗口外和其他进程/网络仍被阻断。
- 只允许真实Linux /dev/null字符设备正常访问，持久写入仍限新运行目录。
- 字体策略没有允许fc-list子进程；Git查询数单独记在git-import-probe.json。
外层原有只读Git检查、CPU主探针和独立本地证据提交保持；没有自动push。
CUDA初始化、真实环境/模型/优化器构造、checkpoint反序列化、物理、PPO和云训练仍禁止。

新运行目录readiness-v7。v1至v6的标记、归档、报告及传输回执前后核验；已采集来源只验证复用。
旧配置成功结果作为基线保留，不重跑任何旧--run，不安装/升级依赖，不改canonical checkout。
当前传输仍是增量证据，不是完整云训练发布包。

三、Stage12进度与何时开训

按工作内容跟踪，不以上传次数计进度：
A. CPU准备：来源冻结、checkpoint策略、候选配置语义和进程内注册已完成；
   观测包装器最后一项等待v7主机回执。v6新增通过项不会因整体BLOCKED而被抹掉。
B. 持续单环境：长期后端、名义reset、终点观测/timeout、资源生命周期及完整新物理预算。
   尚未实现/验收。不能删除G11短轨迹保护后直接长跑。
C. 小规模多环境：隔离、选择性reset、RNG/资源、同步吞吐与峰值内存实测。
   尚未验收；多个独立nworld=1所有者与原生nworld>1是不同能力。
D. 专用runner与checkpoint：端到端、更新计数/预算账本、保存读回和云端发布准备。
   已有源码审查和设计要求；真实端到端及更新边界恢复尚待实现/验收。

正式云PPO最早在B、选定小规模C和D的必要验收通过，并明确云GPU、环境数、更新数、
时间上限、checkpoint策略及完整原生调用总预算后开始。更大规模优化可后续进行，
不必等4096环境。目前还不能给出可靠开训日期或剩余运行次数；v7 PASS仍不等于开训放行。
涉及PPO或云试训前另行明确预算，不能沿用当前PPO=0自动执行。

四、安装版训练源码发现及处理要求

默认入口：mjlab/scripts/train.py直接构造ManagerBasedRlEnv和原RslRlVecEnvWrapper，
并传init_at_random_ep_len=True。仅注册runner_cls不会替换物理工厂；registry没有env_cls。
后续专用入口必须先安装并验收物理钩子，再允许包装器reset，并显式使用False保持名义起点。
runner构造本身会建模型、优化器、storage和logger；因此本次仍禁止构造，不能用0更新绕过。

观测：PPO.act暂存TensorDict引用，下一次process_env_step才复制进storage。
因此v7沿用缓存读取、step和reset出口的快照。开发测试覆盖缓冲区原地复用与reset后的旧观测。
normalizer.forward本身不更新统计；process_env_step使用下一次观测更新actor/critic统计，
若发生auto_reset则其中包含reset观测。保留既有更新节奏，后续实际runner需验证无重复计数。

超时：现有RSL在reward中加gamma*V(s_t)*time_outs，然后按dones切断GAE；
这不是实际终点s_(t+1)的价值。原auto_reset返回的是reset后的观测。
训练前需落实终点观测/价值在reset前的保存、无重复噪声/历史推进、真实终止与timeout同时发生
时的优先级，以及只做一次bootstrap补偿。v7保留原flag，不悄悄修改奖励或成功/终止阈值。

RNN：现有RSL在保存transition后对terminated OR truncated调用actor/critic.reset，
LSTM h/c按done索引归零；后续仍须验证选择性reset不会污染其他环境。

minibatch：此版recurrent generator按num_envs//num_mini_batches分割“环境维”。
B>N会产生空batch，N不能被B整除会遗漏余下环境；必须1<=B<=N且N%B==0。
继承配置B=4不适合候选N=1。v7不擅改PPO超参数，列出N1/B1、N2/B1、N2/B2供未来预算选定，
纯整数校验已实现；尚未执行任何PPO minibatch。

五、checkpoint处理策略

旧文件原路径：
/home/lx/microduck-double-balance/artifacts/double-balance-stage08/returned/5d998fc322345102/evidence-20260930T093645-8955/extension/E-20260929/segment-001/checkpoints/update_004000.pt
Bytes=10073500
SHA256=86d55c3703c18fcf4817db49c6e4dcf839b3f5195d68ae2c559db7e222bded97
SchemaSHA256=e740e6ca4ad74c53a0eaaa9c95dcae9b69fd5b0edf122593743cd015dcb388a7
历史已检查94个有限张量；本轮只校验文件身份，不重新torch.load。
DirectResume=BLOCKED_TASK_AND_TIMING_DIFFER；旧任务成绩不能作为新任务成绩。

未来首次训练按fresh方案：Actor 61→单层LSTM256→MLP512/256/128→14；
Critic 85→MLP512/256/128→1；normalizer、分布参数、Adam和计数重新初始化。
本次未构造这些对象。Actor-only暖启动仍为单独可选研究，不靠键映射或维度相同认定兼容。

现有保存包含actor/critic state_dict（含normalizer buffers）、optimizer、iter；mjlab另存
common_step_counter。默认不保存物理状态、环境/全局RNG、活跃RNN h/c或进行中的rollout。
iter是最后完成更新的零基索引；加载后默认从同一标签继续，不能把iter直接当已完成更新数。
PPO.learning_rate标量也没有被load显式恢复，后续须与optimizer中的LR同步验收。

新增方案为“更新边界继续训练”的设计草案，尚未实现/执行保存：
- 独立记last_completed_index、completed_updates、next_iteration和已消耗/剩余预算。
- 保存task/config/source/dependency身份、run id、checkpoint大小/SHA、seed与reset策略。
- 唯一临时文件、flush/fsync、原子改名，最后写SHA回执；失败保留现场。
- 明确reset/RNG/RNN恢复策略，不宣称中途rollout或逐位等价续跑。
- 中断更新保留attempted计数与started；不自动重试、不返还已消耗预算。

六、固定物理边界与多环境规模准备

真实头部、名义reset、hold=0、61/85观测、14动作、BAM、原奖励/误差/事件/接触阈值、
32/64/128层级、拒绝/回滚/强制精确重放全部保持。
physics_dt=0.002s、decimation=10、public step=0.020s、episode=10s。
原float32的5秒成功边界不加epsilon、不四舍五入。
候选配置移除4个G11诊断事件、测试故障终止和诊断recorder，auto_reset=True；这些声明的新配置
差异尚待真实物理验收。startup和独立工厂继续阻断。未开放随机reset、课程/扰动或实机。

冻结formal_runtime仍限制self.macros<20、tick<20；24步rollout需要240宏步，
10秒episode需要5000宏步。不能删除限制就宣称长期运行已通过。
reset_nominal仅接受[0]、startup要求num_envs=1，私有command RNG draw count=1。
多个隔离的native nworld=1所有者与一个原生nworld>1批量后端是不同能力。

64层接受的积分成本：32+64+64重放=160/宏步。
128层接受的积分成本：32+64+128+128重放=352/宏步。
每公共步10宏步，积分最多3520；宏步forward预留1413。
未授权的短吞吐草案：N=1、2，各预热1公共步、计时2公共步、重复1次；共90宏步，
积分最多31680。这不含完整构造/reset/sense/graph/账本预算，不能据此启动。
24步rollout纯积分上界：N1=84480；N2=168960，均未授权且不等于完整执行预算。

后续记录同步后的transitions/s、aggregate sim-seconds/s、单环境实时率、completed/attempted、
step中位数/p95、接受层级/拒绝/重放、CUDA峰值/驱动显存、RSS、JIT/预热/WAL/写盘分项。
本轮不查询新nvidia-smi，引用v2库存；RTX5060 Laptop约8151MiB不是训练峰值或规模保证。
throughput_measured=null。5秒心跳仅表示进度，不设置任意墙钟截止或自动扩容。

七、现在执行的完整命令

将microduck-stage12-readiness-v7-20261016.zip下载到/home/lx/下载。
若下载名带(1)，使用实际的新v7文件名，不覆盖旧包。先执行以下三行：

unzip -n /home/lx/下载/microduck-stage12-readiness-v7-20261016.zip -d /home/lx/下载 &&
python3 -I -S /home/lx/下载/microduck-stage12-readiness-v7/microduck-stage12-readiness.py --verify-package &&
python3 -I -S /home/lx/下载/microduck-stage12-readiness-v7/microduck-stage12-readiness.py --run

外层只用标准库。自动使用既有/home/lx/microduck-double-balance/workspace/.venv/bin/python
执行CPU子进程，保留该路径，不resolve成系统Python。不pip install，不uv sync。
预期成功状态为STAGE12_PREPARATION_PASS_TRAINING_BLOCKED；TrainingReady仍为False。

待出现Stage12Transfer=LOCAL_TRANSFER_COMMIT_READY_MANUAL_PUSH_PENDING后，另行手动执行：

git -C /home/lx/microduck-double-balance/artifacts/double-balance-stage12/readiness-v7/transfer push origin HEAD

粘贴正常push输出和脚本打印的Stage12TransferCommit即可，无需上传返回ZIP。
即便配置预检失败，只要传输提交已就绪，也推送该失败证据；不要重跑--run。
助手按固定完整commit读取分片，验证文本/原始分片/ZIP/成员SHA后继续。
GitHub的Create a pull request只是自动提示，不需要创建PR；证据分支不合并。

新运行目录：
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/readiness-v7/<实际UTC>-<随机后缀>
新标记：readiness-v7/started-v1.json。
新分支：stage12-readiness-v7-transfer-<实际UTC小写>-<随机后缀>。
真实新commit由运行后输出，不提前编造。

八、状态与恢复命令

只读查看（可重复，不启动主机预检）：
python3 -I -S /home/lx/下载/microduck-stage12-readiness-v7/microduck-stage12-readiness.py --status

仅在返回ZIP和return-receipt.json已生成、传输提交尚未准备好时补齐包装：
python3 -I -S /home/lx/下载/microduck-stage12-readiness-v7/microduck-stage12-readiness.py --prepare-transfer

该恢复不执行主机预检、物理或PPO。完全一致时复用已有commit；不一致则保留现场停止。
push失败只重试同一push；non-fast-forward不强推。没有return-receipt时先贴包装错误。
重复--run遇started只显示已有状态并停止；不删除started、不改SHA、不并发启动第二份。
心跳给出host-probe.log完整路径，按该路径tail -f；Ctrl+C退出日志查看不会重启任务。
依赖源码/GLFW库身份不符时查看headless-setup.json、font-discovery-policy.json和git-import-probe.json；不自行放开子进程限制或安装新包。

九、固定来源与长期规范

LocalRepository=/home/lx/microduck-double-balance/workspace
DownloadDirectory=/home/lx/下载
Artifacts=/home/lx/microduck-double-balance/artifacts
Branch=double-balance
CurrentReviewedHEAD=73ee01930a031d76a9d57419afdfd4464a547ab3
Stage11TagTarget=6ba0dc2de381321576f32310e1995a8e2e11e566
Stage11TagObject=bd1b219be55c813811d4f7cfa25d245b37166688
OriginSSH=git@github.com:liuxue-lab/microduck-double-balance.git
TrackedDiffSHA256=b90daa0274c14d95cec9511ec4ac93568bd604e93ce3c60df1bb1c4155005d7e

仅接受冻结base或已逐项审查的73ee019文档跟进提交，不放行任意后代。
219来源pin与既有dirty patch继续保留；不修改canonical checkout、index或remote。
v2固定传输commit=f6fb64e3ff1b90c7c9d65d1fb29c47df5b76b671。
v2归档SHA=0e1713f616b0a7ad702e0d50f4290e93a60e7dedc5074ba03fa0d125784fc012。
v1归档SHA=97aa588fdad86a012179ec726a1a320bbee5accce1916c2c845afb202d486cab。

继续遵循并保留两份完整原协议：
docs/handoffs/microduck-local-ssh-push-protocol.md
docs/handoffs/microduck-cloud-gpu-and-transfer-protocol.md
原件在baseline/g11/provenance/；Stage11完整交接在baseline/stage11-handoff-closed.txt。
历史文档按原时间点保存，本轮变化以本TXT与prior-readiness-v6/audit.json为准。

默认“你手动Git SSH push → 助手按固定提交读取并校验”。
归档上限49MiB，原始分片每片512KiB，base64 ASCII和分层SHA校验保留。
代码留在double-balance；模型和归档不混入源码分支，不自动fetch/import/push/PR/merge。
继续既有SSH身份，不切换HTTPS用户名/PAT、不重新网页登录、不默认创建新key。
Permission denied(publickey)：按原协议查既有agent/key。
Everything up-to-date但缺文件：核对本次打印的v7 transfer目录、分支和commit，勿用Stage11旧目录。
SHA/标签冲突/non-fast-forward：保留现场审查，不强推、不改预期SHA。
正常push成功不重复整套远端发布核验；传输证据自身的SHA仍逐层检查。

十、云端与验证范围

本轮无云计算或新实例，不使用历史SSH端口。
最近用户确认2026-10-01的5090已关机；本轮未实时查询电源。
Stage09完整head-trace尚未全部返回的限制保留，不宣称所有云端数据都已备份。
本机5060用于开发、仿真、推理；正式PPO只在另行明确预算的云GPU执行。
未来云计算及必要打包/回传通过本地SHA验证后立即提醒手动关机，分别记提醒时间与实际确认；
不运行旧关机daemon，不自动开关实例。

本轮开发测试66项通过：准备/观测契约15、身份/传输12、来源/复用12、导入/runner计算14、
字体边界6、Git边界7。TensorDict迭代回归先复现旧错误，再检验修复和三条适配路径。
v6已通过的两种GitPython源码实际导入开发报告原样保留，不重复执行该导入实验；
两次只读查询也已经在用户v6真实主机回执中通过。导入模块与物理预算代码本轮不修改。
开发证据位于validation/results/；Git导入实验注明reused_from_v6，不能当作新执行次数。
助手环境没有Torch/MuJoCo/Warp/mjlab，未运行真实主机配置、GPU、物理或模型；
开发协议替身不替代主机TensorDict验证，也不构成物理或训练放行。
本轮助手没有执行--run；所有物理/PPO/云训练消耗仍为0。

收到v7后先审查configuration-probe中的prepared_configuration_matches_v6、wrapper_contract
两个cache_kind结果、全部guard_attempts及源文件/旧归档前后保护。
若CPU准备通过，下一工作批次转入持续单环境、终点观测/timeout及完整新物理预算；
不再重复来源采集或已经关闭的旧运行。新增物理执行仍须有完整明确预算，PPO继续为0。
