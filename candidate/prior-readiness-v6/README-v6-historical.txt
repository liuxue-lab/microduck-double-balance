Microduck Double Balance — Stage12 readiness v6
交接日期：2026-10-16（北京时间；运行标识使用UTC）

Stage11=11/11 CLOSED_SCOPED；stage-11-complete 已由用户成功推送。
累计674456 completed / 674457 attempted，历史失败1次保留。
PPO=0、cloud_training=0；原生积分/forward、真实环境/模型/优化器构造及torch.load预算全部为0。
训练预算另行明确。不重跑Stage06/11或Stage12 v1至v5，不删started、旧归档或旧回执，
不改原物理阈值。本包只创建本轮独立证据提交；由用户手动SSH push，不自动PR或合并。

一、v5回执审查结果

TransferBranch=stage12-readiness-v5-transfer-20261015t172533z-c13ab0
TransferCommit=28c52232a220c5538054cd97e17409f22f66fa59
Run=20261015T172533Z-c13ab0
ReturnArchiveBytes=844605
ReturnArchiveSHA256=b5d238d62703f6b605f83547860639ed0225682c5aaca503184e1f0acc3e8f29
ProgramManifestSHA256=7309db97a2e2bbbdfe83f842bafcc9e5e6407cc70df407768b234018777ebbc4

已按固定提交读取2个传输分片，核验分片文本/原始字节、ZIP CRC、238个manifest成员和程序身份。
身份及旧数据保护PASS；249个当前来源、248份保留副本、158项资产验证通过，复制0份。
GLFW修复继续通过。Matplotlib的fc-list调用在exec前被拒绝，并正常进入官方字体扫描回退，
没有执行fc-list。MuJoCo/GLFW/Warp导入与相关零执行预算保护已安装。

v5实际阻塞于GitPython导入：rsl_rl的logger导入git，GitPython探测版本时，
Python subprocess先为标准输入打开/dev/null；预检误把它当作越界持久写入而阻断。
项目entrypoint先捕获并打印同一异常，随后直接导入rsl_rl.runners再次遇到该异常。
配置候选、注册和CPU包装器检查仍未完成，不能将mjlab的部分导入进展当作整体配置PASS。
这是预检对依赖正常导入限制过严，不能算物理方案失败，也不能宣称已经可以训练。
v5=STAGE12_PREPARATION_BLOCKED，新增原生积分/模型/PPO/云训练全部为0。
保留报告和审查在prior-readiness-v5/；旧v5交接原文为README-v5-historical.txt。

二、v6具体修复及执行边界

GitPython3.1.46的正常初始化会调用两次“/usr/bin/git version”：
第一次由Git.refresh检查可执行程序，第二次由FetchInfo.refresh读版本决定解析标志。
后者虽然类名有Fetch，本路径只查版本，不执行fetch、访问仓库或网络。

v6在mjlab前显式导入GitPython，只在这段导入窗口允许上述两次只读查询：
- 核对GitPython版本及git/__init__.py、cmd.py、remote.py完整源码身份。
- 官方setup.py把__version__ = "git"盖章为__version__ = '3.1.46'；只接受准确标签原文
  或准确发布盖章版，二者完整SHA冻结，不接受任意源码差异。
- 核对/usr/bin/git为真实、非符号链接的可执行文件，记录大小/SHA，调用前后复核。
- 严格要求命令、参数、调用栈、新运行目录、禁用Git配置的环境变量以及最多两次计数。
- 必须真实初始化成功、GIT_OK=True；导入结束或异常即关闭窗口，不用quiet模式伪造成功。
- 允许真实Linux /dev/null字符设备(major=1, minor=3)正常打开；不允许符号链接别名、
  普通同名文件或其他/dev设备。该设备不保存数据，持久写入仍限于新运行目录。
- 窗口外git version及其他子进程、Git操作和网络调用仍被CPU预检阻断。

CPU子进程的辅助进程预算因此明确为“两次只读版本查询”，不再笼统称所有子进程为0。
外层原有只读Git身份检查、CPU主探针和独立本地证据提交照旧；无自动push。
字体策略仍禁止fc-list执行，font-discovery-policy.json中的0仅指字体策略允许的子进程数。
新增git-import-probe.json，记录源码形式、二进制身份、允许/完成的查询和/dev/null打开次数。
物理、GPU初始化、环境/模型/优化器构造、checkpoint反序列化、PPO和云训练均未放行。

沿用v4的精确GLFW库路径及已成功导入的二进制身份，沿用v5的Matplotlib原生回退。
只验证和复用已完成来源采集，不重新采集，不安装/升级依赖，不改canonical checkout。
新目录readiness-v6；v1至v5标记、归档、报告及传输回执在执行前后均核验。
当前回传是增量证据，尚不是完整云发布包。

三、Stage12剩余工作与开训条件

按四个工作批次组织，不把上传次数当项目进度，也不保证四次运行内完成：
A. 任务配置、入口契约、checkpoint策略和观测适配CPU预检。本次v6继续修复这一批。
   来源已齐；配置、注册和安装版CPU包装器结果仍等待本次主机回执。
B. 可持续单环境：实现长期后端、名义reset、终点观测/timeout和资源生命周期；
   制定并验收完整新物理预算，不能删除G11短轨迹保护后直接长跑。
C. 小规模多环境：验证隔离、选择性reset、RNG和资源，实测选定规模吞吐/峰值内存。
   多个独立nworld=1所有者与原生nworld>1分别验收。
D. 专用runner与保存恢复：端到端、更新计数/预算账本、保存读回和云端发布准备。
   PPO或云端试训前另行明确预算，当前PPO=0不能自动升级为试训授权。

正式云PPO最早在B、选定小规模C及D必要验收通过，并明确云GPU、环境数、更新数、
时间上限、checkpoint策略及完整原生调用总预算后开始。更大规模优化可以后续推进，
无需等待4096环境；目前无实际吞吐测量，也没有可运行的长期后端，不能给出可靠开训日期。
v6预检PASS仅完成A中相应检查，TrainingReady仍为False。
后续每批报告实际完成项、未完成项及真实物理/PPO消耗，不给无依据的回合数承诺。

四、安装版训练源码发现及处理要求

默认入口：mjlab/scripts/train.py直接构造ManagerBasedRlEnv和原RslRlVecEnvWrapper，
并传init_at_random_ep_len=True。仅注册runner_cls不会替换物理工厂；registry没有env_cls。
后续专用入口必须先安装并验收物理钩子，再允许包装器reset，并显式使用False保持名义起点。
runner构造本身会建模型、优化器、storage和logger；因此本次仍禁止构造，不能用0更新绕过。

观测：PPO.act暂存TensorDict引用，下一次process_env_step才复制进storage。
因此v6沿用缓存读取、step和reset出口的快照。开发测试覆盖缓冲区原地复用与reset后的旧观测。
normalizer.forward本身不更新统计；process_env_step使用下一次观测更新actor/critic统计，
若发生auto_reset则其中包含reset观测。保留既有更新节奏，后续实际runner需验证无重复计数。

超时：现有RSL在reward中加gamma*V(s_t)*time_outs，然后按dones切断GAE；
这不是实际终点s_(t+1)的价值。原auto_reset返回的是reset后的观测。
训练前需落实终点观测/价值在reset前的保存、无重复噪声/历史推进、真实终止与timeout同时发生
时的优先级，以及只做一次bootstrap补偿。v6保留原flag，不悄悄修改奖励或成功/终止阈值。

RNN：现有RSL在保存transition后对terminated OR truncated调用actor/critic.reset，
LSTM h/c按done索引归零；后续仍须验证选择性reset不会污染其他环境。

minibatch：此版recurrent generator按num_envs//num_mini_batches分割“环境维”。
B>N会产生空batch，N不能被B整除会遗漏余下环境；必须1<=B<=N且N%B==0。
继承配置B=4不适合候选N=1。v6不擅改PPO超参数，列出N1/B1、N2/B1、N2/B2供未来预算选定，
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

将microduck-stage12-readiness-v6-20261016.zip下载到/home/lx/下载。
若下载名带(1)，使用实际的新v6文件名，不覆盖旧包。先执行以下三行：

unzip -n /home/lx/下载/microduck-stage12-readiness-v6-20261016.zip -d /home/lx/下载 &&
python3 -I -S /home/lx/下载/microduck-stage12-readiness-v6/microduck-stage12-readiness.py --verify-package &&
python3 -I -S /home/lx/下载/microduck-stage12-readiness-v6/microduck-stage12-readiness.py --run

外层只用标准库。自动使用既有/home/lx/microduck-double-balance/workspace/.venv/bin/python
执行CPU子进程，保留该路径，不resolve成系统Python。不pip install，不uv sync。
预期成功状态为STAGE12_PREPARATION_PASS_TRAINING_BLOCKED；TrainingReady仍为False。

待出现Stage12Transfer=LOCAL_TRANSFER_COMMIT_READY_MANUAL_PUSH_PENDING后，另行手动执行：

git -C /home/lx/microduck-double-balance/artifacts/double-balance-stage12/readiness-v6/transfer push origin HEAD

粘贴正常push输出和脚本打印的Stage12TransferCommit即可，无需上传返回ZIP。
即便配置预检失败，只要传输提交已就绪，也推送该失败证据；不要重跑--run。
助手按固定完整commit读取分片，验证文本/原始分片/ZIP/成员SHA后继续。
GitHub的Create a pull request只是自动提示，不需要创建PR；证据分支不合并。

新运行目录：
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/readiness-v6/<实际UTC>-<随机后缀>
新标记：readiness-v6/started-v1.json。
新分支：stage12-readiness-v6-transfer-<实际UTC小写>-<随机后缀>。
真实新commit由运行后输出，不提前编造。

八、状态与恢复命令

只读查看（可重复，不启动主机预检）：
python3 -I -S /home/lx/下载/microduck-stage12-readiness-v6/microduck-stage12-readiness.py --status

仅在返回ZIP和return-receipt.json已生成、传输提交尚未准备好时补齐包装：
python3 -I -S /home/lx/下载/microduck-stage12-readiness-v6/microduck-stage12-readiness.py --prepare-transfer

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
历史文档按原时间点保存，本轮变化以本TXT与prior-readiness-v5/audit.json为准。

默认“你手动Git SSH push → 助手按固定提交读取并校验”。
归档上限49MiB，原始分片每片512KiB，base64 ASCII和分层SHA校验保留。
代码留在double-balance；模型和归档不混入源码分支，不自动fetch/import/push/PR/merge。
继续既有SSH身份，不切换HTTPS用户名/PAT、不重新网页登录、不默认创建新key。
Permission denied(publickey)：按原协议查既有agent/key。
Everything up-to-date但缺文件：核对本次打印的v6 transfer目录、分支和commit，勿用Stage11旧目录。
SHA/标签冲突/non-fast-forward：保留现场审查，不强推、不改预期SHA。
正常push成功不重复整套远端发布核验；传输证据自身的SHA仍逐层检查。

十、云端与验证范围

本轮无云计算或新实例，不使用历史SSH端口。
最近用户确认2026-10-01的5090已关机；本轮未实时查询电源。
Stage09完整head-trace尚未全部返回的限制保留，不宣称所有云端数据都已备份。
本机5060用于开发、仿真、推理；正式PPO只在另行明确预算的云GPU执行。
未来云计算及必要打包/回传通过本地SHA验证后立即提醒手动关机，分别记提醒时间与实际确认；
不运行旧关机daemon，不自动开关实例。

开发测试63项通过：准备契约12、身份/传输12、来源/复用12、导入/runner计算14、字体边界6、
Git导入边界7，包含临时249来源/158资产复用、旧started保护、传输提交和包装恢复。
另用59份精确上游文件实际导入GitPython3.1.46/gitdb4.0.12/smmap5.0.2：
标签原文与官方版本盖章形式分别通过，每次真实执行且仅执行2次git version；
导入后额外version、status、push、/bin/true、shell命令均在exec前阻断。
开发证据在validation/results/。其中测试机Git身份不作为你主机Git二进制的预期SHA。
上游fixture和盖章转换仅在助手测试目录，不修改用户安装包，不安装任何新依赖。
助手环境没有Torch/MuJoCo/Warp/mjlab，未运行真实主机配置、GPU、物理或模型；
这些开发测试不替代你的v6主机回执或未来物理/runner验收，不能当作训练放行。
本轮助手没有执行--run；所有物理/PPO/云训练消耗仍为0。

收到v6后，先审查git-import-probe、font-discovery-policy、headless-setup、source-reuse前后、
configuration-probe、prepared-env/agent-config及wrapper-contract。
通过后推进持续单环境工厂、终点观测/timeout接入与完整新物理预算；已有批次保持关闭，
不删除保护后复用旧运行标识。
