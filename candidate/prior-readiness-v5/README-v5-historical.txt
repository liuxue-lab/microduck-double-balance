Microduck Double Balance — Stage12 readiness v5
交接日期：2026-10-16（北京时间；运行标识使用UTC）

Stage11=11/11 CLOSED_SCOPED；stage-11-complete 已由用户成功推送。
累计674456 completed / 674457 attempted，历史失败1次保留。
PPO=0、cloud_training=0；原生积分/forward、真实环境/模型/优化器构造及torch.load全部为0。
训练预算另行明确。不重跑Stage06/11或Stage12 v1/v2/v3/v4，不删started或旧归档，
不改原物理阈值。本包只创建本轮独立证据提交；由用户手动SSH push，不自动PR或合并。

一、v4回执已经实际验证的内容

TransferBranch=stage12-readiness-v4-transfer-20261015t171249z-53a731
TransferCommit=f69097d83fa415057b8186c255c23585e93f46d7
Run=20261015T171249Z-53a731
ReturnArchiveBytes=786755
ReturnArchiveSHA256=d214636aae7d694579fb920b8efbc9386cb4abfd4d3f97e01185934d9bcc9d93
ProgramManifestSHA256=618bdd9587b45b5863b2efd32102bd4edb11252835fc2ad46b4f0bea10fe33ef

2个传输分片、ZIP CRC、221个manifest成员和程序包身份均已校验。
来源复用PASS：249个当前来源、248份保留副本、158项资产，复制0份；前后身份一致。
旧源文件/checkpoint/标记/归档保护PASS。GLFW修复在真实主机上已通过，
MuJoCo、GLFW、mujoco.viewer、Warp/mujoco_warp导入完成，相关执行入口仍被阻断。
没有模型、真实环境、原生积分或PPO执行；没有新的云任务。

v4状态仍为STAGE12_PREPARATION_BLOCKED。mjlab导入mediapy，mediapy导入Matplotlib，
Matplotlib首次字体发现调用fc-list --help，被预检的统一子进程限制拦下。
这是预检程序与第三方导入行为的兼容问题，尚未进入候选配置与包装器测试。
不能把这次失败算成原物理方案失败，也不能宣称Stage12环境验收已通过。
完整v4报告在prior-readiness-v4/；旧v4交接按历史原文保留。

二、v5修复范围与预计剩余批次

安装版Matplotlib=3.10.9。已审查对应版本font_manager.py：
_get_fontconfig_fonts()原生支持捕获OSError后返回空fontconfig结果，随后FontManager
继续直接扫描自带和系统字体目录。这不需要外部fc-list，不伪造字体缓存。

v5先核验安装版字体发现源码与准确上游Git blob/SHA一致，并设置MPLBACKEND=Agg。
只有来自已核验_get_fontconfig_fonts的准确fc-list参数才改用FileNotFoundError报告
“可选外部字体发现不可用”，让上游进入已有回退。审计事件发生在exec之前，仍不允许
任何子进程。其他调用者、其他命令、网络、越界写入和重复可选探测均继续阻断。
不修改安装的Matplotlib，不替换mjlab/mediapy，不放开PPO、仿真或模型构造。
GLFW库还需与v4成功导入的二进制身份相同。

新增font-discovery-policy.json记录实际回退、源码身份和子进程数；日志按Torch、
MuJoCo/GLFW、Warp、mjlab/字体、候选配置、缓存包装器分段标明进度。
已完成的来源采集只验证和复用；旧v1至v4标记、归档、报告及传输回执继续保护。
新运行目录为readiness-v5。当前回传仍为增量证据，不是完整云发布包。

后续按四个大批次组织，目的是减少零碎回传，并非保证四次运行内全部通过：
A. 本次v5：完成配置/观测适配的CPU预检。
B. 持续单环境：实现长期后端、名义reset、终点观测/timeout接口和资源生命周期，
   用完整的新预算验收。不能移除G11短轨迹保护后直接长跑。
C. 小规模多环境：隔离/reset/RNG/资源验收，按选定规模实测吞吐与峰值内存。
D. runner和保存恢复：完成端到端、更新计数/预算账本、保存读回和云发布准备。
   涉及PPO或云端试训前必须明确训练预算，不能沿用当前PPO=0自动执行。

若每个批次一次通过，约还需4轮；配置或物理/runner存在新问题则增加修复回合。
目前仍有单环境长期后端尚未实现，所以没有足够依据给出可靠次数上限或开训日期。
通过选定小规模后即可安排获批的云端小预算试训，不要求先优化到4096环境。
后续每轮报告：实际完成项、仍未完成项、是否消耗物理/PPO预算；不按上传次数算进度。

三、Stage12任务与何时正式开训

S12.1 任务配置、专用入口契约、checkpoint策略、观测适配和来源冻结：
      来源已齐；配置/真实CPU包装器预检待v5主机回执。
S12.2 可持续运行的单环境后端、reset、资源生命周期和新配置物理验收：
      尚待实现和新批次验收；旧G11有界运行不能直接当训练环境。
S12.3 多环境隔离、规模/吞吐与峰值内存：
      已有静态计算草案；多所有者和原生nworld>1分别验收，尚无实际吞吐或放行预算。
S12.4 专用runner端到端、timeout处理、保存/恢复、完整依赖/资源包与云启动条件：
      安装版源码审查完成；真实端到端和发布包尚待完成。

正式云PPO可以在单环境持续运行验收、小规模多环境隔离/runner验收通过，并明确云GPU、
环境数、更新数、时间上限、保存/恢复和原生调用总预算后开始。更大规模优化可以后续进行。
当前不能直接开训；也不需要为了等到4096环境才开始。v5预检PASS本身不等于训练放行。
不承诺尚无测量依据的开训日期；不把max_iterations=0当成允许构造模型的dry run。

四、安装版训练源码发现及处理要求

默认入口：mjlab/scripts/train.py直接构造ManagerBasedRlEnv和原RslRlVecEnvWrapper，
并传init_at_random_ep_len=True。仅注册runner_cls不会替换物理工厂；registry没有env_cls。
后续专用入口必须先安装并验收物理钩子，再允许包装器reset，并显式使用False保持名义起点。
runner构造本身会建模型、优化器、storage和logger；因此本次仍禁止构造，不能用0更新绕过。

观测：PPO.act暂存TensorDict引用，下一次process_env_step才复制进storage。
因此v5把缓存读取、step和reset出口都做快照。开发测试覆盖缓冲区原地复用与reset后的旧观测。
normalizer.forward本身不更新统计；process_env_step使用下一次观测更新actor/critic统计，
若发生auto_reset则其中包含reset观测。保留既有更新节奏，后续实际runner需验证无重复计数。

超时：现有RSL在reward中加gamma*V(s_t)*time_outs，然后按dones切断GAE；
这不是实际终点s_(t+1)的价值。原auto_reset返回的是reset后的观测。
训练前需落实终点观测/价值在reset前的保存、无重复噪声/历史推进、真实终止与timeout同时发生
时的优先级，以及只做一次bootstrap补偿。v5保留原flag，不悄悄修改奖励或成功/终止阈值。

RNN：现有RSL在保存transition后对terminated OR truncated调用actor/critic.reset，
LSTM h/c按done索引归零；后续仍须验证选择性reset不会污染其他环境。

minibatch：此版recurrent generator按num_envs//num_mini_batches分割“环境维”。
B>N会产生空batch，N不能被B整除会遗漏余下环境；必须1<=B<=N且N%B==0。
继承配置B=4不适合候选N=1。v5不擅改PPO超参数，列出N1/B1、N2/B1、N2/B2供未来预算选定，
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

将microduck-stage12-readiness-v5-20261016.zip下载到/home/lx/下载。
若下载名带(1)，使用实际的新v5文件名，不覆盖旧包。先执行以下三行：

unzip -n /home/lx/下载/microduck-stage12-readiness-v5-20261016.zip -d /home/lx/下载 &&
python3 -I -S /home/lx/下载/microduck-stage12-readiness-v5/microduck-stage12-readiness.py --verify-package &&
python3 -I -S /home/lx/下载/microduck-stage12-readiness-v5/microduck-stage12-readiness.py --run

外层只用标准库。自动使用既有/home/lx/microduck-double-balance/workspace/.venv/bin/python
执行CPU子进程，保留该路径，不resolve成系统Python。不pip install，不uv sync。
预期成功状态为STAGE12_PREPARATION_PASS_TRAINING_BLOCKED；TrainingReady仍为False。

待出现Stage12Transfer=LOCAL_TRANSFER_COMMIT_READY_MANUAL_PUSH_PENDING后，另行手动执行：

git -C /home/lx/microduck-double-balance/artifacts/double-balance-stage12/readiness-v5/transfer push origin HEAD

粘贴正常push输出和脚本打印的Stage12TransferCommit即可，无需上传返回ZIP。
即便配置预检失败，只要传输提交已就绪，也推送该失败证据；不要重跑--run。
助手按固定完整commit读取分片，验证文本/原始分片/ZIP/成员SHA后继续。
GitHub的Create a pull request只是自动提示，不需要创建PR；证据分支不合并。

新运行目录：
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/readiness-v5/<实际UTC>-<随机后缀>
新标记：readiness-v5/started-v1.json。
新分支：stage12-readiness-v5-transfer-<实际UTC小写>-<随机后缀>。
真实新commit由运行后输出，不提前编造。

八、状态与恢复命令

只读查看（可重复，不启动主机预检）：
python3 -I -S /home/lx/下载/microduck-stage12-readiness-v5/microduck-stage12-readiness.py --status

仅在返回ZIP和return-receipt.json已生成、传输提交尚未准备好时补齐包装：
python3 -I -S /home/lx/下载/microduck-stage12-readiness-v5/microduck-stage12-readiness.py --prepare-transfer

该恢复不执行主机预检、物理或PPO。完全一致时复用已有commit；不一致则保留现场停止。
push失败只重试同一push；non-fast-forward不强推。没有return-receipt时先贴包装错误。
重复--run遇started只显示已有状态并停止；不删除started、不改SHA、不并发启动第二份。
心跳给出host-probe.log完整路径，按该路径tail -f；Ctrl+C退出日志查看不会重启任务。
依赖源码/GLFW库身份不符时查看headless-setup.json和font-discovery-policy.json；不自行放开子进程限制或安装新包。

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
历史文档按原时间点保存，本轮变化以本TXT与prior-readiness-v4/audit.json为准。

默认“你手动Git SSH push → 助手按固定提交读取并校验”。
归档上限49MiB，原始分片每片512KiB，base64 ASCII和分层SHA校验保留。
代码留在double-balance；模型和归档不混入源码分支，不自动fetch/import/push/PR/merge。
继续既有SSH身份，不切换HTTPS用户名/PAT、不重新网页登录、不默认创建新key。
Permission denied(publickey)：按原协议查既有agent/key。
Everything up-to-date但缺文件：核对本次打印的v5 transfer目录、分支和commit，勿用Stage11旧目录。
SHA/标签冲突/non-fast-forward：保留现场审查，不强推、不改预期SHA。
正常push成功不重复整套远端发布核验；传输证据自身的SHA仍逐层检查。

十、云端与验证范围

本轮没有云计算或新实例，不使用历史SSH端口。
最近用户确认：2026-10-01 5090已关机；本轮未实时查询电源。
Stage09完整head-trace尚未全部返回的限制保留，不宣称所有云端数据都已备份。
正式PPO仍只在明确预算的云GPU执行；本机5060用于开发、仿真、推理。
未来云计算和必要打包/回传通过本地SHA校验后立即提醒手动关机，提醒时间与实际确认分别记；
不运行旧关机daemon，不自动开关实例。

开发验证56项通过：12准备契约、12身份/传输、12来源身份/复用、14导入/复用/runner计算、6字体发现边界。
另在独立Python进程中验证真实audit事件可进入Matplotlib的OSError回退，且未启动任何子进程。
包括临时249来源/158资产复用测试，以及上游导入源码在mock CDLL下不启动子进程的分支测试。
助手环境没有Torch/MuJoCo/Warp/mjlab，未运行真实主机配置、GPU、物理或模型；
这些开发测试不替代你主机的v5回执或未来物理/runner验收。

收到v5后，先审查font-discovery-policy、headless-setup、source-reuse-before/after、configuration-probe、
prepared-env/agent-config和wrapper-contract。通过后推进持续单环境工厂、终点观测/timeout
接入与完整新物理预算；已有批次保持关闭，不删除保护后复用旧运行标识。
