Microduck Double Balance — Stage12 readiness v4
交接日期：2026-10-16（北京时间；Git运行标识使用UTC）

Stage11：11/11 CLOSED_SCOPED，stage-11-complete 已由用户成功推送。
累计账本：674456 completed / 674457 attempted；历史失败1次继续保留。
本轮授权：native integrations=0、native forward=0、真实环境构造=0、模型/优化器构造=0、
torch.load=0、PPO=0、云训练=0。训练预算另行明确。
不重跑Stage06、Stage11或Stage12 v1/v2/v3；不删除started，不改原物理阈值。
只准备本轮独立证据提交；不自动push、创建PR或合并。

一、已收到并审查的v3结果

TransferBranch=stage12-readiness-v3-transfer-20261015t164533z-be94cf
TransferCommit=8f54003ac1e511ee3a2fe8b0c3b32c99a74744a9
ReturnRun=20261015T164533Z-be94cf
ReturnArchiveBytes=899225
ReturnArchiveSHA256=bdae9ed59caea5faa09c1172c5b5ae1efb8bec4e89798330867893e31e4c6a2b
ProgramManifestSHA256=a4485050659ab1122f9f6701a88a6b2cd0e3d227002073a99d9ac9ec9da5d91e

2个传输分片、ZIP CRC、全部260个manifest成员及程序包身份均已校验。
身份与现场保护PASS；v3累计/PPO/云训练未增加。
源码采集完成：60份新增、188份复用、1份历史脚本只记身份，共3591016字节已保留源码，
另有158项资产身份。这里只记身份的历史文件是9710625字节的stage10_finalize.py，
其Git blob为90ec480d5cfd783c4d680d7765a1d0c83e111746，不执行、不重新打包。

v3状态仍为STAGE12_PREPARATION_BLOCKED：Torch导入后，MuJoCo导入触发GLFW寻找动态库，
其版本探测试图调用subprocess.Popen，被CPU预检边界拒绝。配置与包装器检查尚未完成。
完整原始报告在prior-readiness-v3/；v1/v2原始事实和限制也继续保留。

二、v4具体新增内容

1. 复用已完成采集：前后核对249个当前来源身份、248份v2/v3保留副本和158项资产。
   检查来源/资产清单是否增删，原文件与旧证据均需匹配；不再次复制或运行这些来源。
2. 导入修复：在导入前设置MUJOCO_GL=disable，并用既有glfw wheel的
   glfw/x11/libglfw.so绝对路径设置PYGLFW_LIBRARY，避免动态库版本探测子进程。
   mjlab会间接导入mujoco.viewer，因此仅MUJOCO_GL=disable不够。
   准确比对MuJoCo 3.10.0及pyGLFW 2.10.0的两份上游导入源码；来源、Git blob、SHA、
   许可证在reviewed-import-sources/。不修改安装包，不mock替代实际安装的包。
   GLFW可能绑定既有动态库以满足viewer导入，但不初始化窗口或图形上下文；相关入口阻断。
   网络、子进程、模型编译、真实环境、原生步进、PPO和反序列化阻断继续保留。
3. 配置预检：对照冻结G11语义、临时注册候选任务并检查深拷贝和启动阻断。
4. 观测适配：get_observations读取已提交缓存，不重新采样噪声/历史；step/reset调用原包装器
   一次后克隆观测，防止下次原地写缓冲区或reset改写尚未存入rollout的前一条观测。
   reward/done/timeout返回值不变。实际Torch/TensorDict包装器配合假CPU环境验证，非物理测试。
5. 已根据安装版RSL 5.0.1/mjlab 1.3.0完成入口、RNN、normalizer、timeout、storage和保存恢复
   源码审查，新增runner-integration-review.json、runner_contract.py与预算/分组草案。

源文件、checkpoint、73个Stage11标记，以及Stage12 v1/v2/v3的标记、归档、报告和传输回执
继续做前后身份核对。新目录只使用readiness-v4，原149份G11 payload逐字节不变。
v4返回仍是增量证据，需要结合固定v2/v3提交；不能称为完整云复现包。

三、Stage12任务与何时正式开训

S12.1 任务配置、专用入口契约、checkpoint策略、观测适配和来源冻结：
      来源已齐；配置/真实CPU包装器预检待v4主机回执。
S12.2 可持续运行的单环境后端、reset、资源生命周期和新配置物理验收：
      尚待实现和新批次验收；旧G11有界运行不能直接当训练环境。
S12.3 多环境隔离、规模/吞吐与峰值内存：
      已有静态计算草案；多所有者和原生nworld>1分别验收，尚无实际吞吐或放行预算。
S12.4 专用runner端到端、timeout处理、保存/恢复、完整依赖/资源包与云启动条件：
      安装版源码审查完成；真实端到端和发布包尚待完成。

正式云PPO可以在单环境持续运行验收、小规模多环境隔离/runner验收通过，并明确云GPU、
环境数、更新数、时间上限、保存/恢复和原生调用总预算后开始。更大规模优化可以后续进行。
当前不能直接开训；也不需要为了等到4096环境才开始。v4预检PASS本身不等于训练放行。
不承诺尚无测量依据的开训日期；不把max_iterations=0当成允许构造模型的dry run。

四、安装版训练源码发现及处理要求

默认入口：mjlab/scripts/train.py直接构造ManagerBasedRlEnv和原RslRlVecEnvWrapper，
并传init_at_random_ep_len=True。仅注册runner_cls不会替换物理工厂；registry没有env_cls。
后续专用入口必须先安装并验收物理钩子，再允许包装器reset，并显式使用False保持名义起点。
runner构造本身会建模型、优化器、storage和logger；因此本次仍禁止构造，不能用0更新绕过。

观测：PPO.act暂存TensorDict引用，下一次process_env_step才复制进storage。
因此v4把缓存读取、step和reset出口都做快照。开发测试覆盖缓冲区原地复用与reset后的旧观测。
normalizer.forward本身不更新统计；process_env_step使用下一次观测更新actor/critic统计，
若发生auto_reset则其中包含reset观测。保留既有更新节奏，后续实际runner需验证无重复计数。

超时：现有RSL在reward中加gamma*V(s_t)*time_outs，然后按dones切断GAE；
这不是实际终点s_(t+1)的价值。原auto_reset返回的是reset后的观测。
训练前需落实终点观测/价值在reset前的保存、无重复噪声/历史推进、真实终止与timeout同时发生
时的优先级，以及只做一次bootstrap补偿。v4保留原flag，不悄悄修改奖励或成功/终止阈值。

RNN：现有RSL在保存transition后对terminated OR truncated调用actor/critic.reset，
LSTM h/c按done索引归零；后续仍须验证选择性reset不会污染其他环境。

minibatch：此版recurrent generator按num_envs//num_mini_batches分割“环境维”。
B>N会产生空batch，N不能被B整除会遗漏余下环境；必须1<=B<=N且N%B==0。
继承配置B=4不适合候选N=1。v4不擅改PPO超参数，列出N1/B1、N2/B1、N2/B2供未来预算选定，
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

将microduck-stage12-readiness-v4-20261016.zip下载到/home/lx/下载。
若下载名带(1)，使用实际的新v4文件名，不覆盖旧包。先执行以下三行：

unzip -n /home/lx/下载/microduck-stage12-readiness-v4-20261016.zip -d /home/lx/下载 &&
python3 -I -S /home/lx/下载/microduck-stage12-readiness-v4/microduck-stage12-readiness.py --verify-package &&
python3 -I -S /home/lx/下载/microduck-stage12-readiness-v4/microduck-stage12-readiness.py --run

外层只用标准库。自动使用既有/home/lx/microduck-double-balance/workspace/.venv/bin/python
执行CPU子进程，保留该路径，不resolve成系统Python。不pip install，不uv sync。
预期成功状态为STAGE12_PREPARATION_PASS_TRAINING_BLOCKED；TrainingReady仍为False。

待出现Stage12Transfer=LOCAL_TRANSFER_COMMIT_READY_MANUAL_PUSH_PENDING后，另行手动执行：

git -C /home/lx/microduck-double-balance/artifacts/double-balance-stage12/readiness-v4/transfer push origin HEAD

粘贴正常push输出和脚本打印的Stage12TransferCommit即可，无需上传返回ZIP。
即便配置预检失败，只要传输提交已就绪，也推送该失败证据；不要重跑--run。
助手按固定完整commit读取分片，验证文本/原始分片/ZIP/成员SHA后继续。
GitHub的Create a pull request只是自动提示，不需要创建PR；证据分支不合并。

新运行目录：
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/readiness-v4/<实际UTC>-<随机后缀>
新标记：readiness-v4/started-v1.json。
新分支：stage12-readiness-v4-transfer-<实际UTC小写>-<随机后缀>。
真实新commit由运行后输出，不提前编造。

八、状态与恢复命令

只读查看（可重复，不启动主机预检）：
python3 -I -S /home/lx/下载/microduck-stage12-readiness-v4/microduck-stage12-readiness.py --status

仅在返回ZIP和return-receipt.json已生成、传输提交尚未准备好时补齐包装：
python3 -I -S /home/lx/下载/microduck-stage12-readiness-v4/microduck-stage12-readiness.py --prepare-transfer

该恢复不执行主机预检、物理或PPO。完全一致时复用已有commit；不一致则保留现场停止。
push失败只重试同一push；non-fast-forward不强推。没有return-receipt时先贴包装错误。
重复--run遇started只显示已有状态并停止；不删除started、不改SHA、不并发启动第二份。
心跳给出host-probe.log完整路径，按该路径tail -f；Ctrl+C退出日志查看不会重启任务。
依赖源码/GLFW库身份不符时查看headless-setup.json；不自行放开子进程限制或安装新包。

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
历史文档按原时间点保存，本轮变化以本TXT与prior-readiness-v3/audit.json为准。

默认“你手动Git SSH push → 助手按固定提交读取并校验”。
归档上限49MiB，原始分片每片512KiB，base64 ASCII和分层SHA校验保留。
代码留在double-balance；模型和归档不混入源码分支，不自动fetch/import/push/PR/merge。
继续既有SSH身份，不切换HTTPS用户名/PAT、不重新网页登录、不默认创建新key。
Permission denied(publickey)：按原协议查既有agent/key。
Everything up-to-date但缺文件：核对本次打印的v4 transfer目录、分支和commit，勿用Stage11旧目录。
SHA/标签冲突/non-fast-forward：保留现场审查，不强推、不改预期SHA。
正常push成功不重复整套远端发布核验；传输证据自身的SHA仍逐层检查。

十、云端与验证范围

本轮没有云计算或新实例，不使用历史SSH端口。
最近用户确认：2026-10-01 5090已关机；本轮未实时查询电源。
Stage09完整head-trace尚未全部返回的限制保留，不宣称所有云端数据都已备份。
正式PPO仍只在明确预算的云GPU执行；本机5060用于开发、仿真、推理。
未来云计算和必要打包/回传通过本地SHA校验后立即提醒手动关机，提醒时间与实际确认分别记；
不运行旧关机daemon，不自动开关实例。

开发验证50项通过：12准备契约、12身份/传输、12来源身份/复用、14导入/复用/runner计算。
包括临时249来源/158资产复用测试，以及上游导入源码在mock CDLL下不启动子进程的分支测试。
助手环境没有Torch/MuJoCo/Warp/mjlab，未运行真实主机配置、GPU、物理或模型；
这些开发测试不替代你主机的v4回执或未来物理/runner验收。

收到v4后，先审查headless-setup、source-reuse-before/after、configuration-probe、
prepared-env/agent-config和wrapper-contract。通过后推进持续单环境工厂、终点观测/timeout
接入与完整新物理预算；已有批次保持关闭，不删除保护后复用旧运行标识。
