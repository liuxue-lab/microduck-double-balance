Microduck Double Balance — Stage12 integration v1 合并接入包
交接日期：2026-10-16 北京时间；运行标识使用 UTC。

当前结论
v7 已完整通过 CPU 准备，可以关闭该准备批次。无需再跑 readiness v1–v7。
本包合并已知的后续代码接入工作和一次聚合检查，不按报错逐项拆包。
代码候选完成不等于物理验收完成；当前 TrainingReady=False。
Stage11=11/11 CLOSED_SCOPED；stage-11-complete 已推送。
累计仍为 674456 completed / 674457 attempted；历史失败1次保留。
本包 --run 的 native_integrations/native_forward/真实环境/模型/优化器/torch.load/PPO/cloud_training 全为0。
不修改旧阈值、旧归档、旧源码补丁、任何 started；不重跑 Stage06、Stage11 或 readiness v1–v7。
本地只创建新的独立证据提交；用户手动 SSH push。没有自动 push、PR 或 merge。

1. 已读取并验证的 v7
Branch=stage12-readiness-v7-transfer-20261015t180254z-1138c6
Commit=472a23f2f83dd43d59b6964bd6233b0745e234dc
Run=20261015T180254Z-1138c6
ReturnArchive=stage12-readiness-return-20261015T180254Z-1138c6.zip
ReturnBytes=994066
ReturnSHA256=12a51c7d1000c67583e888bf55be0075b0dd1de569c39603850e8ddc3344d5ae
ProgramManifestSHA256=a752dae13c3a35bc6c04d2307dea3c6dc9a44bdc152caca66faa21eafc48761a
已校验2个base64分片、全部292个RETURN-MANIFEST成员、ZIP CRC、程序原始字节。
Stage12Readiness=STAGE12_PREPARATION_PASS_TRAINING_BLOCKED
HostReadiness=HOST_READINESS_PASS_TRAINING_BLOCKED
Configuration=CPU_CONFIGURATION_AND_ADAPTER_PASS
配置语义/注册/深拷贝、dict和真实TensorDict缓存两组检查通过；guard_attempts为空，CUDA未初始化。
249个当前来源、248份保留源码、158项资产验证通过，复制0份；身份及旧证据保护PASS。
以上是实际主机证据，不是开发替身结果。prior-readiness-v7/保留原报告及审查身份。

2. 本包集中完成的代码候选

integration/native_runtime.py
- 保留被冻结 G11 的数值驱动、32/64/128试探、拒绝/回滚/精确重放、接触和求解器检查。
- 新 BudgetedRuntime 使用明确会话宏步预算，不再把20宏步当作新任务的生命周期。
- EpisodeAdaptive 只扩展构造时的允许tick范围，保留继承的数值算法；episode上限5000宏步。
- 分离累计宏步、reset后episode tick、已有全局_sim_step_counter起点。
- 完整宏步证据继续落盘，内存只留最近结果；失败停止，不重试。
- 开发协议验证连续25宏步，再reset tick后5宏步，全局计数从90递增到120；这不是30次物理实测。

integration/native_factory.py / rng_policy.py / native_accounting.py
- 独立候选工厂包含eager仿真、真实头部startup、BAM、材质、contact/solver/RNG安装、源身份、reset、cleanup。
- 新配置移除诊断事件、诊断recorder和故障注入；原任务值与阈值保留。
- 旧 install_public_rng 要求3个诊断interval，新候选改为严格空interval，保留9个私有流和原抽样公式。
- 工厂使用 native auto_reset=False，由 TerminalBridge 负责外部自动reset；不伪称原生auto_reset=True。
- 新计数日志使用Stage12累计基线，多个owner的新增量由上层统一汇总，不重复加历史累计。
- 默认调用在任何模型/CUDA/文件构造前拒绝未授权native预算。本次不调用该工厂。
- 构造、长期轨迹、完整episode、资源清理仍需目标主机真实验收；不把源码复用视作新物理PASS。

integration/transitions.py
- 公共step完成后先clone终点观测、reward、terminated、truncated及extras，再做名义reset。
- 下一次get_observations只读已提交缓存；不额外调用observation_manager.compute。
- done=terminated OR truncated；只有 truncated AND NOT terminated 做超时价值补偿。
- 专用process_transition使用gamma*V(terminal)，移除原版RSL的time_outs通道，防止补偿两次。
  原版RSL使用V(s_t)。
- 现有critic是MLP；recurrent critic明确拒绝，不能借用错误的隐藏状态。
- observation normalizer仍由一次process_env_step(next_obs,...)更新；终点critic读取不更新统计。
- 两种缓存×正常/真实终止/纯timeout/二者同时发生，连同N2协议共9个场景集中验证。

integration/durable.py / runner.py
- 旧Stage08 checkpoint只校验SHA，不反序列化、不续训、不复制旧normalizer/Adam/LSTM状态。
- 新checkpoint按完成更新保存：唯一partial、fsync、原子无覆盖发布、SHA回执、最后提交更新账本。
- 开始更新即记录attempt；失败保留started和pending，不自动重试、不返还预算。
- 分开last_completed_index、completed_updates、next_iteration，避免原版resume重复零基索引。
- 续训先核验最终已提交账本/receipt/文件大小/SHA/任务身份；torch.load使用weights_only=True。
- optimizer LR加载后同步algorithm.learning_rate，要求所有param_group LR一致且有限正数。
- 保存actor/critic/optimizer/normalizer及policy sampler RNG；物理、环境私有流、RNN和rollout重新reset。
- 这是“更新边界重置后继续”，不声称中途rollout或逐位等价恢复。
- 独立24步更新控制器不调用stock learn，不随机化episode长度，不重复final-save。
- 本次仅用假算法测试更新计数/失败语义，真实模型/PPO/torch.save/torch.load仍为0。
- 实际模型schema、GPU读回和一次真实更新，须在另行明确的PPO验收预算内验证。

integration/process_vector.py / process_owners.py
- spawn隔离进程，每进程只有一个native nworld=1 owner；目前逻辑owner仅0/1。
- 发出全部动作后再收响应；消息有owner和sequence，跨进程仅传CPU数值，无CUDA句柄共享。
- 选择性reset只触碰所选owner，其他owner只返回缓存；空reset是真正无操作。
- 超时、错序、错误或解码失败会停止整个vector，不重发动作、不替换进程、不自动补跑。
- close确认与清理完成回执分开，子进程退出后仍核验native ledger和resource closeout。
- 这是进程接口候选；尚未做真实N2进程/GPU隔离验收，更不等于原生nworld>1批量后端。

integration/planning.py / source_bundle.py
- 为N1/N2编译保守且完整的命名调用上限，包含setup/reset/forward/sense/graph/cleanup。
- minibatch按环境维校验：N1/B1，N2/B1或B2；继承B=4不允许用于N1/N2。
- 吞吐以同步墙钟下公共transitions/s计，不把内部微步/s当作训练采样吞吐。
- 本次主机CPU任务会生成cloud-source-inputs.zip：248份已核验源码、158项资产（资产46,175,197字节）。
- 9,710,625字节的历史stage10_finalize.py只保留SHA身份，不复制嵌入式旧归档。
- 不包括checkpoint、虚拟环境、依赖wheel或驱动；这不是已安装的云训练环境。
- 每个新包成员做SHA/大小/CRC校验，失败仍保留报告；不安装或升级依赖。

3. 本次主机运行只做什么

使用现有 .venv Python，读取并保护旧来源/started/归档；复用v7成功配置，不再运行旧配置批次。
先执行新的纯契约检查，再用真实Torch2.9.1/TensorDict0.10.0跑假环境CPU场景，最后打包源码/资产。
各独立场景收集失败后继续汇总，避免首个失败掩盖其他问题。
CPU子进程不导入mjlab、GLFW、Matplotlib或GitPython；它们的v7通过证据直接复用。
Torch load/save/CUDA初始化被阻断。无模型、PPO、物理仿真、网络或子进程启动。
外层仍有只读Git身份查询、单个CPU子进程和独立本地证据Git commit；不会push。

预期成功输出：
Stage12Readiness=STAGE12_INTEGRATION_CPU_PASS_PHYSICAL_PENDING
Stage12Transfer=LOCAL_TRANSFER_COMMIT_READY_MANUAL_PUSH_PENDING
Stage12NativeAdded=0; Cumulative=674456 completed / 674457 attempted; PPO=0; cloud_training=0
即使CPU检查失败，也尽可能形成完整失败回包和可手动推送的证据提交。不要删除started重跑。

4. 运行与回传完整命令

下载 microduck-stage12-integration-v1-20261016.zip 到 /home/lx/下载。
解压到新的 microduck-stage12-integration-v1 目录，不覆盖旧readiness包：

unzip -n /home/lx/下载/microduck-stage12-integration-v1-20261016.zip -d /home/lx/下载 &&
python3 -I -S /home/lx/下载/microduck-stage12-integration-v1/microduck-stage12-integration.py --verify-package &&
python3 -I -S /home/lx/下载/microduck-stage12-integration-v1/microduck-stage12-integration.py --run

外层标准库启动器会使用 /home/lx/microduck-double-balance/workspace/.venv/bin/python。
保留该解释器路径，绝不resolve成系统Python。不运行pip install、uv sync、旧smoke或旧--run。
出现LOCAL_TRANSFER_COMMIT_READY_MANUAL_PUSH_PENDING后，单独手动执行：

git -C /home/lx/microduck-double-balance/artifacts/double-balance-stage12/integration-v1/transfer push origin HEAD

粘贴push结果和Stage12TransferCommit即可；不用上传返回ZIP。
本轮transfer分支为stage12-integration-v1-transfer-<UTC小写>-<随机后缀>，运行后才生成真实commit。
GitHub自动显示Create a pull request不是需要创建PR的指令；证据分支不要合并。

只读检查计划（可重复，不启动主机检查或物理）：
python3 -I -S /home/lx/下载/microduck-stage12-integration-v1/microduck-stage12-integration.py --plan

只读查看已有运行：
python3 -I -S /home/lx/下载/microduck-stage12-integration-v1/microduck-stage12-integration.py --status

仅在返回归档和return-receipt.json已经形成时恢复证据提交，不执行CPU/物理：
python3 -I -S /home/lx/下载/microduck-stage12-integration-v1/microduck-stage12-integration.py --prepare-transfer

push失败只重试同一个push；non-fast-forward不强推。不删除任何started，不重启--run。
新目录：/home/lx/microduck-double-balance/artifacts/double-balance-stage12/integration-v1/
started-v1.json覆盖整个新聚合批次。5秒心跳指向host-probe.log，无任意CPU墙钟截止。
本次返回包上限96MiB（因为新增源码/资产包）；每片仍512KiB原始数据，base64 ASCII分片、分层SHA不变。
这只调整新传输包容量，不改任何旧归档、物理阈值或旧证据。

5. 已生成、未授权的物理测量提案

measurement-budget-proposal.json包含全部逐owner caps。当前authorization=PENDING_USER_BUDGET。
建议先N1生命周期候选5公共步，再N2每owner3公共步；总110宏步。
新建真实环境最多3个，reset尝试最多18次。
积分上限38,720；GPU forward上限155,902；CPU forward上限144。
本地原始证据预留12,213,813,248字节（11.375GiB），不是预计实际用量；JIT/cache另计。
各owner完整列出setup/reset/sense、BAM/manager、solver、graph销毁等上限。
PPO=0、云训练=0、checkpoint load=0。提案不是执行许可；本包--run不会执行它。
该短测量仅用于初步接入和审计吞吐，不证明5秒稳定、不证明完整10秒episode，也不保证吞吐已足够训练。
若后续长episode验收需要更多预算，应根据这次真实成本一次给出完整新预算，不能偷偷扩展。
本地factory仍限制5060测量候选；云端factory/device profile和发布组合尚待目标环境验证。

固定成本（未变）：
64层接受：32+64+64重放=160积分/宏步。
128层接受：32+64+128+128重放=352积分/宏步。
每公共步10宏步；最多3520积分。宏步forward预留1413。
单次24步rollout积分上界：N1=84,480；N2=168,960；这些数值不包含完整setup/reset预算。
当前审计后端保留大量同步、快照、WAL与原生检查，不能用短试验直接承诺训练速度。
优化必须另外证明不削弱原标准；不能悄悄去掉mandatory replay、同步或数值检查换吞吐。

6. 什么时候能正式训练

不再以“还要上传几版”回答；v7 CPU准备已关闭。
必要的剩余验收是：
A. 新持续后端、终点观测、reset/RNG和资源生命周期的真实运行，包括足够长的episode范围。
B. 选定N1/N2规模的真实同步吞吐、内存和完整调用成本；可先小规模，不必等待4096环境。
C. 专用runner与真实模型/optimizer/checkpoint更新边界读回的端到端验收。
D. 云目标依赖/源码/资产/设备profile复现，及用户明确云GPU、环境数、更新数、小时数和金额上限。
预算需要把C的小额PPO验收和后续正式训练分开列明。当前PPO=0不能自行执行C或云试训。
本次交付候选代码和聚合CPU验证入口；不提供假装可以直接开训的stock train命令。
因此当前不能给出可信的开训日期或固定剩余轮数，也不承诺“再一次一定开训”。
训练预算尚未明确是授权边界，真实后端/云组合尚未验收是技术边界；二者都要解决。

7. 旧checkpoint与固定物理语义

旧checkpoint：
/home/lx/microduck-double-balance/artifacts/double-balance-stage08/returned/5d998fc322345102/evidence-20260930T093645-8955/extension/E-20260929/segment-001/checkpoints/update_004000.pt
Bytes=10073500
SHA256=86d55c3703c18fcf4817db49c6e4dcf839b3f5195d68ae2c559db7e222bded97
SchemaSHA256=e740e6ca4ad74c53a0eaaa9c95dcae9b69fd5b0edf122593743cd015dcb388a7
历史94个有限张量的检查复用；DirectResume=BLOCKED_TASK_AND_TIMING_DIFFER。
首次训练fresh：Actor61→LSTM256×1→MLP512/256/128→14；Critic85→MLP512/256/128→1。
分布参数、normalizer、Adam和新任务计数均fresh；actor-only transfer未放行。

真实头部、名义reset、hold=0、BAM、actor61/critic85/action14均保持。
physics_dt=.002，decimation=10，public_step=.020，episode=10秒。
float32原5秒成功边界无epsilon、无四舍五入；旧奖励、误差、接触/事件阈值不变。
本候选外部auto-reset由TerminalBridge实现，底层manual reset；顺序变化必须真实验收。
不开放随机reset、课程、扰动或实机。

8. 来源、保护与永久Git/云协议

LocalRepository=/home/lx/microduck-double-balance/workspace
Interpreter=/home/lx/microduck-double-balance/workspace/.venv/bin/python
DownloadDirectory=/home/lx/下载
Artifacts=/home/lx/microduck-double-balance/artifacts
Branch=double-balance
CurrentReviewedHEAD=73ee01930a031d76a9d57419afdfd4464a547ab3
Stage11TagTarget=6ba0dc2de381321576f32310e1995a8e2e11e566
Stage11TagObject=bd1b219be55c813811d4f7cfa25d245b37166688
OriginSSH=git@github.com:liuxue-lab/microduck-double-balance.git
TrackedDiffSHA256=b90daa0274c14d95cec9511ec4ac93568bd604e93ce3c60df1bb1c4155005d7e

只接受原冻结base或已逐项审查的上述文档跟进HEAD，不放开任意后代。
219来源pin、现有dirty patch、Stage11 started和readiness v1–v7标记/归档/回执均前后核验。
baseline/g11的149份冻结文件及所有旧prior-readiness证据保持原字节。
完整Stage11交接：baseline/stage11-handoff-closed.txt。
完整永久协议原件：baseline/g11/provenance/docs/handoffs/
  microduck-local-ssh-push-protocol.md
  microduck-cloud-gpu-and-transfer-protocol.md
旧文档按原时间保存；当前结论以本TXT、stage12-plan.json及prior-readiness-v7/audit.json为准。

默认流程：用户本地SSH push，助手按完整固定commit读取并验证所有分片与ZIP/成员身份。
不让用户来回上传返回ZIP，不自动fetch/import/push/PR/merge，不在源码分支混入模型归档。
沿用既有SSH身份，不改HTTPS/PAT、不重新网页登录、不默认建新key。
Permission denied(publickey)按原协议查agent/key；正常push成功不重复整套远端发布核验。
Everything up-to-date但缺新证据时，检查本轮integration-v1 transfer目录/分支/commit，勿用Stage11旧目录。
SHA/标签冲突/non-fast-forward保留现场审查，不改预期SHA、不强推。

本机5060用于开发/仿真/推理；正式PPO只在另行明确预算的云GPU进行。
不使用历史云SSH端口，不自动开关实例或启动旧关机daemon。
最近用户确认2026-10-01的5090已关机，本轮未实时查询电源；不宣称实时状态。
Stage09完整head-trace尚未全部返回的限制保留，不宣称所有云数据已备份。
未来云计算及必要打包/回传通过本地SHA验证后，应立即提醒用户手动关机，并分别记录提醒及实际确认。

9. 验证证据和限制

既有66项开发回归通过；本次新增34项纯契约检查和9组NumPy张量协议场景通过，共109项。
包括真实临时文件fsync/原子无覆盖/哈希链/坏文件/partial tail、真实临时Git提交及不重跑保护。
新模拟runner覆盖48条假transition/2次假update的计数，以及失败update保留pending。
所有“假算法/假环境/协议替身”都不计为真实PPO、环境或物理运行。
助手环境没有Torch/MuJoCo/Warp/mjlab；未运行用户主机--run、原生GPU或模型。
真实Torch/TensorDict的新场景将在本次聚合CPU命令中一次执行；原v7真实TensorDict PASS已经复用。
新增native factory/process代码经过源码审查与编译，尚无native验收回执。
validation/results/保留本轮日志；validation/inherited-v7-results/只是历史，不重复计算执行次数。

本次审核后不应重新打开readiness配置/导入修补；依据新integration报告聚合处理剩余问题。
