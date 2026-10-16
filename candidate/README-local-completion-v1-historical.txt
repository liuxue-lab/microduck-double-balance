Microduck Double Balance — Stage12 local-completion v1
交接日期：2026-10-16（北京时间）；运行标识使用UTC。

本轮结论
已完整校验 integration-v1 的Git回包，修复唯一已报告的TensorDict进程序列化错误。
新增本地一体化入口：身份保护 → 源包复用 → 真实CPU进程回归 → RTX5060 N1 → N2 → 账本汇总 → Git传输提交。
CPU通过后自动进入本次明确限额的物理验收，中间不需要再上传CPU回包或等待聊天。
本包交付时尚未运行用户RTX5060；Stage12未关闭，TrainingReady=False。
正式训练预算仍待单独明确。PPO=0、cloud_training=0；没有自动push、PR、merge或云开关机。

1. 已核验的来源
GitHub仓库：liuxue-lab/microduck-double-balance
本次回包分支：stage12-integration-v1-transfer-20261015t184102z-6225fa
固定提交：eb5fe0c2371393fe599ed1ba92e078dfe8390f3c
运行：20261015T184102Z-6225fa
返回ZIP：stage12-integration-return-20261015T184102Z-6225fa.zip
返回字节：15609870
返回SHA256：e75d087027812cdb217018d5001f9b08412b949f9bc496d61c2941faf7183e04
程序PACKAGE SHA256：ee6eb6ac9b9d1dadb411b78a978800d72291375b1e953cc1cad95fe961138cdf
30个ASCII base64分片、重建ZIP、CRC和330个返回成员全部核验。
原回包：34/34纯契约PASS；8/8真实Torch/TensorDict终点场景PASS；双owner序列化失败1项。
原始错误：TypeError: must be real number, not dict。
原回包源资产包：406个成员（248源码、158资产），16172847字节。
其SHA256：5d2c4ef1fc8693d328f1f6ac730d7fa35f14b3b032ba9ec28bb44637d5256873。
原回包NativeAdded=0，PPO=0，cloud_training=0，旧证据保护PASS。
prior-integration-v1/保留完整报告和固定身份；不会重跑integration-v1。

固定源码工作区：/home/lx/microduck-double-balance/workspace
固定下载目录：/home/lx/下载
固定归档根：/home/lx/microduck-double-balance/artifacts
源码分支：double-balance
源码HEAD：73ee01930a031d76a9d57419afdfd4464a547ab3（已审查的文档提交）
Stage11关闭基线及stage-11-complete指向：6ba0dc2de381321576f32310e1995a8e2e11e566
标签对象：bd1b219be55c813811d4f7cfa25d245b37166688
原跟踪差异SHA256：b90daa0274c14d95cec9511ec4ac93568bd604e93ce3c60df1bb1c4155005d7e
原差异哈希沿用decode().strip().encode()约定，不改变历史定义。
Stage11=11/11 CLOSED_SCOPED；stage-11-complete成功推送已记录。
当前累计：674456 completed / 674457 attempted；保留历史失败1次。

2. 本次修复及验证边界
TensorDict也有detach、dtype、shape、tolist，旧代码先按这些属性判断Tensor，导致把字典列表送入torch.tensor。
新序列化先识别映射，递归编码真正的张量叶子；解码验证dtype、嵌套shape、标量类型、整数范围和有限值。
另补标量/空张量shape恢复、保留键冲突、观测形状、done与终止标志一致性检查；不传递CUDA句柄。
终点观测/reward/flags/extras先快照再reset，timeout只用gamma*V(terminal)补偿一次。
进程发送全部命令后再收回复；失败后不重发动作、不替换owner、不补跑。
增加只读同步inspect，用于检查缓存、RNG、选择性reset、公共step与内部积分计数以及显存。
原子checkpoint账本、候选更新循环、原数值运行时和RNG策略保留；资源清理错误不会覆盖最初的初始化错误。

本轮开发验证共117项通过：104项单元/契约测试 + 13个真实CPU张量及spawn场景。
另有7项完整Git/源包/云端解包与拒绝错误输入的交付检查通过；测试回执为明确标记的fixture。
真实后端：Torch 2.9.1+cpu、TensorDict 0.10.0，私有开发依赖目录；没有修改用户虚拟环境。
13个场景包含dict/TD终点组合、双owner协议、异常序列化、实际双spawn正常/失败流程，以及生产监督器与完整N1/N2编排。
这些进程使用模拟环境；测试内的合成积分计数明确标为fixture，不计入项目累计。
实际GPU/native、真实策略/优化器/PPO、checkpoint读写仍未在开发环境执行。
旧错误已在真实Torch/TD上复现后再验证修复；没有把NumPy替身等同于真实TensorDict。
全部156份baseline文件与integration-v1逐字节一致；validation/results/保留本轮证据，旧测试记录单独保留。

3. 本轮本地任务的明确限额
--run需要显式参数 --native-integrations 38720 --wall-seconds 43200。
它表示授权本次新的本地RTX5060仿真验收；不是任何PPO或云训练授权。
本轮新建3个真实环境：先1个owner运行5个公共step，再2个独立进程各运行3个公共step。
每个owner仍为native nworld=1；没有声称实现原生批量nworld>1。
N1：1个warmup + 2个计时step + 显式reset + 2个后续step，总计50宏步。
N2：每owner 1个warmup + 2个计时step；随后选择性reset owner0并检查owner1不变；合计60宏步。
每个点均检查缓存重复读取、空reset不改变计数/RNG/观测，并检查关闭回执。

总上限：110宏步；38720内部积分；155902次GPU forward；144次CPU forward；18次reset尝试。
环境构造最多3次；策略构造0、优化器构造0、torch.load/save=0、PPO=0、cloud_training=0。
证据预留12213813248字节（11.375 GiB），另需4 GiB缓存/安全余量；建议至少16 GiB可用磁盘。
运行中可用磁盘低于2 GiB会停止并保留WAL，预留空间用于清理和最终报告。
43200秒是CPU/物理子进程的12小时强制上限，不是预计耗时；身份校验、退出清理和打包另计。
达到上限保留started和不确定计数，不自动重试。
没有测得实际吞吐前，不宣称任务很快，不据此承诺5090训练时长。
完整命名操作限额见measurement-budget-proposal.json和integration/planning.py；用户运行参数绑定完整计划指纹。
不减少32/64/128试探或mandatory exact replay，不放宽阈值，不用eps或rounding改变float32>=5秒成功判定。
physics dt=.002，decimation=10，公共dt=.020，episode上限10秒；actor61/critic85/action14，hold=0、真实头部/BAM保留。
这只是短物理pilot，不证明持续5秒稳定、完整10秒轨迹、持续吞吐或PPO学习效果。

4. 下一条本地完整命令
下载ZIP到 /home/lx/下载；解压到新的目录，不覆盖任何旧包。

unzip -n /home/lx/下载/microduck-stage12-local-completion-v1-20261016.zip -d /home/lx/下载 &&
python3 -I -S /home/lx/下载/microduck-stage12-local-completion-v1/microduck-stage12-local.py --verify-package &&
python3 -I -S /home/lx/下载/microduck-stage12-local-completion-v1/microduck-stage12-local.py --run --native-integrations 38720 --wall-seconds 43200

外层用标准库；子进程用现有 /home/lx/microduck-double-balance/workspace/.venv/bin/python。
保留该解释器路径，不resolve为系统Python。本机不运行pip install或uv sync，不重跑任何旧批次。
新归档目录：/home/lx/microduck-double-balance/artifacts/double-balance-stage12/local-completion-v1
出现 LOCAL_TRANSFER_COMMIT_READY_MANUAL_PUSH_PENDING 后，单独手动执行：

git -C /home/lx/microduck-double-balance/artifacts/double-balance-stage12/local-completion-v1/transfer push origin HEAD

粘贴push结果和Stage12TransferCommit即可；无需上传ZIP，不创建PR。
传输分支为 stage12-local-completion-v1-transfer-<UTC小写>-<随机后缀>，完整commit在本机运行后生成。
push失败只重试同一条push；不要重跑--run，不删除started，不换目录伪装恢复。

只读查看状态：
python3 -I -S /home/lx/下载/microduck-stage12-local-completion-v1/microduck-stage12-local.py --status

只读查看预算：
python3 -I -S /home/lx/下载/microduck-stage12-local-completion-v1/microduck-stage12-local.py --plan --wall-seconds 43200

只读查看本机GPU：
nvidia-smi -l 2

日志路径是 Stage12Run 后的 campaign.log；每5秒心跳给出完整路径。
查看各owner的progress.json可见已完成宏步、公共step、调用上限和WAL pending。
退出后若return-receipt.json已存在，而本地Git传输提交没有准备好，只运行：
python3 -I -S /home/lx/下载/microduck-stage12-local-completion-v1/microduck-stage12-local.py --prepare-transfer

若返回ZIP/return-receipt尚未成功形成，保留完整运行目录和日志；上述命令不会假装补齐缺失归档，也不会重启物理任务。
已有started时--run只报状态并退出。身份、源码、依赖、磁盘或CUDA检查失败均保留报告，不自动升级/改阈值/换后端。

5. 运行结果如何解释
LOCAL_NATIVE_PILOT_PASS_CLOUD_BUDGET_PENDING：两个点、reset隔离、计数、清理、身份保护通过；下一步可讨论云预算和云验收。
LOCAL_COMPLETION_STOPPED_PRESERVE：本批次停止；保留全部证据。失败不会隐藏，未执行点不会被标成完成。
Stage12TrainingReady始终False；上面任一状态都不是正式训练许可。
新增累计按3个owner的WAL只加一次；不是每个owner都重复加Stage11基线。
中断时报告completed/attempted上下界，不把未确认完成算作成功；后续先对账，不返还尝试预算。
若实际新增n次完成/m次尝试，新累计为674456+n / 674457+m；当前交付时n=m=0。

测量单位为成功公共transitions/s，另外单列内部积分/s。N2统计包含两个owner的公共transition。
窗口仅2个计时step；包含IPC、同步、账本及审计开销。启动/JIT与窗口时间分开报告。
Torch peak是各owner自己的峰值；driver free/total是采样时全设备状态，不能相加或冒充驱动显存峰值。
不凭两个点自动扩到4/8/4096环境，不据短pilot自动花费云预算。
大体积原始native快照与缓存留在本机；回包包含WAL、macro结果、清理报告和非缓存证据SHA清单。
源码和资产不重新采集：复用integration-v1的406项包，并补4份构建/许可元数据（README.md、LICENSE、NOTICE、LICENSE-HARDWARE），最终410项。
这修复了只有pyproject/uv.lock而缺readme/license导致云端构建失败的缺口。
这些新增元数据固定到已审查源码HEAD的Git blob；没有把当前默认分支当作来源。

6. 什么时候需要云电脑，什么时候能正式训练
这次本地验收通过后，不再重复readiness或CPU接入准备。
先看实际吞吐与资源情况，再决定是否进入云端验收；更快GPU不能替代对审计/IPC瓶颈的判断。
云端剩余工作：安装锁定环境并核对来源/资产/GPU；在云设备上做明确限额的native生命周期/episode/reset验收；
在明确PPO/model/optimizer/checkpoint预算后验证真实更新与checkpoint读回；通过后才启动批准的正式训练。
目前不能保证还差固定几次，也不能把本地短pilot通过直接写成TrainingReady=True。
需要一次明确的预算信息：GPU型号、GPU最长小时、总费用上限及币种、N/B、PPO更新数、checkpoint读回次数、云native验收限额。
cloud-budget-template.json是待填计划，默认PPO_updates=0、cloud_training=0、模型/优化器/反序列化预算均0。
24步rollout的内部积分上限：N1为84480，N2为168960；这些不是已经获准的运行预算，不包括setup/reset或优化器开销。

本地通过并准备传输提交后会自动生成：
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/local-completion-v1/CLOUD-COMMANDS.sh
其中已经填入实际传输branch、完整commit和新的云目录，不需要手填虚构SHA。
该文件只准备命令，不会自动执行。先完成手动push，再在另行确认预算的云主机上使用。

云端命令逻辑：
- 沿用SSH身份 git@github.com:liuxue-lab/microduck-double-balance.git 克隆独立transfer分支；验证完整commit。
- 执行该分支顶层cloud-stage12.py，逐个验证committed Git文件、base64分片、ZIP、程序清单、410项输入，写入全新目录。
- 仅在新云项目目录运行 uv sync --locked --python 3.12；不改本机环境或原锁文件。
- 使用新 .venv 的未resolve解释器检查Python3.12、219来源pins、完整输入、RSL5.0.1、TD0.10.0及Torch2.9.1 CUDA/GPU库存。
- 输出CLOUD_ENVIRONMENT_IDENTITY_PASS_NATIVE_AND_PPO_PENDING；停在环境校验，不启动PPO。
CLOUD-COMMANDS.sh要求预先设置STAGE12_GPU_HOURS、STAGE12_SPEND_LIMIT、STAGE12_CURRENCY为明确同意的值。
这些值记录预算；脚本不代替平台计费上限/人工关机，也不自动启动或关闭实例。
云端安装会联网获取uv.lock锁定的依赖；本包没有wheelhouse/venv/驱动，不宣称完全离线安装或已完成云复现。
uv --locked的依据：https://docs.astral.sh/uv/concepts/projects/sync/ 和 https://docs.astral.sh/uv/reference/cli/ 。

7. checkpoint策略与正式接入范围
旧checkpoint保留只读：
/home/lx/microduck-double-balance/artifacts/double-balance-stage08/returned/5d998fc322345102/evidence-20260930T093645-8955/extension/E-20260929/segment-001/checkpoints/update_004000.pt
字节10073500；SHA256=86d55c3703c18fcf4817db49c6e4dcf839b3f5195d68ae2c559db7e222bded97
schema SHA256=e740e6ca4ad74c53a0eaaa9c95dcae9b69fd5b0edf122593743cd015dcb388a7
旧任务/特征/时序不同，direct resume及旧normalizer/Adam/LSTM状态迁移继续BLOCKED；不调用torch.load。
正式新任务默认fresh actor LSTM(61->256,1层)+MLP512/256/128->14；critic85->512/256/128->1；新normalizer/Adam/std。
integration/runner.py提供专用24步更新循环；训练构造前需要物理/episode/reset/吞吐/云包验收和显式PPO预算。
RNN minibatch必须整除环境数：N1/B1，N2/B1或B2；不继承不适用的B=4。
每次更新开始先写attempt，完成后唯一partial+fsync+无覆盖原子发布+SHA回执，最后提交账本。
分离last_completed_index、completed_updates、next_iteration；没有原版零基迭代重跑或重复final-save。
仅最后已提交更新可恢复；先核验身份、文件、回执和账本再weights_only=True读回，恢复optimizer LR标量。
继续方式是更新边界重新reset环境/私有流/RNN/rollout，恢复policy采样RNG；不承诺中途或逐位等价续训。
真实策略/优化器构造、真实update、checkpoint GPU读回尚未执行，不能用假算法测试替代。
stock train/旧Stage06 smoke/PreparedRunner路径仍封锁；此包提供的可执行主任务是本地验收和后续云环境校验。

8. 保留项、错误恢复和云关机状态
不修改旧阈值、既有源码补丁、旧归档、checkpoint或任何started；不重跑Stage06、Stage11、readiness v1–v7、integration-v1。
新代码在下载目录，运行证据在新的local-completion-v1；主源码checkout不提交/切分支/reset/merge。
传输提交是独立父less证据分支，不合并到double-balance。不自动推送，不根据GitHub的PR提示创建PR。
错误经验：TD迭代不是key迭代，使用keys/items；TD不是因有detach/dtype就能当Tensor；venv解释器不resolve。
headless GLFW、字体fc-list缺失、GitPython两次git --version等已审查兼容措施继续复用，不扩大子进程/网络权限。
超时/磁盘不足/进程失联：保留started及WAL，核对上下界，不删除后重新跑。

云主机历史：2026-10-01用户确认5090关机；这不是2026-10-16的实时电源证据。
当前云电源状态：UNCONFIRMED；本轮未开启、未连接、未训练，也未自动关机。
当前云SSH主机/端口/身份：未给出新的有效信息，不复用历史端口猜测。
本轮无需立即开云电脑。云计算与必要打包/回传完成，且本地SHA核验通过后，应立即提醒人工关机。
提醒时间与实际关机确认分开记录；GPU利用率0不代表工作结束；不重启失败的自动关机daemon。

必须保留的两份项目协议：
docs/handoffs/microduck-local-ssh-push-protocol.md
docs/handoffs/microduck-cloud-gpu-and-transfer-protocol.md
包内冻结副本见baseline/g11/provenance/。后续交接继续携带固定路径、完整ref/SHA、剩余云任务、错误恢复和关机证据。
