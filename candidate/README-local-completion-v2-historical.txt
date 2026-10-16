Microduck Double Balance — Stage12 local-completion v2
交接日期：2026-10-16（北京时间）；运行标识使用UTC。

当前结论
local-completion-v1已经停止并归档。CPU回归13/13通过；进入N1时，spawn子进程仍在父进程目录，
触发Git导入保护：Git import must run inside the new output directory。
这属于启动代码错误，不是GPU性能不足。原生owner_session尚未进入，物理积分新增0。
累计仍为674456 completed / 674457 attempted，历史失败1次保留；Stage11=11/11 CLOSED_SCOPED。
本包修复启动链，复用已完成CPU报告和410项输入包，只继续尚未执行的物理验收。
Stage12尚未关闭，TrainingReady=False。PPO=0，cloud_training=0；现在不需要开启云电脑。

1. 已核验的本次Git回包
仓库：liuxue-lab/microduck-double-balance
分支：stage12-local-completion-v1-transfer-20261016t001141z-fd8844
固定提交：623484cd5dbb8182205dc035dd13fc95d831b821
运行：20261016T001141Z-fd8844
归档根：/home/lx/microduck-double-balance/artifacts/double-balance-stage12/local-completion-v1
返回ZIP：stage12-local-return-20261016T001141Z-fd8844.zip
字节：15735288
SHA256：01c50a20a5c2db411050db848ef6c16780ea350963a286381357a90ac958e12f
程序PACKAGE SHA256：7a65d7ee18fc4a172f9b1b6ea134061e662a72bfca3d71b7e2f550d57ffe51b2
31个ASCII分片的文本/解码SHA、偏移、重建ZIP、CRC和386个成员全部核验。
返回程序与上一版交付逐文件一致；没有混用移动分支的不同版本。

实际结果：
- CPU13/13 PASS，Torch 2.9.1+cu128，TensorDict 0.10.0；没有CUDA初始化、模型或优化器构造。
- portable cloud-inputs.zip：410项，16181045字节。
  SHA256=24701df9e56face7980a1a2e0544dfebc41a4cf1b84ec4912be795d1a35a43b2。
- single_owner_lifecycle的owner0启动失败，子进程exitcode=1，无强制中断。
- 原生owner目录、native-ledger和原生报告均未建立；N2未执行。
- 新增完成/尝试积分都是0，不把CPU fixture中的模拟计数计入累计。
- preservation=PASS；两份身份快照一致，旧源码、旧checkpoint、started、归档均保留。
本次源证据在prior-local-completion-v1/；integration-v1/readiness/Stage11历史材料继续保留。

2. 修复范围与验证
integration/child_bootstrap.py提供生产子进程共用的启动路径：
创建独占owner-bootstrap/runtime-scratch，先切换cwd，再配置headless/字体/Git的受限导入。
重置Python临时目录缓存，确保临时文件落在该owner的新目录；不改变父进程或源码目录的cwd。
每个owner保存bootstrap-report.json，包含原始/目标cwd、Git查询、字体回退以及失败栈。
报告位于缓存目录之外，返回ZIP包含它；原生失败后可以准确判断停在导入还是物理会话。
仍仅允许GitPython的两次git version；网络、其他子进程和越界写入继续拒绝。
不更改已安装依赖，不自动安装，不修改数值阈值或原生算术。

已完成的本轮开发验证：
- 用真实依赖复现v1的cwd错误；确认失败前没有进入原生会话。
- 真正调用生产native_child和共用bootstrap，实际spawn三个进程，完成N1/N2编排和干净退出。
  仅原生会话替换为明确标注的CPU fixture；生产前置导入、IPC和监督器均实际执行。
- 从生产native_factory提取其完整前置import块，实际导入并通过配置preflight。
  核对45项捕获的mjlab/RSL源码及18项BAM源码；BAM使用uv.lock锁定提交，不能把同版本PyPI轮子当成相同源码。
- 每个进程验证两次Git查询、一次字体回退，以及拒绝额外进程、网络解析、越界写入。
- 23项传输/账本/Git针对性测试通过；7项已完成结果复用、防篡改、防重复启动检查通过。
- 另有8项真实本地Git/返回ZIP/云输入解包和拒绝错误输入检查通过；云命令固定commit且bash语法检查通过。
  这些交付检查使用明确标记的合成物理PASS回执，不是GPU验收证据。
- 156份baseline文件与v1逐字节一致；native_factory/native_runtime/planning/transitions/
  process_vector/runner/durable/native_accounting也逐字节一致。
验证记录在validation/results/；旧117项开发测试及旧交付测试移入历史结果目录，未冒充本轮新结果。
开发依赖只安装在助手的隔离目录，未修改用户虚拟环境；未执行任何真实GPU积分、PPO或checkpoint读写。
这些检查解决了启动覆盖缺口，但不能代替用户RTX5060的物理结果。

3. 已完成任务如何复用
先核验v1固定归档SHA、started所指运行/程序身份、return/transfer回执、退出记录和所有保留报告。
要求v1确实是这次零积分的启动失败，且三个旧原生owner目录都不存在。
已完成CPU报告要求原SHA、13项全PASS；输出cpu-regression-reuse.json，reexecuted_cases=0。
410项输入ZIP先逐成员核验，再复制到新运行；不重新采集源码、资产或4份构建元数据。
若旧证据已变化或旧原生owner目录出现，停止并保留，不按零积分假设继续。
新物理工作只进入local-completion-v2；不重跑local-completion-v1，也不删除其started。

4. 本机完整运行命令
源码工作区：/home/lx/microduck-double-balance/workspace
现有解释器：/home/lx/microduck-double-balance/workspace/.venv/bin/python（不resolve成系统Python）
下载ZIP到 /home/lx/下载；不要覆盖旧解压目录。

unzip -n /home/lx/下载/microduck-stage12-local-completion-v2-20261016.zip -d /home/lx/下载 &&
python3 -I -S /home/lx/下载/microduck-stage12-local-completion-v2/microduck-stage12-local.py --verify-package &&
python3 -I -S /home/lx/下载/microduck-stage12-local-completion-v2/microduck-stage12-local.py --run --native-integrations 38720 --wall-seconds 43200

新证据根：/home/lx/microduck-double-balance/artifacts/double-balance-stage12/local-completion-v2
本条命令显式授权这个修复批次的有限本地物理验收，不授权PPO或云计算。
执行顺序：旧证据/源码保护 → 复用CPU报告与输入包 → N1 → N2 → 对账 → 归档 → 本地Git传输提交。
中间不需要返回聊天确认，不自动重试失败的owner，不把未执行点标为完成。
已有v2 started时只显示状态并退出；不清理标记、不换目录伪装恢复。

看到 LOCAL_TRANSFER_COMMIT_READY_MANUAL_PUSH_PENDING 后，手动执行：

git -C /home/lx/microduck-double-balance/artifacts/double-balance-stage12/local-completion-v2/transfer push origin HEAD

然后粘贴push结果与Stage12TransferCommit；无需上传返回ZIP。
push失败只重试push，禁止再次--run。分支为stage12-local-completion-v2-transfer-<UTC小写>-<随机后缀>。
助手只读取固定完整commit；不自动push、创建PR或merge。GitHub的PR提示不需要操作。

只读查看状态：
python3 -I -S /home/lx/下载/microduck-stage12-local-completion-v2/microduck-stage12-local.py --status

只读查看本次限额：
python3 -I -S /home/lx/下载/microduck-stage12-local-completion-v2/microduck-stage12-local.py --plan --wall-seconds 43200

只读查看GPU：
nvidia-smi -l 2

campaign.log路径随Stage12Run打印；每5秒心跳给出日志路径。
每个owner的progress.json提供macro/public-step/操作计数和WAL pending。
仅当return-receipt.json已形成而传输提交未准备成功时：
python3 -I -S /home/lx/下载/microduck-stage12-local-completion-v2/microduck-stage12-local.py --prepare-transfer
此命令只处理已验证归档，不重启CPU或物理。若归档/receipt尚未形成，保留全部日志和原始文件供审查。
文件名带(1)时用实际名称并加引号，不覆盖旧包。目录或SHA错误时停止，不修改期望SHA绕过核验。

5. 原限额与物理范围
计划SHA256=12ef65c6f34c555f923ecf7b11c555402418612e17eeec28a59ff4dbfcbecd77。
先一个owner运行5个公共step，再两个独立owner各运行3个公共step；每进程native nworld=1。
N1为warmup1 + timed2 + 显式reset + 后续2；N2为每owner warmup1 + timed2，之后选择性reset owner0。
缓存重复读取、空reset、未选中owner的RNG/观测/计数不变、公共step计数及关闭回执均检查。
总上限：3个真实环境，110宏步，38720内部积分，155902 GPU forward，144 CPU forward，18次reset尝试。
策略/优化器构造0，torch.load/save=0，PPO=0，cloud_training=0。
证据预留11.375 GiB，另需4 GiB余量；建议至少16 GiB可用磁盘，运行中低于2 GiB会停止保留。
43200秒是CPU/物理子进程的12小时强制上限，不是耗时承诺；身份、清理和打包另计。
超时只结束本次创建的进程组并保留证据，不结束无关进程，不自动补跑。
物理dt=.002，decimation=10，公共dt=.020，episode上限10秒；61/85/14布局、真实头部、BAM、hold=0不变。
32/64/128试探及mandatory exact replay保持；float32>=5秒成功判断不增加epsilon或rounding。
这是短pilot，不证明持续5秒稳定、完整10秒episode、长期吞吐或PPO学习效果。

6. 结果及何时需要云电脑
LOCAL_NATIVE_PILOT_PASS_CLOUD_BUDGET_PENDING：本地两个点及保护检查通过，进入云预算/云验收准备。
LOCAL_COMPLETION_STOPPED_PRESERVE：停止并保留，查看bootstrap、原生报告和WAL，不重跑。
传输提交准备成功只代表证据可上传，不代表物理PASS。
各owner的WAL只加一次Stage11基线；中断保留completed/attempted区间，不推断成功，不返还预算。
本次已知累计仍为674456/674457；v2结果必须根据新账本加总后更新。
公共transitions/s与内部积分/s分开，启动/JIT与2个timed step分开，计时含IPC、审计及同步。
Torch峰值按owner记录；driver free/total为全设备采样，不相加冒充驱动峰值。
不从两个点自动扩到4/8/4096环境，也不假设换5090就能解决CPU审计或IPC瓶颈。

现在无需开启云电脑。v2本地物理通过后，下一步集中明确云端预算和执行云端验收。
需要的预算项：GPU型号、最长GPU小时、总费用与币种、N/B、云native积分/forward限额、
真实模型/优化器构造次数、PPO更新数、checkpoint写入/读回次数。
cloud-budget-template.json仍默认零训练/未填写；历史A800/5090可用时长不代替本轮授权。
本地PASS后自动生成以下命令文件（只生成，不执行）：
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/local-completion-v2/CLOUD-COMMANDS.sh
其中使用真实transfer分支和commit。完成手动push并明确预算后，才在新的云主机运行。
命令要求STAGE12_GPU_HOURS、STAGE12_SPEND_LIMIT、STAGE12_CURRENCY均为已同意值。
它用Git SSH独立克隆，核对commit/分片/程序/410输入，运行uv sync --locked --python 3.12，
再检查来源、219 pins、Python、RSL5.0.1、TD0.10.0和Torch2.9.1 CUDA；不会自动训练或开关机。
此云端入口停在CLOUD_ENVIRONMENT_IDENTITY_PASS_NATIVE_AND_PPO_PENDING。
正式训练还需要云设备上真实native生命周期/episode/reset验收，以及预算内真实更新和checkpoint读回。
现有native_factory限定5060；云GPU适配及物理放行仍待云预算和设备信息，不能直接套用本地命令。
只有上述实际验收通过并明确训练预算后才能正式开训，不把CPU检查或短pilot写成TrainingReady=True。

7. checkpoint与正式训练接入
旧checkpoint继续只读保留，不反序列化、不上传、不直接resume：
/home/lx/microduck-double-balance/artifacts/double-balance-stage08/returned/5d998fc322345102/evidence-20260930T093645-8955/extension/E-20260929/segment-001/checkpoints/update_004000.pt
10073500字节；SHA256=86d55c3703c18fcf4817db49c6e4dcf839b3f5195d68ae2c559db7e222bded97
schema SHA256=e740e6ca4ad74c53a0eaaa9c95dcae9b69fd5b0edf122593743cd015dcb388a7。
任务、特征和时序不同，旧normalizer/Adam/LSTM状态迁移仍BLOCKED。
正式方案为fresh actor LSTM61->256单层、MLP512/256/128->14；critic85->512/256/128->1；全新统计与优化器。
专用24步更新循环、原子checkpoint发布及账本控制器已准备；stock train/Stage06 smoke/PreparedRunner仍封锁。
N1/B1，N2/B1或B2；不继承不整除环境数的B4。
更新前写attempt，完成后唯一partial+fsync+无覆盖原子发布+SHA回执，最后提交更新账本。
只恢复最后已提交更新；加载前检查身份、hash和账本；使用weights_only=True并恢复optimizer LR标量。
恢复发生在更新边界，重新reset环境/RNN/rollout，恢复policy采样RNG；不宣称中途或逐位等价续训。
24步rollout的内部积分上限N1=84480、N2=168960，另加setup/reset；这里只是后续预算算术，不是运行授权。

8. 固定身份、保留范围与云状态
源码分支double-balance；SSH origin=git@github.com:liuxue-lab/microduck-double-balance.git。
已审查HEAD=73ee01930a031d76a9d57419afdfd4464a547ab3。
Stage11关闭commit=6ba0dc2de381321576f32310e1995a8e2e11e566；tag对象=bd1b219be55c813811d4f7cfa25d245b37166688。
原tracked diff SHA256=b90daa0274c14d95cec9511ec4ac93568bd604e93ce3c60df1bb1c4155005d7e，保持decode().strip().encode()定义。
不改主checkout，不清理旧补丁，不重跑Stage06/Stage11/readiness v1–v7/integration-v1/local-completion-v1。
所有started和归档保留。原始native快照与缓存留本机；回包包含关键报告、WAL与非缓存证据SHA清单。
单个传输ZIP不超过49 MiB；证据分支独立，不合并到double-balance。

必须持续保留两份协议及其命令/错误经验：
docs/handoffs/microduck-local-ssh-push-protocol.md
docs/handoffs/microduck-cloud-gpu-and-transfer-protocol.md
包内完整冻结副本位于baseline/g11/provenance/。

2026-10-01用户曾确认5090关机；不能作为现在的电源证据。当前云电源=UNCONFIRMED。
本轮未启动、连接或训练云主机；新的SSH地址/端口/身份尚未给出，不猜测复用旧端口。
今后云计算和必要打包/回传结束且本机SHA核验通过，应立即提醒用户手动关机；提醒与实际确认分别记录。
GPU利用率0不等于任务结束；不自动开关实例，不重启失败的关机daemon。

新对话开场语
继续microduck-double-balance Stage12，读取local-completion-v2交接TXT及最新Git传输固定提交。
已审查v1回包623484cd5dbb8182205dc035dd13fc95d831b821：CPU13/13 PASS，因子进程cwd在原生会话前失败，新增积分0。
v2已修复共用启动链并补真实依赖/spawn验证，复用旧CPU和410输入；请从新回包对账继续至云预算/验收准备。
保留全部started和旧包，不重跑完成批次，不改阈值；PPO=0、云训练=0，训练预算另行明确，不自动push/PR/merge。
