Microduck Stage12 cloud-execution-v1 — 2026-10-16

本包状态：离线编排检查通过；真实云端执行尚未发生；PPO=0、cloud_training=0。
不能把 Package PASS、Git push 成功或环境安装成功当成正式训练放行。

已核验的输入
1. cloud-inventory-v2 固定提交：530adc361e93d013c04d5a80176819f03150d1d9。
   盘点时间 2026-10-16T04:11:26.846737Z。MANIFEST 7/7 字节和哈希匹配，
   collector.py 与交付程序 SHA256 546ec2810d3ce3f803a89227cc1e3272aeb4961954e315b018fba3e65ca99f2f 一致。
2. 同一台 RTX5090，GPU UUID GPU-f231d519-f45e-6111-fc31-33dbb8b98f8f，
   显存 32607 MiB，驱动 580.105.08。盘点当时未列出 compute process。
   盘点工具从宿主侧看到 208 CPU / 约 754 GiB RAM；这些不是套餐配额。
   用户平台截图套餐是 25 vCPU / 90GB RAM；执行时补记 cgroup 限额。
3. 数据盘实际剩余 21,006,852,096 B（约 19.6 GiB），系统盘剩余 31,560,310,784 B。
4. 旧项目 /root/autodl-tmp/microduck-double-balance/workspace，HEAD
   61060f0684aabb460727a0d226b8d17225ee7954，盘点时工作树干净。
   旧 .venv：Python 3.12.3；torch 2.9.1、numpy 2.4.1、mujoco 3.10.0、
   mujoco-warp 3.8.1、warp-lang 1.12.0、mjlab 1.3.0、mjlab-microduck 0.1.0、
   rsl-rl-lib 5.0.1、tensordict 0.10.0、better-actuator-models 1.0.1，10/10 主依赖版本匹配。
   镜像默认 conda 中的 torch 2.8.0+cu128 不能代替上述 .venv。
5. uv 未在盘点进程的 PATH 中找到；Git SSH 失败原因是 GitHub known_hosts 缺记录，
   不是已证实的私钥失效。历史 /root/.ssh/id_ed25519_microduck_stage09 存在。

本包具体处理
- 使用固定 GitHub 官方 ED25519 主机公钥和指纹，在本次新目录写专用 known_hosts；
  StrictHostKeyChecking=yes，复用已有 Stage09 私钥，不读取或传输私钥内容，
  不删除 ~/.ssh/known_hosts，不改为 HTTPS/PAT，不创建新密钥。
- 源包复用已审定 cloud-inputs.zip，410 个 payload 文件 + 清单；
  SHA256 24701df9e56face7980a1a2e0544dfebc41a4cf1b84ec4912be795d1a35a43b2；不重做源码采集。
- 保留旧项目及 .venv。在新系统盘目录 /root/md12-runtime-<run-id> 安装 uv 0.13.1，
  使用 uv sync --locked --no-dev --python 3.12.14 建立独立固定环境。
  新解释器、venv 和 uv 缓存在同一新目录；允许 uv 使用该次缓存续接，旧环境不升级。
  Python 3.12.14 是已审查 processor fallback 的原运行时要求，不能直接用 3.12.3 冒充一致。
  本项目 Git origin 全程 Git SSH；uv.lock 中公开第三方依赖的原下载来源保持不变。
- 旧 factory 的本机 5060 准入改成独立 cloud scope + 5090 准入；入口先核对上述 GPU UUID。
  计数基线改为 704664 completed / 704665 attempted；入口不调用本机历史 campaign。
  继承代码仅改 native_factory/native_accounting/native_runtime 三份文件，另加 cloud_limits。
  数值验收、原生计算、状态回滚、精确 replay 和原阈值保持；adaptation.diff 可审阅。
- 子进程继续使用已验收的短 /tmp/md12-* 编译目录、headless/import 限制和资源销毁流程。
- 每阶段独立 started/done/failed 回执；完成的阶段复用。云任务在后台运行，关闭本机终端
  不会主动结束作业。--launch 已有 launch 回执时只报告状态，不再次启动。
- --resume 允许继续尚未开始的后续阶段，并允许仅安装阶段用缓存继续。
  失败安装另留 attempt 回执；原 started/failed 不删除。native 阶段一旦开始却未完成，
  不从头重跑、不重新提交未确认的 step、不退款、不声称存在 bitwise 中途恢复。

本轮五个条件执行点（均是 native 验收，PPO=0）
阶段                   单 owner 公共步数    owners   integrations 总上限
n1-pilot                3                   1        45,120
n2-pilot                3                   2        90,240
n1-rollout              24                  1       360,960
n2-rollout              24                  2       721,920
n1-full-episode         500                 1     7,520,000
合计硬上限：8,738,240 integrations；35,003,654 GPU forwards；4,816 CPU forwards；
最多 7 个新环境 owner；每个进程 native nworld=1。
这是上限，不是预计消耗；通过或预算用尽就停，不需要耗满预算。
首步作为 warmup，短测计两步，24 步样本计后 23 步；记录 IPC、同步、审计和证据 I/O。
每个已完成 macro 在 owner 关闭后独立重算原验收判据，并检查 accepted replay。
前一项失败，后一项不启动。长样本/完整回合按已测最大步时和证据体积进行时间/磁盘准入。
空间估算包含原始证据及 ZIP、ASCII、Git 回传副本；运行中保留 8 GiB 数据盘回传空间，
系统 /tmp 留 2 GiB。无空间时停并保留证据，不删旧资料、不截断账本。
估算不是完成保证，不能因为有执行计划就把长回合列为通过。

完整回合语义
保持真实终止条件。只有从真实 reset 连续达到第 500 步（10 秒），此前没有 done，
末步为 truncated 且不是 terminated，且 terminal/reset 桥正确，才写 full_10s_episode_proven=true。
若零动作早于 500 步自然终止，写 FULL_EPISODE_NOT_COMPLETED，不强改 flags、不取消终止条件。
连续 24 步属于吞吐样本，不是生产训练速度保证。N>2、nworld>1/4096、B=4 均未放行。

统一预算与时间
用户已确认保留旧 5090、上限 24 小时，无需再问 GPU 型号和这次小时数。
采用保守固定截止：2026-10-17 03:53:14 UTC（北京时间 2026-10-17 11:53:14），
即本次 24h 确认时刻 +24h。native 最迟在 UTC 03:23:14 停，预留半小时回传。
安装、等待、断线、重连共同占用这一个窗口，不在每个进程或 --resume 时重新给 24h。
确认之前实际开机了多久未知，平台实际计费和作业时钟分开；截止不等于平台自动关机。
当时截图显示 ¥2.88/h 和余额不足 24h；24h 单 GPU 算术费用 ¥69.12，未确认已充值。
脚本不充值、不购买/新建实例、不自动开关机。过期后不自动扩大预算。

本机完整操作（解压目录假设为 /home/lx/下载/microduck-stage12-cloud-execution-v1）
1) 下载 ZIP 到 /home/lx/下载；外部交接 TXT 给出该 ZIP 的最终 SHA256。
   unzip -n /home/lx/下载/microduck-stage12-cloud-execution-v1-20261016.zip -d /home/lx/下载
   python3 /home/lx/下载/microduck-stage12-cloud-execution-v1/local.py --prepare

2) 手动推送输入包，然后直接启动，无需把输入 push 输出先贴回来等下一轮。
   git -C /home/lx/microduck-double-balance/artifacts/double-balance-stage12/cloud-execution-v1/transfer push origin HEAD
   python3 /home/lx/下载/microduck-stage12-cloud-execution-v1/local.py --launch

3) 只读查看 integrations 次数、阶段、elapsed_s、公共步和剩余预算。
   python3 /home/lx/下载/microduck-stage12-cloud-execution-v1/local.py --status
   python3 /home/lx/下载/microduck-stage12-cloud-execution-v1/local.py --commands
   当前 SSH：ssh -p 46171 root@connect.bjb2.seetacloud.com
   云任务已经在后台。查状态和退出监控不重新运行训练。

4) 出现 return-receipt.json 后，复制 --commands 打印的完整
   Stage12ReturnUploadCommand（ssh ... 'git -C <本次 cloud work>/return-transfer push origin HEAD'），
   在本机终端手动执行。它使用云机既有 Git SSH 身份推送独立 return 分支。
   本助手和程序都不会自动执行 push。

5) 推送成功后，在本机取回并验证全部新证据。
   python3 /home/lx/下载/microduck-stage12-cloud-execution-v1/local.py --receive
   将 return push 输出和本命令输出的固定 commit、状态贴回对话；无需上传 ZIP。
   input 分支、return 分支都不能创建 PR/合并到 double-balance。

断点与失败恢复（只执行对应的一条）
- 输入 push 失败：只重试步骤 2 的 git push；不重跑本机已完成批次。
- 下载 Git 输入失败或 SSH 断线：再次 --launch；若后台作业已开始，仅报告已有 launch。
- 安装失败 / SETUP_INTERRUPTED_NATIVE0：看 --status 中 last_setup_interruption；
  修复网络或磁盘外部问题后执行以下命令，复用已完成安装步骤和本次缓存：
  python3 /home/lx/下载/microduck-stage12-cloud-execution-v1/local.py --resume
- native started 后失败：保留进度和 WAL，不重跑该点。程序汇总精确值或上下界并打包。
  若整个监督进程意外消失，确认没有仍运行的 worker 后 --resume 只会继续未开始步骤，
  遇到未完成 native marker 会停止归档，不将失败点重新开跑。
- 决定结束本轮安装故障、或异常后需要只汇总回传：
  python3 /home/lx/下载/microduck-stage12-cloud-execution-v1/local.py --finalize
  它拒绝与仍在运行的本次 worker 并发，不补做物理步。
- 打包失败：
  python3 /home/lx/下载/microduck-stage12-cloud-execution-v1/local.py --package-only
- return push 失败：只重试已打印的 Stage12ReturnUploadCommand。
- 取回中断：只重试 --receive；固定第一次 fetch 的 commit，保留已验证文件和旧 partial。
- GitHub host key 不匹配：停止，不关闭校验、不盲删 known_hosts。
- Private key permission/publickey 错误：复用已有 key 与权限，不新建密钥/PAT；禁止 force push。
- 实例已关机或 SSH 地址变了：用户先在平台确认实例；本包不替用户开机/重建实例。
- 日志里的检查失败不等于训练失败：本包没有正式训练；以具体 source/setup/native 状态区分。

证据、完整性和旧结果
- 每个新 ZIP <49 MiB；长 WAL 按字节分片且记录 offset/完整原文件哈希，单 ASCII 分片 <=512 KiB。
- 大型证据索引同样压缩分片；本机按原文件分组重建，避免逐文件重复遍历完整索引。
- --receive 从独立 Git SSH 缓存固定完整提交，校验 MANIFEST、归档、所有分片和重组的原始文件；
  缺一项都不能写 FULL_NEW_CLOUD_EVIDENCE_RETURN_VERIFIED。
- 新 native 目录全部返回：macro 边界、microstep qpos/qvel/time、WAL、接触/排序审计样本、
  资源关闭回执、运行配置、安装命令输出和阶段报告。
- 沿用 G11 的证据范围：并未新增长期保存每个内部 microstep 的完整 Data，不能冒称已保存。
- 新安装的 wheels、解释器、编译缓存和实时监督控制台不打包；命令日志/持久回执保留。
  可复现源码/执行包已在 input commit，不重复塞进 return。
- Stage09 历史完整 head-trace/trace 仍有只在云端的资料；本次不声称已回传全部旧云文件。
- 本机 Stage11 11/11 CLOSED_SCOPED、CPU13、N1、v6 离线审查和 v7 N2 60/60 全部复用，
  不重跑 Stage06/Stage11、local v5/v6/v7。
  v7 固定提交 d506be014253a2bb0439ec03b3282497a33f2e2b，新增11136，累计704664/704665。
  原 v5 失败仍归档为失败；256 tier 通过不覆盖旧失败，不扩大数值阈值。
  local v7 仅每 owner 0.06s 短试验，不能当作完整10s或云端持续吞吐证明。

checkpoint 与正式训练放行
Stage08 checkpoint update_004000.pt 继续只读保留、不加载、不直接 warm-start：
SHA256 86d55c3703c18fcf4817db49c6e4dcf839b3f5195d68ae2c559db7e222bded97；
schema SHA256 e740e6ca4ad74c53a0eaaa9c95dcae9b69fd5b0edf122593743cd015dcb388a7。
未来策略：全新 LSTM actor(61->256，1层)+MLP512/256/128->14，
critic85->512/256/128->1；全新 normalizer、Adam、std，24步 rollout。
N1 recurrent minibatches=1；N2=1或2，不沿用 B=4。
只有 truncated 且非 terminated 时用 terminal V bootstrap 一次。
未来 checkpoint 只在完整 update + durable WAL 之后独占写 partial、fsync、哈希和原子发布；
只从最后 committed update 恢复权重/优化器/学习率及消耗账本，fresh env + zero RNN boundary。
中断更新不退还次数，不承诺 mid-rollout bitwise 恢复。
当前 PPO=0、cloud_training=0；不构造策略/优化器、不做 PPO、不载入旧 checkpoint。
原 task registry 的 PreparedRunner 阻断保留，不能使用 stock train 命令绕开 dedicated factory。
要开始正式训练，还要本次真实云端数值/完整回合/吞吐验收过关，然后单独明确有限 PPO
update 数与 checkpoint 保存/恢复验收预算，再给正式训练总量。24h 是准备窗口，不是
已经批准 24h PPO。全部 native 点成功时也只会写 PPO_BUDGET_PENDING，training_ready=false。
若时间/磁盘不足、零动作早终止或验收失败，先按报告处理实际瓶颈，不宣称再跑固定几次就能训练。

仓库与长期规范（继续保留）
LocalRepository=/home/lx/microduck-double-balance/workspace
DownloadDirectory=/home/lx/下载
LocalArtifacts=/home/lx/microduck-double-balance/artifacts
CodeBranch=double-balance
LocalSourceHEAD=73ee01930a031d76a9d57419afdfd4464a547ab3
OriginSSH=git@github.com:liuxue-lab/microduck-double-balance.git
Stage11CompleteCommit=6ba0dc2de381321576f32310e1995a8e2e11e566
Stage11TagObject=bd1b219be55c813811d4f7cfa25d245b37166688
stage-11-complete 标签已正常推送，不重复推送/重做验证。
protocols/AGENTS.md
protocols/microduck-local-ssh-push-protocol.md
protocols/microduck-cloud-gpu-and-transfer-protocol.md
以上三份完整原文包含在本包；正式仓库规范位于 docs/handoffs/。

关机提醒状态
当前必要云端任务：本包独立环境与条件 native 验收。实际关机确认仍为 UNCONFIRMED。
完成必要计算和新证据打包、手动 Git push、--receive 全部 SHA 校验后立即在平台手动关机，
保留实例和所需数据；不用等助手写完后续文档。保留/删除实例、释放存储是不同动作。
等待代码或不继续执行时，可手动关机减少空转；脚本退出、预算到期、SSH断线都不是关机证据。
不执行旧关机守护，也不冒称已建立后台空闲通知。

官方技术依据（2026-10-16 核对）
https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints
https://docs.astral.sh/uv/concepts/projects/sync/
https://docs.astral.sh/uv/concepts/cache/
https://pypi.org/project/uv/
