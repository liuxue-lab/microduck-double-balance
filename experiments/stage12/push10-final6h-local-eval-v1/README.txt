Stage12：最后一轮训练后的本机选模与独立验证（2026-10-24）

这一入口只读取已回传的三个训练检查点，在本机 RTX 5060 上推理和 MuJoCo 仿真。
新增 PPO=0，不连接云主机，不重启训练，不修改旧六小时预算和训练记录。
5090 不再有本轮必要工作；如仍开机，请手动关机并保留存储。实际电源状态尚未确认。

先下载 microduck-stage12-push10-final6h-local-eval-v1-20261024.py 到 /home/lx/下载。
在本机终端执行：

python3 "/home/lx/下载/microduck-stage12-push10-final6h-local-eval-v1-20261024.py" --install-code &&
git -C "/home/lx/microduck-double-balance/workspace" push origin double-balance &&
python3 "/home/lx/下载/microduck-stage12-push10-final6h-local-eval-v1-20261024.py" --run

--install-code 仅导入内嵌增量 Git bundle；只新增本阶段目录。
它检查固定 SSH origin 和 double-balance 分支，保留不冲突的已暂存、未暂存和未跟踪改动。
如果新增路径冲突或 HEAD 已有分歧，会停止并保留。不会 stash、clean、reset 或 force push。
推送仍由用户执行。正常成功的 git push 输出足够，不必重复远端核验。
--run 一次启动本机后台验证，并每 2 秒刷新进度；重复 --run 只连接原任务。
本机必须保持开机、唤醒和供电。Ctrl+C 退出监控不停止后台任务。
关闭终端不等于关闭电脑；关机、睡眠或系统终止用户会话可能中断本机任务。

重新查看（每 2 秒刷新；结束后显示报告路径并退出）：
python3 "/home/lx/下载/microduck-stage12-push10-final6h-local-eval-v1-20261024.py" --watch

单次状态 / 启动前只检查 / 明确停止 / 结束后导出审查包：
python3 "/home/lx/下载/microduck-stage12-push10-final6h-local-eval-v1-20261024.py" --status
python3 "/home/lx/下载/microduck-stage12-push10-final6h-local-eval-v1-20261024.py" --check
python3 "/home/lx/下载/microduck-stage12-push10-final6h-local-eval-v1-20261024.py" --stop
python3 "/home/lx/下载/microduck-stage12-push10-final6h-local-eval-v1-20261024.py" --pack

--check 需先 --install-code；不会初始化 GPU、执行仿真或安装/升级依赖。
--pack 只在本机打包验证结果、逐回合记录和日志，不重新下载模型，也不自动上传。
最终输出 LOCAL_FINAL6H_EVAL_REPORT_SAVED= 后的 report.json 可直接发回对话。
需要完整诊断时，--pack 输出 LOCAL_FINAL6H_EVAL_REVIEW= 后的 review.zip。

固定验证范围
1. candidate-01（新增1553/累计7563更新）、candidate-02（3428/9438）、
   candidate-final（5304/11314），各用相同三回合选模：
   无外力：环境种子2026102407；
   抗扰：环境种子2026102407，外力种子2026102500（原失败回归）和2026102700。
2. 只有三回合都合格的模型才有资格进入独立验证；按原排序选定一个并锁定。
   18回合：无外力种子2026102406、2026102407；
   16条未用于选模的外力序列2026102800至2026102815，环境种子2026102407。
3. 最多27回合、75分钟计算，从本机启动时计时；退出清理最多另给90秒。
   每回合最多15分钟墙钟初始化/仿真，且受75分钟总上限约束。
   不自动延长、续跑、重试、换种子或重新训练；中断、超时均保留记录。
4. 每个完整回合60秒，3000控制步/30000物理步。真实水平外力0.1*m*g，
   m为当前机器人本体质量（不含两球）；作用于trunk_base惯性质心，无额外力矩。
   从2秒开始，每2秒施力0.1秒，到58秒，共29次；每次固定50个2毫秒物理子步。
   每次方向独立按水平面均匀角度生成，序列预先固定，不根据稳定状态推迟。
5. 61D actor、85D critic、14个BAM舵机、原几何接触和动作路径均保留；
   不加动作滤波，不加扶持，不学习或更新归一化器；每500Hz物理步检查模型边界。
   越界、跌落、丢球判失败，不放宽容差；非有限值/运行异常停止任务。
   选模阶段有限物理失败仍记录后续预定候选；独立验证首次失败即停止。
6. 全部通过仅证明这个锁定模型通过本次固定集合；不宣称所有随机序列必过、
   原64回合套件通过、严格接触专项通过或实机验收完成。

证据与路径
本机仓库：/home/lx/microduck-double-balance/workspace
下载目录：/home/lx/下载
代码分支：double-balance
代码origin：git@github.com:liuxue-lab/microduck-double-balance.git
增量基线提交：8270763ce09f842e3ed6a6d2eab19ed2545670ec
新增代码：experiments/stage12/push10-final6h-local-eval-v1/
新提交/包身份/构建验证详情见审查包 build-receipt.json、release-validation.json。
不创建阶段完成标签：Stage12尚待本次实际验收和后续审查。

已回传分支：stage12-push10-final6h-return-20261023t181130z-1b79efbd-s002
已回传提交：0cbee30e5a83904371076ca09a8cfb4afd1fa3f4
本机回传根目录：
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-push10-final6h-v1-client/returned/0cbee30e5a83904371076ca09a8cfb4afd1fa3f4
其中 files/ 是模型和原始账本。旧记录只读，复核原365个冻结源码和所用回传文件SHA256。
云端原始索引SHA256：1e006b5e706d5c5767ff1cc6033013ef6c3fbfec3c8205c8862f42405c08f64b
本机既有排版索引SHA256：4519472a91b4b583b3602975e6184d96ba2a954e73beaaad02aa06254dbe8288
两者格式不同、数据相同；原回传程序将解析后的JSON用indent=2另存，因此分别锁定。
三个候选各10041821字节，位于 files/run/checkpoints/：
final6h_001553.pt：2ba6f78bc907236bb546d6929b0dbfd8841379dd4c34231133dd488f268f8e50
final6h_003428.pt：89d3afa6ec96576ee0c5a2d95304f6c42317501b53af0122d87d40f55ed0bcdf
final6h_005304.pt：cf7ce9297672943687019afe071d94918d3eb2620c18a1e8fe689512f8c08111

本机新增输出目录：
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-push10-final6h-local-eval-v1
报告：该目录/report.json
监督日志：该目录/logs/supervisor.log
逐回合证据：该目录/evaluations/
独立本机账本：该目录/budget.json

当前事实与剩余任务
用户报告记录：训练完成5304次更新，消耗5305次尝试，累计策略更新11314；
末尾未完成的第5305次rollout仍计入旧消耗，不续跑、不退款、不计入已保存模型。
训练worker已保存完整快照并记录TRAINING_PHASE_COMPLETE/train_window_complete；
监督器随后报Training phase did not complete: formal。原代码没有保存实际worker退出码，
因此只能确认自动选模没有执行，不能声称已证实某个SIGTERM退出码。
源代码显示训练退出清理与截止SIGTERM存在竞态。本入口绕过已结束训练调度，
让保存结果的worker获得有上限的清理时间，保留退出处理器并独立记录真实退出码。
没有修改、重跑或伪造旧训练结果。实际训练的独立验收回合数仍为0，等待本次运行。

必要资料已于用户本机2026-10-24 09:38:11（北京时间）验证：400文件、5包，
合计45987487字节压缩数据。已提醒手动关闭5090；实际关机状态UNCONFIRMED。
历史SSH为ssh -p 32185 root@connect.bjb2.seetacloud.com，本入口不使用、不探测。
未声称整个云存储均已备份。无本轮剩余云端计算/传输需求；不要为本机验证开启5090。
Stage13/14已取消；本次结果不能代替视频目视审查、严格接触专项或实机部署验收。

常见错误
- 找不到已验证回传：先检查上述固定目录，保留错误输出；不要重新训练或改期望哈希。
- 本机冻结源码不符：不覆盖本机修改；发回错误中的路径，以便核对具体差异。
- 另一GPU计算进程活动：用nvidia-smi只读确认已有任务，待其完成后运行；脚本不会杀它。
- 已有孤立预算、部分输出或后台异常：保留日志，重复--run不会重置预算或免费重试。
- 进度初始化阶段短时间为0：读取日志；不要并发开启第二份。--watch每2秒读本机状态。
- 时间到/运行错误：状态不是验收失败的充分证据，也不是验收通过；发回report.json。
- 新目录与本机文件冲突或Git分歧：保留双方，不force、不clean，发回install-report.json。
- 推送认证失败：使用既有Git SSH身份，不切HTTPS，不要求PAT。推送失败时&&不会启动验证。
- 文件名下载后带(1)：使用实际路径或明确改回上述名称，避免启动旧版本。

构建侧只运行CPU契约、边界、真实CPU子进程信号和Git安装测试。
隔离环境的PID映射与/proc不同，子进程生命周期测试使用进程身份读取夹具；
进程组、信号、wait和退出码使用真实CPU子进程，并另测PID复用/组不符拒绝发信号。
没有在构建环境执行MuJoCo/CUDA或真实模型验收；是否通过只能看本机实际报告。

必须保留的长期交接规范：
docs/handoffs/microduck-local-ssh-push-protocol.md
docs/handoffs/microduck-cloud-gpu-and-transfer-protocol.md
