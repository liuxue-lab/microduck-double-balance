Microduck Stage12：保留4500结果，补齐6000独立批次评估
2026-10-20；云端RTX5090；SSH端口52763。

本次修复依据
用户回传 microduck-stage12-native-eval-20261020T085438.316463Z-10548.json，
SHA256=d2215ca3173c1664b32dd5d08351d73cbacea5be3d9bb8bd93f4a18797186557。
原评估4500模型的两组、256个首次回合已完成；6000在第3组、采样0步时停止。
错误来自评估器额外增加的初始状态/观测/声明DR字段逐字节一致性检查。
旧报告只留整体指纹，不能确定哪个字段不同，不能归因于浮点误差或模型损坏。
旧报告未带回4500逐组汇总；目前不能据此报告其成功率。
原训练文件未变，原评估进程已退出；本轮6000次PPO训练早已完成。

只运行这个新入口
下载 microduck-stage12-evaluate-recovery-v1-20261020.py 到 /home/lx/下载。
python3 "/home/lx/下载/microduck-stage12-evaluate-recovery-v1-20261020.py" --resume

它核验原训练和原评估的身份、源码、模型、已完成回合、原故障位置及进程状态。
读取4500已完成的256个首次回合，核对逐回合行与汇总一致，不重跑。
只为6000执行2种子×128环境的256个首次回合，每回合最多500步/10秒。
新增PPO=0；不训练、不改奖励或成功阈值，不安装环境、不下载大模型。
云端代码传输约18.2 KiB；复用原项目、Python环境和Warp缓存。

时间与断线
继承原评估 started_utc=1792486397.1664722，deadline_utc=1792488197.1664722。
原截止时间为北京时间2026-10-20 17:23:17；不是启动修复脚本后再给30分钟。
已过截止时间则返回 ORIGINAL_EVALUATION_DEADLINE_EXPIRED，带回4500汇总，新增仿真0步。
到期停止本次恢复worker，最多90秒清理；不自动延长，不自动重新开始。
后台任务不依赖本机SSH持续在线。断线后只读接回：
python3 "/home/lx/下载/microduck-stage12-evaluate-recovery-v1-20261020.py" --watch
只收一次报告可将--watch改为--collect。重复--resume只接回已启动的恢复任务。
请回传 NATIVE_EVALUATION_REPORT_SAVED= 后的小JSON；不用上传模型或审查ZIP。
本机报告目录：/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-evaluation-recovery-v1-reports

比较含义与未解决的问题
恢复后明确使用相同任务/reset/DR/噪声配置的独立批次描述性比较。
相同种子标签不再被当作初始物理状态、观测和DR逐字节相同的证据。
报告 paired_comparison_allowed=false，保留原不一致事实，差异原因仍为UNKNOWN。
不计算逐状态配对胜负，不声称因果优劣；排序只供后续选模型复核。
6000会增加每个初始字段的SHA、形状、dtype、有限性记录，并记录与旧整体指纹是否一致。
4500旧证据没有逐字段指纹，不能通过新增记录反推出此前具体差异。
成功仍要求连续250控制步/5秒满足全部原条件，且首次回合结束前无物理失败；超时允许。
策略用确定性动作、冻结权重和归一化；保留原观测噪声与DR。
这是独立于训练rollout的模型选择评估，不是最终留出验收或视频/实机验收。

证据与文件保留
原训练：/root/autodl-tmp/microduck-double-balance/artifacts/double-balance-stage12/native-training-v2
原评估：/root/autodl-tmp/microduck-double-balance/artifacts/double-balance-stage12/native-evaluation-v1
恢复评估：/root/autodl-tmp/microduck-double-balance/artifacts/double-balance-stage12/native-evaluation-recovery-v1
不覆盖原评估源码、started、progress、exit、初始指纹和完成组JSON。
原训练文件、两模型和旧评估关键证据运行前后校验；四个相关目录共享launch锁。
原包和源码身份不符、旧任务存活、证据损坏或新失败，均保留现场并停止；不自动重试。

核对范围
17项标准库离线恢复检查通过：完成回合复用、错误行拒绝、截止时间、重复接回、锁、
只评估6000、无配对结论、确定性/零PPO、嵌入包和旧控制器不变等。
Observer实现与原评估相同；原评估的28项离线检查记录保留。
这些检查不是GPU仿真通过。原4500两组已在用户5090实际运行；本恢复版本尚待用户运行。
构建回执记录准确字节数、SHA256和包身份。完整源码、离线检查和原始错误报告在审查ZIP。

后续约定
本次128环境仅用于评估。下一轮正式训练恢复4096并行环境，不能再默认为2048。
性能同步开销优化尚未部署、未测量；本入口不启动新训练，不重跑已通过短测。
完成评估并收回必要结果后，如无其他云任务，可手动关闭5090并保留存储；脚本不操作电源。
