Microduck Stage12 评估恢复v2：统一Python与汇总兼容检查
2026-10-20；用户17:10要求一次处理完整恢复流程。

当前已知事实
正式训练6000/6000更新完成；4500两组256个首次回合评估完成；6000尚未采样。
原评估在初始整体指纹精确一致检查失败，具体差异字段未知。
恢复v1在preflight的 g['summary']==summarize(rows) 处失败。
该失败发生在写入恢复started和启动worker之前，故这次未新增仿真或PPO。
用户提供的v1报告身份4f77019d524e7bf9803234ced216b4f7a09b23bb2bc9dbfd69f052478cad69b5。
报告文件名microduck-stage12-native-eval-recovery-20261020T090822.375013Z-10627.json；
用户终端给出的SHA256=436b9c8eca69b580e1c63a1f8cc9a1eb476a317b92ee75fa0dd94c8cfc54cfc2。
尚未取得旧4500逐组文件，因此不能断言已证实哪项浮点值不同或哪版Python导致。

修复内容
1. 包安装由系统Python处理，汇总preflight、后台监督、实际评估和只读监看统一用
   /root/autodl-tmp/microduck-double-balance/workspace/.venv/bin/python。
   记录bootstrap和实际运行Python版本，不再跨解释器盲目要求整个汇总字典精确相等。
2. 原始回合的成功/失败布尔值、数量、步数和5秒定义保持精确检查。
   派生汇总浮点值采用abs_tol=1e-12、rel_tol=2e-12；每处差异记录路径、旧值、新值、差值和容差。
   这只用于检查统计汇总是否一致，不用于放宽任何物理成功阈值，也不改写旧数据。
   NaN/Inf、类型/结构变化、次数变化和超出舍入范围的差异仍拒绝启动。
3. 一次检查两组已完成数据；失败报告包含两组诊断和原汇总（无效时明确标记未核验）。
4. 远端程序错误不再全部包装成“SSH返回不完整”；终端直接显示具体错误，报告保留结构化明细。
5. 新目录native-evaluation-recovery-v2，不覆盖v1；持有训练v1/v2、原评估、恢复v1/v2五处锁。
   如果恢复v1已有started，拒绝另开任务；v2重复--resume只接回，不重新采样。

唯一运行命令（下载新版到/home/lx/下载后执行）
python3 "/home/lx/下载/microduck-stage12-evaluate-recovery-v2-20261020.py" --resume

SSH仍为 ssh -p 52763 root@connect.bjb2.seetacloud.com，复用共享连接。
只复用4500的256个已完成回合，并为6000补2种子×128环境，共256个首次回合，每回合最多10秒。
新增PPO=0；策略和归一化冻结；原物理、观测噪声、DR和成功阈值不变。
保持独立批次描述性比较，不声称严格初始状态配对、不计算逐状态配对胜负。
这不是最终留出验收、视频验收或实机验收。

原截止时间保持北京时间2026-10-20 17:23:17（1792488197.1664722）。
若已过期，只收回现有结果，不启动6000仿真，不自动增加30分钟；最多90秒清理。
断线接回：同一新版文件 --watch。只收一次：同一新版文件 --collect。
只回传 NATIVE_EVALUATION_REPORT_SAVED= 后的小JSON，不用传模型或审查ZIP。
报告目录：/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-evaluation-recovery-v2-reports

验证范围
37项离线检查通过，包括真实preflight/launch/status代码搭配隔离文件与进程替身，
浮点舍入差异记录、显著差异拒绝、计数精确比较、两组错误同时收集、原deadline、
重复启动、旧恢复started保护、共享锁、原文件不变、RPC报错、终端显示与相同解释器路径。
Observer实现和物理阈值与原评估完全一致；不宣称本助手运行了Torch、CUDA或物理仿真。
云端原4500两组已运行；新版6000仍待用户运行确认。

参考依据（机制说明，不代表已经证明本次具体原因）
Python官方文档：Python 3.12变更了sum对浮点数的求和算法。
https://docs.python.org/3.12/library/functions.html#sum
因此同一段sum代码在不同解释器下可能出现末位差异；本修复统一解释器，并保留数值差异审计。

后续正式训练恢复4096环境的约定不变；本次128只是评估批次。
性能候选未部署、未测量加速比。无需重跑已完成训练或smoke。
收回必要结果后若无其他云任务，请手动关闭5090并保留存储；脚本不操作电源。
