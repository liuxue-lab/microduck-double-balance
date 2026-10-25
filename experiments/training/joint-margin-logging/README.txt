Microduck Stage12 v4 运行修复（同一v3任务）

报告证据：2026-10-23 22:27:58北京时间启动，完成1次PPO后，日志统计zero_()触发PyTorch inference tensor异常，22:28:14保留停止。
这1次没有保存模型；云端已回传49个文件并在用户本机核对，原49文件SHA256在本包复核。
这是统计代码的兼容性错误，不是训练目标验收失败。已在真实PyTorch 2.9.1+cpu复现旧代码，并验证新代码。

修复
1. 统计累积和清零使用新张量，支持采样inference_mode到日志普通模式的切换；原物理步统计语义不变。
2. 每个小规模训练更新完成即保存，然后才调用日志；普通日志异常也尽力保存已经提交且有限的优化器状态。
3. 云端建真实环境前，在原安装PyTorch上做CUDA统计张量模式检查；它不做物理积分或PPO。
4. 原v3奖励、物理模型、原25°关节边界、14关节、61/85维观测、BAM、归一化与选模/留出验收均不改。

本次不新开预算
原总上限1505次，已完成但未保存的1次继续记为消耗。恢复父模型headroom_000505.pt（累计5505策略更新）。
剩余最多1504次：先5次64环境smoke，再4096环境500+500+499正式更新。
账本smoke目标6（旧1 + 新5），候选账本位置506、1006、1505；实际策略更新计数为5505+账本完成次数-1。
原started.json和budget.json在安装修复时逐字节保持，原49个文件全部先备份；之后训练按原账本追加。
原截止：北京时间2026-10-24 00:27:58（UTC 2026-10-23 16:27:58）。不因修复延长，不退款，不自动开新任务。
保留原最大10个60s评估、先2选模再4留出，通过提前结束；留出失败不继续调参。
超过原截止时只安装修复/采集，不开始训练；不得删除账本绕过。

运行
下载单文件microduck-stage12-headroom-v4-20261023.py到/home/lx/下载；在本机终端执行：
python3 "/home/lx/下载/microduck-stage12-headroom-v4-20261023.py" --install-code &&
git -C /home/lx/microduck-double-balance/workspace push origin double-balance &&
python3 "/home/lx/下载/microduck-stage12-headroom-v4-20261023.py" --repair --ssh "ssh -p 53885 root@connect.bjb2.seetacloud.com"

本机base必须是v3提交8e1fd13c262e13e76fa9e40efaf00f8e92059bff；增量只新增experiments/stage12/headroom-runtime-repair-v4。
不冲突本机修改保留，冲突/提交分歧停止；不stash/reset/force、不改变Git SSH远端、不发模型到代码分支。
云目录仍为/root/autodl-tmp/microduck-double-balance/artifacts/double-balance-stage12/native-headroom-constraint-v3。
本机仍为/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-headroom-constraint-v3-client。
--repair首次核对已知错误、无活进程、账本1/1且无模型，然后备份/更换代码并续接session-2。重复执行不会另开训练。
--status/--watch只查看原任务，--stop停作业但不关机；Ctrl+C只退出本机监控。
--resume不越过新的运行错误。此次已知错误必须用--repair，不能直接用旧v3 --train/--resume。

回传
沿用原Git SSH分卷回传；先保存轻量报告并打包。--pack只打包并打印平台下载路径；--return只回传；有云push回执用--download。
平台下载文件可完全离线导入：--import-return --index <实际索引路径> --archive <实际单分卷路径>；支持(1)后缀。
多分卷原名同目录省略--archive。新索引与分卷也可直接上传对话分析。
原session-1 artifact commit=4895ca7d3bfbd55991a159381482a2112944e7e7。
原父headroom505模型SHA256=a3b2dda26bf99ee6676aef8a750e045b1d7243ef04d0d2272634395d0b9e5565。

电源与范围
5090已在线时，当前紧接修复/续训期间保持开机；用户表示关闭后需等两小时才能重新打开。
助手没有连接/启动/关闭云实例；电源状态仍UNCONFIRMED。
完成计算与回传校验后，如果没有紧接云工作再提醒手动关机并保留存储。
Stage13/14额度冻结；只续训脚踩篮球头顶球，无轨迹或抗扰训练。
保留docs/handoffs/microduck-local-ssh-push-protocol.md、microduck-cloud-gpu-and-transfer-protocol.md、microduck-power-update-20261023.txt。

检查范围
真实Torch CPU张量检查、原故障复现、预算及恢复流程、真实本地Git和嵌入bootstrap通道、回传与保留改动检查。
本机没有mjlab/MuJoCo/CUDA环境；没有声称已做真实GPU烟雾/物理验证或新PPO。完整结果见CHECKS.json。
