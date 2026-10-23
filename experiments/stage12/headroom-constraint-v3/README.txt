Microduck Stage12 关节约束 v3 — 有限续接运行包

用户要求：修改好继续训练；5090 关闭需等两小时才能打开。
当前准备期间若实例已在线，请保持开机。助手没有连接或启停云实例。

修复内容
1. 复用旧 headroom_000505.pt（累计5505次）的策略、价值网络、优化器与归一化器。
2. qpos 余量惩罚从20°开始；在25°处位置成本加权为-6，并加重越界成本。
3. 增加0.08s速度预测成本：在接近边界前对快速外向运动产生压力。
4. 原双球主奖励乘以余量系数，在20°以内为1，至25°衰减至0。
5. 训练统计直接读取每个2ms积分后的qpos；终止和超时重置前的越界不会消失。
6. 原14关节范围、BAM、61D actor/85D critic、物理/控制频率、原验收规则和归一化约定保持。
7. 学习率从1e-5调为固定3e-5。新损失设计不保证训练成功，以云端选择/留出结果判断。

预算
独立的最多1505次新增PPO（64环境5次smoke + 4096环境三段各500次）；从新账本建立起最多7200s计算、90s收尾。
每段末最多2个60s选模回合；首个合格候选再做4个新留出种子，合格或留出失败即结束。总评估最多10回合。
失败尝试计入预算；中断不刷新时间/次数，不重放丢失的PPO，不借用旧505或冻结Stage13/14额度。
原505次账本及旧模型保持原样。只训练无外力静态双球，不新增轨迹或抗扰任务。
此数值为本次实现采用并已向用户说明的有限上限；用户执行明确的 --train 命令启动这一轮。

运行
下载 release/microduck-stage12-headroom-v3-20261023.py 到 /home/lx/下载，无需解压完整包。
在本机 Ubuntu 终端执行以下一组命令；使用当前实例地址（最后提供端口53885）：

python3 "/home/lx/下载/microduck-stage12-headroom-v3-20261023.py" --install-code &&
git -C /home/lx/microduck-double-balance/workspace push origin double-balance &&
python3 "/home/lx/下载/microduck-stage12-headroom-v3-20261023.py" --train --ssh "ssh -p 53885 root@connect.bjb2.seetacloud.com"

本机只导入、推送、发送约百KB源码并监控；训练在云5090原虚拟环境内运行。共享SSH会话有效时复用，不保存密码。
原v2代码已导入后的父提交要求：88bd1890d33243eb8d94cfde42882771aefd20b3。
安装保留不冲突的本机改动；冲突/分歧保留并停止。不stash/reset/clean/force push；不新打阶段完成标签。
新云目录：/root/autodl-tmp/microduck-double-balance/artifacts/double-balance-stage12/native-headroom-constraint-v3
本机控制目录：/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-headroom-constraint-v3-client

观察与停止
同一入口 --status 或 --watch：只读；--stop：停止本轮作业而不关实例。
监控显示本段物理采样头部越界百分比、角度峰值、结束的完整60s合规回合；这些训练统计不代替候选验收。
Ctrl+C 只退出本机监控，不停止已分离的云任务。重复 --train 返回原任务，不另开预算。
--resume 只处理可核对的中断，不越过错误、未提交更新或截止时间；--status/--collect 不启动训练。

回传
训练结束先保存轻量状态报告，再生成可下载的ZIP/JSON，然后走既有Git SSH独立资料分支正常回传。
如果Git下载再次很慢，已有云打包不会丢失，可用 --pack 仅打印包路径，再在平台下载相应分卷和索引。
已有云push回执时 --download 仅下载；--return 仅回传，不启动PPO。
手动下载后支持完全离线导入（一个分卷时）：
python3 "/home/lx/下载/microduck-stage12-headroom-v3-20261023.py" --import-return --index "/home/lx/下载/return-index.json" --archive "/home/lx/下载/part-0001.zip"
如果有多个分卷，把所有原名分卷和索引放在同一目录，省略 --archive；每个包都按索引校验。
(1)重名后缀可直接通过 --index / --archive 指定，不修改期望SHA256。不需要重启训练来修复传输。
若用户本地已有包，也可直接把新索引与分卷上传到对话分析。不要将资料分支合并到double-balance。

验证范围
本地只做标准库/NumPy代数、接口替身、真实本地Git/子进程通道、预算与回传验证。
助手环境未安装Torch/mjlab/MuJoCo/CUDA；没有运行真实物理回合或PPO。
新配置5次GPU smoke包含在1505预算里，成功后才进入正式段；训练效果尚待实际运行。
详细检查结果随 CHECKS.json 和检查日志交付，不把CPU通过写成GPU或模型验收通过。

长期协议
保留 docs/handoffs/microduck-local-ssh-push-protocol.md 与 microduck-cloud-gpu-and-transfer-protocol.md。
本次电源补充见 microduck-power-update-20261023.txt；电源状态始终UNCONFIRMED，除非用户/平台确认。
父回传为192文件、505/505更新、4回合保持双球但边界0/4；20个中间模型和3个TensorBoard文件仍未回传。
本包只带代码、测试和必要父账本，不包含模型，模型留在原云产物路径。
