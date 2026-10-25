Microduck Stage12 性能优化候选与只读统计
日期：2026-10-20，北京时间；连接：ssh -p 52763 root@connect.bjb2.seetacloud.com

当前训练状态（来自用户截图，而非本工具重新查询）
2026-10-20 13:08:50 左右，正式训练802/6000，13.4%；前置5更新短测已通过。
当前脚下几何稳定90.3%，双平衡几何稳定69.2%。这些不是独立任务成功率。
13:10:02，RTX5090利用率59%，显存12330/32607 MiB，45°C，168W。
PID1562的Python占12320 MiB。SSH断开不等于后台训练停止。
当前运行包身份：dd6d1e1e697bcaa348dbdd92dd8c7b30ea31ae9c32599338106859ae7657523a。
当前控制器SHA256：5b57ef3a761757b9d2bd2c39e7faaa6713b1f4e29b54500f5e1ded13f33438cb。
本次没有修改当前训练包、原控制器、云端文件，没有停止/重启训练。

用户现在需要运行的唯一新命令
下载 microduck-stage12-perf-readonly-20261020.py 到 /home/lx/下载，然后在本机执行：
python3 /home/lx/下载/microduck-stage12-perf-readonly-20261020.py
复用现有共享SSH会话；会话已断开时需由OpenSSH重新认证一次。
从已有日志各读取至多1MiB尾部，在云端用标准库解析，然后仅返回小JSON。
不导入Torch/MuJoCo，不安装软件，不运行仿真、基准测试、评估或PPO。
不发送start/stop/resume，不覆盖checkpoint、started或budget文件。
输出PERF_READONLY_REPORT_SAVED=路径；用户回传该JSON。

时间指标如何解释
1. 最近最多30轮Collection time：原runner的采样段墙钟时间。
2. 最近最多30轮Learning time：原runner的回报计算、PPO更新及其包装器墙钟时间。
3. 两段的比例仅以这两段之和为分母，不包括外围日志和checkpoint保存。
4. metrics.jsonl中elapsed_seconds相邻记录差：包含中间日志和保存工作的实际更新间隔。
5. 原日志计时没有CUDA事件分解，不可当成纯GPU内核耗时，也不能直接精确分离同步/IO。
6. 采集时训练持续运行，各文件不是同一原子快照，更新号差1可来自正常在途更新。
7. 以样本吞吐和训练效果判断优化，不以GPU-Util是否100%作为目标。

最小性能候选（尚未部署）
candidate/train.py 以及 train-performance-candidate.patch 仅用于离线审阅。
不是可直接覆盖运行目录的部署包；现有包有完整性校验，不能直接替换文件。
后续是否值得部署，应先依据只读统计结果确定；本包不含停止、换包或重启入口。

已实现的候选改动
1. 当前2个观测组时，正常路径每控制步的显式主机标量读取：
   动作1 + nan_state1 + 观测2 + reward1 + 上游NaN检查4 = 9。
   候选保留动作执行前检查1次，环境步完成后合并检查1次，总计2次。
   这个计数仅指所审阅的检查语句，不包括仿真器和PPO内部同步。
2. 每步仍检查所有动作、观测、奖励、done，以及nan_state重置；没有延到整批末尾。
   合并后的输出检查覆盖NaN和Inf，原上游check_nan所覆盖的done也包含在内。
   关闭重复上游检查只因对应判断已在env.step返回前完成；不是取消异常检测。
3. 每次PPO更新后，将所有参数的有限值判断在设备端合并，再读一次结果。
4. 自定义训练指标按所在设备批量复制到主机，保留每条指标float32均值、顺序和非有限值过滤。
5. 直接把已有collection/learning计时写入候选progress，不添加GPU同步计时。

保持的设置
2048环境、24步rollout、PPO minibatch/epoch、学习率、Stage08初始化、归一化策略、
物理步长、求解器、奖励、终止条件、课程与物理参数保持原值。
预算预约/提交fsync、模型保存及校验、检查点频率、日志频率与原8小时时限均未削减。
未新增短测或PPO预算，未启用AMP、torch.compile或新的并行环境数。

离线检查范围与限制
offline-review.json记录NumPy模拟张量的判断逻辑、单次主机读取、指标顺序/均值、
截断日志解析、只读输入不变、模拟SSH下小报告生成，以及关键保存函数AST不变。
build-receipt.json记录原文件与候选SHA256。
没有安装PyTorch或执行真实CUDA；没有性能收益数字，也不声称逐轨迹数值等价。
部署候选需要单独处理包身份和续训来源，保存Adam、归一化、课程计数、剩余预算和原截止时间。
不能再次从Stage08重头训练，不能重复已经通过的短测，不能替换仍运行进程的源文件。

后续决策顺序
先读已存在的计时；如果采样开销大，优先减少重复检查和日志同步。
如果PPO更新开销大，再定位小批次/RNN开销，不凭GPU占用率直接增加环境数。
增加环境数会增加每次更新的样本量，改变采样预算和优化行为，不能作为无副作用开关。
日志刷盘节流要区分普通进度和关键预算/checkpoint；后者本候选保持原行为。

参考
PyTorch性能指南：https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
NVIDIA-SMI利用率定义：https://docs.nvidia.com/deploy/nvidia-smi/index.html
代码核对使用随上一交接保留的固定版本源码；没有升级任何依赖。
