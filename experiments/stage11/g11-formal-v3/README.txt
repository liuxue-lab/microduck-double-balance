Stage11 G11 已审计正式候选（名义、native nworld=1）

payload/ 的149份文件与已执行 v3 程序原字节一致，包括92份已接受物理辅助源。
保存的 JSONL 是1792次v2积分的轻量账本；990字节NPZ仅包含一次正常步的公开观测/奖励，不是完整轨迹或模型。
此目录不安装生产任务，不解锁随机训练，不是新的物理运行授权。不要直接调用worker重跑已完成批次。

CPU容器/键类型回归（12项；不重跑旧29项，也不导入GPU模拟器）：
python3 experiments/stage11/g11-formal-v3/validate_v3.py

原始v3复核需先将固定ZIP解压为return-root/extracted，return-root保留ZIP与g11-report.json：
review_return_v3.py --help 展示输入参数。使用已安装NumPy的Python，输出只写return-root/audit.json。
原始归档及独立报告位于交付收尾ZIP；固定commit和SHA见docs/audits/stage11/g11-closure-review.json。
运行期attempted/completed不可因再次CPU审计而增加。原报告中的OPEN和历史账本均保持不变。

v2的完整步后owner快照缺失仍属真实限制；v3不宣称恢复该快照。
旧checkpoint已检查94张量，但TASK_AND_TIMING_DIFFER，禁止直接resume。
