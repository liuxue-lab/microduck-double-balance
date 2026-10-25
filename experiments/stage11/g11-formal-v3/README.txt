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

2026-10-15 CI 文档链接补充：
payload/ 原149份已执行文件全部保持原字节，payload-inventory.json 保持原样。
额外补入3份仅用于文档导航的历史依赖，与仓库 docs/ 对应原件逐字节相同：
  audits/stage-10-publication-confirmation.json
  provenance/microduck-stage-10-ci-followup.md
  provenance/microduck-stage-11-opening-prompt.txt
这3份文档不是已执行程序的新增成员，不纳入原149份执行清单，不重新制作旧回传包。
历史文档中的运行命令只供追溯；当前没有待运行的G11物理脚本。
补记见 docs/handoffs/microduck-stage-11-ci-followup.txt（仓库根目录下）。
