Microduck Stage12 最后一次六小时训练：奖励预检修复 v2
2026-10-24，北京时间

结论与当前状态
原任务在02:11:30启动；24秒后奖励预检失败，0次PPO尝试、0次PPO完成、0次评估。
错误是预检未识别mjlab 1.3.0用scale变量表示dt；尚未进入新奖励的真实rollout。
已核验Git SSH返回的49个文件。原模型仍是6010更新的headroom_000506.pt。
本包修复并续接同一任务，不新建预算，不刷新时钟，不代表云端已经开始训练。

固定截止（2026-10-24，Asia/Shanghai）
原始起点：02:11:30.031889；训练停止：06:56:30.031889；全部计算截止：08:11:30.031889。
对应UTC epoch：1792779090.031889 / 1792796190.031889 / 1792800690.031889。
计算后最多90秒保存清理；之后CPU打包回传最多30分钟，不再PPO或物理积分。
之前建议的09:30定时关机仍留有余量；定时关机由用户在平台设置，本脚本不关机。
必要文件下载并通过SHA256后应手动关机保留存储；电源状态UNCONFIRMED。

一次粘贴：安装、由用户推送、续接
把 microduck-stage12-push10-final6h-repair-v2-20261024.py 下载到 /home/lx/下载，然后运行：
python3 "/home/lx/下载/microduck-stage12-push10-final6h-repair-v2-20261024.py" --install-code &&
git -C "/home/lx/microduck-double-balance/workspace" push origin double-balance &&
python3 "/home/lx/下载/microduck-stage12-push10-final6h-repair-v2-20261024.py" --repair --ssh "ssh -p 32185 root@connect.bjb2.seetacloud.com"

当前新5090地址由用户提供，脚本复用共享SSH会话。
--repair会核对已知失败的预算、起点、退出信息、worker错误和源码，保存旧文件后安装修复。
64环境×5次更新的真实烟测及ONNX检查通过，4096环境正式PPO完成至少1次更新，才输出READY_TO_LEAVE。
看到READY_TO_LEAVE后，可Ctrl+C退出监控、关闭终端或本机，云端任务独立运行。
重复--repair只监控已经发起的本次续接，不再次训练或计6小时；--train是同一修复入口的别名。
预留续接标记后出现启动不确定或错误会保留停止，不自动重试。过06:56:30不再启动训练。

实时查看
python3 "/home/lx/下载/microduck-stage12-push10-final6h-repair-v2-20261024.py" --watch
单次状态：同脚本 --status；停止计算：--stop；仅重试回传：--return；下载核验：--download。
错误报告现在包含worker_result中的真实异常，不必仅凭最后几行初始化日志判断。
训练已结束时watch会保存报告并退出，这是正常终态。监控断开不会停止云端。

修复内容与边界
奖励验证直接调用当前安装的RewardManager.compute，在独立CPU缓冲区用0.02/0.04秒两个dt验证
正负奖励、零权重、累计值和重复调用；不调用真实奖励函数，不修改运行中的manager，不推进物理或PPO。
50Hz下物理失败仍为100×(-1)×0.02=-2；关闭dt缩放、频率错误、倍率错误仍被拒绝。
termination-before-reward-before-reset检查保留。奖励权重、PPO参数、外力和验收种子不改。
测试使用官方v1.3.0实际compute源码，来源见 reference/runtime-source.json。
旧started.json和budget.json逐字节保持，旧attempt=0；新会话使用session-003/004，session-001保留。
旧49文件在repairs/reward-scaling-v2/before/保留；supervisor在打包后追加的最终日志另存完整版本。
旧package移至retired-package，旧结束/进程/回传控制记录先备份再清理。
旧transfer/final保留；本次结果使用transfer/final-reward-scaling-v2和新的artifact分支。

固定来源与证据
父模型：10037598 bytes，SHA256 ca2a82acbe939169367272ffb50242bbfc1b4a39c06ed186bfc4cf3009ed4130。
原源码提交 c9d5f704a5ca9b91e0d40ead9fb40b5c5727f321。
原package de7afba37fd8f807905005a828644b3dfd7e1bcb1404b0e884b3a76019f2640f。
失败回传分支 stage12-push10-final6h-return-20261023t181130z-de7afba3-s001。
失败回传提交 caa3266b69522e7c06d19d8e766339083e89839e。
返回索引SHA256 666e3fa0fc45d43bfbe3291c6c031ce317c16fe145842dc77d2897b9ba278a87。
原分卷SHA256 f8c6a3f2b413f7588c00a2c6e305b409431b6789525ba30c61354cb25fc8a6ab。
本修复新增源码目录 experiments/stage12/push10-final6h-repair-v2；增量bundle基于上面的原源码提交。
本次构建提交及package身份以build-receipt.json为准。

任务与验收（沿用原方案）
固定61D actor/85D critic、14 BAM、原几何接触动作链、50Hz控制/500Hz物理，不加滤波。
25%回合无扰动，75%有扰动；外力按原墙钟阶段5%→7.5%→10%自重。
每个60秒回合在2,4,...58秒施0.1秒水平力，每次独立随机方向；实际力=fraction*m_robot*g。
总上限10005次尝试：5次烟测、10000次正式保险上限；到原墙钟即停。
最多3个候选各3回合选模；锁定合格模型后最多18回合独立验证，合计最多27回合。
失败或超时如实记录，不换种子补过，不再自动训练。有限序列通过不表示所有随机序列均通过。
原详细训练方案保留在reference/original-final6h-v1-README.txt，仅作历史设计说明；其中初次启动命令已被本修复入口替代。
Stage13/14继续取消；最终模型导出/部署一致性、严格接触检查、实机标定属于后续，未宣称完成。

固定路径与协议
本机仓库 /home/lx/microduck-double-balance/workspace，分支double-balance。
Git SSH远端 git@github.com:liuxue-lab/microduck-double-balance.git，不改HTTPS/PAT、不强推。
本机报告 /home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-push10-final6h-v1-client。
云端任务 /root/autodl-tmp/microduck-double-balance/artifacts/double-balance-stage12/native-push10-final6h-v1。
遵循 docs/handoffs/microduck-local-ssh-push-protocol.md。
遵循 docs/handoffs/microduck-cloud-gpu-and-transfer-protocol.md。
本地改动保留，源码由用户SSH推送；模型及证据走单独artifact分支，不合并到源码。
目前尚未收到修复后的云端READY_TO_LEAVE或训练结果；不能声称训练正在运行或已通过。
