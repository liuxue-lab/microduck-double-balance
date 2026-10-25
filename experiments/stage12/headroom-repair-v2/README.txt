Stage12 静态双球 — 缺失依赖与原任务恢复修复 v2
2026-10-23，Asia/Shanghai

本轮真实结果
固定回传提交 4823151e3a856eeec82a7bcef6396bfdbe05d0d7，44 个成员已逐文件校验。
任务 CAMPAIGN_ERROR_PRESERVE：加载配置时报 No module named 'reward_spec'。
PPO 尝试/完成均 0，候选评估 0，没有新模型。回传成功不是训练通过。
原预算 505 次不变；原开始 2026-10-23 20:08:58.523970+08:00，截止 22:08:58.523970+08:00。
修复不自动续期，也不把一次启动失败变成一份新的 505 次预算。

改动范围
1. 从 Stage12 原始已提交代码恢复 reward_spec.py；Git blob 56321d96901a12b7fe88f798eb10dc99511e22e0。
   现有 reward-spec.json 是关节余量说明，不是 Python 导入所需的 reward_spec 模块。
2. 补充本地模块依赖闭包检查，并在恢复前导入训练/评估依赖；不构建环境，不积分，不安装库。
3. 显式 --repair 只处理回传证据所指的零更新故障；比对旧文件哈希、活进程和 GPU 租约。
   全部 44 个旧文件复制校验后，才切换到新训练包。原包和旧 source-commit.json 留在修复审计目录。
   budget.json 与 started.json 逐字节保留，只有源代码指针更新。session 从 1 到 2，worker 从 2 继续编号。
   重复 --repair 不二次启动。正在运行、已消耗更新、不同故障、原截止已过都不能绕过停止规则。
4. 回传目录按 session 分开；修复后用新归档和新资料分支，旧失败归档不覆盖。
5. 本机回传提示显示真实任务状态、目标是否通过和原始错误，避免把传输成功当作训练成功。

明确不变
train.py、evaluate.py、native_task.py、speed_task.py、headroom_task.py、headroom_costs.py、
budget.py、模型边界观察器和原 Stage12 模型保持字节一致。没有修改奖励、物理、观测或验收标准。
本轮预算仍只修 head_roll 余量和无外力保持；10% 每 2 秒随机扰动仍是之后的独立工作。
Stage13/14 预算和失败记录保持冻结，不推进轨迹，不重训原 Stage12，不重跑旧诊断。

运行
本机 /home/lx/microduck-double-balance/workspace，下载 /home/lx/下载。
云端复用 /root/autodl-tmp/microduck-double-balance 的原 .venv、冻结源码、原模型和原任务目录。
新源目录 experiments/stage12/headroom-repair-v2；只新增代码，不覆盖旧源码目录。
源分支仍为 double-balance，由用户导入新增量 bundle 并以既有 Git SSH origin 推送。
新版单文件入口包含 bundle，当前精确命令和完整提交哈希见交接 TXT 与 build-receipt.json。
正常已验证的旧回传不用重做；云端取源码继续经现有共享 SSH 发送小型固定提交包，不 fetch GitHub。
本轮修复包需要新的代码提交，因此必须先 --install-code，再正常 git push，最后用 --repair。
不能再用旧脚本的 --train 期待恢复已保留的运行错误。

验证限制
CPU 检查只能证明代码和恢复逻辑，不能证明 GPU 环境构建、PPO 或双球效果已通过。
实际 GPU 执行仍由用户运行；如果原截止时间已过，保留同一预算停止，需明确的新时间授权后才能继续。
新检查不代表新的物理诊断、训练预算或对实机安全的承诺。

回传和电源
原回传分支 stage12-headroom-v1-return-20261023t120858z-c0c23353 已验证，原 ZIP 为 86858 字节。
本轮没有中间模型遗漏（索引 omitted 为空），旧云端全量历史资料仍未全部备份。
已提醒用户可以手动关 GPU，实际是否关机 UNCONFIRMED；脚本没有任何实例开关机动作。
计算及必要回传结束、本机 SHA256 通过后立即手动关云 GPU，不能等本机视频和文档完成。

长期约定
保留并遵循 docs/handoffs/microduck-local-ssh-push-protocol.md 和
 docs/handoffs/microduck-cloud-gpu-and-transfer-protocol.md。
当前 SSH 为用户最后提供的 ssh -p 53885 root@connect.bjb2.seetacloud.com；不据此宣称实例仍在线。
如已换实例，不能用同一脚本自动迁移预算；需先证明原账本、修复证据和截止时间都保留。
