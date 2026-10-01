# Microduck Stage 10 — 初始化批次 C 首组失败审查与恢复

审查日期：2026-10-01。结论：第一组完整采到了预期的 33 个初始化快照，随后因诊断误用辅助力缓存检查而停止。保留原失败记录，按 SHA256 复用这组轨迹；仅执行原计划尚未开始的第二、三组。Stage 10 未完成。

## 证据核验

输入 `review-20261001T142003Z-196ef3-part01.zip`，16,307,719 字节，SHA256：

`6d285fb6c6a627d27dcfc334f730892460bd18db86c11792fb8985576394fc4a`

28 项清单文件逐项 size/SHA256 通过；4 个数组 ZIP 可完整解压，507 个独立数组的内容哈希、dtype、shape、引用及条目一致。33 个快照共 30,476 次数组引用，独立数值数组原始数据共 61,295,000 字节。物理数组与已返回观测数组均有限。

所有快照的 `context/assistance_hold`、`physical/xfrc_applied`、`physical/qfrc_applied`、`physical/time` 均为零。所有快照的 `hold_action/_force` 为 `None`，另三个缓存均未创建。最终回合计数、common step 和 sim step 为零。原始调用计数准确匹配预期，包括最终 reset 内的两次原有 forward，BAM compute 一次、observation compute 两次、policy.reset 一次。

构造期记录了 Python 层 `mjwarp.step/forward/reset_data` 各三次，属于原有 CUDA graph 构造路径。它们不是 rollout 次数，也不是全部 GPU kernel/graph 执行计数。此次 rollout、加载后的策略推理和新增 PPO 均为零。

## 直接错误及源码依据

子日志及 `result.json` 一致记录：

```text
ValueError: Residual assistance cache: _force
```

诊断入口第 184 行调用了冻结源码 `double_balance_stage08_state.py` 的 `assert_zero_assistance`。该函数第 203 行同时要求缓存已存在且为零；`None` 因而触发这条消息。

冻结的 `tasks/mdp.py` 中，`BallHoldAction.__init__` 第 8706 行将 `_force` 初始化为 `None`；`process_actions` 在第 8729–8741 行才产生四个缓存；`apply_actions` 第 8747–8748 行在 `_force is None` 时直接返回。本次没有处理动作，返回轨迹与此生命周期完全一致。错误消息中的“Residual”不等于实际测得了非零外力。

已复核交接源码中的 101 项冻结文件，且本次回传的九个原工具文件与交付版本逐字节一致。没有修改 Stage 03 物理、Stage 04 任务/验收、奖励、头部约束或正式训练代码。

## 修复内容与执行额度

只修改独立下载目录中的初始化诊断：末尾改用严格只读的“尚未处理动作”检查。要求 `_force` 精确为 `None`、其他三个缓存不存在、hold 配置 `(0.,)`、零宽 hold 动作、hold 及所有 body/DOF 的实际外力均有限且为零。缓存已物化即停止待审查，即使其数值为零。检查不创建/清空缓存、不调用 process/apply、不增加 reset/forward/step。旧正式 rollout 的 `assert_zero_assistance` 保持原样。

新工具安装到 `/home/lx/下载/microduck-stage10-initialization-recovery`，不覆盖旧下载工具。固定继续原批次：

`/home/lx/microduck-double-balance/artifacts/double-balance-stage10/initialization/20261001T141949Z-37e84aba`

原 `plan.json`、`run.json`、`01-initialization/attempt-772ee431/result.json`、日志与轨迹全部固定哈希保留。新恢复记录写入 `initialization-recovery-v1/`。首组维持 `FAILED_REVIEW_REQUIRED`，没有伪造完成凭据。

剩余额度为 2 个新进程，各 128 环境；每个仅原有的一次最终 reset（含两次原 forward）、零 rollout、零策略推理、零 PPO、零云预算。已完成组验证凭据后只复用；任何新构造后失败均停止待审查，不自动重做。第一组不会重跑。

加载既定 Stage 08 主模型及 normalizer/Adam，LR `5.062500000000001e-05`、迭代 4999、加载 common step 120000；评估进度按旧流程清零、使用原 policy.reset，禁止 optimizer.step。工具及任务配置、运行时和源文件沿用原固定校验。

## 明确保留的证据缺口

首组在缓存断言处停止，因此没有执行到：零 twist 断言、observer.rows 空检查、末尾 actor/critic 参数与完整 state_dict 比较、末尾 Adam 比较、末尾 LR/迭代检查、backend_after 活读及比较、最终源码/plan 复核。恢复时的校验只描述恢复时刻，不能补成第一组的历史检查。

三组比较的最终状态会显式包含 `WITH_POSTCHECK_GAP`。复用第一组仅用于带此限制的诊断比较，不能称为全部运行后检查已完成。单组轨迹也不能定位跨进程数值差异的因果根源。

15 项模型/工作区采集缺口继续保留于 JSON：12 项未枚举对象、2 项数组过大或 object 类型、1 项超过 8 MiB。EFC 的填充存储、未捕获的设备内部状态和 GPU 读回同步效应均限制解释。程序状态不等于策略成功。

## 验证及下一条命令

10 项恢复回归检查和 3 项安装入口检查通过。覆盖懒缓存只读判定、非零/NaN/Inf 拒绝、配置与作用域检查、旧证据哈希、篡改拒绝、新失败阻止重试、只启动第二/三组、重复执行零新增构造、带历史缺口的三组比较、重复安装和文件/符号链接保护。

测试使用返回证据、NumPy、假对象与临时目录。这里未运行 Torch、MuJoCo 或 CUDA；不把离线检查写成 GPU 实测。

下载单文件入口到 `/home/lx/下载` 后执行：

```bash
python3 '/home/lx/下载/microduck-stage10-initialization-recovery-start.py'
```

运行结束后上传全部打印的 review ZIP 分卷。再次失败时保留目录，先返回新的审查包；不要删除旧尝试或调用旧入口重跑。恢复及仅打包快捷命令见工具 README。

## 持续操作约束

本机仓库 `/home/lx/microduck-double-balance/workspace`，下载目录 `/home/lx/下载`。分支 `double-balance`、HEAD `00e34c2038771c5d4ad49fe45dff828c0232e60c`；Stage 09 标签仍在技术收尾提交，不移动、不重复完整远端核验。

保留协议 `docs/handoffs/microduck-local-ssh-push-protocol.md` 和 `docs/handoffs/microduck-cloud-gpu-and-transfer-protocol.md`。后续传输默认 Git SSH，代码与模型归档分支分开，不合并归档分支。本机创建提交及注释标签，由用户 SSH 推送；此次仅交付诊断恢复工具，没有创建本机提交或标签。

云端 RTX 5090 为 `OFF_CONFIRMED_BY_USER`；此次不访问云端、不运行自动关机守护。Stage 09 原始轨迹尚未全部回传，云数据不得删除。以后云工作和必要回传 SHA256 完成后立即提醒手动关机，提醒和实际确认分别记录。RTX 5060 只作代码/仿真/推理/视频，不重跑 Stage 06 smoke。

主模型仍为 Stage 08 `E/20260929/update_004000.pt`，SHA256 `86d55c3703c18fcf4817db49c6e4dcf839b3f5195d68ae2c559db7e222bded97`；历史独立成功率仍为 652/768（84.90%），不合并开发重复试验。无辅助双球平衡优先，头部纠偏暂停；不锁头、不默认全程朝前。拆板迁移尚未授权，旧托盘任务和旧结果继续保留。
