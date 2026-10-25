# Stage 10 初始化诊断恢复 v1

只恢复已审查的 `20261001T141949Z-37e84aba` 批次。旧下载工具、原始 plan/run/result、第一组轨迹均保留；恢复信息写入该批次的 `initialization-recovery-v1/`。不写源码仓库。

运行下载到 `/home/lx/下载` 的单文件入口：

```bash
python3 '/home/lx/下载/microduck-stage10-initialization-recovery-start.py'
```

入口校验并安装到 `/home/lx/下载/microduck-stage10-initialization-recovery`，使用仓库现有 `.venv`，不安装依赖。不要继续调用旧 initialization-start 或旧诊断入口。

恢复命令：

```bash
python3 '/home/lx/下载/microduck-stage10-initialization-recovery/microduck-stage10-initialization-recovery.py' --resume /home/lx/microduck-double-balance/artifacts/double-balance-stage10/initialization/20261001T141949Z-37e84aba
```

只打包，不构造环境：

```bash
python3 '/home/lx/下载/microduck-stage10-initialization-recovery/microduck-stage10-initialization-recovery.py' --package-only
```

第一组固定复用 `01-initialization/attempt-772ee431`，原状态仍为 `FAILED_REVIEW_REQUIRED`，不生成假的完成凭据。第二、三组各最多一次环境构造；已完成并通过哈希的组只复用。新的构造后失败会阻止自动重试，需上传全部打印的 review ZIP 分卷。已完成后重复运行只复用和重新打包。

原 `assert_zero_assistance` 用于缓存已生成后的检查，此次初始化尚未处理动作，`_force=None`、另外三个缓存尚不存在。恢复工具只在诊断末尾检查这一精确状态、零 hold、所有 body 的 `xfrc_applied` 和所有 DOF 的 `qfrc_applied` 均有限且为零。已物化的缓存（即使全零）也会停止待审查；不会创建/清零缓存、处理动作、额外 forward 或 step。Stage 08/09 的原断言及旧无辅助验收完全保留。

本次剩余额度：2 个新进程，每个 128 环境，1 次原有最终 reset（含原有 2 次 forward）；0 rollout、0 策略推理、0 PPO、0 云 GPU 小时。环境构造中的原 CUDA graph 捕获内部调用另计，不称为“零内部物理调用”。检查点恢复模型/normalizer/Adam、LR `5.062500000000001e-05`、迭代 4999、common step 120000；评估计数器按原流程归零，原 policy.reset；不执行优化器 step。

最终批次状态保留 `WITH_POSTCHECK_GAP`：第一组缺少的运行末尾模型/Adam/后端/命令等检查不能重建。此结果不评估策略成功，Stage 10 尚未完成。

固定仓库 `/home/lx/microduck-double-balance/workspace`；分支 `double-balance`，HEAD `00e34c2038771c5d4ad49fe45dff828c0232e60c`，101 项冻结文件需校验。主检查点 `E/20260929/update_004000.pt`，SHA256 `86d55c3703c18fcf4817db49c6e4dcf839b3f5195d68ae2c559db7e222bded97`。

云端状态 `OFF_CONFIRMED_BY_USER`。不启动/停止云实例，不调用旧自动关机守护，不重跑 Stage 06 smoke；未全部回传的 Stage 09 云数据继续保留。以后云计算及必要回传 SHA256 完成后立即提醒用户手动关机，提醒与确认分开记录。

后续交接保留仓库协议 `docs/handoffs/microduck-local-ssh-push-protocol.md`、`docs/handoffs/microduck-cloud-gpu-and-transfer-protocol.md`。传输默认 Git SSH，代码与模型归档分支分开，不合并归档分支。本机创建提交及注释标签、用户 SSH 推送；正常成功后不重复完整远端核验。此次没有提交、标签或推送。

离线验证使用假对象、返回轨迹和临时目录，未运行 Torch/MuJoCo/CUDA。见审查包中的测试日志；本机执行后仍需审查全部返回分卷。
