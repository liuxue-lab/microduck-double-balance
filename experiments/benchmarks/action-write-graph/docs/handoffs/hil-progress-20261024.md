# Microduck Stage12：PC动作写入图与累计优化交接

更新：2026-10-24，Asia/Shanghai。当前仍是PC性能优化，项目未最终验收。

长期遵循[本地SSH推送规范](microduck-local-ssh-push-protocol.md)及[云GPU与传输规范](microduck-cloud-gpu-and-transfer-protocol.md)，两份全文均保留。历史命令仅供追溯。

## 已核实的状态

旧 `TIMELINE_ARM_FAILED_PRESERVE` 已解决。根因为旧CUPTI12.8.90的 `CUPTI_ERROR_INVALID_DEVICE`；隔离使用12.9.79产生完整GPU时间线，两组160步、73440状态差异为零。固定入口 `native-pc-timeline-fixed-v2` 已验证，用户截图确认提交 `f33210ad7c474161b85e9f4cb5dc6db65f877db7` 正常SSH推送，不重复修复或远端全套检查。

最新 `report(20261024-153814).json`：PC_SYNC_MEASURED，errors=[]，4组/640控制步/6400物理步。三个候选各73440公共状态标量完全一致，161个延迟/RNG快照全部一致。结论KEEP_INFERENCE_ONLY，无候选被采用。

| 已测方案 | 均值ms | p95ms |
|---|---:|---:|
| 已采用基准 | 25.185270 | 26.276674 |
| 固定形状四元数 | 24.644871 | 25.881295 |
| 固定形状延迟 | 25.386679 | 26.952261 |
| 两者组合 | 24.469622 | 25.982217 |

组合的均值仅改善约2.84%，p95约1.12%，不满足均值/p95均至少5%的标准。每组150个正式计时步全部超过20ms。因此这次是测量成功、性能未达标，不是新timeline故障。

此前ScalarAudit+PackedChecks改善均值3.50%/p95 3.14%，单独未采用。新累计组仅测量叠加收益，不假定收益相加，不放宽采用门槛。

最新报告位于 `/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-pc-sync-v1/report.json`，SHA256 `afe6ba189a35733df2470d8e95f4f7600070e5752eab99669e851cccdd8a167a`。
修复成功报告SHA256 `1c12ea0d8602dfc3aeb83b9662936b5145a1ef659f936ae1e52e8d1fc2f8e114`；固定CUPTI库SHA256 `2fdab19dc3fccdd4b2f5eba137aabefc68e685f6497bb38eea649866c5672a2a`。

## 本轮方案、范围与执行

原动作写入host耗时约1.68ms/控制周期。新ActionGraph把每个物理步的原始动作写入合并提交；处理后的动作和辅助力输入每控制步暂存一次，仍每个物理步写入。编码器偏置、关节映射、零辅助力写入及原处理逻辑保留。
预热仅写独立私有缓冲；重放先读取当前目标/外力的完整内容，保留未涉及关节和其他物体外力。原 `[0]` 索引在捕获副本中用同值CUDA int64索引表示；参考路径保留原Python列表。前40次完整目标和外力逐位比较，无额外物理步或RNG抽样。

三组：baseline、action-graph、cumulative。累计组额外采用StaticQuaternion+ScalarAudit+PackedChecks；延迟优化、CommandViews、ReadGraph、HeadGraph不启用。500Hz检查的阈值、频率、顺序和完整记录不变，扰动及只清除自有外力的异常处理保留。
每组160控制步/1600物理步（3.2秒，含完整2.0–2.1秒扰动），排除前10控制步计时。逐项检查73440公共状态、完整Audit/Trial/扰动及161次动作历史/目标/辅助力输入/延迟/RNG快照；不一致停止并保留。均值和p95都改善至少5%才反序复测候选和基准，以更快基准作最终判断。最多5组/800控制步/8000物理步。

下一条命令（下载到 `/home/lx/下载` 后）：

```bash
python3 "/home/lx/下载/microduck-stage12-pc-actiongraph-v1-20261024.py" --apply &&
git -C /home/lx/microduck-double-balance/workspace push origin double-balance
```

保持主终端开启，只回传：

```text
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-pc-actiongraph-v1/report.json
```

同目录固定入口为 `microduck-stage12-pc-actiongraph-v1.py`；支持 `--status`、`--watch`（2秒刷新，完成后仍保持）、`--stop`。`--run`不自动重跑；必要时`--retry-run`建立新目录保留旧证据。本包不自动激活生产方案或启动最终60秒HIL。

## 不变模型与剩余任务

范围仅PC仿真↔有线ROS2 Humble↔Orin Nano Super FP32推理。不接电机或实物机器人，不做真机演示；Stage13/14行走取消。允许位置漂移，不宣称严格持续接触；原头壳及双球模型保持。无新增PPO、重导出、INT8或未经验证FP16。

- 已采用：InferenceOnly+PhaseCache+BAM Graph v2+BatchedAuditReads。
- 最终检查点final6h_005304.pt，本轮PPO5304/总11314；heldout18/18，campaign24/27，不能写27/27。
- 检查点SHA256 `cf7ce9297672943687019afe071d94918d3eb2620c18a1e8fe689512f8c08111`。
- ONNX SHA256 `8c1c1250b9b88d9a91e3eb095fea8bce21daa73c251b43f3cd8adc1327f03c4e`。
- 冻结导出 `/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-ros2-humble-v1/frozen-export`。
- actor61/critic85/action14；h/c各[1,1,256]；归一化已烘焙，无关节重排、裁剪或动作滤波。
- 扰动：机器人自重10%，水平独立均匀随机方向，每2秒持续0.1秒，60秒回合，2至58秒固定起始。
- 既有ROS2/Orin功能验证2×3000控制步/30000物理步、6000次ONNX比较通过；原完整周期约39.9ms。FP32平均0.352815ms/p95 0.388907ms；JSON调度到验证平均2.593678ms/p95 3.052529ms。这些不是最终完整闭环实时通过。
- PC192.168.50.1/enp4s0/jetson-direct；Orin192.168.50.2/enP8p1s0/jetson-wired；domain62，FastDDS reliable/depth1/volatile/wired-only。后续才连接jetson@192.168.50.2并复用已有SSH连接。

待办：PC性能→最终方案完整60秒ROS2/Orin HIL（完整周期均值/p95/p99/最大值/超20ms次数，按既定截止处理）→部署冻结→文档、仿真演示与归档。没有新的stage12-complete标签。

## Git、回传与云GPU

固定仓库 `/home/lx/microduck-double-balance/workspace`，分支double-balance，远端 `git@github.com:liuxue-lab/microduck-double-balance.git`。
本包基于已运行的pc-sync提交 `289e3fd260f8c603c3d8e0337d740b0806c6255f`，仅增加experiments/stage12/pc-actiongraph-v1；新提交及包身份由安装回执记录。最新报告能证明本机安装，不能证明pc-sync已推送；正常用户SSH push可一次推送尚未上传的两个提交。不自动push，不reset/stash/clean，不覆盖冲突，不force push；正常push输出足够。

回传分支stage12-push10-final6h-return-20261023t181130z-1b79efbd-s002，完整提交 `0cbee30e5a83904371076ca09a8cfb4afd1fa3f4`；400文件/5归档已校验，并非全部云端原始轨迹均回传。
云端剩余任务无；5090已由用户确认关闭，OFF_USER_CONFIRMED。先前已提醒关机，确切提醒与确认时刻不在当前证据中，不补造时间。不开机、不探测旧云主机、不启动失败过的关机守护。本轮仅笔记本RTX5060，不需Orin。

## 常见错误与快捷操作

| 现象 | 处理 |
|---|---|
| KEEP_INFERENCE_ONLY且PC_ACTIONGRAPH_MEASURED | 测量完成但收益不足，保持原方案、回传报告，不放宽门槛 |
| ACTIONGRAPH_SAMPLE_COMPLETE | 160步计划停止，不是完整60秒验收 |
| 捕获失败、源哈希变化、状态不一致 | 保留并停止，一份报告含日志尾部，不替换GPU库或跳过检查 |
| 路径冲突、本地分支分歧 | 保留install-report及双方文件，不强制合并 |
| 下载名带(1) | 使用实际路径，避免运行同名旧脚本 |
| 旧timeline再次报错 | 旧入口仅作历史；使用已验证fixed-v2，不重装修复 |
| 主终端中断 | 查看本轮report或--status；不并发重跑，重试用新目录 |
| Git要求PAT/密码 | 沿用SSH远端和已有key/agent，详见全文规范 |

CPU回归使用原方法体及合成积分器，覆盖1600步动作写入、三组160步流程、故障保留、门槛判定和真实Git导入；不能冒充RTX5060实际CUDA捕获或性能结论。下一步等待本机报告。
