Microduck Stage12：已验证 CUPTI 修复的固定入口 v2（2026-10-24）

本次关闭的问题
TIMELINE_ARM_FAILED_PRESERVE 的直接原因是原 CUPTI 12.8.90 在当前 RTX5060
环境中返回 CUPTI_ERROR_INVALID_DEVICE，无法采集 GPU 活动。原 GPU 计算、图重放、
CUDA Event 计时均正常，调整 Warp 初始化顺序无效。

用户报告 report(20261024-133007).json 已确认隔离 CUPTI 12.9.79 成功：
SHA256 1c12ea0d8602dfc3aeb83b9662936b5145a1ef659f936ae1e52e8d1fc2f8e114
状态 PC_CUPTI_REPAIR_TIMELINE_MEASURED；profiling_repair_verified=true；errors=[]。
8 个原始环境步均有 CUDA 核函数，窗口内共 35522 个；160 步、73440 个共同状态
标量差异全部为零；完整审计、Trial、扰动相同；原模型、环境和旧报告未改动。
这是这台电脑上的直接验证；不声称已查明 NVIDIA 内部的全部失效机制。

本入口固化该已验证方案，不继续尝试其他 CUDA/Torch 版本。
--apply 将已验证的 5 个库和 NVIDIA License 按 SHA256 复制到独立固定目录；
只在分析子进程中设置 LD_PRELOAD/LD_LIBRARY_PATH。不改 .venv、驱动、shell 配置，
不运行 pip/apt、不下载库、不连接云端或 Orin。先后核对冻结输入，自动运行两个
小型 GPU 采集测试验证新固定路径。复用已有完整项目数值对照，不重复跑 160 步。

安装与一次执行
下载 microduck-stage12-timeline-fixed-v2-20261024.py 至 /home/lx/下载，然后执行：

python3 "/home/lx/下载/microduck-stage12-timeline-fixed-v2-20261024.py" --apply &&
git -C /home/lx/microduck-double-balance/workspace push origin double-balance

两个命令一次粘贴；前者成功才执行后者。脚本内嵌增量 Git bundle，仅添加本目录，
以 02d34b43d930a688d57904e90015298bb3492886 为父提交，保留不冲突的工作树改动。
远端保持 git@github.com:liuxue-lab/microduck-double-balance.git，不 force push。
脚本自身不 push；第二条由用户终端沿用已有 SSH 身份执行。
若旧 SSH agent 尚未缓存口令，使用既有 agent；不用 GitHub 密码/PAT。
本项未完成 Stage12 全部验收，因此不建立 stage12-complete 标签。

完成标志：TIMELINE_FIX_STATUS=TIMELINE_FIX_INSTALLED_VERIFIED。
本次主终端保持开启，5090 保持关闭；无需连接 Orin。
成功不需要为同一个 CUPTI 错误继续发报告。再次 --apply 只核对文件，不重复 GPU 测试。
失败保留所有结果并停止，不回退到旧库伪造成功；失败时发回下面的 fix-report.json。

固定路径（删除下载目录里的脚本后仍可用）
ROOT=/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-pc-timeline-fixed-v2
入口：ROOT/microduck-stage12-timeline-fixed-v2.py
修复回执：ROOT/fix-report.json
固定库：ROOT/runtime/cupti-12.9.79/libcupti.so.12
固化证据：ROOT/evidence/reviewed-repair-report.json
已验证项目轨迹：
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-pc-cupti-repair-v1/runs/20261024T132900Z-205d6d0f

后续需要新时间线时使用固定入口 --run；已有新报告时不自动重复，显式重测 --retry-run。
--status 单次查看；--watch 每 2 秒刷新，观察窗口 Ctrl+C 只退出观察。
--stop 只停止该入口所属且 PID/start_ticks 一致的活动任务，并保留报告。
正在运行的主终端 Ctrl+C 会停止其子进程；重开后先 --status，不凭旧 RUNNING 判断存活。
原 microduck-stage12-pc-timeline-v1-20261024.py 作为历史入口保留，后续不要再用它采集。
旧 native-pc-timeline-v1/report.json 中的失败是历史记录，不能改写为成功。

--run 的范围与门槛
固定入口使用原包 ba0b16e7acde7bc8484d9c9519c490110a3556fe528d9276f0cf611732456b54。
原包先核验全部源文件，之后才导入；原 worker 不改动。
基准进程保留原 CUPTI，timeline 进程加载固定库；各最多 160 控制步/1600 物理步，
每个 worker 墙钟最多 230 秒。保留 500Hz 模型边界、电机、外力审计和扰动顺序。
对照完整状态与审计，8 步完整 GPU 覆盖、实际加载库映射和输入未变均通过才成功。
新报告为 ROOT/report.json，状态 PC_TIMELINE_FIXED_MEASURED，不覆盖任何旧报告。

仍未完成的项目任务
正常未开启分析器的 PC 环境步均值 24.9927876ms，p95 26.278897ms，最大26.770447ms，
150/150 个计时步超过20ms。故 50Hz 尚未达标，本修复不声称已提高正常运行速度。
分析器下 8 步均值40.846096ms含采集开销，不能作为正常性能或加速依据。
时间线显示大量小核函数和同步：8步共1192次 cudaStreamSynchronize、14274次
cudaLaunchKernel；CUDA活动并集76.953ms、环境窗口326.887ms。剩余未记录活动的
时间不能直接视为纯Python或证明GPU空闲。后续应检查同步来源与小操作提交成本，
保留500Hz检查并以无分析器对照衡量，不能把嵌套宿主时长直接相加。
顺序仍为：PC性能优化 → 最终60秒ROS2/Orin完整周期验收 → 部署定版 → 材料归档。
最终完整周期还包含通信和Orin推理，不能仅凭PC均值低于20ms宣称通过。

冻结约定
范围：电脑仿真 ↔ ROS2 Humble有线通信 ↔ Orin Nano Super FP32 ONNX；无真机电机或真机演示。
保留 InferenceOnly + PhaseCache + BAM Graph v2 + BatchedAuditReads、JSON + FP32。
固定 actor61/critic85/action14，LSTM h/c=[1,1,256]；不变更关节顺序、归一化或动作滤波。
final6h_005304.pt SHA256 cf7ce9297672943687019afe071d94918d3eb2620c18a1e8fe689512f8c08111
final6h_005304.onnx SHA256 8c1c1250b9b88d9a91e3eb095fea8bce21daa73c251b43f3cd8adc1327f03c4e
冻结导出：/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-ros2-humble-v1/frozen-export
此前 heldout18/18，campaign24/27；功能闭环6000步通过，不能表述为50Hz实时验收通过。
云端任务 NONE；5090 为 OFF_USER_CONFIRMED（用户2026-10-24确认），不探测/启动。
实际关机时刻未知，不补造平台回执。回传400文件/5归档已核验，非全部云端原始轨迹。
回传分支 stage12-push10-final6h-return-20261023t181130z-1b79efbd-s002；
回传提交 0cbee30e5a83904371076ca09a8cfb4afd1fa3f4，不重复传输或重新训练。

长期规范（完整副本附于本目录）
docs/handoffs/microduck-local-ssh-push-protocol.md
docs/handoffs/microduck-cloud-gpu-and-transfer-protocol.md
本机固定仓库 /home/lx/microduck-double-balance/workspace，分支 double-balance。
新增目录冲突、库/报告SHA变化、Git分歧均停止保留；不覆盖、reset、stash、clean或force。
正常 push 成功输出足够，无异常无需重复远端验证。下载带(1)后缀时使用正确实际路径。

本地测试范围
CPU 回归覆盖固定目录复制/幂等、库损坏/路径冲突、仅子进程环境、缺失GPU事件拒绝、
固定路径成功/失败流程、超时回收、PID复用拒绝、Git bundle保留工作树改动及新增路径冲突。
本开发环境无CUDA，不将CPU测试或模拟进程结果冒充新固定入口的RTX5060实测。
--apply 自动完成最终路径的两个实际GPU验证；已有完整项目GPU证据如上。
