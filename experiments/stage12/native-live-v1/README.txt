Microduck Stage12 — 运行中的实时三维窗口

本次新增 native-live-v1。原 native-mujoco-v3 与 hil-replay-v1 的代码和记录保持独立。
沿用冻结 final6h_005304、FP32 ONNX、原生MuJoCo 5ms、20ms控制、原BAM/2ms延迟时钟。
不训练，不量化剪枝，不连接5090，不接入真实机器人电机。5090保持用户确认的关闭状态。

请在笔记本图形桌面的终端运行，连接Orin网线并保持终端与窗口开启：
python3 "/home/lx/下载/microduck-stage12-native-live-v1-20261025.py" --apply

先校验原通过记录，创建并预热实时窗口，再进行两组各60秒、各3000周期的Orin闭环。
没有桌面、可用图形后端或独立第三物理CPU核时，在正式控制前保留错误报告并停止。
使用现有项目Python/MuJoCo/glfw，不联网安装或升级依赖。

窗口：1 全身/双球；2 头部/上球；F 跟随；鼠标左键旋转、右键平移、滚轮缩放。
Esc或关闭窗口只结束显示，控制和原始记录继续；停止HIL使用主终端Ctrl+C或 --stop。
没有暂停控制、鼠标施力、复位或动作写入接口。请保持窗口可见以验收实际显示。
窗口标明状态时间、最新已完成周期耗时与超时计数；最后一帧的周期可能尚未结束，最终以报告为准。

显示独立进程、独立物理CPU核，目标25fps，关闭阴影反射，显示进程降低自身优先级。
控制进程以50Hz向私有非阻塞Unix数据报通道发送状态。缓冲区满只跳过显示消息。
所有状态原始记录、所有控制周期与审计保持完整；状态发送计入state_capture_ms及完整20ms期限。
窗口只从状态计算几何位置，不执行物理步进、策略或ROS。渲染与绘图不进入控制进程。

新一轮必须重新满足6000个固定20ms期限，首周期照计，不跳步、不平移期限、不自动重试。
另外每组至少500个实际显示的新状态，起点<=第10步、终点>=2990步，最大状态年龄500ms、
相邻所见状态最大跨度25控制步；显示统计和实时验收分开记录。
完整qpos/qvel/ctrl和时间索引必须与已通过v3逐位一致。
旧v3的最大14.519/14.471ms不能冒充新窗口模式的实测结果。

固定入口：
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-live-v1/microduck-stage12-native-live-v1.py
--status 一次状态；--watch 每2秒进度；--stop 停止本轮并保留日志。
--run 复用已有报告；只有显式 --retry-run 才新建一次闭环。每轮保存到独立时间戳目录。
报告：/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-live-v1/report.json
只有 NATIVE_LIVE_REALTIME_PASSED 且 live_hil_acceptance=true 才表示本次显示与实时验收都通过。

本机仓库：/home/lx/microduck-double-balance/workspace，分支 double-balance。
单文件内嵌增量Git bundle，只新增 experiments/stage12/native-live-v1。
保留本机暂存/未暂存/未跟踪文件；冲突或分歧保留停止；不reset、stash、clean或自动推送。
用户仍使用原SSH发布：git -C /home/lx/microduck-double-balance/workspace push origin double-balance
遵循 docs/handoffs/microduck-local-ssh-push-protocol.md 与
 docs/handoffs/microduck-cloud-gpu-and-transfer-protocol.md。本次未生成交接文档。

实现使用MuJoCo经典mjv/mjr可视化接口，窗口独立于控制与动力学：
https://mujoco.readthedocs.io/en/stable/programming/visualization.html
测试使用合成35/32/14维模型与测试显示服务器，不替代用户实际Orin/图形桌面的验收。
