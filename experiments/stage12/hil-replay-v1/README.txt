Microduck Stage12 — 已验收 HIL 数据的离线回放

这一步读取 native-mujoco-v3/runs/20261025T044453Z-d194fd47。
该运行已完成两组各60秒、共6000次控制周期的20ms验收：
quiet-2407 最大14.518736ms，regression-2500 最大14.471036ms，均零超时。
首周期计入，未丢弃周期、未平移期限；状态与v2逐位一致。
原报告SHA256：513c9add8b0029dfb039efba38fc1d1390a97ed310766b8c184d42d87abfd079。

本入口只生成这次记录的回放；它不重新执行物理积分、策略、ROS通信或训练。
不连接Orin或云GPU。5090保持用户确认的关闭状态。
物理机器人、电机接入与真机演示已从交付范围移除。
不会重新导出、量化或剪枝固定FP32模型，也不更改已经通过的v3运行代码。

用法（先下载单文件入口到 /home/lx/下载）：

python3 "/home/lx/下载/microduck-stage12-hil-replay-v1-20261025.py" --apply

--apply 从内嵌增量Git bundle导入仅新增 experiments/stage12/hil-replay-v1 的提交，
从固定Git对象安装运行包，然后生成视频。保持主终端开启。
不需要网络、Orin或新的密码。使用现有项目 .venv/bin/python 和现有 ffmpeg。
依次探测已有 EGL / 图形桌面 / OSMesa 后端；显式 MUJOCO_GL 设置优先。
不会自动安装依赖或改变系统配置。单次生成上限15分钟，持续输出帧数进度。

产物：
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-hil-replay-v1/report.json
报告内 index 指向本轮 index.html；可用以下命令在浏览器打开：
python3 "/home/lx/下载/microduck-stage12-hil-replay-v1-20261025.py" --open

每组一个完整60秒 H.264 MP4，1280x480、50fps、3000帧。
左全身/下球、右头部/上球，两个相机跟随记录位置。
视频逐一使用正式采样1..3000；初始状态单独保存PNG。
HTML页面嵌入全量耗时数据，不依赖HTTP服务器、外部字体或CDN。
支持点击曲线定位视频、逐帧前后、0.25/0.5/1/2倍速。
完整周期、环境步、ROS往返均显示；嵌套区间不能直接相加。
初始、中间、最终PNG另存，编码后完整解码核对帧数。

原始CSV、NPZ、MJB、报告等读取前后验证SHA256；CSV全量重新核对20ms
固定释放、期限、计数、均值/p95/p99/最大值。NPZ要求3001个有限float64状态，
顺序和时间严格匹配。绘图只更新几何位置，不调用mj_step、策略或接触力求解。
没有BAM内部状态和完整积分状态，不能将此NPZ称为可恢复动态仿真的检查点。
50Hz回放不能代替200Hz子步的接触审计，也不新宣称任何硬件/总体成功率验收。

运行、状态与异常处理：
--run：复用已有报告，不自动重复生成。
--status：一次查看；--watch：每2秒查看，Ctrl+C只结束观察。
--stop：停止本次所属绘图进程，保留全部文件。
--retry-run：用户显式新建一次绘图尝试；旧尝试保留，仍不重跑HIL。
Ctrl+C、关闭主终端或进程异常均保留报告/日志；已有RUNNING记录会核验进程身份。
重新启动不会自行覆盖或清空结果。错误摘要直接包含在小report.json中。

v3控制入口仍是（已有通过记录时 --run 仅显示结果）：
python3 /home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-mujoco-v3/microduck-stage12-native-mujoco-v3.py --status
其他原有 --run/--watch/--stop/--retry-run 语义保留；本步骤不触发它们。

Git保持原SSH远端。脚本不自动推送：
git -C /home/lx/microduck-double-balance/workspace push origin double-balance

本机修改、暂存、未跟踪/忽略文件与分歧均保留；不reset/stash/clean/强推。
模型、完整轨迹、视频不进入代码分支。本目录只有源码、必要测试与验收摘要。
遵循 docs/handoffs/microduck-local-ssh-push-protocol.md 与
docs/handoffs/microduck-cloud-gpu-and-transfer-protocol.md。

实现依据：MuJoCo 3.10.0 Python Renderer、mj_kinematics/mj_comPos/mj_camlight。
https://mujoco.readthedocs.io/en/stable/python.html
https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html
本地验证使用合成MJB/轨迹检验回放与编码，不冒充用户的实际双球视频。
实际双球画面需在本机读取已保存的116MB MJB与轨迹后生成。
