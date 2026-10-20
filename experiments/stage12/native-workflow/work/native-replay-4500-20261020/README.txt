Microduck Stage12 / selected 4500 / 本机有限视觉回放 v1
日期：2026-10-20

下载根目录的 microduck-stage12-replay-4500-v1-20261020.py 到 /home/lx/下载。
本机Ubuntu运行：
python3 "/home/lx/下载/microduck-stage12-replay-4500-v1-20261020.py" --record

本机4500已取回且SHA核验通过；365份源码匹配，复用既有项目.venv。
这是确定性策略推理+视频，新增PPO=0，不需要SSH或5090，不自动安装依赖。
不运行已完成的训练、smoke、512回合比较或模型下载。

最多两段单环境首回合，每段500步/10秒，种子20261021和20261022。
保留训练任务、噪声与随机化，冻结策略和归一化。首回合终止即停止，不补录成功样本。
最多1000控制步/10000物理子步；两段只是视觉示例，不是云端128环境的精确轨迹回放。
不会重新计算或替代4500的212/256=82.81%评估成绩，也不宣称旧G11物理验收或实机通过。

每段MP4左侧全身/脚下球，右侧头壳/上球；两视角来自同一GPU状态。
使用mjlab1.3.0官方OffscreenRenderer、ViewerConfig公开接口；CPU渲染数据仅forward，不额外积分。
固定镜头不是追踪镜头；主体离开特写范围时以全身为准。
1280×480、25fps，libx264；已有ffmpeg或已安装imageio_ffmpeg二进制，缺失则停并保存报告。
输出每段初始/中间/最后PNG、逐段统计JSON，以及可在本机浏览器直接打开的index.html。
MP4完成后会解码验证。实际仿真和渲染仍需用户在5060执行，助手没有GPU验证结果。

输出目录：
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-replay-4500-v1
输入模型：
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-selected-4500/checkpoints/update_004500.pt
模型SHA256：df03202acf010680b63d3277ec879f63e2d4808e5d1286653923a314f125ac6a

终端每20秒显示阶段。首次初始化计入15分钟原始墙钟时限；停止清理最多约20秒。
这是本机前台任务：保持本机开机及终端。Ctrl+C会停止该录像，保留文件，不重置时限。
重复--record不会另起录像。失败不自动重试。不要删除started.json来重开预算。
另开终端可只读监看：
python3 "/home/lx/下载/microduck-stage12-replay-4500-v1-20261020.py" --watch
Ctrl+C退出这个只读监看不停止原启动终端的录像。

执行结束回传LOCAL_VIDEO_REPORT_SAVED=后的JSON以及成功生成的两段MP4；不回传模型。
如果失败，只发小JSON，其中含worker/ffmpeg日志尾部；不要重跑训练或云评估。
--watch和重复--record可再次收集报告，不延长预算。

已完成26项离线代码检查，详细结果见offline-review.json。
原Observer和fingerprint从已成功完成的评估代码逐字复制；源码来源、固定版本API参考均在包中。
进程停止测试使用替身；助手环境没有实际5060/EGL验证，不宣称GPU录像已通过。
构建：python3 work/native-replay-4500-20261020/build.py
离线核对：python3 work/native-replay-4500-20261020/checks.py
核对过程不导入Torch/MuJoCo、不加载模型、不运行仿真/PPO。

官方固定版本参考：
https://github.com/mujocolab/mjlab/blob/v1.3.0/src/mjlab/viewer/viewer_config.py
https://github.com/mujocolab/mjlab/blob/v1.3.0/src/mjlab/viewer/offscreen_renderer.py
https://github.com/mujocolab/mjlab/blob/v1.3.0/src/mjlab/scripts/play.py
完整参考源码位于reference/。渲染不修改已取回任务文件、原模型或仓库源码。

状态与后续：本轮2048环境训练6000/6000完成，独立比较512/512完成，4500模型及必要证据已取回。
原始初始指纹不配对的问题仍保留，不将独立批次差异直接定为过拟合的因果证据。
下一轮正式训练恢复4096环境；性能候选未部署，不自动追加本轮预算。
5090已无必要任务：请手动关机并保留存储；当前电源状态UNCONFIRMED。
6000模型、完整云端日志与旧证据仍留云端，不宣称全部备份。
最后使用的云连接：ssh -p 52763 root@connect.bjb2.seetacloud.com；本录像不使用它。
本轮未创建git提交/tag/push；代码分支double-balance，既有SSH远端：
git@github.com:liuxue-lab/microduck-double-balance.git

必须继续保留的长期交接规范：
docs/handoffs/microduck-local-ssh-push-protocol.md
docs/handoffs/microduck-cloud-gpu-and-transfer-protocol.md
本机仓库：/home/lx/microduck-double-balance/workspace；下载目录：/home/lx/下载。
