Microduck Stage12 最后一次 6 小时抗扰微调
2026-10-24，Asia/Shanghai

用户授权
“那就继续训练 4个小时 这是最后一次训练 奖励调好 开始训练后 我就去睡了”，随后改为“6小时吧”。
最新云 RTX5090：ssh -p 32185 root@connect.bjb2.seetacloud.com（用户已克隆实例）。
本次是独立新预算，旧 v3/v4、Stage13/14 的预算与结果保持关闭。
尚未在此工作区运行真实 GPU 仿真或 PPO；本次交付是启动准备，不是训练成功证明。

一次粘贴启动
将 microduck-stage12-push10-final6h-v1-20261024.py 下载到 /home/lx/下载。
在本机 Linux 终端一次粘贴以下三行，&& 确保前一步成功才继续：
python3 "/home/lx/下载/microduck-stage12-push10-final6h-v1-20261024.py" --install-code &&
git -C "/home/lx/microduck-double-balance/workspace" push origin double-balance &&
python3 "/home/lx/下载/microduck-stage12-push10-final6h-v1-20261024.py" --train --ssh "ssh -p 32185 root@connect.bjb2.seetacloud.com"

安装只增加 experiments/stage12/push10-final6h-v1 子目录，保留本机已有修改；由用户现有 Git SSH 身份推送。
启动脚本复用共享 SSH 会话；首次云主机认证后不逐次询问密码。
观察到 READY_TO_LEAVE 才表示新配置烟测/ONNX检查通过且正式抗扰 PPO 已完成至少一次更新。
只有 SSH 连接成功、初始化或 smoke 字样，并不表示正式训练已开始。
READY_TO_LEAVE 后可 Ctrl+C 退出监控、关闭终端或关闭本机；云端独立继续。
重复 --train 只读取/监控已启动任务，不能重新消费预算。

时间安排与停止
总墙钟上限 21600 秒（6 小时），从云端预检成功、不可变启动账本落盘时起算。
前 17100 秒（约4小时45分钟）包括初始化、64env×5更新的新配置烟测，以及4096env正式训练。
最后最多 4500 秒（约1小时15分钟）用于候选选模和独立留出验证，均计入6小时。
最多10005次新PPO尝试（5次smoke+10000保险上限），墙钟为主要上限；不会因一次评估通过提前结束正式训练。
最多90秒额外清理只用于保存完整状态和退出，不能再做物理积分或更新。
每25更新或120秒落盘；每个完整optimizer更新保留一致CPU快照，到时即使有半回合采样也保存最后完整模型。
失败/中断尝试仍记已消耗，不退款、不重新抽样、不自动续训。
烟测、数值、电机或环境错误会保留证据并停止，不能保证本轮一定训练达标。
到时自动停止计算、打包并尝试通过既有Git SSH回传新结果。
云实例不会自动关机，停止训练后实例仍可能计费；必要结果本地SHA256核验后应手动关机并保留存储。
目前云电源状态未独立核实，不执行任何实例开关机操作。

之后查看
python3 "/home/lx/下载/microduck-stage12-push10-final6h-v1-20261024.py" --watch --ssh "ssh -p 32185 root@connect.bjb2.seetacloud.com"
只读单次状态：同一脚本 --status（同样可带 --ssh）
手动请求停止：同一脚本 --stop（不会开启新预算）
回传失败后仅重试结果回传：同一脚本 --return；不会重新训练。
终端输出 FINAL6H_REPORT_SAVED= 指向的JSON可先上传分析；完整回传通过 FINAL6H_LOCAL_RETURN_VERIFIED= 确认。
来源工程 /home/lx/microduck-double-balance/workspace，分支 double-balance，远端 git@github.com:liuxue-lab/microduck-double-balance.git。
本机新任务输出在 /home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-push10-final6h-v1-client。
云端输出在 /root/autodl-tmp/microduck-double-balance/artifacts/double-balance-stage12/native-push10-final6h-v1。

固定父模型
headroom_000506.pt，累计6010次更新，10037598字节。
SHA256 ca2a82acbe939169367272ffb50242bbfc1b4a39c06ed186bfc4cf3009ed4130。
云路径 native-headroom-constraint-v3/run/checkpoints/headroom_000506.pt；父训练package ID 0bb3633f91cea2c119e0c63eb4d5f06f17a80819636500015dafdd8368d12857。
本机已核验回传提交 67842a986fb33309334379a883dfcc533a4ddd4e。
旧预算506/506/1505已结束，其中1次完成但未保存更新仍在旧账本；新预算从0独立开始。
父模型和旧证据不覆盖，新 checkpoint 使用新目录。
原始冻结源码365文件、native_task、speed_task、运行库版本都在云预检核对，不升级或重新安装依赖。

为什么继续训练
旧模型在 force_seed=2026102410 的一次测试通过。
之后新64组验证在首组 force_seed=2026102500 失败：45.076s首次左髋yaw名义越界，45.36s top_ball_lost。
三髋最大超限约1.204°/2.194°/2.051°；head_roll 23.339°本身仍在25°内。
该批只运行1组，其余63组未运行。不能说64组都失败，也不能说已具备可靠抗扰。
两份实测报告保留在 reference/，不计入新验收。

训练设计
- 继续现有actor/critic/optimizer与归一化器，保持原common_step_counter成熟度，不从零训练。
- 61D actor/85D critic、14 BAM舵机、原头壳接触/球体/几何/动作链、50Hz控制/500Hz物理保持一致。
- 不增加动作滤波、不夹物理状态、不修改名义关节限位、不要求回到起点。
- 每个episode 60s；每env独立整数年龄，reset后重新从0计时。
- 25%回合无扰动以保持已有能力，75%回合施规定外力。
- 前10%训练窗口为5%自重，10%到25%为7.5%，后75%为10%；阶段变化只影响新episode。
- 每次脉冲独立均匀随机水平方向；每episode t=2,4,...58s施0.1s，最多29次/1450施力物理步。
- 真实F=fraction*m_robot*g作用于trunk_base惯性质心，当前机器人mass排除上下两球。
- 在scene/action写入之后、实际积分之前施加；只撤销自身力，未知外力报错保留。
- 使用私有随机生成器，训练/选模/留出随机种子分离。
- PPO保留原4096env、24step rollout、5epochs/4minibatches；初始LR3e-5，最后1/3训练窗降1e-5；clip=.15，entropy=.003。

奖励调整
保留原双球主奖励、速度匹配、动作平滑以及 head_roll 的5°余量和0.08s外向预测约束。
其余13关节增加直接测得qpos的3°限位余量成本及0.08s预测外向成本；内区和朝内恢复不收相应成本。
位置权重-2，外向预测权重-.25；失败的left_hip_yaw/right_hip_yaw/right_hip_roll重要度2，其余为.25。
physical_failure对top_ball_lost或fell只罚一次：raw100×weight(-1)×dt(.02)=-2，不罚time_out。
所有成本非负配负权重，云启动核对实际reward manager按dt缩放且termination先于reward；训练日志核查惩罚不变正奖励。
这些是针对已有失败的保守修改，不能把奖励配置等同于物理硬保证。

自动选模与验收
在训练窗口约1/3、2/3、末尾保存最多3个不同模型候选。
每候选selection固定3回合：无扰动2407、已见失败2500回归、全新2700扰动序列。
三回合全部符合60s双球保持、14关节/电机边界及实际外力要求才有资格；漂移不作失败。
合格候选按实际余量等同一既定规则排序；锁定一个sha后才开始独立heldout。
heldout最多18回合：2个无扰动初始种子2406/2407，16个新force seed2026102800..2815（环境2407）。
heldout首失败停止，不换模型、不换种子补成通过；无人合格只输出诊断最优，不写通过。
selection+heldout共最多27回合，全部受6小时截止约束；时间不够写未完成，不冒充失败或通过。
此处有限留出验证不等于此前64组已完成、不证明所有随机序列、一般域随机化或实机成功。
最终模型ONNX导出与部署一致性、G11严格接触和实机标定仍属后续；旧smoke.onnx不能冒充最终模型。

工程与交接
只向原源码提交 ddfc350a8f07d92b0ca6991b371145e0fe81b15d 增加新实验目录，提供增量Git bundle。
遵循 docs/handoffs/microduck-local-ssh-push-protocol.md。
云GPU与结果传输遵循 docs/handoffs/microduck-cloud-gpu-and-transfer-protocol.md。
模型/大资料走独立artifact分支，不混入double-balance源码分支；只返回新候选/最新模型与必要证据。
训练日志分片；中间恢复模型和TensorBoard大资料可仅保留云端SHA索引，不宣称全部存储已备份。
Stage13/14轨迹任务继续取消；不借用或刷新旧预算。
CHECKS.log/CHECKS.json记录离线验证；真实GPU烟测包含在本次预算内并由启动任务自动完成。

