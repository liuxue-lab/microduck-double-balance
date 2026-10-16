Microduck Stage11 G11 v3 — 剩余正式环境接口批次
2026-10-15（北京时间）

本文件与current-state-v3.json、plan.json是当前状态。previous-state.json和reused-v1/v2中旧数字是历史原件，不覆盖历史记录。
G01–G10 CLOSED_SCOPED；G11 OPEN；10/11＝90.9%。
当前累计670872 completed / 670873 attempted；仍只有1次历史失败native integration。
本包开发/审计/主机验证新增原生积分0，PPO=0、云训练=0。

一、v2返回结论与可复用范围
分支 stage11-g11-formal-environment-v2-transfer-20261015t141720z-e3e8fb
提交 731ea8022f350f4468518e12aef5ad479124da6b
tree d41a97210bc7d40dea64fa14e1a2c6a6a7ee5cf6
ZIP 4900903 bytes，SHA256 8021857bc9b0375f1ea66398f4cc602106c4d1ad3ecc49b17ba4c34657599152
全部10片传输、615成员、135程序文件和92既有helper已核验。219源码及源Git状态前后不变。
3682条WAL完整配对，实际新增1792 completed/1792 attempted；不是新失败积分。
正常控制步10宏步完成：9个在64分辨率接受，1个在128接受；原判据离线重算一致，全部强制精确重放通过。
核对20个宏步边界的8289个Data/owner数组哈希，并比较候选与重放逐微步qpos/qvel/time/stage指纹。
真实环境构造、21/21配置预检、默认RNG隔离、完整/选择/空/非法reset、全局interval相位检查通过。
正常步实际Reward/Termination/Observation各1次，Action.apply和Metrics.substep各10次，61/85维输出及reward有限。
随后step-0-after全owner归档前，roots_for拒绝env._double_balance_tray_site_ids的元组键缓存。
因此正常步的完整步后owner快照缺失；日志formal_steps_completed=0是归档提交计数，不表示实际10宏步未执行。
自动超时/手动终止尚未运行，不能把v2算作G11矩阵全部通过。资源42/42及接触图已释放。
运行239.651秒，WAL落盘8.815秒；这是实测分项，不构成对不同批次的提速倍数证明。

二、v3修复
原函数_double_balance_tray_site_id用(id(robot),站点名)作键，奖励首次调用时才创建缓存。
键名含tray属于旧函数命名；实际查找stage11_head_material_frame，未添加托盘实体。
新增formal_owner_keys.py，仅扩展私有候选的owner快照/恢复和证据导出。
保留原字典对象、原元组键对象、键类型和数值；不跳过缓存、不把运行时元组键改成字符串。
字典证据用带类型且无分隔符冲突的路径编码，附完整typed_keys。1、"1"、(1,)等不合并。
只接受原str/int及有限深度、有限长度的str/int/tuple组合；拒绝bool/float/frozenset等未知键。
accepted/92文件不改字节；显式绑定所有实际引用快照与owner_state的模块，close时恢复绑定。
积分前预热原生站点查询，改变该缓存数值、插入诊断键、重绑定，再验证恢复。
该预检要求全Data/owner、默认RNG、全部native/manager计数不变，不调用奖励或观测做预热。
每宏步边界和公开步后继续全owner覆盖，不能把主机容器测试当成CUDA通过。

三、本次仅运行剩余两工况
使用新建的单native world和fresh reset；不恢复v2缺失的步后owner，不声称三步连续轨迹。
复用v2正常步10宏步、reset矩阵和旧29/29主机/G09/G10证据，不重跑这些批次。
A. 自动超时：episode计数设为499，保留原10秒上限，执行10宏步后检查timeout和原生auto reset。
   验证真实recorder pre_reset/post_reset/post_step顺序、前观测和本步动作、复位后的动作清零与history。
B. 手动终止：独立测试终止项注入True，执行10宏步，检查terminated与pending；再次step必须在动作处理前拒绝。
   最后显式reset[0]，检查pending清除；这是故障注入，不是宣称发生了真实跌落。
两个正式步都使用幅值不超过0.001的14维动作，方向相反；不调用策略、不载入optimizer/runner。
step/local函数interval/local类interval/global interval均在每个正式步真实到期各一次。
固定mjlab没有pre_step钩子；保存显式step-before边界，不宣称测试不存在的原生钩子。
运行成功输出G11_REMAINING_MATRIX_MEASURED_REVIEW_REQUIRED；仍待与v2合并审阅，不自动关闭G11。

四、时序、几何和原阈值
formal_task.py、formal_geometry.py、formal_config.py、formal_command.py和checkpoint_review.py与v2字节一致。
实际top_head_shell三角网格、stage11_head_material_frame、无新托盘；头球半径0.020m，名义间隙0.0005m。
顶球失落基于实际网格投影并集的半径膨胀和真实上包络，不使用AABB/旧矩形。
稳定阈值保持：顶球中心0.012m、间隙误差0.006m、相对速度0.08m/s；下层偏移0.06m、
高度0.20–0.29m、倾角20°、下球速度0.15m/s、hold<=1e-6、float32连续稳定5s；无epsilon补偿。
公开宏步.002s、10倍decimation、控制步.020s；32/64/128原接受阈值、拒绝回滚、每接受宏步精确重放不变。
宏步独占counter/action/BAM/Scene.write/update/substep metrics；BAM每次真实Scene.write各1次，包括reset。
Reward/Termination读取已接受端点的9字段哈希，不再采用stock最后一个子步的派生状态滞后语义。
快照覆盖numeric Data、真实管理器/任务/实体/BAM/传感器、9条私有随机流、别名/Model/默认RNG。
失败恢复的是私有试算起点，不是完整RL step；无自动重试或新增积分档位。
本轮只覆盖名义单世界接口，不声明长时双平衡成功、任意随机重置、多世界正式回归或训练成绩。

五、明确预算
本次2个20ms控制步，共40ms物理时间，跨auto reset分成两个episode；最大20宏步。
每宏步最坏32+64+128+128重放=352次native integration；总上界7040。
GPU forward<=20×1413+80=28340；CPU forward<=8；reset<=3。
BAM/Scene.write<=23；Action.apply/Scene.update/Metrics.substep<=20。
Observation.compute<=4，原4项噪声×4=16；Command.compute<=4；Event.apply<=7。
上界不是实际计数；64即可接受时每宏步160积分，总计仍以真实WAL记账。
全预算见plan.json；没有总墙钟截止；5秒心跳；禁止预估数覆盖实测账本。

六、审计开销与诚实边界
保留每次积分前后fsync WAL及必要native同步；积分外GPU forward和CPU forward另记WAL。
内层kernel计数在内存更新，宏步/退出时checkpoint，progress最多每5秒刷新。
保留一个持久接触排序图；不逐积分创建临时数值图；数值算术仍为G10已验证的eager路径。
实时接触排序oracle、变化约束oracle和数值/状态判据不弱化。
保存所有macro/trial/拒绝/回滚/端点指纹、宏步起止全Data/owner、每微步qpos/qvel/time和stage哈希。
每宏步保存原始接触/约束样本及完整运行摘要；不重复生成G10的每内迭代全量blob。
摘要不足以离线重建所有内部oracle；旧G09/G10原始证据仍保留。没有宣称归档强度完全不变。
旧G10大包不重新扫描；只流式校验小型v1/v2归档和旧started身份。
硬丢进程的未配对attempt按完成区间报告，不推断成功，不删除标记重试。

七、checkpoint和固定环境
checkpoint路径：
/home/lx/microduck-double-balance/artifacts/double-balance-stage08/returned/5d998fc322345102/evidence-20260930T093645-8955/extension/E-20260929/segment-001/checkpoints/update_004000.pt
SHA256 86d55c3703c18fcf4817db49c6e4dcf839b3f5195d68ae2c559db7e222bded97；10073500 bytes。
v1已读94个finite张量；复用其固定报告/schema，本次只校验文件SHA，torch.load=0。
直接resume仍BLOCKED_TASK_AND_TIMING_DIFFER；维度61/85和14动作一致不证明语义/normalizer/recurrent兼容。
不修改/转换/导出旧模型，不加载optimizer/runner，不继承旧任务成功率。
源路径 /home/lx/microduck-double-balance/workspace；Python使用其中.venv/bin/python原符号链接路径。
branch double-balance；HEAD 056eaf2b0c78261f34e3fa581858c8573cab981e。
tracked diff SHA256 b90daa0274c14d95cec9511ec4ac93568bd604e93ce3c60df1bb1c4155005d7e。
Torch2.9.1(+cu128)、MuJoCo3.10.0、mujoco-warp3.8.1、Warp1.12.0、mjlab1.3.0、bam1.0.1。
保留原dirty mdp.py和全部未跟踪诊断文件，不reset/clean，不覆盖注册任务/安装源码。
新程序仅在独立program目录作用于新建env实例；源码219项/HEAD/工作区状态前后核对。

八、保护、协议和剩余任务
新根 /home/lx/microduck-double-balance/artifacts/double-balance-stage11/g11-formal-environment-v3
新started-v1.json中的v1表示标记格式，绝不复用v1/v2目录标记。
源仓库不git add/commit；独立transfer只本地commit；用户SSH push，不自动推送、PR或合并。
保留docs/handoffs/microduck-local-ssh-push-protocol.md和docs/handoffs/microduck-cloud-gpu-and-transfer-protocol.md，原文在provenance/。
remote git@github.com:liuxue-lab/microduck-double-balance.git；下载目录/home/lx/下载。
旧5090最近2026-10-01由用户确认关闭，只是历史状态；本次无新云工作，不启动/停止云机。
本阶段剩余：真实CUDA验证元组键owner修复及两工况；回传独立审计完整新增账本/精确重放/资源释放；
将v1检查点、v2正常步与v3剩余证据合并判定G11范围关闭条件。
此处未执行Torch/CUDA；不将主机测试、push成功或程序候选完成冒称G11已通过。
