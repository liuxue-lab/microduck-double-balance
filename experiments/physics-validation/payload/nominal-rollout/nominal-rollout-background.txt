Microduck Stage11 / G11 formal-environment-v2 候选说明
日期：2026-10-15（北京时间）

本包是下一次本机运行，不是G11已通过的报告。
当前G01–G10=CLOSED_SCOPED，G11=OPEN；10/11=90.9%。
累计669080 completed / 669081 attempted，包含1次历史失败尝试。
开发和主机检查新增原生积分0，PPO=0，云训练=0。

1. 来源与修改范围
完整读取本轮交接TXT，安全解压并核验357项清单。
accepted/保留92项G10程序原字节；host-candidate/保留29/29检查候选及固定源码。
不执行旧worker、旧G09/G10批次或旧29项主机批次。
运行前后核验219项项目/安装源码；保持项目HEAD、分支、tracked diff及工作区状态。
正式候选仅存在本轮独立program目录，绑定新建ManagerBasedRlEnv实例；不改任务注册、不写项目源码。
原静态任务的startup/top_ball_lost阻断未被回写删除；本候选用实际几何检查和新终止定义替换其私有配置。
复用固定G10关闭报告65d20e39f4681202829906091e07da70674f2eb5。
旧started标记/大归档保留，只读G10与v1标记、校验570139字节v1返回包，不重新扫描G10大归档。
本轮固定返回分支stage11-g11-formal-environment-v1-transfer-20261015t134827z-0442dc；
提交5f01cdacd565c0be9c93ccbf592821b1224f47b3；返回SHA256=23b388333f6410dbd52b41fa1e12d9683a5f890da69cda0409f8a2fbc46e706c。
两片ASCII、144成员、126程序成员、219源码前后、源Git状态均已核验；v1新增积分0/0。

2. 新正式头部候选
使用真实top_head_shell三角网格，转换到已审计的stage11_head_material_frame。
名义放置通过实际三角面最近点求半径0.020m球的无穿透中心，间隙0.0005m。
top_ball_lost定义：球心投影距实际网格投影并集>球半径，或低于最近投影处真实上包络一球半径。
不是托盘矩形、AABB或新碰撞托盘；本终止几何是新任务语义候选，不冒称旧终止函数等价。
保留旧数值阈值：顶球中心0.012m、间隙误差0.006m、相对速度0.08m/s；下层偏移0.06m、
高度0.20–0.29m、倾角20°、下球速度0.15m/s、hold<=1e-6、连续稳定5s。
稳定高度使用真实曲面距离；观测/奖励保留已审计的头部参考特征：切平面高度误差不是曲面间隙，
旋转参考系速度不是接触点滑移。保留旧float32稳定计时，不添加epsilon或改为四舍五入计步。
头球材料tau=.003，脚/下球既有支持材料tau=.0025，摩擦与接触参数未改。
仅名义无噪声摆放、单native world；随机训练重置仍阻断。

3. 一次执行的矩阵
A. 真实环境构造与启动校验；9个管理器接口（nominal curriculum为原生NullCurriculumManager）。
B. 在真实Torch定时器上检查已通过主机评审的混合函数/类interval reset修复，然后接入9个私有RNG流。
C. 全reset、选择性[0]reset、空reset、重复/越界/错误布尔长度/非整数选择器；全局interval相位跨reset保留。
D. 正常20ms控制步：10个接受宏步；完整奖励、终止、动作、BAM、观测历史、step/interval回调。
E. 自动超时控制步：将episode计数设置为499，仍保持原10s上限；再运行10宏步后检查timeout和自动reset。
   这是超时边界注入，不是声称真的运行了10秒物理轨迹。
F. 手动终止控制步：单独测试term注入True，运行10宏步，检查真实终止返回和pending；再次step必须在动作处理前拒绝。
   这是接口故障注入，不是声称观察到了真实跌落；随后显式reset[0]。
G. recorder真实pre_reset/post_reset/post_step；auto pre_reset保留上一观测和本步动作，post_reset动作清零。
   固定mjlab源码没有pre_step钩子，本包另存显式step-before快照，不冒称测到了不存在的原生钩子。
H. step、普通local interval、类local interval、global interval均真实到期，要求每正式步各一次。
I. 复用v1已实读94个有限张量的checkpoint检查，仅校验当前checkpoint文件身份；本批torch.load=0。
J. 图/最后一个传感上下文实际资源释放；运行后源文件与Git状态复核；封存并生成SSH归档分支。

4. 接入语义与回滚范围
完整复用已通过主机检查的step AST候选：仅把10次循环内的完整6条语句替换为唯一宏步所有者。
每宏步只递增一次counter、apply_action、Scene.write、BAM、Scene.update与substep metrics。
Scene.write在复位时也会推进BAM，所以整个矩阵预算BAM=34，不错误地写成30。
公开时间2ms，decimation10，正式step20ms。32/64/128原阈值与拒绝规则不变；每个接受宏步强制精确重放。
选择accepted_endpoint时序：奖励/终止读已接受末端，区别于stock的一子步滞后；不声明旧checkpoint语义等价。
奖励/终止前核对qpos/qvel/time/xpos/xquat/site_xpos/site_xmat/cvel/sensordata端点哈希。
宏步保存全部numeric Data、显式真实管理器/任务/BAM/实体/传感器所有者和9流；旧严格别名/Model/默认RNG核验保留。
动作处理/公共counter/BAM发生在私有试算快照之前；失败恢复的是试算起点，不声称回滚整个RL step。
失败或硬中断停止，不恢复后重试；证据保留供离线审核。
仅选择性[0]，不冒称单native nworld=2或任意部分世界回归。

5. 预算（上界不是已执行量）
真实控制步3；宏步最多30；真实时间共60ms（跨自动复位拆成两个episode）。
每宏步最坏32+64+128+128重放=352次native integration；最多10560。
每宏步GPU forward上界1413，另留80次构造外/reset/外层刷新余量；总上界42470。
CPU forward<=8，native reset<=4，Scene.write/BAM<=34，Action.apply/Scene.update/Metrics.substep<=30。
候选64即可通过时，每宏步实际160积分；仍以真实attempted/completed记账，提前失败可更少。
全部细项见plan.json；没有墙钟截止；5秒只是心跳轮询。
累计账本只加实际native integration API调用；forward、图、kernel计数分别记录。

6. 审计开销调整与证据边界
保留每次积分调用前后同步和fsync WAL；嵌套kernel/图计数留在内存，宏步/退出时持久化。
额外CPU forward及积分之外的GPU forward也记WAL。
不再逐内核刷progress和fsync；progress每5秒更新，无全量逐内核JSON。
取消临时数值图反复capture，使用G10已经逐字节比较通过的eager算术；只保留1个持久接触排序图。
每次接触16字段/全容量排序oracle、变化约束列表核验和必要同步仍执行；每宏步保存一个原始接触/约束样本及全程hash摘要。
保留每个宏步起点和提交前完整Data/owner原始数据、所有trial/拒绝/回滚/端点指纹、每次微步qpos/qvel/time和stage哈希。
本批不保存每次内迭代全Contact/Constraint原始blob，属于G11接口审计的新归档范围；G10原始证据仍是旧范围依据。
不能由本批hash摘要重新推导每次内部oracle全过程；不能称存档强度完全不变。
硬丢进程时，无配对完成的积分attempt按completed区间[n-1,n]报告；部分WAL尾段保守增加attempt上界，禁止猜测完成。
嵌套kernel计数在正常退出时精确；硬丢失只保留最后checkpoint与每积分上界。
operation-profile.json保存实际耗时，WAL另记写入/同步时间；没有提前承诺提速倍数或运行分钟数。

7. 实际checkpoint
/home/lx/microduck-double-balance/artifacts/double-balance-stage08/returned/5d998fc322345102/evidence-20260930T093645-8955/extension/E-20260929/segment-001/checkpoints/update_004000.pt
SHA256=86d55c3703c18fcf4817db49c6e4dcf839b3f5195d68ae2c559db7e222bded97
来源：固定056eaf2的Stage10交接，旧Stage08 E托盘模型；本包附原文。
v1已以CPU weights_only=True实读10073500字节、94个张量，全部finite；严格键映射无改名。
v2复用固定提交中的检查清单和schema，不重复torch.load；只流式校验当前文件SHA。
当前文件或证据身份不匹配即停止，避免混用检查点；不替换文件、不丢未知键。
保存旧托盘配置与新头部配置的特征/动作/奖励/终止/时间清单，以及新任务解析后的61/85维与14关节顺序/scale/offset/clip/BAM延迟。
旧checkpoint训练配置尚未逐字段重建，维度相同不证明特征、normalizer或循环记忆兼容。
本轮直接resume结论固定为BLOCKED_TASK_AND_TIMING_DIFFER；后续warm-start需独立明确normalizer/recurrent/optimizer/counter策略。
不改写旧模型，不转换/导出，不继承旧成功率作为新任务成绩。

8. 固定路径与保护
项目/home/lx/microduck-double-balance/workspace；branch double-balance；HEAD 056eaf2b0c78261f34e3fa581858c8573cab981e。
tracked diff SHA256=b90daa0274c14d95cec9511ec4ac93568bd604e93ce3c60df1bb1c4155005d7e。
解释器使用项目.venv/bin/python原路径，不resolve到系统Python。
新根目录/home/lx/microduck-double-balance/artifacts/double-balance-stage11/g11-formal-environment-v2。
新started-v1.json只创建一次；未开始/已开始/失败不混淆。禁止删除该标记以重跑。
新transfer固定为上述新根目录/transfer；独立新Git分支；只本地commit，不push、不merge、不创建PR。
主项目无git add/commit/reset/clean操作；旧归档和旧started不改。
--describe仅验内嵌包并展示计划；--status只读进度；--package-only只封存现有结果，积分0。

9. 保留协议与云状态
docs/handoffs/microduck-local-ssh-push-protocol.md
docs/handoffs/microduck-cloud-gpu-and-transfer-protocol.md
两份固定原文放在provenance/；后续交接必须带上。
SSH remote=git@github.com:liuxue-lab/microduck-double-balance.git。
正常git push成功输出已经足够，不要求反复跑远端全量核验。
旧5090最近由用户于2026-10-01确认关闭；这是历史状态，本轮未查询电源，也没有云工作。
不启动/停止云机、不重跑旧Stage06、无需新关机动作。

10. 验收与下一步
此处环境没有Torch/MuJoCo/Warp/CUDA，无法声称真实环境矩阵已通过。
已做新增主机检查、静态编译、内嵌包/传输/单次保护验证；29/29旧检查证据直接复用。
运行输出G11_FORMAL_MATRIX_MEASURED_REVIEW_REQUIRED也不自动关闭G11。
回传后独立核对：身份/源码/账本/完整3步矩阵/每宏步精确重放/真实资源释放/语义清单/检查点复用身份。
通过后再按实际证据确定G11剩余项及关闭条件；仍不自动进入PPO、云训练或规模吞吐验收。

11. v2修复和新增运行前检查
v1在6.828秒处、环境构造之前被候选断言拒绝：错误假设play模式关闭整个actor噪声组。
CPU/GPU forward及native integrations均为0；只读checkpoint检查完成。
继承actor组enable_corruption=True，critic=False；原噪声范围逐项保留：
base_ang_vel ±0.03（3维）、projected_gravity ±0.01（3维）、joint_pos ±0.001（14维）、joint_vel ±0.25（14维）。
只在实际ObservationManager深拷贝后的四个noise对象上绑定私有obs_noise/0采样，保留noise→clip→scale→delay/history顺序。
参数缓存预热后纳入真实noise owner快照；默认CPU/CUDA RNG跨公开reset/step必须不变。
Observation_noise预算改正为4项×最多7次compute=28；这是计数上界修正，不改物理积分预算或数值阈值。
命令继承rel_forward_envs=0.2，原先要求0也是错误假设。v2编译固定原生_resample_command方法，
仅将该方法的uniform采样接入command_value流；保留分桶标志、前向clamp与body/world参考顺序。
原生_update_command仍执行standing=1的最终清零；原turn-in-place扩展p=0，原生即为no-op。
构造时一次性IMU挂载偏差由原生观测shape probe建立，其缓存保留并纳入env owner；不伪称全配置无随机性。
configuration-preflight.json在环境构造之前一次检查cadence、actions、events、commands、obs/noise/delay及spawn。
构造后再核对真实manager roster、BAM与delay、实际owner和默认RNG，之后才进入新物理矩阵。
v2的目录是新建g11-formal-environment-v2；内部started-v1.json名称表示标记格式版本，绝不复用/删除旧目录标记。
保留v1所有原件，本包reused-v1/携带已核验报告和检查点检查证据。
本批仍是首个待执行的G11正式物理矩阵；不是重新跑v1已完成的物理积分，因为v1实际为零。
本轮NumPy对固定原生噪声公式及命令AST作回归验证；这不替代本机Torch/CUDA验证。
