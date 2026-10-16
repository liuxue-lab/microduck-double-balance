Microduck Stage11 · G11 主机契约与候选修补
日期：2026-10-15

本包的结果
29/29 项主机检查通过。其中包含 6 种环境外层重置/终止分支、7 种语义不兼容反例。
验证执行了固定 mjlab 源码中 step、reset、_reset_idx、EventManager.apply/reset 和 runner.load 的方法体。
运行时使用明确标注的 NumPy 替身；没有导入 Torch、Warp、MuJoCo、mjlab 或 PPO，没有运行真实环境或物理积分。
新增原生积分＝0；实际累计仍为 667608 completed / 667609 attempted，历史失败尝试保留。
G01–G09 CLOSED_SCOPED；G10/G11 OPEN；固定门进度 9/11＝81.8%。

一、已复现并修补的事件重置缺陷
固定版本 EventManager.reset 遍历 _mode_class_term_cfgs["interval"]，却用其子集索引访问按全部 interval 事件排列的定时器。
当普通函数事件排在类事件前面时，类事件的重置会改写前一个事件的定时器，自身定时器保持旧值；普通函数事件也未被正确重采样。
当全局事件排在局部类事件前面，选择 env_ids=[1] 时，还可能索引到仅含一项的全局定时器并抛 IndexError。
event_manager_reset.patch 只把这处枚举对象改为全部 interval 配置，保持类事件 reset 回调、全局时钟保留和未选中世界保持不变。
检查包含：混合事件、仅普通函数、全量/选择性/空重置、数组所有者保持、全局到期仅触发一次、局部到期仅触发选中世界、最强 recompute 仅执行一次、reset 最小步数与首次触发。
此补丁是候选文件，尚未安装到你的项目或虚拟环境。当前 G10 脚本仍依赖原始固定源码身份。

二、环境外层接入候选
source_contracts.build_step_candidate 保留固定 ManagerBasedRlEnv.step 的整个外层控制流，只把循环内的完整六条子步语句替换为一个明确的宏步回调。
宏步回调负责一次公共子步计数、动作/BAM 应用、Scene 写入、已接受的物理宏步、Scene 更新和子步指标；外层仍执行 10 个宏步，正式步长仍为 20 ms。
候选检查回调的计数所有权，避免仅替换 sim.step 后产生重复动作、重复计数或重复 Scene 更新。
回归核对奖励/终止、观测历史、记录器 pre/post reset、自动重置、手动重置阻塞、command dt=0 的 reset，以及 step/interval 事件只按正式步提交一次。
该模块不自动绑定真实环境，不创建正式任务，不移除 startup 或 top_ball_lost 的发布阻断，也未绑定已验收的真实物理驱动。

仍需正式裁定的时序差异：stock step 在奖励/终止之后才 forward，读取的是滞后一子步的派生数据；自适应接受端点已完成刷新。
主机反例用合成宏步代数 9 与 10 展示了该差异，数值不是物理测量。候选要求显式声明 accepted_endpoint 语义；这不代表与旧奖励或旧 checkpoint 已经等价。

三、严格 checkpoint 键迁移候选
checkpoint_candidate.migrate_checkpoint_keys 接受已经检查过的内存映射，不反序列化、不保存、不加载实际 checkpoint。
它保留原张量对象与训练元数据，拒绝未知旧键、新旧模型字典混合、分布参数覆盖和 std/log_std 双重参数化；仅替换键的开头命名空间。
主机回归也执行了原 runner.load 方法体，用内存替身复现未知键被丢弃、当前 actor 字典被覆盖的情况。
compare_semantics 单独比较 actor/critic 特征顺序、含义、历史展开、normalizer、动作关节/缩放/偏置/延迟、步长、奖励终止与派生状态时序。
相同维度不会绕过语义差异；清单相同仍要求实际来源核验，实际 checkpoint 兼容性继续为未验证。
本包没有访问实际训练 checkpoint，不能用于宣称完整训练恢复、优化器状态兼容或 formal rollout 恢复已通过。

四、包内文件
source_contracts.py：固定源码检查、最小事件修补和环境外层候选。
checkpoint_candidate.py：严格键迁移与语义差异比较。
validate_host.py：29 项可复现的主机回归。
event_manager_reset.patch：供后续正式源码接入评审的单行补丁。
regression-review.json：实际检查结果、反例和明确结论边界。
pinned-source/ 与 source-pins.json：5 份已回传源码原件及哈希；用户源文件没有被修改。
INVENTORY.json：包内文件字节数和 SHA256。

如需本地重现（可选，不涉及 GPU）：
解压后使用项目虚拟环境运行包内 g11-candidate/validate_host.py，例如：
/home/lx/microduck-double-balance/workspace/.venv/bin/python -I -B /home/lx/下载/stage11-g11-host-contracts/g11-candidate/validate_host.py
它只在解压目录更新 regression-review.json；需要现有 NumPy，不安装依赖，不运行项目入口。

五、与当前 G10 批次的衔接
刚交付的 microduck-stage11-g10-native-blocks-recovery.py 保持原版本。
SHA256：20f9281ceb3028f1aceee3802a876823dc1dcdfdff2865ac1947740fbd2ac129
协议 g10-native-blocks-endpoint-recovery-v3，最多 1472 次积分、5918 次 GPU forward；这些仍是待本机运行/回传的上限，不计入实际完成账本。
截至本包生成，没有收到该批新的回传分支；已有 cache-resume 回传 6/6 原始比较结论继续复用。
后续先独立复核 G10 新批次原始结果并做覆盖评审，再把本包候选接入正式环境，完成真实奖励/终止/重置/动作延迟/记录器/事件及 checkpoint 回归。
PPO＝0，云训练＝0；本包没有执行推送、创建 PR、合并或物理重试。
