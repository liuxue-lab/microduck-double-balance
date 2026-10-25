Microduck Stage12：4500与6000模型的确定性比较
日期：2026-10-20；SSH：ssh -p 52763 root@connect.bjb2.seetacloud.com

本轮已完成的事实
原训练6000/6000（含唯一短测5次），约239.0分钟，进程退出码均为0。
用户回传的16:37收尾报告确认5995条正式更新日志，无无效行、无重复更新。
4500与6000两个模型均在云端通过实际文件SHA256核对，不再重复训练或重开原预算。
同样250更新窗口的5秒训练日志指标：4251–4500为11.43%，5751–6000为6.44%。
后期瞬时联合几何稳定仍约93.45%，因此先检查实际策略，不凭奖励上升认定任务成功。
训练日志指标是重置组/日志步的均值，不是严格逐回合成功率。

现在只需运行一个文件
保存 microduck-stage12-evaluate-models-v1-20261020.py 到 /home/lx/下载，运行：
python3 "/home/lx/下载/microduck-stage12-evaluate-models-v1-20261020.py" --evaluate
现有云端5090需要开机；本机仅控制、监看与接收小报告，不运行仿真/PPO。
发送约16 KiB代码，复用云端项目、现有Python依赖、两个检查点与Warp缓存。
不安装依赖、不下载整个项目、不再次执行训练前短测。
首次认证一次，后续复用共享SSH会话；实际断线后重连可能再次提示密码。

固定评估协议
- 候选仅update_004500.pt与update_006000.pt，不临时添加其他模型。
- 每个候选两个新种子20261021、20261022，每种子128环境。
- 每个模型256个首次回合，两个模型共512个；每回合最多500控制步/10秒。
- 最多256000条全环境控制转移（包括为了维持批次运行而重置的已结束环境）；10物理子步/控制步。
  只有首次回合计入成功分母和诊断，第二回合不会重复计数。
- 使用原生训练任务make_cfg，不替换真实头壳、物理求解器、观测、阈值、观测噪声或DR。
  仅把auto_reset关闭以准确读取首次回合终止状态，随后手动重置已结束环境。
- 加载已训练actor/critic与归一化统计；不加载优化器、不更新任何参数；actor明确使用确定性输出。
  两个模型使用同一个完成奖励爬坡的时钟，脚下辅助为0，速度命令为0。
- 在原metrics_manager.compute之后、reset和forward之前读取缓存，避免重复调用有状态的成功计数器。
- 成功：首次回合曾连续250个50Hz控制步满足现有全部几何阈值，且该回合结束时无物理失败。
  timeout允许；即使先满足5秒、随后跌倒/丢球，该回合仍按失败计数，同时报告ever_5s计数。
- 记录每回合最长连续稳定时长、所有阈值的越界次数、稳定段被打断次数、物理失败原因。
- 每个种子核对两个模型的初始身体/球状态、初始观测与已声明模型DR字段的哈希。
  这保证上述初始内容相同；不同终止/重置后，后续噪声序列未必完全相同，不声称逐轨迹精确重放。
- 汇总每个模型的成功数/256及Wilson区间，提供配对胜负；这是模型比较/选择集，并非最终独立留出验收。
  不放宽成功阈值，不把结果写成旧G11严格物理审计通过，不自动宣称任务达标。

范围与保留
此次新增PPO=0。原6000更新预算、started标记、训练代码、模型和全程日志均保持不变。
启动与整个运行期间持有原v1/v2训练启动锁，防止误开训练与评估争抢GPU。
后台独立于SSH；评估总时限30分钟，含四次环境构建，超限最多另留90秒结束清理。
started标记使重复--evaluate只接回；失败保留现场，不自动重跑已完成分组、不重开时限。
复用的Warp缓存可写入新128环境编译缓存；它不是训练证据文件。
对原训练标记、预算、配置、进度、完整metrics日志和两个模型在评估前后核对哈希。
未更改任何云端电源设置，无自动关机。云计算和必要回传结束后应手动关机并保留存储。

接回与输出
断网或Ctrl+C后接回：
python3 "/home/lx/下载/microduck-stage12-evaluate-models-v1-20261020.py" --watch
只查询一次/收小报告：
python3 "/home/lx/下载/microduck-stage12-evaluate-models-v1-20261020.py" --collect
只需发回NATIVE_EVALUATION_REPORT_SAVED=后的JSON。
本机报告目录：/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-evaluation-v1-reports
云端评估目录：/root/autodl-tmp/microduck-double-balance/artifacts/double-balance-stage12/native-evaluation-v1
逐回合group-*.json和模型本体保留云端；返回报告只含摘要、配对结果和哈希。

固定模型身份
4500：df03202acf010680b63d3277ec879f63e2d4808e5d1286653923a314f125ac6a
6000：f89a16643e949bc0e79ae6be77c5e12635cd7538b5bfe0bdef9a1cb37c802ee3
每个10036060字节；原训练包identity：dd6d1e1e697bcaa348dbdd92dd8c7b30ea31ae9c32599338106859ae7657523a
收尾报告SHA256：8d53efa1e0dd0da8936ae3922bbf805d1084cb6be1c8236c92867b4210d6f5ef

离线检查范围
28项检查通过：实际Observer在NumPy替身上的249/250边界、单步越界清零、首次回合掩码、
物理失败覆盖、诊断统计与配对计数；不可变指纹；确定性调用；嵌入代码一致性；状态命令；
重复启动不重跑；中断保留；PID复用检查；原训练控制器未变和已返回模型身份。
没有在助手环境执行真实Torch/MuJoCo/CUDA评估。实际GPU结果等待用户执行上述入口。
接口核对使用交接包冻结的mjlab/rsl-rl源码和官方mjlab v1.3.0 EventManager源码。
官方参考：https://github.com/mujocolab/mjlab/blob/v1.3.0/src/mjlab/managers/event_manager.py

后续约定
用户13:34要求“这一轮先这样，后续记得恢复”：后续正式训练恢复4096并行环境，
同时实施已准备的重复同步合并优化。这里的128仅是固定评估批次，不改变该约定。
先依据两个模型的确定性表现及越界诊断决定下一轮起点/奖励调整；不会自动开启新一轮训练。
已有训练和评估证据不重跑；新训练也不静默扩展原6000次/8小时预算。
本次没有Git提交/tag/push。长期规范继续保留：
docs/handoffs/microduck-local-ssh-push-protocol.md
docs/handoffs/microduck-cloud-gpu-and-transfer-protocol.md
