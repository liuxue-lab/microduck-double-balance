# Microduck 云 GPU 关机提醒与文件传输长期规范

生效：2026-10-01 05:06:09，Asia/Shanghai。用户要求后续所有交接持续保留。

用户原话：“以后不用就提醒我关机 并且 写进之后的所有交接文档 上传下载文件 都通过之前的ssh的方式 这个也写进交接文档 和把常见的错误和小快捷点也写进去 并一直保持”。

本规范与 [本地提交与 SSH 推送固定规范](microduck-local-ssh-push-protocol.md) 一起长期执行。它约束后续协作流程，不授权新增训练、改任务、改变验收或实例电源操作。

## 1. 云 GPU 不再需要时立即提醒

当云端训练、评估和其他必要计算结束，必要打包与传输完成，回传文件已在本机通过 SHA256 校验时，主动告诉用户：“云 GPU 现在不再需要，请在平台手动关机并保留所需数据”。不要等本机录像、视频目视审查、文档编写、提交或阶段标签全部完成才提醒。

GPU 利用率为 0 不是单独的关机依据；此时可能仍在初始化、CPU 评估、打包或传输。先结合阶段状态和正在进行的任务判断。

- 尚有必要云端工作时，指出具体剩余工作；完成后立即提醒。
- 只需本机 RTX 5060 推理、MuJoCo 仿真、录像或文档时，不为这些工作保留云 GPU 计算需求。
- 提醒用户手动关机，不代为启动、停止、关机，不再执行已经失败并被禁止的旧关机守护。
- 把“云端计算结束”“文件已回传”“已经提醒”“用户确认关机”分别记录。没有实际确认就写 `UNCONFIRMED`。
- 无 SSH 连接、日志停止或训练进程退出，均不能单独证明实例关机。
- 云端还有未回传的完整轨迹时必须说明。手动关机与删除实例、释放存储不是同一项操作；不得把后两者附加到关机提醒中。
- 作业预算墙钟和实例实际计费时长分别记录；作业完成或预算守护退出不等于平台停止计费。

这是一条每阶段协作中的主动提醒规则。当前没有云平台状态连接器，不宣称能够在用户离开对话后自动监测实例空闲，也不以文档规则冒充已经建立后台通知任务。

## 2. 固定位置、身份和传输方式

| 项目 | 固定规则 |
|---|---|
| 本机代码仓库 | `/home/lx/microduck-double-balance/workspace` |
| 本机下载目录 | `/home/lx/下载` |
| 本机产物目录 | `/home/lx/microduck-double-balance/artifacts/`，放在源码仓库外 |
| 正式代码分支 | `double-balance` |
| GitHub SSH 远端 | `git@github.com:liuxue-lab/microduck-double-balance.git` |
| 阶段代码与注释标签 | 本机创建，用户通过既有 SSH 身份推送 |
| 文件传输 | 默认经 GitHub Git SSH 上传和取文件，使用专用临时传输分支和仓库外缓存 |
| 云主机连接 | 使用用户当阶段提供的 SSH 地址和端口，不能把旧实例地址当作永久有效 |

Stage 09 本次成功回传的实际路径是“云端 Git SSH push → GitHub → 本机 HTTPS 下载并校验”。它不是全程 SSH。用户本次补充要求后，未来生成的上传及取文件脚本默认均使用 Git SSH；已经完成且已校验的回传不重做。

本规范中的 Git SSH 用于远端认证和 Git 数据交换。浏览器下载助手交付的小脚本、以及用户主动在对话中上传审查包，是不同的交付环节，不需要因此申请 GitHub 密码或重新登录。

## 3. 源码、模型和评估资料分开传输

源码通过 `double-balance` 的既有 SSH 路径分发。云端只接收获批并已提交的代码版本，不用临时修改的云代码覆盖正式分支。隔离环境产生的阶段代码按本地提交规范导出增量 bundle，在本机导入并创建阶段提交及注释标签。

已批准回传的模型、评估和视频使用阶段专用传输分支，例如 Stage 09 的 `stage09-review-20261001`。在独立传输目录中打包，不在训练源码目录里创建资料提交。传输分支不得合并进 `double-balance`，传输成功也不表示阶段已经验收完成。

后续传输脚本必须：

1. 先写明需要的文件范围、大小、SHA256 和尚未包含的文件；清单不等于已回传。
2. 对重复检查点按哈希去重；保留必要的恢复断点、进度、优化器来源、账本和回执。
3. 每个压缩包使用项目约定的 49 MiB 上限；更大时按模型或文件分包。不要未经说明把全部大型原始轨迹放进 Git。
4. 云端使用已经成功的 SSH 身份向专用传输分支正常 push，不 force push，不操作正式阶段标签。
5. 本机在源码仓库外建立独立 Git SSH 缓存，只取指定传输分支。固定完整提交 SHA 后，从相应 Git 对象导出文件到临时路径；按清单校验后才标记回传完成。不要从可变分支上的混合时刻下载一组文件。
6. 解包前后检查路径和哈希，保留既有文件。发生 SHA 不符时停止使用该文件，不训练、不覆盖已验证模型、不修改期望哈希来通过检查。
7. 为传输失败提供仅重试上传或仅重试下载的命令，不重新启动训练、不清空进度或预算。
8. 成功 push 的终端输出足以确认正常推送，不重复完整远端核验；本机文件哈希校验仍必须完成。

不要默认切回慢速的大文件本机↔云端 SCP，不引入 gh 网页登录、PAT 或 GitHub 密码。确需其他传输方式时说明具体限制和替代方法，由用户决定；不能静默改变长期约定。

## 4. Stage 09 已验证路径与证据

这些是本阶段事实，不是自动适用于新实例的固定配置。

```text
CloudSSH=ssh -p 41381 root@connect.bjb1.seetacloud.com
CloudProject=/root/autodl-tmp/microduck-double-balance
CloudArtifacts=/root/autodl-tmp/microduck-double-balance/artifacts/double-balance-stage09
CloudTransferDirectory=/root/autodl-tmp/stage09-review-transfer-20261001
CloudTransferKey=/root/.ssh/id_ed25519_microduck_stage09
TransferBranch=stage09-review-20261001
TransferCommit=e7730090daaf6c65fc4d688417d39f9380f0c691
CheckpointManifestSHA256=9d95a48e015a8870727ca664fd667274b026ff7698a973ae6c9c16e14afcd347
PilotReviewSHA256=5fe9295b66131be7a067d3380b8fbe05a8bac1cd025415bd9fbfaec4044bbe62
LocalReturn=/home/lx/microduck-double-balance/artifacts/double-balance-stage09/returned-pilot-20261001
LocalReturnReceipt=/home/lx/microduck-double-balance/artifacts/double-balance-stage09/returned-pilot-20261001/return-verified.json
```

2026-10-01 04:53:32 用户提供终端证据：push 成功，`Stage09CheckpointReturn=PASS`，`Stage09VerifiedCheckpoints=13`。归档包含每组 0/250/500 步及 F2 的 150 步恢复模型，共 13 个唯一检查点。无需重做这些传输。

完整云端 head-trace/trace 文件没有全部回传；审查包含报告、选定回合数据和完整轨迹哈希。它们仍需在交接里标为留存云端，不能写成“云端所有文件均已备份”。

2026-10-01 05:03:28 已在对话中提醒用户手动关闭 5090。到本规范创建时，用户尚未确认实际关机，状态为 `UNCONFIRMED`。后续确认应追加时间与证据。

## 5. 常见错误及固定处理

| 现象 | 解释与处理 |
|---|---|
| 点击 `.sha256` 后只有一行文字 | 正常，这是校验文本；将其中值与本机文件 SHA256 比较，不需要另找“下载按钮” |
| GitHub 要求用户名、密码或 PAT | 检查是否误用了 HTTPS Git 远端；恢复既有 `git@github.com:...` SSH 地址，按本地推送规范处理 |
| 云主机 SSH 提示 root 密码 | 这是云主机身份验证，不是 GitHub 密码；不要混淆两套连接 |
| `Permission denied (publickey)` | 核对当前机器的 SSH 身份和既有 agent/仓库权限；先复用已成功的 key，不默认重建一套 |
| `gh auth login` 报 `unknown flag` | Stage 09 云机旧版 gh 2.4.0 不支持当时使用的参数；既有 Git SSH 回传已成功，后续不再走这条登录支路 |
| gh 登录出现 `unexpected EOF`、`i/o timeout` | Stage 09 的设备登录接口访问失败；网页能打开不能证明该接口正常。沿用已成功的 Git SSH，不反复更换登录参数 |
| `Stage09Uploading=GitHub SSH artifact branch` 暂时无新行 | 仅表示开始上传，既不是完成也不能单凭静默认定卡死；只读查看现有 git/ssh 进程或日志，不并发启动第二份上传 |
| 大文件 SCP 速度很低 | Stage 09 已观察到这一问题；后续采用上述 GitHub Git SSH 中转，不反复从头 SCP |
| `Everything up-to-date` 但缺预期文件 | 核对本地是否已经提交或导入了对应资料；不要把旧分支无变化当成新文件已上传 |
| `non-fast-forward`、标签冲突 | 停止 force 操作，按本地 SSH 规范做一次必要的分歧检查；保留两边提交 |
| 下载/解包 SHA 不匹配 | 保留已验证文件和错误记录，仅重取失败包；不能改校验值、跳过校验或重新训练 |
| 下载文件名变成 `文件名(1)` | 使用实际文件路径或明确重命名到命令指定的名称；不要把旧文件当作刚下载的新版本 |
| `Address ... changed` / SSH 主机密钥变化 | 先核对平台实例和主机身份；不关闭主机密钥验证，也不盲目删除全部 known_hosts |
| 训练状态仍写 RUNNING，但无进程 | 结合退出日志、检查点与预算判断是否为断电后遗留状态；不据此认定仍在训练，也不直接重新开训 |
| 欠费中断后保存进度少于已计次数 | 保留已计消耗；恢复检查点、Adam/LR/进度及预算需单独审计，不能把丢失更新当成免费重试 |
| 关闭监控窗口后担心训练停止 | `tail -f` 或只读监控中的 Ctrl+C 只退出该监控；前台训练/录像主程序中的 Ctrl+C 则可能中断作业，先辨认当前终端 |
| 本机视频长时间未显示结果 | 看脚本每 15 秒心跳和该任务日志；首次初始化可能较慢，不并发重开同一脚本 |

原有 Git 分支、bundle、标签与 SSH agent 的完整报错处理仍见 [本地提交与 SSH 推送固定规范](microduck-local-ssh-push-protocol.md)，以后追加案例，不删掉已有解决办法。

## 6. 常用快捷操作

回到固定源码仓库：

```bash
cd /home/lx/microduck-double-balance/workspace
git status --short --branch
```

查看本机 GPU 使用情况（只读，Ctrl+C 退出查看）：

```bash
nvidia-smi -l 2
```

本机计算文件校验值（将示例文件名替换为实际文件）：

```bash
sha256sum /home/lx/下载/pilot.zip
```

Stage 09 视频审查入口：

```bash
python3 /home/lx/下载/microduck-stage09-videos.py
```

仅当此前录像中断时，复制该次日志已经打印的 `Stage09VideoResume=` 后完整命令；它带有准确输出目录。不要新建输出目录来假装恢复，也不要用训练恢复脚本代替视频恢复。

Stage 09 旧回传脚本曾打印 `Stage09DownloadRetry=`，它可仅重试当次未完成下载，但使用的是历史 HTTPS 下载实现。该次回传现在已成功，不要重跑；后续新传输脚本改用本规范的 Git SSH 取文件方式。

代码阶段正常推送仍为：

```bash
cd /home/lx/microduck-double-balance/workspace
git push origin double-balance
```

只有本机对应注释标签确实已建立，才执行 `git push origin stage-XX-complete`，其中 XX 必须替换为实际阶段。正常成功后不重复完整远端核验。

## 7. 所有后续交接必须保留的内容

每份新交接同时引用本规范和本地 SSH 推送规范，并明确填写：

- 当前阶段、范围、已获授权、未解决问题及下一条可执行命令。
- 本机仓库与下载目录、代码分支、阶段提交/标签及正常 push 是否已成功。
- 当前云主机连接信息及其适用阶段；不能无证据沿用旧端口、身份或电源状态。
- 传输分支及完整提交 SHA、必要文件/模型清单、哈希、回传验证状态、未回传的原始资料。
- 剩余云端任务、是否还需要云 GPU、提醒时间、实际关机确认和证据；未确认写 `UNCONFIRMED`。
- 本阶段新增常见错误、已验证解决办法、只读监控/仅重试失败步骤的快捷命令。
- 本机仅代码/仿真/推理/视频，正式训练只能使用另行批准预算的云 GPU；不重跑 Stage 06 smoke。

不得把规则简化成一句“沿用之前方式”而遗漏关键路径、状态和错误处理。历史冻结文档不改写为新事实；新的交接追加最新状态并指出历史材料仅供追溯。
