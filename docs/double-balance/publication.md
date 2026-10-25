# GitHub 首次文档发布记录

> **历史记录：此轮已经完成。** 文档提交 `f4ad984` 及两次后续 CI 修复已经推送；
> `33e4006` 的分支和标签完整检查均已通过，见 [CI 记录](ci-followup.md)。
> 下文保留当时的导入步骤以便审计，不是当前维护者需要重复执行的指令。
> 本次页面整理使用新的独立交付脚本；展示设置见 [repository-settings.md](repository-settings.md)。

本次提交更新根 README，新增双球实验入口、本目录的报告和证据，以及独立的文档导入工具。
父提交固定为 `056657aa36158fb81dca6fd2092e525222d6e632`，分支仍为 `double-balance`。
采用注释标签 `stage-12-hil-docs-20261025`；旧标签保持原样。
此标签标记文档归档版本，不表示严格接触、真机或完整备份已验收。

## 原笔记本操作

将 `microduck-stage12-github-publication-20261025.py` 下载到 `/home/lx/下载`，执行：

```bash
python3 /home/lx/下载/microduck-stage12-github-publication-20261025.py --apply
```

该文件内嵌增量 Git bundle。导入器核对分支、父提交、SSH 远端、已跟踪文件状态和新增路径冲突，
验证包内容后快进到文档提交，并安装该注释标签。它保留现有不冲突的未跟踪条目，
不执行 `git add .`、reset、clean 或 stash，也不会启动旧 HIL 发布器。
重复执行已成功的导入会验证当前文件并返回已安装。

导入成功后，使用既有 SSH 身份推送：

```bash
git -C /home/lx/microduck-double-balance/workspace push origin double-balance
git -C /home/lx/microduck-double-balance/workspace push origin stage-12-hil-docs-20261025
```

导入器只处理本机 Git，不访问 GitHub、不连接 Orin，也不启动训练、仿真或录像。
当前没有剩余云计算任务，5090 保持已确认关闭状态；文档操作不需要 Orin 开机。
推送需以用户终端的实际成功输出为准，正常成功后不要求重复远端验证。

如只需预览本机条件，可使用 `--check`；该选项不写入仓库。
不带参数会显示帮助；`--status` 显示预期版本和本机版本。
导入回执和解出的 bundle 位于源码仓库外：

```text
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/github-publication-v1/
```

## 发生冲突时

| 提示 | 处理 |
| --- | --- |
| 分支、HEAD 或 SSH 远端不匹配 | 保留现场，核对实际 checkout；不要自动换远端或强制回退 |
| 已跟踪文件有改动 | 先审查自己的修改，再决定如何提交或处理 |
| 新文档路径已存在或被符号链接占用 | 比较文件来源；导入器不会覆盖它们 |
| 同名标签指向不同对象 | 保留两边证据，不强制覆盖标签 |
| 推送 non-fast-forward | 按既有协议检查分歧，不 force push |

## 手动增量包方式

审阅 ZIP 同时提供独立 `microduck-stage-12-hil-docs-from-native-live-v1.bundle`，
仍遵循已有增量 bundle 流程。优先使用单文件导入器完成本地前置核对；
需要人工审阅时可在仓库内使用 `git bundle verify`、`git bundle list-heads` 和 `git show`。
不要将审阅 ZIP 整体解压覆盖源码仓库，也不要把模型资料分支合并到源码分支。

长期规范：[本地提交与 SSH 推送](../handoffs/microduck-local-ssh-push-protocol.md)、
[云 GPU 与资料传输](../handoffs/microduck-cloud-gpu-and-transfer-protocol.md)。
