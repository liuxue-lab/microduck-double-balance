# 原生训练与部署 文档发布后的 CI 修复记录

日期：2026-10-25（Asia/Shanghai）

CI 是推送后的自动检查，负责测试、构建、导出验证和文档链接检查。
下面两次提交均已上传成功；邮件通知的失败发生在 GitHub Actions 检查阶段。

## 已发生的失败

| 提交 | 分支运行 | 标签运行 | 结果 |
| --- | --- | --- | --- |
| 文档发布 `f4ad984e40ee4498e86c510364bb407804c74710` | [40182466565](https://github.com/liuxue-lab/microduck-double-balance/actions/runs/40182466565) | [40182472201](https://github.com/liuxue-lab/microduck-double-balance/actions/runs/40182472201) | pytest 收集阶段 206 个错误 |
| 首次 CI 修复 `e9396b3a17f31b03556d971fd90ec1aff0709010` | [40185853002](https://github.com/liuxue-lab/microduck-double-balance/actions/runs/40185853002) | [40185859902](https://github.com/liuxue-lab/microduck-double-balance/actions/runs/40185859902) | 391 通过、1 跳过、1 失败，另有 4 个 subtest 通过 |

首次失败来自原命令 `uv run --with pytest pytest -q` 未限定收集范围，
将历史版本归档的测试一起导入。它们依赖原主机上的固定运行包、复用测试模块名，
部分还在导入时设置 `MUJOCO_GL=egl`，引起环境污染。
对应报错包括 `Pinned baseline package missing`、`import file mismatch` 和 `eglQueryString`。

首次修复已经解决收集问题，但将 `testpaths` 写入 `pyproject.toml`，
触发了 [Stage09 冻结契约](../audits/head-balance/approved-plan.json)：该契约校验整个文件的 SHA256，
即使只添加测试配置也会失败。这是上一版修复遗漏的约束。

失败用例为 [test_frozen_mdp_prefix_and_old_factories](../../tests/test_posture_contract.py)。
期望的 `pyproject.toml` SHA256 是
`5965e21013e1670884f79fbed14068e63864973a2bcd503821eada88b413783d`，
首次修复后的值为 `fe02f799540be36cba83e321bdabcf21104d0b948e93bf8f7dfe8260938b6394`。
两轮 CI 均在 CPU 测试步骤退出，后面的构建和导出等步骤尚未执行。

## 当前修复

1. 将 [pyproject.toml](../../pyproject.toml) 恢复为冻结版本的原始字节。
2. 用独立的 [pytest.ini](../../pytest.ini) 设置 `testpaths = tests`，保持默认测试范围。
3. 让 [收集回归测试](../../tests/test_ci_test_discovery.py) 使用实际 `pytest.ini`，
   继续验证默认隔离、显式选择阶段以及无范围限制时能复现归档冲突。
4. 保留前两次 CI 失败证据；推送后的完整 CI 结果补记在下文。

沿用首次修复后的 [CI 工作流](../../.github/workflows/ci.yml)：
`uv run --frozen python -m pytest tests/ -q` 运行全部根目录测试，
文档导入器在独立进程中运行标准库 unittest。构建、任务入口、策略导出、ONNX 对照、
硬件生成和文档链接等步骤保持原样。

冻结清单和冻结测试均保持原样，历史阶段测试仍可在相应环境中显式运行。
没有忽略失败、跳过冻结测试、修改模型或调整依赖版本。

## 验证与发布

本次在隔离环境使用 pytest 9.1.1 验证 3 项收集回归测试，并验证 20 项离线导入器测试。
针对失败项，从固定提交获取并核对 Git blob 后，执行原测试方法的未修改断言：
80 个冻结文件的完整 SHA256 和 `mdp.py` 原始前缀哈希均通过。
这项检查只运行原冻结测试方法，不冒充完整项目测试。

增量 bundle 验证包括实际导入、重复导入、旧标签保留和 132 个模拟未跟踪条目的保留。
Markdown 链接按完整 Git 路径表及实际 Markdown 内容检查；其他文件不按完整源码克隆处理。

打包修复时，此环境未安装项目完整 Torch/MuJoCo/BAM 依赖，因此当时尚不能确认完整 CI。
随后推送的 `33e40060921c8da45143ac4daaded345acead4df` 已完成下列全部检查。

本次基线为 `e9396b3a17f31b03556d971fd90ec1aff0709010`，
新注释标签为 `stage-12-hil-ci-fix-v2-20261025`。
文档标签和第一次修复标签保持原样，旧失败记录不被覆盖。
继续采用本地增量 bundle 导入、用户 SSH 推送方式。

固定规范：[本地 SSH 推送](../handoffs/microduck-local-ssh-push-protocol.md)、
[云 GPU 与资料传输](../handoffs/microduck-cloud-gpu-and-transfer-protocol.md)。
本机仓库仍为 `/home/lx/microduck-double-balance/workspace`，下载目录为 `/home/lx/下载`。
当前没有云计算任务，5090 保持已确认关闭；此次不需要 Orin 开机，不重跑训练或 HIL。

## 已完成的完整 CI

| 引用 | Actions | 结果 |
| --- | --- | --- |
| `double-balance`，提交 `33e4006` | [40187216971](https://github.com/liuxue-lab/microduck-double-balance/actions/runs/40187216971) | success，2026-10-25 11:12:33 UTC |
| `stage-12-hil-ci-fix-v2-20261025`，同一提交 | [40187224532](https://github.com/liuxue-lab/microduck-double-balance/actions/runs/40187224532) | success，2026-10-25 11:12:18 UTC |

两次均为 **392 passed、1 skipped、3 warnings，另有 4 个 subtest 通过**；
20 项离线导入器测试、源码与 wheel 构建、任务 CLI、ONNX 导出与对照、
高跷与秋千座生成、349 个相对文档链接检查全部完成。
机器可读回执见 [github-ci-20261025.json](evidence/github-ci-20261025.json)。

这确认了修复提交的完整 CI，不代表真实机器人验收，也不替代后续文档整理提交的检查。
本次主页整理不改动工作流、依赖、冻结测试或运行代码；新提交仍需在用户推送后运行 Actions。
