# Microduck 本地提交与 SSH 推送固定规范

状态：长期有效，后续所有阶段沿用

本规范用于 Microduck 双重平衡项目的阶段封口。除非用户明确修改，后续 Codex
对话、桌面端任务和交接文档都必须沿用本规范，不再临时改用 HTTPS、网页登录或
新的推送流程。

## 1. 固定位置与远端

```text
LocalRepository=/home/lx/microduck-double-balance/workspace
DownloadDirectory=/home/lx/下载
Branch=double-balance
OriginSSH=git@github.com:liuxue-lab/microduck-double-balance.git
RemoteRepository=https://github.com/liuxue-lab/microduck-double-balance.git
```

检查点等大文件固定放在仓库外的 artifacts 目录，不提交到 Git：

```text
SourceCheckpoint=/home/lx/microduck-double-balance/artifacts/basketball-release-6d8f74b/checkpoint.pt
Stage05Checkpoint=/home/lx/microduck-double-balance/artifacts/double-balance-stage05/checkpoint.pt
```

## 2. 默认方式：桌面端 Codex 直接操作本地仓库

桌面端 Codex 直接打开：

```text
/home/lx/microduck-double-balance/workspace
```

Codex 在本地完成代码、测试、文档、提交和注释标签，但停在推送之前。用户使用
既有 SSH 身份执行：

```bash
cd /home/lx/microduck-double-balance/workspace

git push origin double-balance
git push origin stage-XX-complete
```

其中 `stage-XX-complete` 替换为当前阶段标签，例如
`stage-05-complete`、`stage-06-complete`。

不要为了推送切换到 HTTPS，不要要求用户提供 GitHub 密码或 PAT，也不要通过
网页重建提交或标签。

## 3. 网页或隔离环境的回退方式：Git bundle

只有当 Codex 不在上述本地仓库中，而是在网页端或隔离工作区中完成提交时，才
生成增量 Git bundle。文件名固定为：

```text
microduck-stage-XX-from-stage-YY.bundle
```

用户把下载文件保存到：

```text
/home/lx/下载
```

以 Stage 05 为例：

```text
/home/lx/下载/microduck-stage-05-from-stage-04.bundle
```

导入时使用临时远端跟踪引用，避免直接 fetch 到当前已检出的分支：

```bash
cd /home/lx/microduck-double-balance/workspace

git bundle verify /home/lx/下载/microduck-stage-05-from-stage-04.bundle

git fetch /home/lx/下载/microduck-stage-05-from-stage-04.bundle \
  refs/heads/double-balance:refs/remotes/stage05/double-balance \
  refs/tags/stage-05-complete:refs/tags/stage-05-complete

git switch double-balance
git merge --ff-only refs/remotes/stage05/double-balance

git push origin double-balance
git push origin stage-05-complete
```

后续阶段只替换 bundle 文件名、临时引用名和阶段标签，不重新设计流程。

## 4. 正常成功时不重复完整验证

以下输出足以确认正常推送：

```text
To github.com:liuxue-lab/microduck-double-balance.git
   <old>..<new>  double-balance -> double-balance
```

以及新标签或标签已存在：

```text
* [new tag] stage-XX-complete -> stage-XX-complete
```

出现上述正常结果后，不再要求用户每个阶段重复执行完整的 fetch、分歧计数、标签
解引用和工作树验收。Codex 也不应在没有异常的情况下反复核验同一远端状态。

只在以下情况执行远端核验：

1. push 报错；
2. 出现 non-fast-forward、标签冲突或 SHA 不一致；
3. bundle 导入结果不确定；
4. 用户明确要求核验。

需要时使用一次最小核验：

```bash
git fetch origin double-balance --tags
git rev-list --left-right --count origin/double-balance...double-balance
git rev-parse double-balance
git rev-parse origin/double-balance
git rev-parse 'stage-XX-complete^{}'
git status --short --branch
```

正常期望为分歧 `0 0`、本地分支等于远端分支、阶段标签解引用到阶段提交、工作树
干净。

## 5. 常见问题与固定处理

| 现象 | 原因 | 处理 |
|---|---|---|
| `Everything up-to-date`，但期望的新提交没有上传 | 本地分支仍停在上一阶段，常见于尚未导入 bundle | 先检查 `git rev-parse HEAD`，再导入 bundle 并 `git merge --ff-only` |
| `src refspec stage-XX-complete does not match any` | 本地不存在该标签 | 检查 `git tag -l 'stage-XX-complete'`；隔离环境产物先从 bundle 导入标签 |
| `unknown revision` 或“有歧义的参数” | 提交或标签尚未导入本地 | 不继续 push；先导入 bundle 或切到包含该对象的正确仓库 |
| GitHub 要求用户名或密码 | `origin` 被设置成 HTTPS | 执行 `git remote set-url origin git@github.com:liuxue-lab/microduck-double-balance.git` |
| `Permission denied (publickey)` | SSH key 未加载或 GitHub 未配置公钥 | 运行 `eval "$(ssh-agent -s)"`、`ssh-add ~/.ssh/id_ed25519`，再用 `ssh -T git@github.com` 检查 |
| 每次 push 都要求输入一段口令 | 输入的是 SSH 私钥口令，不是 GitHub 密码；agent 未缓存 | 登录会话中先运行 `ssh-add ~/.ssh/id_ed25519`；需要跨终端缓存时使用系统 keyring/keychain |
| `non-fast-forward` | 远端出现本地没有的新提交 | 禁止 force push；先停止并审计 `git fetch origin double-balance` 后的分歧 |
| `would clobber existing tag` 或标签 SHA 不同 | 同名标签已经指向另一对象 | 禁止强制覆盖；比较本地/远端标签并停止处理 |
| `couldn't find remote ref` | 分支名或标签名写错，或尚未推送 | 使用固定分支 `double-balance` 和本阶段准确标签 |
| 找不到 bundle | 下载位置或文件名不一致 | 先检查 `/home/lx/下载`；bundle 固定使用 `microduck-stage-XX-from-stage-YY.bundle` 命名 |
| 无法 fetch 到已检出的分支 | fetch 目标直接写成了 `refs/heads/double-balance` | 按本规范 fetch 到 `refs/remotes/stageXX/double-balance`，再执行 fast-forward merge |

## 6. 禁止事项

1. 不因普通推送改用 HTTPS、网页登录、PAT 或网页端重建提交。
2. 不对 `double-balance` 或阶段标签执行 force push。
3. 不在 bundle 尚未导入时把 `Everything up-to-date` 当成阶段已上传。
4. 不提交 `.pt` 检查点或其他大模型文件。
5. 正常推送成功后不重复要求用户执行全套验收。
6. 后续交接文档不得删除本规范的引用。
