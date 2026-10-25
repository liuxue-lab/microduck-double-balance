# 文档增量发布工具

[English](README.md) · **简体中文**

该目录只负责将已完成的 HIL 文档纳入 Git，不包含新的控制或训练逻辑。
运行方式与发布范围见 [文档发布说明](../../../docs/double-balance/publication.md)。

- `install.py` 是离线导入器源码；直接运行只显示帮助，没有内嵌发布数据。
- `build_release.py` 将本次允许的文档与工具文件写入隔离 Git 对象库，生成单文件导入器和增量 bundle。
- `test_install.py` 使用临时完整 Git 仓库验证导入与拒绝路径，使用 Python 标准库。

构建需要固定父提交 `056657aa36158fb81dca6fd2092e525222d6e632`、其目录树和 README 对象。
构建器不检出分支、不推送；输入应为本次审阅后的源码目录，输出应在源码仓库外。

本目录的本地检查命令为：

```bash
python3 -m unittest discover -s experiments/publication/documentation -p 'test_install.py' -v
```

它不连接 Orin、云主机或 GitHub，不运行 MuJoCo 或 PPO。
整个仓库的 CI 保留原流程；该工具的单元检查与仓库 CI 分别执行。
