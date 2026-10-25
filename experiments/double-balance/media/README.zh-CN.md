# 演示媒体

[项目首页](../../../README.zh-CN.md) · [实验说明](../README.zh-CN.md) · [English](README.md)

完整录像包含同步的全身视角和头顶球近景，规格均为 **60 秒、1280×480、50 fps**。

| 演示 | 完整原始录像 | 页面预览 |
| --- | --- | --- |
| 平衡 | [balance.mp4](balance.mp4) | [12 秒 WebP](balance-demo.webp) |
| 扰动恢复 | [disturbance-recovery.mp4](disturbance-recovery.mp4) | [12 秒 WebP](disturbance-demo.webp) |

完整 MP4 与原始录像的文件字节完全一致，保留原始构图。公开文件名与来源文件名、
SHA256 的对应关系见 [original-videos.json](original-videos.json)。

页面动图为第 6–18 秒的裁边预览，规格为 960×320、20 fps。点击预览即可打开完整录像。
此前制作的 12 秒 MP4 片段仍可通过 [balance-demo.mp4](balance-demo.mp4) 和
[disturbance-demo.mp4](disturbance-demo.mp4) 访问。

## 来源

录像呈现 PC–Jetson HIL 运行中保存的状态，来源见
[录像报告](../../../docs/double-balance/evidence/reports/recording-replay-report.json)。
性能表使用另一轮带实时查看器的
[计时报告](../../../docs/double-balance/evidence/reports/live-control-report.json)。

[demo-provenance.json](demo-provenance.json) 记录预览的裁剪、时间区间、编码设置及输出哈希。
使用原始文件名、Python 3 和 FFmpeg 可重新生成预览：

```bash
python3 experiments/double-balance/media/build_demos.py --source-dir /path/to/recordings
```

完整录像通过原文件复制发布，不由上述预览生成命令重新编码。
