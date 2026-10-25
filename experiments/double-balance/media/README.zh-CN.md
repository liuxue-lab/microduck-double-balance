# 演示媒体

[项目首页](../../../README.zh-CN.md) · [实验介绍](../README.zh-CN.md) · [English](README.md)

完整演示视频为 **2560 × 1440、50 fps、60 秒**。同步双视角分别展示全身与
头顶球接触细节，重新调整的构图铺满画面，无顶部黑边。

| 演示 | 完整高清 MP4 | 原生 PNG |
| --- | --- | --- |
| 自然平衡 | [下载 balance.mp4](https://raw.githubusercontent.com/liuxue-lab/microduck-double-balance/refs/heads/double-balance/experiments/double-balance/media/balance.mp4) | [第 10 秒画面](balance-poster.png) |
| 抗扰恢复 | [下载 disturbance-recovery.mp4](https://raw.githubusercontent.com/liuxue-lab/microduck-double-balance/refs/heads/double-balance/experiments/double-balance/media/disturbance-recovery.mp4) | [第 10 秒画面](disturbance-recovery-poster.png) |

视频直接使用已保存的原生模型和状态轨迹重新渲染，上传后的 MP4 与 PNG 均保持
导出文件的原始字节。[hd-media.json](hd-media.json) 记录规格和哈希；
[导出报告](hd-export-report.json) 包含全轨迹构图、逐帧边缘检查、来源身份和完整解码结果。

## 页面预览

现有 12 秒 WebP 动图保留原布局：960 × 320、20 fps，截取此前录像的第 6–18 秒。
裁切方式与哈希见 [demo-provenance.json](demo-provenance.json)。此前的 12 秒 MP4 片段也保留：
[自然平衡](https://raw.githubusercontent.com/liuxue-lab/microduck-double-balance/81e50d8f933cde8ae02fcaba3fb35d1a519d07e1/experiments/double-balance/media/balance-demo.mp4) · [抗扰恢复](https://raw.githubusercontent.com/liuxue-lab/microduck-double-balance/81e50d8f933cde8ae02fcaba3fb35d1a519d07e1/experiments/double-balance/media/disturbance-demo.mp4)。

## 来源记录

1440p 视频来自[原生 HIL 运行](../../../docs/double-balance/evidence/reports/headless-control-report.json)
保存的状态。性能表采用另一次[实时窗口运行的计时报告](../../../docs/double-balance/evidence/reports/live-control-report.json)。

[original-videos.json](original-videos.json) 记录此前 1280 × 480 导出的身份，原文件保留在源码快照中：
[自然平衡](https://raw.githubusercontent.com/liuxue-lab/microduck-double-balance/81e50d8f933cde8ae02fcaba3fb35d1a519d07e1/experiments/double-balance/media/balance.mp4) · [抗扰恢复](https://raw.githubusercontent.com/liuxue-lab/microduck-double-balance/81e50d8f933cde8ae02fcaba3fb35d1a519d07e1/experiments/double-balance/media/disturbance-recovery.mp4)。
原始[回放报告](../../../docs/double-balance/evidence/reports/recording-replay-report.json)与页面动图继续保留。
