# Demonstration media

[Project](../../../README.md) · [Experiment](../README.md) · [简体中文](README.zh-CN.md)

The complete recordings show two synchronized views: the whole robot and a close-up
of the head and upper ball. Each MP4 is 60 seconds at 1280×480 and 50 frames per second.

| Demonstration | Download original MP4 | Inline preview |
| --- | --- | --- |
| Balance | [balance.mp4](https://raw.githubusercontent.com/liuxue-lab/microduck-double-balance/81e50d8f933cde8ae02fcaba3fb35d1a519d07e1/experiments/double-balance/media/balance.mp4) | [12-second WebP](balance-demo.webp) |
| Disturbance recovery | [disturbance-recovery.mp4](https://raw.githubusercontent.com/liuxue-lab/microduck-double-balance/81e50d8f933cde8ae02fcaba3fb35d1a519d07e1/experiments/double-balance/media/disturbance-recovery.mp4) | [12-second WebP](disturbance-demo.webp) |

The complete MP4 files retain the exact bytes of the original recordings, including
the original framing. Their descriptive filenames are mapped to the source filenames
and SHA256 hashes in [original-videos.json](original-videos.json).

The inline animations are cropped 960×320, 20 fps excerpts covering seconds 6–18.
Use the original MP4 links above to download the full recordings for your video player. The earlier 12-second MP4 excerpts
remain available as [balance-demo.mp4](https://raw.githubusercontent.com/liuxue-lab/microduck-double-balance/81e50d8f933cde8ae02fcaba3fb35d1a519d07e1/experiments/double-balance/media/balance-demo.mp4) and
[disturbance-demo.mp4](https://raw.githubusercontent.com/liuxue-lab/microduck-double-balance/81e50d8f933cde8ae02fcaba3fb35d1a519d07e1/experiments/double-balance/media/disturbance-demo.mp4).

## Provenance

The recordings visualize saved states from the PC–Jetson HIL run identified in the
[replay report](../../../docs/double-balance/evidence/reports/recording-replay-report.json).
The performance table uses the separate
[live-viewer timing report](../../../docs/double-balance/evidence/reports/live-control-report.json).

[demo-provenance.json](demo-provenance.json) records the preview crops, time interval,
encoding settings and output hashes. To reproduce those previews from the original
source filenames, use Python 3 and FFmpeg:

```bash
python3 experiments/double-balance/media/build_demos.py --source-dir /path/to/recordings
```

The complete recordings are published by copying the source files; they are not
outputs of that preview-building command.
