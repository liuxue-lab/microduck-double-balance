# Demonstration media

[Project](../../../README.md) · [Experiment](../README.md) · [简体中文](README.zh-CN.md)

The full demonstrations are **2560 × 1440, 50 fps, 60 seconds**. Two synchronized
camera views show the whole robot and the head–ball contact, with edge-to-edge framing.

| Demonstration | Full-resolution MP4 | Native PNG |
| --- | --- | --- |
| Balance | [Download balance.mp4](https://raw.githubusercontent.com/liuxue-lab/microduck-double-balance/refs/heads/double-balance/experiments/double-balance/media/balance.mp4) | [10-second frame](balance-poster.png) |
| Disturbance recovery | [Download disturbance-recovery.mp4](https://raw.githubusercontent.com/liuxue-lab/microduck-double-balance/refs/heads/double-balance/experiments/double-balance/media/disturbance-recovery.mp4) | [10-second frame](disturbance-recovery-poster.png) |

The videos were rendered directly from the saved native model and recorded states.
The uploaded MP4 and PNG files are copied here byte-for-byte. Specifications and
checksums are recorded in [hd-media.json](hd-media.json); the
[export report](hd-export-report.json) includes the camera fit, all-frame edge checks,
source identities and full video decode results.

## Inline previews

The existing 12-second WebP animations retain their layout: 960 × 320, 20 fps,
covering seconds 6–18 of the earlier recordings. Their crops and hashes are in
[demo-provenance.json](demo-provenance.json). The original 12-second MP4 excerpts
are also available: [balance](https://raw.githubusercontent.com/liuxue-lab/microduck-double-balance/81e50d8f933cde8ae02fcaba3fb35d1a519d07e1/experiments/double-balance/media/balance-demo.mp4) · [disturbance recovery](https://raw.githubusercontent.com/liuxue-lab/microduck-double-balance/81e50d8f933cde8ae02fcaba3fb35d1a519d07e1/experiments/double-balance/media/disturbance-demo.mp4).

## Source records

The 1440p videos visualize the recorded states from the
[native HIL run](../../../docs/double-balance/evidence/reports/headless-control-report.json).
The performance table uses the separate
[live-viewer timing report](../../../docs/double-balance/evidence/reports/live-control-report.json).

[original-videos.json](original-videos.json) documents the earlier 1280 × 480
exports preserved in the source snapshot:
[balance](https://raw.githubusercontent.com/liuxue-lab/microduck-double-balance/81e50d8f933cde8ae02fcaba3fb35d1a519d07e1/experiments/double-balance/media/balance.mp4) · [disturbance recovery](https://raw.githubusercontent.com/liuxue-lab/microduck-double-balance/81e50d8f933cde8ae02fcaba3fb35d1a519d07e1/experiments/double-balance/media/disturbance-recovery.mp4).
Their original [replay report](../../../docs/double-balance/evidence/reports/recording-replay-report.json)
and the WebP previews remain available.
