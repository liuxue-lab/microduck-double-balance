# Recorded-state replay

[Project](../../../README.md) · [Experiments](../../../experiments/README.md) · **English** · [简体中文](README.zh-CN.md)

Video rendering from saved native states, with synchronized timing data.

## Original demonstrations

Each recording contains **60 seconds**, **3,000 frames**, **1280 × 480** pixels and
**50 fps**. The paired camera views show the whole robot and the upper ball.

- [Balance — original MP4](https://raw.githubusercontent.com/liuxue-lab/microduck-double-balance/81e50d8f933cde8ae02fcaba3fb35d1a519d07e1/experiments/double-balance/media/balance.mp4)
- [Disturbance recovery — original MP4](https://raw.githubusercontent.com/liuxue-lab/microduck-double-balance/81e50d8f933cde8ae02fcaba3fb35d1a519d07e1/experiments/double-balance/media/disturbance-recovery.mp4)

The renderer reads the recorded native model and state arrays, updates geometry,
and aligns the video with the saved control timing. It replays an existing run;
it does not advance physics or request new policy actions.

The [replay report](../../../docs/double-balance/evidence/reports/recording-replay-report.json)
records source identities, frame checks and timing-data verification.

## Source map

- [Run controller](controller.py)
- [Implementation](package/)
- [Focused tests](tests/)

[Setup and assets](../../../docs/double-balance/getting-started.md) · [Detailed implementation record](README.txt)
