# Live visualization

[Project](../../../README.md) · [Experiments](../../../experiments/README.md) · **English** · [简体中文](README.zh-CN.md)

The 50 Hz PC–Jetson control loop with an independent interactive 3D viewer.

## Control and visualization

The PC advances native MuJoCo at **200 Hz** and exchanges actions with the Jetson at
**50 Hz**. A separate process displays the saved state stream, keeping rendering
independent of the control and physics work.

Both 60-second runs completed 3,000 control cycles with zero missed 20 ms deadlines.
The maximum complete-cycle times were **15.310 ms** for balance and **14.676 ms** for
disturbance recovery. See the [original measurements](../../../docs/double-balance/evidence/reports/live-control-report.json).

The viewer supports a whole-body view, a head view, following, orbiting, panning
and zooming. Its state stream and window implementation are in
[live_stream.py](package/live_stream.py) and [live_viewer.py](package/live_viewer.py).

## Source map

- [Run controller](controller.py)
- [Implementation](package/)
- [Focused tests](tests/)

[Setup and assets](../../../docs/double-balance/getting-started.md) · [Detailed implementation record](README.txt)
