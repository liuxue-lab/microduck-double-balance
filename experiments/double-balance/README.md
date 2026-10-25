# Double Balance

[Home](../../README.md) · **English** · [简体中文](README.zh-CN.md)

A recurrent policy coordinates Microduck's 14 joints to balance on a basketball
while supporting a second ball directly on the head shell.
The PC–Jetson hardware-in-the-loop system combines native MuJoCo dynamics with
FP32 ONNX inference on a Jetson Orin Nano Super.

## Balance

[![Microduck maintaining double-ball balance](media/balance-demo.webp)](media/balance.mp4)

12-second preview · [Original recording · 60 s · 50 fps](media/balance.mp4)

Whole-body coordination keeps the robot and upper ball balanced as the lower basketball moves.

## Disturbance recovery

[![Microduck maintaining balance under repeated horizontal disturbances](media/disturbance-demo.webp)](media/disturbance-recovery.mp4)

12-second preview · [Original recording · 60 s · 50 fps](media/disturbance-recovery.mp4)

Repeated horizontal force pulses test the policy's response to changes in support and body motion.
Each pulse is equivalent to 10% of the robot's body weight.

[Media details](media/README.md) · [Model card](../../docs/double-balance/model-card.md)

## Evaluation

The live-viewer evaluation consists of two 60-second cases with a 20 ms control deadline.
Timing covers the complete cycle: communication, reference action checks, physics and status publication.

| Scenario | Cycles | Mean | p95 | p99 | Maximum | Missed deadlines |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Balance | 3,000 | 10.888 ms | 12.380 ms | 13.279 ms | 15.310 ms | 0 |
| Disturbance recovery | 3,000 | 11.317 ms | 12.214 ms | 12.634 ms | 14.676 ms | 0 |

Both cases retained the lower and upper balls for the full episode. The task allows
the robot and lower ball to translate while maintaining balance.

The balance case uses report ID `quiet-2407`. The disturbance case, `regression-2500`,
applies 29 horizontal pulses to the trunk's center of mass: one pulse every two seconds
from 2 s to 58 s, lasting 0.1 s, with randomized horizontal direction and a nominal force of 0.728590 N.

The recordings and their previews show recorded HIL trajectories. The table reports the separate evaluation
with the live viewer enabled; their exact source mapping is in the [media notes](media/README.md).

[Raw evaluation report](../../docs/double-balance/evidence/reports/live-control-report.json) ·
[Detailed technical report (中文)](../../docs/double-balance/project-report.md) ·
[Supporting data (中文)](../../docs/double-balance/evidence/README.md)

## System

| Component | Implementation |
| --- | --- |
| Physics | Native MuJoCo on the PC CPU; 200 Hz, four substeps per control cycle |
| Control | 50 Hz over wired ROS2 Humble |
| Policy | Recurrent PPO actor; 61D observations and 14D actions |
| Deployment | FP32 ONNX, ONNX Runtime CPUExecutionProvider on Jetson Orin Nano Super |
| Action checks | Every applied Jetson action compared with the frozen PyTorch actor |
| Visualization | Independent 3D viewer at approximately 25 fps |

The actor retains observation normalization, recurrent state and the established BAM actuator contract.
Its final checkpoint was trained over 11,314 cumulative updates.
[Model identity and deployment assets](../../docs/double-balance/model-card.md)

## Explore the implementation

- [Code guide](../../docs/double-balance/code-guide.md): training tasks, physics, communication and rendering.
- [Setup guide](../../docs/double-balance/getting-started.md): source installation and runtime prerequisites.
- [Reproduction guide (中文)](../../docs/double-balance/reproduction-guide.md): commands for the established PC–Jetson environment.
- [Documentation](../../docs/double-balance/README.md): results, model and technical references.

## Future direction

How can trajectory tracking be integrated with whole-body control so that Microduck follows a desired path while balancing on a basketball and keeping a second ball on its head?

Built on the [Microduck Playground basketball task](../basketball/README.md).
[Upstream and contributions](../../README.md#upstream-and-contributions) · [License](../../README.md#license)
