# Code guide

[Home](../../README.md) · [Getting started](getting-started.md) · [简体中文](code-guide.zh-CN.md)

The project combines a PPO training stack with a PC–Jetson control runtime.
Use the map below to follow an observation from MuJoCo through the policy and back
to the simulated robot.

## Control and inference

| Component | Source |
| --- | --- |
| Runtime entry and package loading | [bootstrap.py](../../experiments/stage12/native-live-v1/package/bootstrap.py) |
| Native MuJoCo dynamics | [native_backend.py](../../experiments/stage12/native-live-v1/package/native_backend.py) |
| Control loop and session | [native_runtime.py](../../experiments/stage12/native-live-v1/package/native_runtime.py) · [native_session.py](../../experiments/stage12/native-live-v1/package/native_session.py) |
| ROS2 observations and action replies | [loop_ros.py](../../experiments/stage12/native-live-v1/package/loop_ros.py) · [loop_contract.py](../../experiments/stage12/native-live-v1/package/loop_contract.py) |
| Policy inference and PyTorch reference | [native_inference.py](../../experiments/stage12/native-live-v1/package/native_inference.py) · [reference_overlap.py](../../experiments/stage12/native-live-v1/package/reference_overlap.py) |
| Task criteria and model limits | [native_criteria.py](../../experiments/stage12/native-live-v1/package/native_criteria.py) · [native_model_limits.py](../../experiments/stage12/native-live-v1/package/native_model_limits.py) |
| State stream and interactive viewer | [live_stream.py](../../experiments/stage12/native-live-v1/package/live_stream.py) · [live_viewer.py](../../experiments/stage12/native-live-v1/package/live_viewer.py) |
| Recording playback and visualization | [Replay package](../../experiments/stage12/hil-replay-v1/package/) |

The runtime is retained in versioned packages so that reports and model identities
continue to point to their original implementation. `bootstrap.py` resolves the
associated modules and assets; the [dependency map — Chinese](reproduction-guide.md#3-必须保留的依赖链)
describes that setup.

## Training and robot model

| Source | What to look for |
| --- | --- |
| [Task definitions](../../src/mjlab_microduck/tasks/) | Environment configuration, observations, rewards, terminations and randomization |
| [Double-balance training](../../src/mjlab_microduck/double_balance_training.py) | Training utilities in the project source stack |
| [Native training workflow](../../experiments/stage12/native-workflow/README.md) | Native-workflow entry point and the archived training progression |
| [BAM actuator model](../../src/mjlab_microduck/actuator/) | Actuator dynamics used by the robot |
| [Robot assets](../../src/mjlab_microduck/robot/) | MJCF, joint layout, collisions and hardware geometry |
| [CPU regression tests](../../tests/) | Configuration checks and behavioral invariants |

The top-level source includes the inherited tasks and earlier double-balance
implementations. For the demonstrated system, follow the runtime map above and the
associated package manifests. The [training report — Chinese](project-report.md#3-训练与模型冻结)
connects the training progression to the released policy identities.

## Results and supporting records

- [Model card](model-card.md): actor inputs, action outputs, inference format and artifact hashes.
- [Demonstrations](../../experiments/double-balance/README.md): behavior and measured cycle timing.
- [Raw reports](evidence/reports/): machine-readable performance and evaluation records.
- [Detailed report — Chinese](project-report.md): methodology, test conditions and the source version index.
- [Audit records](../audits/): fixed configuration, model identities and development checks.

To prepare a change, see [Contributing](../../CONTRIBUTING.md). Source imports and
artifact references use the established package paths shown above.
