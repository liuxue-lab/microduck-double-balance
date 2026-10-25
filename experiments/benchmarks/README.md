# Runtime benchmarks

[Project](../../README.md) · [Experiments](../../experiments/README.md) · **English** · [简体中文](README.zh-CN.md)

Focused comparisons of simulation, inference, synchronization and data movement. Each package retains its own test setup and measurements; the final complete-cycle results are linked below.

| Component | Focus |
| --- | --- |
| [Runtime profiling](runtime-profile/README.md) | Host-side call profiling of the established control-runtime configuration. |
| [CPU and CUDA timeline](cpu-cuda-timeline/README.md) | Correlated CPU and CUDA activity traces for locating control-loop costs. |
| [Isolated CUPTI support](cupti-isolation/README.md) | A scoped CUPTI library setup for collecting GPU activity on the profiling workstation. |
| [Jetson inference](jetson-inference/README.md) | Inference, transport and optimization measurements for the Jetson deployment. |
| [Kinematics caching](kinematics-cache/README.md) | Reuse of phase-local kinematic values with comparison against the original computations. |
| [BAM graph output ownership](bam-graph-shared-output/README.md) | The shared-output CUDA graph candidate for BAM actuator calculations. |
| [BAM graph input snapshots](bam-graph-snapshot/README.md) | Separate input, graph-output and committed-state buffers for BAM graph replay. |
| [Action write graphs](action-write-graph/README.md) | CUDA graph capture of action application with live destination-state checks. |
| [Head kinematics graphs](head-kinematics-graph/README.md) | CUDA graph capture of head-geometry calculations. |
| [Inference-only execution](inference-only-loop/README.md) | Actor-only observation work and removal of training reward computation during frozen-policy inference. |
| [Packed audit reads](packed-audit-reads/README.md) | Grouping audit-field reads while preserving the full validation set. |
| [Scalar audit processing](scalar-audit/README.md) | Scalar-side audit and packed-read comparisons for the control path. |
| [Tensor read graphs](tensor-read-graph/README.md) | Graph-based capture of audit reads from simulation tensors. |
| [Bridged simulation reads](bridged-read-graph/README.md) | Shared tensor views for graph reads from the simulation array wrappers. |
| [Command views with int64 indices](command-views-int64/README.md) | Contiguous command-field views with int64 address assumptions. |
| [Command views with mixed indices](command-views-mixed-index/README.md) | Contiguous command views preserving the actual int32 and int64 index layouts. |
| [Synchronization costs](synchronization/README.md) | Comparisons of synchronization, quaternion and delay-handling costs. |
| [Native MuJoCo baseline](native-mujoco-baseline/README.md) | Initial native CPU-physics integration with the existing policy and ROS2 transport. |
| [Buffered native control](native-mujoco-buffered/README.md) | Preallocated trace storage and separated progress reporting for native control. |

[Demonstrations](../double-balance/README.md) · [Model card](../../docs/double-balance/model-card.md) · [Code guide](../../docs/double-balance/code-guide.md)
