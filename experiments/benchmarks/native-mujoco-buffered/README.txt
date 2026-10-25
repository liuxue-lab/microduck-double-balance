Native MuJoCo CPU / Stage12 PC-Orin validation

This isolated runtime uses the existing double-ball scene, frozen final6h_005304 policy, FP32 ONNX and validated ROS2 transport. Physics is native mujoco.mj_step at 5ms; control stays at 20ms. CPU float32 task views retain the existing actor/noise/observation delays and BAM model. Native MjData integrates in float64. Foot rays use native mj_ray. No MuJoCo-Warp integration or CUDA graph executes.

The original BAM delay buffer still advances at 2ms (lags 6/8/10/12ms). Its delayed command is sampled by the 5ms solver with zero-order hold. The original 2ms Warp trajectories and 500Hz validation remain historical evidence; they are not relabelled as native evidence. CPU random sequences and closed-loop trajectories are not claimed identical to CUDA.

--apply imports only experiments/stage12/native-mujoco-v2, preserves local edits, installs the pinned package and runs the two 60-second cases once. --run reuses any existing report; --retry-run explicitly creates a new retained run. --status, --watch and --stop use the preserved controller. No package upgrades, training, model export, cloud GPU access, physical motor I/O or automatic Git push.

Acceptance retains retention, every-step joint/motor/force checks and every Orin-action comparison. At 200Hz this is 12,000 physics steps and 580 force steps in the push case. Realtime acceptance additionally requires all 3,000 fixed-release 20ms deadlines per case, including first ticks and logging. Late ticks are not skipped or rebased. Functional-only success is reported separately.

Each case exports timing.csv, state-data.npz (float64 native states), native-model.mjb and the original audit trace. State replay is not an exact restart or another simulation. The operation report includes failed-worker log tails and output hashes.

Canonical checkout: /home/lx/microduck-double-balance/workspace, branch double-balance. Downloads: /home/lx/下载. User performs the existing SSH push. Conventions: docs/handoffs/microduck-local-ssh-push-protocol.md and docs/handoffs/microduck-cloud-gpu-and-transfer-protocol.md. The 5090 stays off (user confirmed); no new handoff document is generated.

Version 2 recovers the completed v1 run into a separate receipt using pure standard-library checks. V1 files stay byte-identical. Progress persistence moves to the supervisor process; the control worker emits compact live counters, retaining every physics audit and all 3000 timing deadlines.

The 20ms path uses preallocated state storage and bounded trace buffers, nonblocking progress notifications, scoped cyclic-GC suspension, and scoped disabling of runtime JIT optimization. Large model images are saved in memory at reset and exported after both cases. Local frozen-PyTorch inference runs after sending the Orin request and before receiving its reply; it consumes the original RPC timeout and cannot substitute a local action. All per-substep electrical, joint, force and finite-state checks remain synchronous.

Native tensor mirrors avoid temporary copies. Pure entity properties reuse the original computation until the next native publication and return independent copies; forward/reset/step each invalidate them. The one-world actuator history uses CPU ring copies with the original RNG/lag-update function and its exact state sequence. The supervisor places its simulator and ROS bridge on separate eligible physical cores, preferring available higher-capacity/frequency cores. This honors inherited affinity and changes no system governor or priority.

The 20ms pass rule has not been relaxed. Local paced CPU checks include an emulated 4.5ms request interval, not actual Orin/DDS timing; full PC-Orin realtime acceptance is granted only from the new laptop run with zero missed deadlines in both 60-second cases. A functional-only result must never be described as a realtime pass. Live UI messages may coalesce under backpressure, but no control, state, audit or timing sample is dropped. Buffered traces survive handled errors and controlled stops; power loss or SIGKILL can lose the current in-memory trace prefix.
