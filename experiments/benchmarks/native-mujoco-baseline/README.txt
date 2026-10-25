Native MuJoCo CPU / Stage12 PC-Orin validation

This isolated runtime uses the existing double-ball scene, frozen final6h_005304 policy, FP32 ONNX and validated ROS2 transport. Physics is native mujoco.mj_step at 5ms; control stays at 20ms. CPU float32 task views retain the existing actor/noise/observation delays and BAM model. Native MjData integrates in float64. Foot rays use native mj_ray. No MuJoCo-Warp integration or CUDA graph executes.

The original BAM delay buffer still advances at 2ms (lags 6/8/10/12ms). Its delayed command is sampled by the 5ms solver with zero-order hold. The original 2ms Warp trajectories and 500Hz validation remain historical evidence; they are not relabelled as native evidence. CPU random sequences and closed-loop trajectories are not claimed identical to CUDA.

--apply imports only experiments/stage12/native-mujoco-v1, preserves local edits, installs the pinned package and runs the two 60-second cases once. --run reuses any existing report; --retry-run explicitly creates a new retained run. --status, --watch and --stop use the preserved controller. No package upgrades, training, model export, cloud GPU access, physical motor I/O or automatic Git push.

Acceptance retains retention, every-step joint/motor/force checks and every Orin-action comparison. At 200Hz this is 12,000 physics steps and 580 force steps in the push case. Realtime acceptance additionally requires all 3,000 fixed-release 20ms deadlines per case, including first ticks and logging. Late ticks are not skipped or rebased. Functional-only success is reported separately.

Each case exports timing.csv, state-data.npz (float64 native states), native-model.mjb and the original audit trace. State replay is not an exact restart or another simulation. The operation report includes failed-worker log tails and output hashes.

Canonical checkout: /home/lx/microduck-double-balance/workspace, branch double-balance. Downloads: /home/lx/下载. User performs the existing SSH push. Conventions: docs/handoffs/microduck-local-ssh-push-protocol.md and docs/handoffs/microduck-cloud-gpu-and-transfer-protocol.md. The 5090 stays off (user confirmed); no new handoff document is generated.
