Microduck Stage12 cumulative PC / Orin loop — 2026-10-25 Asia/Shanghai

The original TIMELINE_ARM_FAILED_PRESERVE is resolved. CUPTI12.9.79 isolated
repair is installed; no driver, Torch or system-library change is needed.
The latest cumulative candidate passed exact state/audit/RNG/action-history
comparisons and two independent unprofiled timing measurements:
baseline25.091286/26.285986 mean/p95 ms;
cumulative22.452714/23.670329; confirmation22.917374/24.269924.
Qualified by both >=5% improvement gates, but all measured steps exceeded20ms.
Reviewed report SHA256:
a48093e63cce12fa50052f00b425e8eb28542e16aed78210c374e7655505a3c7

This package validates that exact candidate in the existing wired PC-Orin loop.
Two cases only: quiet-2407 and regression-2500, each3000 controls /30000 physics
steps /60 simulated seconds. Actual wall time may be longer. The latter keeps
29 original10%-weight horizontal pulses at2,4,...58s, each0.1s. All original
500Hz limits and wrench checks remain. Translation is allowed; strict continuous
contact is not claimed. No retries, local fallback, motors, new PPO or export.

Runtime scoped to this new loop:
InferenceOnly + PhaseCache + BAM Graph v2 + StaticQuaternion + ScalarAudit +
PackedChecks + ActionGraph. Qualified action/quaternion sources are exact copies
and checked against the reviewed installation. Original delay sampling remains.
Initial reset still computes the original61D actor /85D critic; after reset only
actor work is needed, and zero rewards are explicitly compatibility placeholders.
Every applied14D action comes from Orin FP32 and is checked against the frozen
local CPU reference; no action filtering, normalization change, remapping or clipping.
Recurrent h/c resets only at request0 and3000. Three transcript chains must match.

Connect Orin to power and the existing Ethernet cable. Keep5090 OFF.
Save the deliverable in /home/lx/下载, then run:

python3 "/home/lx/下载/microduck-stage12-cumulative-loop-v1-20261025.py" --apply &&
git -C /home/lx/microduck-double-balance/workspace push origin double-balance

Keep the main terminal open. Existing Orin SSH authentication is reused; if the
shared connection expired, the normal Orin login may prompt once. No credentials
are read or stored by this package. No software installation/download is needed.
Installation adds only experiments/stage12/cumulative-loop-v1 to the fixed branch,
via an embedded incremental Git bundle based on25378e7fccbdba8b127b593eeaa47403c56ca866.
The launcher never pushes, stashes, resets, cleans or overwrites a path conflict.

Return ONE file:
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-cumulative-loop-v1/report.json

It embeds both cases, adapter checks, three-party transport evidence, separated
timings and log tails. Raw data remains beside it in the unique run folder:
case-*/timing.csv — one row per completed control tick;
case-*/state-data.npz — initial state plus every50Hz qpos/qvel/ctrl snapshot;
case-*/state-data.json — schema and replay caveat;
case-*/push10-trace.jsonl — original metrics and disturbance evidence;
loop-transcript.jsonl — all original observations and returned actions.
These can be plotted or rendered offline using the same frozen model. Recorded
state replay is neither a new closed-loop experiment nor a simulator checkpoint.
No online rendering is added. Exported files are hashed in the final report.

Timing interpretation
Orin inference, ROS round trip, IPC round trip, CPU reference comparison, PC plant
step, state capture, active cycle and start-to-start intervals are separate.
Nested timing intervals must not be summed. PC plant timing includes CUDA
completion and the full original checks. State capture is outside that plant
span but inside the recorded active cycle. Start-to-start also includes outer
checks/progress logging; its last value is empty because there is no next tick.
All3000 steps are reported, plus a separate diagnostic view after10 initial steps.

Success status:
CUMULATIVE_LOOP_FUNCTIONAL_PASSED_REALTIME_PENDING
This is functional60s validation of the optimized closed loop. It is NOT50Hz
real-time HIL acceptance. The existing20ms target is retained; lockstep execution
does not certify wall-clock scheduling, even if a latency statistic is below20ms.
The original1s RPC failure guard is unchanged and is not a50Hz deadline.
PC simulation latency is not a lower bound on real-robot Orin deployment latency.
Real-robot sensor/motor integration is outside this project scope.

Failure status:CUMULATIVE_LOOP_FAILED_PRESERVE. The first failure stops the run,
preserves evidence and shuts down only its owned workers. Historical reports,
frozen model and training sources are checked before/after and never replaced.
A success receipt validated-pipeline.json pins only this isolated runtime.
No global production switch or new stage12-complete tag is created.

Stable launcher after installation:
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-cumulative-loop-v1/microduck-stage12-cumulative-loop-v1.py
--status reads; --watch refreshes every2s and stays open after completion;
--stop requests this run's shutdown; --run reuses prior evidence without rerun;
--retry-run explicitly creates a new preserved run. Do not launch concurrent runs.

CPU tests cover6000 real socket packets with synthetic recurrence/in-memory ROS,
60000 synthetic physics substeps through the actual adapter methods, first-case
failure, timing/state export and genuine Git bundle installation preservation.
CPU fixtures do not prove actual DDS, ONNX, CUDA speed or real MuJoCo behavior.
Those results remain pending the laptop/Orin report.

Both full protocols and current handoff are retained in docs/handoffs:
microduck-local-ssh-push-protocol.md
microduck-cloud-gpu-and-transfer-protocol.md
microduck-stage12-hil-handoff-20261025.md
