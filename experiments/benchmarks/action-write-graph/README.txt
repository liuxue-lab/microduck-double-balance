Microduck Stage12 action-write CUDA graph comparison — 2026-10-24 Asia/Shanghai

Current status
The CUPTI fault is fixed and pushed as f33210ad7c474161b85e9f4cb5dc6db65f877db7.
Latest pc-sync-v1: PC_SYNC_MEASURED, errors=[], 4 arms, 640 controls/6400 physics.
Mean/p95 ms: baseline25.185270/26.276674; quat24.644871/25.881295;
delay25.386679/26.952261; combined24.469622/25.982217.
All 73440 common-state scalars and 161 delay/RNG snapshots matched per candidate.
Neither mean nor p95 gains reached both 5% gates. KEEP_INFERENCE_ONLY.
All 150 measured controls per arm exceeded20ms. No real-time pass is claimed.
Reviewed report SHA256:
afe6ba189a35733df2470d8e95f4f7600070e5752eab99669e851cccdd8a167a

Experiment
Three unprofiled arms: baseline, action-graph, cumulative.
ActionGraph stages original process_action outputs once per control and replays
original apply_action method bodies ten times per control. Encoder bias, joint
mapping and both hold-wrench writes remain. Each replay starts from current
live destination contents, preserving every untouched joint/body/torque.
Warmup writes only to independent private destinations. The graph never
advances physics, draws RNG or applies the evaluation disturbance. Original
BallHoldAction's literal [0] index is represented once as device int64 [0] in
the capture copy, avoiding CPU index conversion. Reference uses the original
Python-list path. No torch.compile or precision change.
First40 writes compare full destination FP32 bits against original methods on
the same inputs, within10 excluded warmup controls. Original processing,
delay sampling, force lifecycle and all500Hz audits remain active.

Cumulative additionally uses StaticQuaternion + ScalarAudit + PackedChecks,
whose earlier individual comparisons were exact but had insufficient speed
benefits. This measures their combined effect; it does not assume gains add.
Delay optimization, CommandViews, ReadGraph and HeadGraph remain disabled.
Original500Hz thresholds, error order, records and owned-force cleanup stay.

Every arm:160 controls/1600 physics,3.2seconds including the full2.0–2.1second
10%-weight pulse. Compare73440 common-state scalars, full Audit/Trial/pulses,
and161 action-history/targets/hold-input/delay/RNG snapshots. A discrepancy
stops and preserves; no silent fallback. Both mean andp95 must improve>=5%.
If qualified, repeat the selected candidate then baseline in fresh processes;
both candidate results must beat the faster baseline. Maximum5 arms,
800 controls/8000 physics. No production activation or60s HIL acceptance.

Laptop command (save the deliverable in /home/lx/下载)
python3 "/home/lx/下载/microduck-stage12-pc-actiongraph-v1-20261024.py" --apply &&
git -C /home/lx/microduck-double-balance/workspace push origin double-balance

Keep the terminal open. No Orin;5090 stays OFF_USER_CONFIRMED. This embeds an
incremental Git bundle based on289e3fd260f8c603c3d8e0337d740b0806c6255f and only
adds experiments/stage12/pc-actiongraph-v1. It does not push, reset, stash,
clean, overwrite conflicts, install dependencies, train or re-export models.
The prior report establishes local installation, not a successful push.
The user's normal push sends both commits if the previous one was not pushed.

Return one file:
/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-pc-actiongraph-v1/report.json
The report embeds relevant log tails. Raw logs/traces remain in its run folder.
Historical reports are checked and left intact. --status reads; --watch keeps
refreshing every2s, even when complete; --stop affects only this owned run.
--run reuses existing evidence; --retry-run explicitly creates a new run.

CPU tests use original pinned methods and a synthetic integrator. They cannot
establish CUDA capture support, GPU speed, real MuJoCo equivalence or Orin HIL.
All GPU conclusions remain pending the laptop report. Existing reference
package/tests are experiments/stage12/pc-timeline-v1, unchanged.
Official graph constraints:
https://docs.pytorch.org/docs/2.9/notes/cuda.html#cuda-graphs
Additional fixtures: official mjlab1.3.0 PyPI wheel, SHA256
e3b544bf3e7b69d6de0641c0f98fa3c9bb9d8ff662951ad83dcaba4db4fe47cf.

Both full protocols retained:
docs/handoffs/microduck-local-ssh-push-protocol.md
docs/handoffs/microduck-cloud-gpu-and-transfer-protocol.md
Current handoff:docs/handoffs/microduck-stage12-hil-handoff-20261024.md
No new stage12-complete tag. Final full-loop50Hz acceptance remains pending.
