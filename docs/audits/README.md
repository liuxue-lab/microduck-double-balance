# Technical reviews

[Documentation](../README.md)

Reviews and captured results are grouped by method and subsystem.

| Area | Contents |
| --- | --- |
| [Policy runner](policy-runner/) | Runner initialization, checkpoint reload and CPU/CUDA checks. |
| [Assisted training](assisted-training/) | Capacity measurements, training progress and candidate evaluation. |
| [Assistance removal](assistance-removal/) | Unassisted training, replication and final evaluation. |
| [Head pose and balance](head-balance/) | Reward comparisons, head-motion diagnosis and recovery records. |
| [Replay and contact diagnostics](replay-diagnostics/) | Replay differences, initialization, substeps and head-contact analysis. |
| [Native physics integration](native-physics/) | Native integration, state ownership, reset behavior and recovery. |

Current system measurements are indexed in the [double-balance evidence guide](../double-balance/evidence/README.md).
