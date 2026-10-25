# Double-balance policy

[Home](../../README.md) · [Demonstrations](../../experiments/double-balance/README.md) · [Code guide](code-guide.md)

A PPO-trained recurrent policy for balancing Microduck on a basketball while
supporting a second ball on its original head shell. The deployment combines
native MuJoCo dynamics on the PC with FP32 ONNX inference on a Jetson Orin Nano Super.

## Model and deployment

| Property | Value |
| --- | --- |
| Training method | PPO with a recurrent actor and observation normalization |
| Actor input | 61 observations |
| Actor output | 14 actions, using the fixed joint order, scale and offsets |
| Training critic | 85 observations |
| Inference | FP32 ONNX · ONNX Runtime CPUExecutionProvider |
| Compute | Jetson Orin Nano Super |
| Physics | Native MuJoCo CPU · 200 Hz |
| Control | Wired ROS2 Humble · 50 Hz |

The final checkpoint contains 11,314 cumulative training updates, including
5,304 updates from the final six-hour run. Deployment retains the model's normalization,
recurrent state and BAM actuator behavior.

FP32 is the selected deployment format. Quantization experiments and the training
details are documented in the [project report — Chinese](project-report.md#3-训练与模型冻结).

## Evaluation

The system was measured in two fixed 60-second cases, one without external force
and one with periodic horizontal disturbance pulses. Each completed 3,000 control
cycles within the 20 ms deadline. All 6,000 applied Jetson actions were compared
with the PyTorch reference.

The evaluation covers simulated robot and ball dynamics with real Jetson inference
and communication. It measures ball retention and timing while allowing spatial
drift. Physical-robot deployment and broader seed-based success statistics are
outside these two demonstrations.

[Results](../../experiments/double-balance/README.md) ·
[Raw measurements](evidence/reports/native-live-v1-report.json) ·
[Test conditions and drift — Chinese](project-report.md#6-保球几何稳定与漂移)

## Artifacts and availability

| Artifact | Bytes | SHA256 |
| --- | ---: | --- |
| `final6h_005304.pt` | 10,041,821 | `cf7ce9297672943687019afe071d94918d3eb2620c18a1e8fe689512f8c08111` |
| `final6h_005304.onnx` | 2,501,039 | `8c1c1250b9b88d9a91e3eb095fea8bce21daa73c251b43f3cd8adc1327f03c4e` |
| `contract.json` | See inventory | `e0283bf7f00c8e2c7b57128b91ca3ced3abb4698e8f4c4aa11e74a745ad84929` |
| `golden.json` | See inventory | `b7ebe914c3933477d35bc0cc70c6b1bb4c1c83e67bd61dc22cd8f0775b1609e7` |

The distribution package names these same model bytes `checkpoint.pt` and
`policy.onnx`. It also includes `contract.json`, `golden.json`, a file manifest,
checksums and the complete demonstration recordings. Original filenames and hashes
remain recorded in the package manifest.

The full policy weights are maintained as external artifacts; this repository does
not currently provide a public download for the double-balance policy.
[assets-known.json](assets-known.json) records the original locations and
[assets-verified.json](assets-verified.json) contains the verification records.
The upstream single-ball basketball policy has a separate identity.

## Using the model

Use the policy together with its normalization, recurrent reset behavior,
deployment contract and fixed robot model. The [getting-started guide](getting-started.md)
describes source installation; the [PC–Jetson environment guide — Chinese](reproduction-guide.md)
documents the additional assets and runtime packages used by the demonstrated system.
