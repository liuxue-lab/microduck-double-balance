# Documentation

[Home](../README.md) · **English** · [简体中文](README.zh-CN.md)

Explore the double-balance task, its learning pipeline and the PC–Jetson control system.

## Start here

| Guide | Contents |
| --- | --- |
| [Demonstrations](../experiments/double-balance/README.md) | Balance, disturbance recovery and measured results |
| [Double-balance documentation](double-balance/README.md) | A focused guide to the system, model and implementation |
| [Getting started](double-balance/getting-started.md) | Source installation, CPU checks and HIL prerequisites |
| [Model card](double-balance/model-card.md) | Policy architecture, exact artifact identities and availability |
| [Code guide](double-balance/code-guide.md) | Where to find training, control, inference and visualization |

## Technical references

- [Technical report (中文)](double-balance/project-report.md): training, execution timing, disturbances and evaluation definitions.
- [Reproduction guide (中文)](double-balance/reproduction-guide.md): the established PC–Jetson environment and its commands.
- [Data index (Chinese)](double-balance/evidence/README.md): original reports and provenance.
- [Media notes](../experiments/double-balance/media/README.md): demonstration sources and processing.
- [Asset inventory (Chinese)](double-balance/archive-scope.md): model and runtime asset records.

## Development

[Contributing](../CONTRIBUTING.md) · [CI history (Chinese)](double-balance/ci-followup.md) · [Citation](../CITATION.cff)

The [upstream overview](upstream/README.md) introduces the original Microduck Playground experiments and hardware.
Research records are grouped by method and subsystem in [audits/](audits/) and [handoffs/](handoffs/).
Maintainer procedures are documented in the [local SSH workflow (Chinese)](handoffs/microduck-local-ssh-push-protocol.md)
and [compute and transfer guide (Chinese)](handoffs/microduck-cloud-gpu-and-transfer-protocol.md).
