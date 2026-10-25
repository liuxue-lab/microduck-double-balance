# Contributing

The active branch is `double-balance`. Start with the [code guide](docs/double-balance/code-guide.md)
and [getting-started guide](docs/double-balance/getting-started.md).
The demonstrated system combines simulated robot dynamics with real Jetson inference and wired ROS2 communication.

## Changes and verification

- Explain the concrete behavior or documentation problem and keep each change focused.
- Preserve the deployment contract, locked dependencies and original MDP behavior. A dependency or runtime change needs an explicit new baseline and appropriate validation.
- Run `uv run --frozen python -m pytest tests/ -q` for runtime changes in the supported environment. Research packages have separate pinned environments and test entry points.
- For documentation changes, run `python3 scripts/check_relative_links.py .`, check rendered pages, and keep English and Chinese entry points consistent.
- Label simulation, recorded replay, live HIL and physical robot evidence accurately. Cite the raw report for every new result; do not generalize fixed-seed measurements into success rates.
- Keep model checkpoints, credentials and machine-specific generated files outside the source tree. The two published original demonstration recordings are versioned media assets; preserve their bytes and provenance. Include hashes and availability notes for external model assets.

The CI workflow additionally checks packaging, CLI entry points, ONNX export/parity and inherited hardware generators.
An existing green run validates its exact commit; a later commit receives its own check.

## Media and attribution

Keep small preview assets with their experiment. Record their source hashes and extraction settings;
see [the two double-balance previews](experiments/double-balance/media/README.md).
Changes to inherited experiments must retain original attribution and the relevant software/hardware license.
Update [CITATION.cff](CITATION.cff) only with verified project metadata.

For this maintainer's existing checkout, follow the established
[incremental bundle and local SSH protocol](docs/handoffs/microduck-local-ssh-push-protocol.md).
Do not sweep unrelated untracked files into a commit. Cloud training is a separately scoped activity under
the [compute and transfer protocol](docs/handoffs/microduck-cloud-gpu-and-transfer-protocol.md).

## Contributing upstream

Open a focused change against the repository that owns the affected component, after checking its current contribution rules.
This project's README publication, an accepted upstream PR, a shared model and an official website feature are separate outcomes.
See [upstream contribution notes](docs/double-balance/upstream-contribution.md) for the verified example and sources.
