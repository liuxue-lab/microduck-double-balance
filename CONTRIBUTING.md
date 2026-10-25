# Contributing

The active branch is `double-balance`. Start with the [code guide](docs/double-balance/code-guide.md)
and [getting-started guide](docs/double-balance/getting-started.md).
The completed scope is simulation HIL; the repository does not claim physical robot validation.

## Changes and verification

- Explain the concrete behavior or documentation problem and keep each change focused.
- Preserve the Stage09 frozen contract, including byte-level hashes of `pyproject.toml` and the original MDP prefix. A dependency or runtime change needs an explicit new baseline and appropriate validation.
- Run `uv run --frozen python -m pytest tests/ -q` for runtime changes in the supported environment. Archived stage tests require their own pinned packages and must not be collected indiscriminately.
- For documentation changes, run `python3 scripts/check_relative_links.py .`, check rendered pages, and keep English and Chinese entry points consistent.
- Label simulation, recorded replay, live HIL and physical robot evidence accurately. Cite the raw report for every new result; do not generalize fixed-seed measurements into success rates.
- Keep large checkpoints, full recordings, credentials and machine-specific generated files out of the source change. Include hashes and availability notes for external artifacts.

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
