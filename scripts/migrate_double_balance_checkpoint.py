"""Create a Stage-05 double-balance checkpoint without running PPO."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch

from mjlab_microduck.double_balance_checkpoint import (
    RELEASE_CHECKPOINT_SHA256,
    TARGET_LEARNING_RATE,
    audit_source_schema,
    audit_target_schema,
    file_sha256,
    migrate_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--expected-source-sha256", default=RELEASE_CHECKPOINT_SHA256
    )
    parser.add_argument("--learning-rate", type=float, default=TARGET_LEARNING_RATE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if source == output:
        parser.error("source and output must be different files")
    if not source.is_file():
        parser.error(f"source checkpoint does not exist: {source}")
    if output.exists() and not args.force:
        parser.error(f"output already exists (use --force to replace it): {output}")

    source_digest = file_sha256(source)
    if source_digest != args.expected_source_sha256:
        parser.error(
            "source SHA-256 mismatch: "
            f"expected {args.expected_source_sha256}, got {source_digest}"
        )

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    source_schema = audit_source_schema(checkpoint)
    migrated = migrate_checkpoint(
        checkpoint,
        source_sha256=source_digest,
        learning_rate=args.learning_rate,
    )
    target_schema = audit_target_schema(migrated)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".stage05-", dir=output.parent) as tmp:
        temporary_output = Path(tmp) / output.name
        torch.save(migrated, temporary_output)
        os.replace(temporary_output, output)

    if file_sha256(source) != source_digest:
        raise RuntimeError("source checkpoint changed during migration")
    report = {
        "status": "PASS",
        "ppo_executed": False,
        "source": {
            "path": str(source),
            "size_bytes": source.stat().st_size,
            "sha256": source_digest,
            "schema": source_schema,
        },
        "output": {
            "path": str(output),
            "size_bytes": output.stat().st_size,
            "sha256": file_sha256(output),
            "schema": target_schema,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
