"""Validate Stage-05 checkpoint migration without constructing or running PPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mjlab_microduck.double_balance_checkpoint import (
    RELEASE_CHECKPOINT_SHA256,
    file_sha256,
)
from mjlab_microduck.double_balance_checkpoint_validation import validate_migration


def _load_checked(path: Path, expected_sha256: str | None) -> tuple[dict, str]:
    if not path.is_file():
        raise ValueError(f"checkpoint does not exist: {path}")
    digest = file_sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {digest}"
        )
    return torch.load(path, map_location="cpu", weights_only=False), digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("migrated", type=Path)
    parser.add_argument(
        "--expected-source-sha256", default=RELEASE_CHECKPOINT_SHA256
    )
    parser.add_argument("--expected-migrated-sha256")
    args = parser.parse_args()

    source, source_digest = _load_checked(
        args.source.resolve(), args.expected_source_sha256
    )
    migrated, migrated_digest = _load_checked(
        args.migrated.resolve(), args.expected_migrated_sha256
    )
    report = validate_migration(source, migrated, source_digest)
    report["source_sha256"] = source_digest
    report["migrated_sha256"] = migrated_digest
    report["source_unmodified"] = (
        file_sha256(args.source.resolve()) == source_digest
    )
    if not report["source_unmodified"]:
        raise RuntimeError("source checkpoint changed during validation")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
