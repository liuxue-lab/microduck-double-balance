#!/usr/bin/env python3
"""Write a Stage 08 experiment manifest; never starts training or cloud jobs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mjlab_microduck.double_balance_stage08_plan import campaign_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", choices=("A800", "5090"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = campaign_manifest(args.gpu)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Stage08Manifest={args.output.resolve()}")
    print("FormalTrainingStarted=False")


if __name__ == "__main__":
    main()
