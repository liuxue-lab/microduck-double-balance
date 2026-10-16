"""Installed metadata and source identities; no simulator or training imports."""
from pathlib import Path
import hashlib
import importlib.metadata as meta
import json
import os
import platform
import sys
import traceback

from .contracts import REPO, require
from .source_reuse import source_roster, verify_rows


def verify(root, out, phase):
    result = dict(status="SOURCE_REUSE_STARTED", phase=phase)
    try:
        prior = json.loads((root / "prior-readiness-v3/audit.json").read_bytes())
        frozen = root / "prior-readiness-v3/captured-source-manifest.json"
        require(hashlib.sha256(frozen.read_bytes()).hexdigest() == prior["source_manifest_sha256"],
                "Frozen completed capture identity differs")
        manifest = json.loads(frozen.read_bytes())
        require(len(manifest["files"]) == 249 and len(manifest["assets_hashed_not_packaged"]) == 158,
                "Completed capture scope differs")
        versions = json.loads((root / "baseline/g11/accepted/plan.json").read_bytes())["versions"]
        for name, expected in versions.items():
            require(meta.version(name) == expected, "Pinned distribution version changed: " + name)
        require(meta.version("torch").split("+")[0] == "2.9.1", "Pinned Torch core differs")
        for name, expected in {"rsl-rl-lib": "5.0.1", "tensordict": "0.10.0",
                               "glfw": "2.10.0", "imageio-ffmpeg": "0.6.0"}.items():
            require(meta.version(name) == expected, "Reviewed distribution version differs: " + name)
        pins = json.loads((root / "baseline/source-pins.json").read_bytes())
        mjlab_runner = next(Path(p) for p in pins if p.endswith("/mjlab/rl/runner.py"))
        repo = Path(REPO)
        rsl_root = Path(meta.distribution("rsl-rl-lib").locate_file("rsl_rl"))
        roster = source_roster(repo, rsl_root, mjlab_runner.parents[1])
        result = verify_rows(manifest, Path(prior["base"]) / prior["run"], roster, repo)
        result.update(transfer_commits=["f6fb64e3ff1b90c7c9d65d1fb29c47df5b76b671", prior["transfer_commit"]])
        if phase == "before":
            installed = sorted([{"name": d.metadata.get("Name"), "version": d.version}
                                for d in meta.distributions()], key=lambda x: (x["name"] or "").lower())
            with (out / "environment.json").open("x", encoding="utf-8") as f:
                json.dump(dict(interpreter=sys.executable, python=sys.version, platform=list(os.uname()),
                               machine=platform.machine(), packages=installed,
                               CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
                               no_install=True, no_uv_sync=True, native_imports_in_verification=False),
                          f, ensure_ascii=False, sort_keys=True, indent=2)
                f.write("\n")
        return result
    except BaseException as exc:
        result.update(status="SOURCE_REUSE_BLOCKED", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        with (out / ("source-reuse-" + phase + ".json")).open("x", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
