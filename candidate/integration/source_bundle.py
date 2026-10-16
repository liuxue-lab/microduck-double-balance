"""Bundle the reviewed source/asset allowlist, without checkpoint or environment.

This is portable input material. It is not an installed cloud environment,
dependency wheelhouse, or evidence that a cloud GPU can run the native backend.
"""
from pathlib import Path
import hashlib
import json
import zipfile
from .durable import require, encoded, file_digest, publish, write_json


def checked_relative(name):
    p = Path(name)
    require(isinstance(name, str) and name and not p.is_absolute() and
            ".." not in p.parts and "\\" not in name and not name.endswith("/"), "Unsafe bundle path")
    return name


def build_zip(destination, rows, notes):
    require(len({r["name"] for r in rows}) == len(rows), "Duplicate source bundle member")
    require(sum(r["bytes"] for r in rows) < 80 * 1024**2, "Bundle allowlist exceeds declared size")
    index = {}
    for r in rows:
        checked_relative(r["name"])
        p = Path(r["source"])
        require(p.is_file() and not p.is_symlink(), "Source missing/symlinked: " + str(p))
        require(p.stat().st_size == r["bytes"] and file_digest(p) == r["sha256"],
                "Bundle source identity changed: " + r["name"])
        index[r["name"]] = {k: r[k] for k in ("bytes", "sha256")}
    require("SOURCE-BUNDLE-MANIFEST.json" not in index, "Reserved manifest name")
    def writer(f):
        with zipfile.ZipFile(f, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for r in rows:
                z.write(r["source"], r["name"])
            z.writestr("SOURCE-BUNDLE-MANIFEST.json", encoded(dict(files=index, **notes)))
    target = publish(destination, writer)
    with zipfile.ZipFile(target) as z:
        require(z.testzip() is None and set(z.namelist()) == set(index) | {"SOURCE-BUNDLE-MANIFEST.json"},
                "Source bundle CRC/roster differs")
        for name, row in index.items():
            raw = z.read(name)
            require(len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"],
                    "Source changed while bundling: " + name)
    return dict(path=str(target), bytes=target.stat().st_size, sha256=file_digest(target),
                members=len(index), expanded_bytes=sum(r["bytes"] for r in rows), **notes)


def prepare(root, output, repo):
    manifest = json.loads((root / "prior-readiness-v3/captured-source-manifest.json").read_bytes())
    rows, skipped = [], []
    for r in manifest["files"]:
        if r["disposition"] == "REVIEWED_HISTORICAL_IDENTITY_ONLY":
            skipped.append({k: r[k] for k in ("relative", "bytes", "sha256", "disposition")})
        else:
            rows.append(dict(name=r["relative"], source=r["source"], bytes=r["bytes"], sha256=r["sha256"]))
    for r in manifest["assets_hashed_not_packaged"]:
        rel = checked_relative(r["path"])
        rows.append(dict(name="project/" + rel, source=str(Path(repo) / rel), bytes=r["bytes"], sha256=r["sha256"]))
    require(len(rows) == 406 and len(skipped) == 1, "Reviewed source/asset roster differs")
    report = build_zip(output / "cloud-source-inputs.zip", rows,
                       dict(format="stage12-source-inputs-v1", sources=248, assets=158,
                            identity_only_historical_files=skipped, checkpoints=0,
                            cloud_environment_complete=False, cloud_training=0,
                            omitted=["dependency wheels", "installed virtualenv", "GPU driver/runtime",
                                     "Stage08 checkpoint", "historical Stage10 embedded archive"],
                            integration_code="This delivery's verified program directory is a separate input"))
    write_json(output / "cloud-source-inputs.json", report)
    return report
