"""Verify the completed v2/v3 capture in place; never recapture those sources."""
from pathlib import Path
import hashlib
import json

from .contracts import require
from .source_capture import file_identity, relative_name

FIELDS = ("bytes", "sha256", "git_blob_sha1")
ASSET_EXTENSIONS = (".xml", ".stl", ".obj", ".ply", ".png", ".jpg", ".json")


def source_roster(repo, rsl_root, mjlab_root):
    rows = {}
    for p in sorted(rsl_root.rglob("*.py")):
        rows["rsl_rl/" + str(p.relative_to(rsl_root))] = str(p)
    for folder in ("scripts", "rl"):
        for p in sorted((mjlab_root / folder).rglob("*.py")):
            rows["mjlab/" + str(p.relative_to(mjlab_root))] = str(p)
    for name in ("pyproject.toml", "uv.lock", "AGENTS.md"):
        rows["project/" + name] = str(repo / name)
    for folder in ("src", "scripts"):
        for p in sorted((repo / folder).rglob("*.py")):
            if "__pycache__" not in p.parts:
                rows["project/" + str(p.relative_to(repo))] = str(p)
    return rows


def retained_path(row, prior_run):
    mode = row["disposition"]
    if mode == "CAPTURED_NEW":
        return prior_run / relative_name(row["saved"])
    if mode == "REUSED_VERIFIED_V2":
        path = Path(row["prior_file"])
        require(path.is_absolute(), "Prior v2 evidence path must be absolute")
        return path
    require(mode == "REVIEWED_HISTORICAL_IDENTITY_ONLY", "Incomplete prior source row")
    return None


def verify_rows(manifest, prior_run, roster, repo):
    require(manifest["status"] == "CAPTURE_COMPLETE_WITH_VERIFIED_REUSE", "Prior capture was not complete")
    rows = manifest["files"]
    expected = {r["relative"]: r["source"] for r in rows}
    require(len(expected) == len(rows) and expected == roster, "Current source roster changed")
    retained, source_bytes = 0, 0
    for row in rows:
        identity = {k: row[k] for k in FIELDS}
        require(file_identity(row["source"]) == identity, "Current source changed: " + row["source"])
        old = retained_path(row, prior_run)
        if old is not None:
            require(file_identity(old) == identity, "Retained source changed: " + str(old))
            retained += 1
            source_bytes += row["bytes"]
    assets = manifest["assets_hashed_not_packaged"]
    actual_assets = {str(p.relative_to(repo)) for p in (repo / "src").rglob("*")
                     if p.is_file() and p.suffix.lower() in ASSET_EXTENSIONS}
    require(len({r["path"] for r in assets}) == len(assets) and
            actual_assets == {r["path"] for r in assets}, "Current asset roster changed")
    for row in assets:
        require(file_identity(repo / relative_name(row["path"])) == {k: row[k] for k in FIELDS},
                "Asset identity changed: " + row["path"])
    return dict(status="VERIFIED_IN_PLACE_NO_RECAPTURE", current_source_identities=len(rows),
                retained_source_files=retained, retained_source_bytes=source_bytes,
                assets_verified=len(assets), source_files_copied=0,
                source_manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
                cloud_reproduction_bundle_complete=False)
