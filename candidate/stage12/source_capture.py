"""Bounded source evidence collection with verified reuse and a durable partial index."""
from pathlib import Path
import hashlib
import json
import os

from .contracts import require


def file_identity(path):
    """Stream both SHA256 and the exact Git blob identity; never execute source."""
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Source missing/symlinked: " + str(path))
    before = path.stat()
    sha, blob = hashlib.sha256(), hashlib.sha1()
    blob.update(b"blob " + str(before.st_size).encode() + b"\0")
    total = 0
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(block)
            blob.update(block)
            total += len(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino) and total == before.st_size,
            "Source changed while reading: " + str(path))
    return dict(bytes=total, sha256=sha.hexdigest(), git_blob_sha1=blob.hexdigest())


def relative_name(name):
    p = Path(name)
    require(not p.is_absolute() and ".." not in p.parts and "\\" not in name and p.parts,
            "Unsafe source evidence path")
    return p


class SourceCollector:
    def __init__(self, out, policy, reuse, prior_run):
        self.out, self.policy = Path(out), policy
        self.reuse, self.prior_run = reuse, Path(prior_run)
        self.rows, self.seen = [], set()
        self.captured_bytes = self.reused_bytes = 0
        self.progress = self.out / "source-capture-progress.jsonl"
        # This run is one shot. An existing progress file is not a retry license.
        with self.progress.open("x", encoding="utf-8"):
            pass

    def record(self, row):
        self.rows.append(row)
        with self.progress.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def add(self, source, relative):
        source, rel = Path(source), relative_name(relative)
        require(relative not in self.seen, "Duplicate source evidence target: " + relative)
        self.seen.add(relative)
        row = dict(source=str(source), relative=relative, disposition="STARTED")
        try:
            row.update(file_identity(source))
            identity_only = self.policy["identity_only_sources"].get(relative)
            if identity_only is not None:
                require(row["bytes"] == identity_only["bytes"] and
                        row["git_blob_sha1"] == identity_only["git_blob_sha1"],
                        "Reviewed historical utility changed: " + str(source))
                row.update(disposition="REVIEWED_HISTORICAL_IDENTITY_ONLY", reason=identity_only["reason"],
                           source_commit=identity_only["source_commit"], copied=False, executed=False)
            else:
                require(row["bytes"] <= self.policy["source_file_cap_bytes"],
                        "Source file too large: " + str(source) + "; bytes=" + str(row["bytes"]) +
                        "; cap=" + str(self.policy["source_file_cap_bytes"]))
                require(self.captured_bytes + self.reused_bytes + row["bytes"] <=
                        self.policy["captured_and_reused_total_cap_bytes"],
                        "Source capture/reuse total exceeds 40 MiB at " + str(source))
                old = self.reuse.get(relative)
                if old is not None:
                    require(str(source) == old["source"] and row["bytes"] == old["bytes"] and
                            row["sha256"] == old["sha256"], "Previously captured source changed: " + str(source))
                    prior_file = self.prior_run / relative_name(old["evidence_relative"])
                    previous = file_identity(prior_file)
                    require(previous["bytes"] == old["bytes"] and previous["sha256"] == old["sha256"],
                            "Prior v2 source evidence differs: " + str(prior_file))
                    self.reused_bytes += row["bytes"]
                    row.update(disposition="REUSED_VERIFIED_V2", prior_file=str(prior_file), copied=False,
                               prior_transfer_commit="f6fb64e3ff1b90c7c9d65d1fb29c47df5b76b671",
                               prior_archive_sha256="0e1713f616b0a7ad702e0d50f4290e93a60e7dedc5074ba03fa0d125784fc012")
                else:
                    dest = self.out / "captured-source" / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with source.open("rb") as src, dest.open("xb") as dst:
                        for block in iter(lambda: src.read(1024 * 1024), b""):
                            dst.write(block)
                    actual = file_identity(dest)
                    require(actual == {k: row[k] for k in ("bytes", "sha256", "git_blob_sha1")},
                            "Captured source changed between hashing/copying: " + str(source))
                    self.captured_bytes += row["bytes"]
                    row.update(disposition="CAPTURED_NEW", saved=str(dest.relative_to(self.out)), copied=True)
            self.record(row)
            return row
        except BaseException as exc:
            row.update(disposition="BLOCKED", error=repr(exc))
            self.record(row)
            raise

    def assert_reuse_complete(self):
        require(set(self.reuse) <= self.seen, "Previously captured source disappeared from the source roster")

    def summary(self):
        modes = ("CAPTURED_NEW", "REUSED_VERIFIED_V2", "REVIEWED_HISTORICAL_IDENTITY_ONLY", "BLOCKED")
        return dict(counts={mode: sum(r["disposition"] == mode for r in self.rows) for mode in modes},
                    copied_bytes=self.captured_bytes, reused_bytes=self.reused_bytes,
                    copied_and_reused_bytes=self.captured_bytes + self.reused_bytes)

    def finalize(self, status, details):
        result = dict(status=status, files=self.rows, **self.summary(), **details,
                      cloud_reproduction_bundle_complete=False,
                      reference_note="Reused v2 files remain in the pinned earlier return; this ZIP is incremental source evidence.")
        with (self.out / "captured-source-manifest.json").open("x", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
        return result
