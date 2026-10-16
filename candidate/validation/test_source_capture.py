"""Exercise the observed oversized-archive failure and verified incremental capture."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from stage12.source_capture import SourceCollector, file_identity
from stage12 import host_sources


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.out = self.base / "out"
        self.out.mkdir()
        self.prior = self.base / "prior"
        self.prior.mkdir()
        self.policy = dict(source_file_cap_bytes=16, captured_and_reused_total_cap_bytes=32,
                           identity_only_sources={})

    def source(self, name="a.py", raw=b"value = 1\n"):
        p = self.base / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
        return p

    def collector(self, reuse=None):
        return SourceCollector(self.out, self.policy, reuse or {}, self.prior)

    def test_new_source_is_copied_and_hashed(self):
        src = self.source()
        c = self.collector()
        row = c.add(src, "project/a.py")
        self.assertEqual(row["disposition"], "CAPTURED_NEW")
        self.assertEqual((self.out / row["saved"]).read_bytes(), src.read_bytes())

    def test_exact_reviewed_oversized_utility_is_identified_without_copy(self):
        src = self.source(raw=b"a" * 64)
        row = file_identity(src)
        self.policy["identity_only_sources"]["project/a.py"] = dict(row, reason="closed utility", source_commit="fixture")
        c = self.collector()
        actual = c.add(src, "project/a.py")
        self.assertEqual(actual["disposition"], "REVIEWED_HISTORICAL_IDENTITY_ONLY")
        self.assertFalse((self.out / "captured-source").exists())
        self.assertEqual(c.summary()["copied_bytes"], 0)

    def test_changed_reviewed_utility_still_fails(self):
        src = self.source(raw=b"a" * 64)
        self.policy["identity_only_sources"]["project/a.py"] = dict(file_identity(src), reason="closed utility", source_commit="fixture")
        src.write_bytes(b"b" * 64)
        with self.assertRaisesRegex(RuntimeError, "utility changed"):
            self.collector().add(src, "project/a.py")

    def test_unknown_large_file_records_path_size_and_preserves_partial_manifest(self):
        c = self.collector()
        good = self.source()
        c.add(good, "project/a.py")
        bad = self.source("too-large.py", b"b" * 64)
        with self.assertRaisesRegex(RuntimeError, r"too-large.py; bytes=64; cap=16"):
            c.add(bad, "project/too-large.py")
        c.finalize("PARTIAL_BLOCKED", {})
        m = json.loads((self.out / "captured-source-manifest.json").read_bytes())
        self.assertEqual(m["counts"]["CAPTURED_NEW"], 1)
        self.assertEqual(m["counts"]["BLOCKED"], 1)
        self.assertEqual(len((self.out / "source-capture-progress.jsonl").read_text().splitlines()), 2)
        self.assertEqual(good.read_bytes(), (self.out / "captured-source/project/a.py").read_bytes())

    def reuse_fixture(self):
        src = self.source()
        old = self.prior / "captured-source/project/a.py"
        old.parent.mkdir(parents=True)
        old.write_bytes(src.read_bytes())
        row = dict(file_identity(src), source=str(src), evidence_relative="captured-source/project/a.py")
        return src, old, {"project/a.py": row}

    def test_matching_v2_source_is_reused_without_copying(self):
        src, old, reuse = self.reuse_fixture()
        before = old.read_bytes()
        c = self.collector(reuse)
        row = c.add(src, "project/a.py")
        c.assert_reuse_complete()
        self.assertEqual(row["disposition"], "REUSED_VERIFIED_V2")
        self.assertFalse((self.out / "captured-source").exists())
        self.assertEqual(before, old.read_bytes())

    def test_changed_current_source_is_rejected(self):
        src, old, reuse = self.reuse_fixture()
        src.write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "captured source changed"):
            self.collector(reuse).add(src, "project/a.py")

    def test_changed_prior_evidence_is_rejected(self):
        src, old, reuse = self.reuse_fixture()
        old.write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "evidence differs"):
            self.collector(reuse).add(src, "project/a.py")
        self.assertEqual(old.read_bytes(), b"changed")

    def test_reused_sources_still_count_toward_total_cap(self):
        src, _, reuse = self.reuse_fixture()
        self.policy["captured_and_reused_total_cap_bytes"] = src.stat().st_size
        c = self.collector(reuse)
        c.add(src, "project/a.py")
        with self.assertRaisesRegex(RuntimeError, "total exceeds"):
            c.add(self.source("other.py"), "project/other.py")

    def test_missing_prior_roster_member_is_not_silently_dropped(self):
        _, _, reuse = self.reuse_fixture()
        with self.assertRaisesRegex(RuntimeError, "disappeared"):
            self.collector(reuse).assert_reuse_complete()

    def test_existing_progress_prevents_retry_and_source_symlink_is_rejected(self):
        c = self.collector()
        with self.assertRaises(FileExistsError):
            self.collector()
        src = self.source()
        link = self.base / "link.py"
        link.symlink_to(src)
        with self.assertRaisesRegex(RuntimeError, "symlinked"):
            c.add(link, "project/link.py")

    def test_path_escape_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Unsafe"):
            self.collector().add(self.source(), "../outside.py")
        self.assertFalse((self.base / "outside.py").exists())

    def test_completed_capture_host_verification_249_sources_158_assets(self):
        from stage12.source_reuse import source_roster
        program, repo = self.base / "program", self.base / "repo"
        for d in (program / "prior-readiness-v3", program / "baseline/g11/accepted", repo / "src", repo / "scripts"):
            d.mkdir(parents=True)
        for name in ["pyproject.toml", "uv.lock", "AGENTS.md"] + ["src/module_" + str(i) + ".py" for i in range(241)] + ["scripts/stage10_finalize.py"]:
            (repo / name).write_text("# fixture\n")
        mjlab, rsl = self.base / "mjlab", self.base / "rsl_rl"
        for path in (mjlab / "rl/runner.py", mjlab / "scripts/train.py", rsl / "runners/on_policy_runner.py", rsl / "model.py"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# installed fixture\n")
        roster = source_roster(repo, rsl, mjlab)
        self.assertEqual(len(roster), 249)
        rows, copied = [], 0
        for relative, source in roster.items():
            path = Path(source)
            row = dict(source=source, relative=relative, **file_identity(path))
            if relative.endswith("stage10_finalize.py"):
                row["disposition"] = "REVIEWED_HISTORICAL_IDENTITY_ONLY"
            else:
                old = self.prior / "captured-source" / relative
                old.parent.mkdir(parents=True, exist_ok=True)
                old.write_bytes(path.read_bytes())
                if copied < 60:
                    row.update(disposition="CAPTURED_NEW", saved=str(old.relative_to(self.prior)))
                else:
                    row.update(disposition="REUSED_VERIFIED_V2", prior_file=str(old))
                copied += 1
            rows.append(row)
        assets = []
        for i in range(158):
            path = repo / "src" / (str(i) + ".xml")
            path.write_text("<mujoco/>")
            assets.append(dict(path=str(path.relative_to(repo)), **file_identity(path)))
        frozen = program / "prior-readiness-v3/captured-source-manifest.json"
        frozen.write_text(json.dumps(dict(status="CAPTURE_COMPLETE_WITH_VERIFIED_REUSE", files=rows, assets_hashed_not_packaged=assets)))
        (program / "prior-readiness-v3/audit.json").write_text(json.dumps(dict(
            base=str(self.prior.parent), run=self.prior.name, transfer_commit="fixture",
            source_manifest_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest())))
        (program / "baseline/g11/accepted/plan.json").write_text('{"versions":{}}')
        (program / "baseline/source-pins.json").write_text(json.dumps({str(mjlab / "rl/runner.py"): "fixture"}))
        dist = SimpleNamespace(version="5.0.1", locate_file=lambda name: rsl)
        versions = {"torch": "2.9.1", "rsl-rl-lib": "5.0.1", "tensordict": "0.10.0", "glfw": "2.10.0", "imageio-ffmpeg": "0.6.0"}
        meta = SimpleNamespace(version=lambda name: versions[name], distributions=lambda: [], distribution=lambda name: dist)
        with patch.object(host_sources, "REPO", str(repo)), patch.object(host_sources, "meta", meta):
            before = host_sources.verify(program, self.out, "before")
            after = host_sources.verify(program, self.out, "after")
        self.assertEqual(before, after)
        self.assertEqual((before["current_source_identities"], before["retained_source_files"], before["assets_verified"]), (249, 248, 158))
        self.assertFalse((self.out / "captured-source").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
