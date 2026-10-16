"""Regression checks for the actual publication and transport contracts and the new manual-Git transport."""
from pathlib import Path
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from stage12 import contracts as c
from stage12 import publication as p
from stage12 import transfer as t

spec = importlib.util.spec_from_file_location("stage12_launcher", ROOT / "microduck-stage12-local.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.review = json.loads((ROOT / "publication-review.json").read_bytes())
        self.base = dict(head=c.BASE_COMMIT, branch="double-balance", tag_commit=c.BASE_COMMIT,
                         tag_object=c.TAG_OBJECT, origin=p.ORIGIN)
        self.followup = dict(self.base, head=p.REVIEWED_HEAD, parent=c.BASE_COMMIT,
            documentation_changes={r["path"]: dict(status={"added": "A", "modified": "M"}[r["status"]],
                                                   git_blob_sha1=r["git_blob_sha1"]) for r in self.review["files"]})

    def test_frozen_base_and_exact_reviewed_followup_accepted(self):
        p.validate_publication(self.base, self.review)
        p.validate_publication(self.followup, self.review)

    def test_other_head_tag_branch_or_origin_rejected(self):
        for field, wrong in (("head", "a" * 40), ("tag_commit", p.REVIEWED_HEAD),
                             ("tag_object", "a" * 40), ("branch", "other"), ("origin", "https://example.invalid")):
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                p.validate_publication(dict(self.followup, **{field: wrong}), self.review)

    def test_extra_change_wrong_blob_or_wrong_parent_rejected(self):
        mutations = [copy.deepcopy(self.followup) for _ in range(3)]
        mutations[0]["documentation_changes"]["src/physics.py"] = dict(status="M", git_blob_sha1="f" * 40)
        next(iter(mutations[1]["documentation_changes"].values()))["git_blob_sha1"] = "f" * 40
        mutations[2]["parent"] = "f" * 40
        for observed in mutations:
            with self.assertRaises(RuntimeError):
                p.validate_publication(observed, self.review)

    def test_failed_identity_retains_raw_local_refs(self):
        values = {("rev-parse", "HEAD"): "e" * 40, ("branch", "--show-current"): "double-balance",
                  ("rev-parse", "refs/tags/stage-11-complete"): c.TAG_OBJECT,
                  ("rev-parse", "stage-11-complete^{}"): c.BASE_COMMIT,
                  ("config", "--get", "remote.origin.url"): p.ORIGIN}
        with tempfile.TemporaryDirectory() as folder, patch.object(launcher, "git", side_effect=lambda *a: values[a].encode()):
            with self.assertRaises(RuntimeError):
                launcher.identity(Path(folder))
            saved = json.loads((Path(folder) / "identity-observed-before.json").read_bytes())
            self.assertEqual(saved["head"], "e" * 40)
            self.assertEqual(saved["tag_object"], c.TAG_OBJECT)
            self.assertIn("failure", saved)


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.archive = self.folder / "return.zip"
        self.archive.write_bytes(bytes(range(256)) * 2100)
        self.destination = self.folder / "transfer"
        self.branch = "stage12-local-completion-v5-transfer-20261015t180000z-abcdef"
        self.summary = dict(archive_sha256=t.digest(self.archive.read_bytes()), archive_bytes=self.archive.stat().st_size)

    def prepare(self):
        return t.prepare_transfer(self.archive, self.destination, self.branch, self.summary)

    def test_real_git_commit_roundtrip_and_idempotence(self):
        first = self.prepare()
        second = self.prepare()
        self.assertEqual(first, second)
        manifest = json.loads((self.destination / "return-manifest.json").read_bytes())
        self.assertEqual(len(manifest["parts"]), 2)
        rebuilt = b"".join(t.decode_part((self.destination / row["file"]).read_bytes(), row) for row in manifest["parts"])
        self.assertEqual(rebuilt, self.archive.read_bytes())
        self.assertEqual(t.local_git(self.destination, "config", "--local", "--get", "push.default").strip(), b"current")
        self.assertEqual(first["upload_command"], "git -C " + str(self.destination) + " push origin HEAD")
        self.assertFalse(first["automatic_push"])

    def test_interrupted_commit_can_finish_without_recreating_payload(self):
        real = t.local_git
        def interrupted(folder, *args, **kwargs):
            if args[0] == "commit":
                raise RuntimeError("simulated local commit interruption")
            return real(folder, *args, **kwargs)
        with patch.object(t, "local_git", side_effect=interrupted), self.assertRaises(RuntimeError):
            self.prepare()
        manifest_before = (self.destination / "return-manifest.json").read_bytes()
        result = self.prepare()
        self.assertEqual(manifest_before, (self.destination / "return-manifest.json").read_bytes())
        self.assertEqual(len(result["commit"]), 40)

    def test_unknown_existing_file_is_preserved(self):
        self.destination.mkdir()
        unknown = self.destination / "keep-me.txt"
        unknown.write_text("original")
        with self.assertRaises(RuntimeError):
            self.prepare()
        self.assertEqual(unknown.read_text(), "original")
        self.assertFalse((self.destination / ".git").exists())

    def test_existing_changed_part_is_not_overwritten(self):
        self.prepare()
        part = next(self.destination.glob("*.base64.txt"))
        part.write_bytes(b"changed")
        with self.assertRaises(RuntimeError):
            self.prepare()
        self.assertEqual(part.read_bytes(), b"changed")

    def test_archive_identity_and_network_commands_fail_closed(self):
        self.archive.write_bytes(b"changed")
        with self.assertRaises(RuntimeError):
            self.prepare()
        for command in ("push", "fetch", "clone", "merge", "reset"):
            with self.assertRaises(RuntimeError):
                t.local_git(self.folder, command)
        self.assertFalse(self.destination.exists())

    def test_existing_started_marker_never_launches_child(self):
        base = self.folder / "run-base"
        base.mkdir()
        raw = t.json_bytes(dict(run=str(base / "old-run")))
        (base / "started-v1.json").write_bytes(raw)
        with patch.object(launcher, "BASE", base), patch.object(launcher.subprocess, "Popen", side_effect=AssertionError("child launched")):
            self.assertEqual(launcher.run(), 2)
        self.assertEqual((base / "started-v1.json").read_bytes(), raw)

    def test_zip_manifest_corruption_is_rejected(self):
        package, report = b'{"files":{}}', t.json_bytes(dict(status="test"))
        rows = {"program/PACKAGE.json": package, "evidence/stage12-report.json": report}
        manifest = {k: dict(bytes=len(v), sha256=t.digest(v)) for k, v in rows.items()}
        zip_path = self.folder / "checked.zip"
        with zipfile.ZipFile(zip_path, "x") as z:
            for name, raw in rows.items():
                z.writestr(name, raw)
            z.writestr("RETURN-MANIFEST.json", t.json_bytes(manifest))
        self.assertEqual(t.verify_return(zip_path)["members"], 2)
        bad = self.folder / "bad.zip"
        with zipfile.ZipFile(bad, "x") as z:
            for name, raw in rows.items():
                z.writestr(name, raw + b"x")
            z.writestr("RETURN-MANIFEST.json", t.json_bytes(manifest))
        with self.assertRaises(RuntimeError):
            t.verify_return(bad)

    def test_blocked_new_run_still_prepares_manual_transfer_and_can_recover_packaging_only(self):
        program = self.folder / "program"
        program.mkdir()
        (program / "PACKAGE.json").write_bytes(t.json_bytes(dict(files={})))
        (program / "prior-readiness-v7").mkdir()
        (program / "prior-readiness-v7/audit.json").write_bytes((ROOT / "prior-readiness-v7/audit.json").read_bytes())
        (program / "prior-readiness-v6").mkdir()
        (program / "prior-readiness-v6/audit.json").write_bytes((ROOT / "prior-readiness-v6/audit.json").read_bytes())
        (program / "prior-readiness-v5").mkdir()
        (program / "prior-readiness-v5/audit.json").write_bytes((ROOT / "prior-readiness-v5/audit.json").read_bytes())
        (program / "prior-readiness-v4").mkdir()
        (program / "prior-readiness-v4/audit.json").write_bytes((ROOT / "prior-readiness-v4/audit.json").read_bytes())
        (program / "prior-readiness-v3").mkdir()
        (program / "prior-readiness-v3/audit.json").write_bytes((ROOT / "prior-readiness-v3/audit.json").read_bytes())
        (program / "prior-readiness-v2").mkdir()
        (program / "prior-readiness-v2/audit.json").write_bytes((ROOT / "prior-readiness-v2/audit.json").read_bytes())
        repo = self.folder / "source"
        interpreter = repo / ".venv/bin/python"
        interpreter.parent.mkdir(parents=True)
        interpreter.write_text("unused fixture")
        base = self.folder / "integration-v1"
        with patch.object(launcher, "BASE", base), patch.object(launcher, "ROOT", program), \
             patch.object(launcher, "REPO", str(repo)), \
             patch.object(launcher, "identity", side_effect=RuntimeError("injected identity mismatch")), \
             patch.object(launcher.subprocess, "Popen", wraps=launcher.subprocess.Popen) as popen:
            self.assertEqual(launcher.run(38720, 43200), 1)
            self.assertTrue(all(call.args[0][0] == "git" for call in popen.call_args_list))
            original = json.loads((base / "transfer-receipt.json").read_bytes())
            report = t.verify_return(next(base.glob("stage12-local-return-*.zip")))["report"]
            self.assertEqual(report["status"], "LOCAL_COMPLETION_STOPPED_PRESERVE")
            self.assertEqual(report["canonical_before"], report["canonical_after"])
            self.assertEqual(launcher.prepare_existing_transfer(), original)
            self.assertEqual(launcher.run(), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
