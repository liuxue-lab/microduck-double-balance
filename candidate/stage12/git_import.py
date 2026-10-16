"""Permit only the two reviewed GitPython import-time `git version` queries."""
import importlib
import importlib.metadata as meta
import json
import os
from pathlib import Path
import stat
import sys

from .contracts import require
from .source_capture import file_identity

GIT = "/usr/bin/git"
GIT_ENV = {"GIT_PYTHON_GIT_EXECUTABLE": GIT, "GIT_CONFIG_NOSYSTEM": "1",
           "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null",
           "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"}


def is_null_device(path):
    """Recognize only the literal Linux null character device, never an alias."""
    if path not in ("/dev/null", b"/dev/null"):
        return False
    info = os.lstat(path)
    return (stat.S_ISCHR(info.st_mode) and
            os.major(info.st_rdev) == 1 and os.minor(info.st_rdev) == 3)


class GitImportProbe:
    def __init__(self):
        self.active = False
        self.sources = {}
        self.binary_identity = None
        self.cwd = None
        self.module_version = None
        self.report = dict(status="NOT_CONFIGURED", command=[GIT, "version"],
                           maximum_queries=2, permitted_queries=[], completed_queries=0,
                           null_device_write_opens=0, repository_commands=0,
                           network_commands=0, installed_files_modified=False)

    def configure(self, root, out, distribution=meta.distribution):
        require(self.binary_identity is None, "Git import probe cannot be configured twice")
        require(not any(n == "git" or n.startswith("git.") for n in sys.modules),
                "GitPython must not be imported before its guarded window")
        row = json.loads((root / "reviewed-import-sources/gitpython-provenance.json").read_bytes())
        require(row["read_only_import_command"] == [GIT, "version"] and row["import_probe_count"] == 2,
                "Reviewed Git import contract differs")
        dist = distribution(row["distribution"])
        require(dist.version == row["version"] == "3.1.46", "GitPython version differs from reviewed 3.1.46")
        reviewed = []
        for src in row["sources"]:
            reference = root / "reviewed-import-sources" / src["reviewed"]
            installed = Path(dist.locate_file(src["installed"]))
            expected, actual = file_identity(reference), file_identity(installed)
            require(expected == {key: src[key] for key in ("bytes", "sha256", "git_blob_sha1")},
                    "Bundled GitPython source identity differs")
            if src["installed"] == "git/__init__.py":
                # The reviewed upstream build stamps one version line in release
                # wheels/sdists. Accept only the two complete known byte strings.
                stamp = row["release_version_stamp"]
                setup = root / "reviewed-import-sources" / stamp["source"]
                require(file_identity(setup)["git_blob_sha1"] == stamp["source_git_blob_sha1"],
                        "GitPython version stamp build source differs")
                raw = reference.read_bytes()
                require(raw.count(stamp["upstream_line"].encode()) == 1, "GitPython version stamp is ambiguous")
                release = root / "reviewed-import-sources" / stamp["reviewed"]
                require(release.read_bytes() == raw.replace(stamp["upstream_line"].encode(), stamp["release_line"].encode())
                        and file_identity(release) == stamp["identity"], "GitPython release stamp differs")
                require(actual == expected or actual == stamp["identity"],
                        "Installed GitPython import source differs: " + str(installed))
                self.module_version = "git" if actual == expected else "3.1.46"
                self.report["source_form"] = "upstream_tag" if actual == expected else "release_version_stamp"
            else:
                require(actual == expected, "Installed GitPython import source differs: " + str(installed))
            self.sources[src["installed"]] = (str(installed), actual)
            reviewed.append(dict(path=str(installed), identity=actual))
        require(set(self.sources) == {"git/__init__.py", "git/cmd.py", "git/remote.py"},
                "GitPython reviewed source roster differs")
        require(is_null_device("/dev/null"), "Expected the real Linux /dev/null character device")
        require(os.access(GIT, os.X_OK), "Reviewed absolute Git binary is not executable")
        self.binary_identity = file_identity(GIT)
        self.cwd = str(out.resolve())
        require(Path.cwd().resolve() == out.resolve(), "Git import must run inside the new output directory")
        require(not any(k.startswith("GIT_") for k in os.environ),
                "Launcher must clear inherited Git environment variables")
        os.environ.update(GIT_ENV)
        self.report.update(status="VERIFIED_READ_ONLY_IMPORT_READY", version=dist.version,
                           reviewed_sources=reviewed, binary_identity=self.binary_identity,
                           cwd=self.cwd, git_environment=dict(GIT_ENV))

    def _reviewed_caller(self):
        expected = {("git.cmd", "execute", self.sources["git/cmd.py"][0]),
                    ("git", "refresh", self.sources["git/__init__.py"][0])}
        seen = set()
        frame = sys._getframe(1)
        while frame is not None:
            seen.add((frame.f_globals.get("__name__"), frame.f_code.co_name, frame.f_code.co_filename))
            frame = frame.f_back
        return expected <= seen

    def handle(self, event, args):
        if event != "subprocess.Popen" or not self.active:
            return False
        executable, argv, cwd, env = args
        if executable != GIT or not isinstance(argv, (list, tuple)) or list(argv) != [GIT, "version"]:
            return False
        require(cwd == self.cwd, "Git version query cwd differs from the new run")
        require(isinstance(env, dict) and {k: v for k, v in env.items() if k.startswith("GIT_")} == GIT_ENV,
                "Git version query environment differs")
        require(self._reviewed_caller(), "Git version query did not originate in reviewed import refresh")
        require(len(self.report["permitted_queries"]) < 2, "Additional Git version query exceeds import allowance")
        self.verify_unchanged()
        self.report["permitted_queries"].append(dict(argv=list(argv), cwd=cwd,
                                                      audit_event="subprocess.Popen"))
        return True

    def preload(self):
        require(self.report["status"] == "VERIFIED_READ_ONLY_IMPORT_READY", "Git import window is not ready")
        self.active = True
        try:
            module = importlib.import_module("git")
            require(module.__version__ == self.module_version and module.GIT_OK is True and
                    module.Git.GIT_PYTHON_GIT_EXECUTABLE == GIT,
                    "GitPython did not initialize normally with the reviewed binary")
            require(len(self.report["permitted_queries"]) == 2,
                    "GitPython import did not complete the two reviewed version queries")
            self.verify_unchanged()
            self.report.update(status="READ_ONLY_GITPYTHON_IMPORT_PASS", completed_queries=2,
                               git_ok=True)
        except BaseException as exc:
            # Popen's audit event precedes exec. If initialization failed after a
            # permit, do not invent either success or zero actual child execution.
            self.report.update(status="READ_ONLY_GITPYTHON_IMPORT_BLOCKED", error=repr(exc),
                               completed_queries=None if self.report["permitted_queries"] else 0)
            raise
        finally:
            self.active = False

    def verify_unchanged(self):
        for path, identity in self.sources.values():
            require(file_identity(path) == identity, "GitPython source changed during import/preflight")
        if self.binary_identity is not None:
            require(file_identity(GIT) == self.binary_identity, "Git binary changed during import/preflight")
