"""Use Matplotlib's own no-fontconfig fallback while still denying all children."""
from pathlib import Path
import errno
import importlib.metadata as meta
import json
import os
import sys

from .contracts import require
from .source_capture import file_identity


class FontconfigFallback:
    def __init__(self):
        self.source = None
        self.identity = None
        self.report = dict(status="NOT_CONFIGURED", external_processes_allowed=0,
                           optional_fontconfig_denials=[], graphics_backend="Agg",
                           graphics_contexts_created=False, installed_files_modified=False)

    def configure(self, root, distribution=meta.distribution):
        require("matplotlib.font_manager" not in sys.modules, "Configure font discovery before its import")
        row = json.loads((root / "reviewed-import-sources/font-provenance.json").read_bytes())
        dist = distribution(row["distribution"])
        require(dist.version == row["version"], "Matplotlib import version differs from reviewed 3.10.9")
        expected = file_identity(root / "reviewed-import-sources" / row["reviewed"])
        source = Path(dist.locate_file(row["installed"]))
        actual = file_identity(source)
        self.report.update(source=str(source), expected=expected, actual=actual, version=dist.version)
        require(expected["git_blob_sha1"] == row["git_blob_sha1"] and actual == expected,
                "Installed Matplotlib font discovery source differs from reviewed version")
        self.source, self.identity = str(source), actual
        os.environ["MPLBACKEND"] = "Agg"
        self.report["status"] = "VERIFIED_OPTIONAL_FONTCONFIG_FALLBACK_READY"

    def handle(self, event, args):
        if event != "subprocess.Popen" or self.source is None:
            return
        executable, argv, _cwd, _env = args
        # Do not let an unrelated caller turn a forbidden operation into a handled
        # error. Exact caller source, function and argv are all required.
        if executable != "fc-list" or not isinstance(argv, (list, tuple)) or list(argv) not in (
                ["fc-list", "--help"], ["fc-list", "--format=%{file}\\n"]):
            return
        frame = sys._getframe(1)
        reviewed_caller = False
        while frame is not None:
            if (frame.f_code.co_filename == self.source and frame.f_code.co_name == "_get_fontconfig_fonts"
                    and frame.f_globals.get("__name__") == "matplotlib.font_manager"):
                reviewed_caller = True
                break
            frame = frame.f_back
        if not reviewed_caller:
            return
        require(file_identity(self.source) == self.identity, "Matplotlib font source changed during import")
        require(not self.report["optional_fontconfig_denials"], "Unexpected repeated optional fontconfig probe")
        self.report["optional_fontconfig_denials"].append(dict(argv=list(argv), spawned=False,
                exception="FileNotFoundError", handling="reviewed upstream except OSError -> []"))
        self.report["status"] = "OPTIONAL_FONTCONFIG_UNAVAILABLE_NATIVE_FONT_SCAN_FALLBACK"
        # font_manager.py explicitly treats OSError as an unavailable optional
        # fontconfig tool, then searches its packaged/system fonts itself. The
        # audit event occurs before exec: no subprocess is actually started.
        raise FileNotFoundError(errno.ENOENT, "Stage12 CPU preflight disables optional external font discovery", "fc-list")

    def verify_unchanged(self):
        if self.source is not None:
            require(file_identity(self.source) == self.identity, "Matplotlib font source changed during preflight")
