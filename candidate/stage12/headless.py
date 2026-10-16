"""Configure import discovery only; no viewer, context, process or native step."""
from pathlib import Path
import importlib.metadata as meta
import json
import os
import sys

from .contracts import require
from .source_capture import file_identity


def bundled_glfw(dist):
    # Use an exact wheel member, never a system search or subprocess probe.
    candidates = [Path(dist.locate_file(f)) for f in dist.files or []
                  if str(f) == "glfw/x11/libglfw.so"]
    require(len(candidates) == 1, "Expected exactly one bundled glfw/x11/libglfw.so")
    path = candidates[0]
    require(path.is_absolute(), "Bundled GLFW path must be absolute")
    identity = file_identity(path)
    require(identity["bytes"] > 0, "Bundled GLFW library is empty")
    return path, identity


def prepare(root, out, distribution=meta.distribution):
    """Validate reviewed discovery code before importing installed native packages."""
    report = dict(status="STARTED", graphics_contexts=0, subprocess_probes=0,
                  reviewed_sources=[], library_loaded_by_this_function=False)
    try:
        require(not any(n in sys.modules for n in ("mujoco", "glfw")),
                "Headless setup must precede MuJoCo/GLFW imports")
        provenance = json.loads((root / "reviewed-import-sources/provenance.json").read_bytes())
        for row in provenance["sources"]:
            dist = distribution(row["distribution"])
            require(dist.version == row["version"], "Import dependency version differs: " + row["distribution"])
            installed = Path(dist.locate_file(row["installed"]))
            reference = root / "reviewed-import-sources" / row["reviewed"]
            expected, actual = file_identity(reference), file_identity(installed)
            require(expected["git_blob_sha1"] == row["sha"], "Bundled upstream source identity differs")
            report["reviewed_sources"].append(dict(path=str(installed), expected=expected, actual=actual))
            require(actual == expected, "Installed import source differs from reviewed exact version: " + str(installed))
        library, identity = bundled_glfw(distribution("glfw"))
        os.environ["MUJOCO_GL"] = "disable"
        os.environ["PYGLFW_LIBRARY"] = str(library)
        report.update(status="HEADLESS_DISCOVERY_CONFIGURED", MUJOCO_GL="disable",
                      PYGLFW_LIBRARY=str(library), library_identity=identity,
                      native_library_binding_expected=True,
                      note="Installed GLFW may bind its bundled DLL for mujoco.viewer imports; init/window/context/physics remain blocked.")
        return report
    except BaseException as exc:
        report.update(status="HEADLESS_DISCOVERY_BLOCKED", error=repr(exc))
        raise
    finally:
        with (out / "headless-setup.json").open("x", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, sort_keys=True, indent=2)
            f.write("\n")
