"""Reuse the verified integration-v1 archive; add packaging metadata only."""
from pathlib import Path
import hashlib
import json
import zipfile
from integration.durable import require, file_digest, publish, encoded, write_json


def prepare_inputs(root, out, repo):
    prior = json.loads((root / "prior-integration-v1/audit.json").read_bytes())
    row = prior["source_bundle"]
    source = Path(row["path"])
    require(source.is_file() and not source.is_symlink() and source.stat().st_size == row["bytes"] and
            file_digest(source) == row["sha256"], "Completed source bundle changed; do not recapture")
    with zipfile.ZipFile(source) as z:
        manifest = json.loads(z.read("SOURCE-BUNDLE-MANIFEST.json"))
        require(len(z.namelist()) == 407 and len(set(z.namelist())) == 407 and z.testzip() is None,
                "Prior source ZIP roster/CRC differs")
        files = manifest["files"]
        require(set(z.namelist()) == set(files) | {"SOURCE-BUNDLE-MANIFEST.json"}, "Prior source roster differs")
        for name, identity in files.items():
            p = Path(name)
            require(not p.is_absolute() and ".." not in p.parts and "\\" not in name, "Unsafe source path")
            raw = z.read(name)
            require(len(raw) == identity["bytes"] and hashlib.sha256(raw).hexdigest() == identity["sha256"],
                    "Prior source member differs: "+name)
        # uv_build requires readme/license metadata. They were outside the
        # original Python/asset allowlist and are added here before cloud setup.
        supplements = {}
        build_pins = json.loads((root / 'cloud-build-metadata-pins.json').read_bytes())
        for name in ("README.md", "LICENSE", "NOTICE", "LICENSE-HARDWARE"):
            path = repo / name
            require(path.is_file() and not path.is_symlink() and path.stat().st_size < 1024**2,
                    "Project build metadata missing: "+name)
            raw = path.read_bytes()
            require(hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == build_pins[name]['git_blob_sha1'],
                    'Pinned build metadata changed: '+name)
            supplements["project/"+name] = raw
        final = dict(files)
        final.update({k: dict(bytes=len(v), sha256=hashlib.sha256(v).hexdigest()) for k,v in supplements.items()})
        def writer(f):
            with zipfile.ZipFile(f, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as dest:
                for name in sorted(files):
                    dest.writestr(name, z.read(name))
                for name, raw in supplements.items():
                    dest.writestr(name, raw)
                dest.writestr("SOURCE-BUNDLE-MANIFEST.json", encoded(dict(
                    format="stage12-portable-inputs-v2", files=final,
                    reused_source_archive_sha256=row["sha256"], captured_sources_again=0,
                    build_metadata_added=list(supplements), cloud_environment_complete=False)))
        target = publish(out / "cloud-inputs.zip", writer)
    with zipfile.ZipFile(target) as z:
        require(z.testzip() is None, "Cloud inputs CRC differs")
    result = dict(path=str(target), bytes=target.stat().st_size, sha256=file_digest(target), members=len(final),
                  reused_input_sha256=row["sha256"], newly_collected_build_metadata=list(supplements),
                  prior_sources_recaptured=0, checkpoints=0, installed_cloud_environment=False)
    write_json(out / "cloud-inputs.json", result)
    return result
