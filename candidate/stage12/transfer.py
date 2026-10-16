"""Local, isolated evidence commit. This module has no push/fetch/network operation."""
from pathlib import Path
import base64
import hashlib
import json
import os
import re
import subprocess
import zipfile

from .contracts import REPO, require
from .publication import ORIGIN

PART_BYTES = 512 * 1024
MAX_ARCHIVE_BYTES = 96 * 1024 * 1024


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def same_or_create(path, raw):
    path = Path(path)
    require(not path.is_symlink(), "Transfer file is symlinked: " + str(path))
    if path.exists():
        require(path.is_file() and path.read_bytes() == raw, "Existing transfer file differs; preserve it: " + str(path))
    else:
        with path.open("xb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())


def verify_return(archive):
    """Check every ZIP member before creating a recoverable transport receipt."""
    archive = Path(archive)
    require(archive.is_file() and not archive.is_symlink() and
            0 < archive.stat().st_size <= MAX_ARCHIVE_BYTES, "Return archive missing/unsafe/over 96 MiB")
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        require(len(names) == len(set(names)), "Duplicate return ZIP names")
        for name in names:
            p = Path(name)
            require(not p.is_absolute() and ".." not in p.parts and "\\" not in name and not name.endswith("/"),
                    "Unsafe return ZIP member")
        require(sum(i.file_size for i in z.infolist()) <= MAX_ARCHIVE_BYTES + 1024 * 1024,
                "Expanded return ZIP exceeds bounded manifest allowance")
        manifest = json.loads(z.read("RETURN-MANIFEST.json"))
        require(set(names) == set(manifest) | {"RETURN-MANIFEST.json"}, "Return member roster differs")
        for name, row in manifest.items():
            raw = z.read(name)
            require(len(raw) == row["bytes"] and digest(raw) == row["sha256"], "Return member SHA differs: " + name)
        report = json.loads(z.read("evidence/stage12-report.json"))
        package_sha = digest(z.read("program/PACKAGE.json"))
    return dict(archive=str(archive), bytes=archive.stat().st_size, sha256=digest(archive.read_bytes()),
                members=len(manifest), report=report, package_manifest_sha256=package_sha)


def decode_part(raw, row):
    require(row["encoding"] == "base64-ascii-lines" and len(raw) == row["text_bytes"] and
            digest(raw) == row["text_sha256"], "Encoded shard validation failed")
    raw.decode("ascii")
    block = base64.b64decode(b"".join(raw.split()), validate=True)
    require(base64.encodebytes(block) == raw and 0 < len(block) <= PART_BYTES and
            len(block) == row["bytes"] and digest(block) == row["sha256"], "Decoded shard validation failed")
    return block


def payloads(archive, branch, summary):
    require(re.fullmatch(r"stage12-local-completion-v5-transfer-[0-9]{8}t[0-9]{6}z-[0-9a-f]{6}", branch) is not None,
            "Unexpected Stage12 transfer branch")
    archive = Path(archive)
    require(archive.is_file() and not archive.is_symlink(), "Archive missing or symlinked")
    raw = archive.read_bytes()
    require(0 < len(raw) <= MAX_ARCHIVE_BYTES, "Archive must be nonempty and at most 96 MiB")
    require(summary["archive_sha256"] == digest(raw) and summary["archive_bytes"] == len(raw),
            "Archive differs from the verified transfer summary")
    files, parts = {}, []
    for index, offset in enumerate(range(0, len(raw), PART_BYTES)):
        block = raw[offset:offset + PART_BYTES]
        encoded = base64.encodebytes(block)
        name = archive.name + ".part" + str(index).zfill(3) + ".base64.txt"
        row = dict(file=name, index=index, offset=offset, bytes=len(block), sha256=digest(block),
                   encoding="base64-ascii-lines", text_bytes=len(encoded), text_sha256=digest(encoded))
        require(decode_part(encoded, row) == block, "Shard round trip differs")
        files[name] = encoded
        parts.append(row)
    manifest = dict(format="microduck-return-ascii-parts-v2", file=archive.name, bytes=len(raw),
                    sha256=digest(raw), part_bytes_max=PART_BYTES, parts=parts, branch=branch,
                    scope="Stage12 local completion bounded native pilot evidence; independent archive branch; never merge; PPO=0 cloud=0")
    files["return-manifest.json"] = json_bytes(manifest)
    files["transfer-summary.json"] = json_bytes(summary)
    files["cloud-stage12.py"] = (Path(__file__).resolve().parents[1] / "cloud-stage12.py").read_bytes()
    files["transfer-owner.json"] = json_bytes(dict(format="microduck-stage12-transfer-owner-local-completion-v5", branch=branch,
                                                   archive_sha256=digest(raw)))
    files["README.txt"] = ("Microduck Stage12 local completion v5 evidence transport.\n"
                          "Separate transfer branch; never merge into double-balance.\n"
                          "No model/checkpoint or native training run is started by preparing this commit.\n"
                          "Read return-manifest.json and all parts at one full commit SHA.\n"
                          "Verify encoded SHA/size, strict base64 round trip, decoded SHA/size,\n"
                          "reconstructed ZIP SHA/size and RETURN-MANIFEST.json before review.\n"
                          "Pushing is manual. A transfer commit is not Stage12 acceptance.\n").encode()
    return files


def local_git(folder, *args, missing_ok=False):
    # No remote commands are included; all writes target the isolated transfer repository.
    require(args[0] in {"init", "symbolic-ref", "rev-parse", "config", "add", "commit",
                        "status", "ls-tree", "rev-list"}, "Transfer Git operation is not allowed")
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null", GIT_TERMINAL_PROMPT="0",
               GIT_OPTIONAL_LOCKS="0", LC_ALL="C")
    cmd = ["git", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false", "-c",
           "core.attributesFile=/dev/null", "-c", "commit.gpgsign=false", "-C", str(folder), *args]
    result = subprocess.run(cmd, env=env, capture_output=True, timeout=60)
    if missing_ok and result.returncode == 1:
        return None
    require(result.returncode == 0, "Local transfer Git failed: " + result.stderr.decode(errors="replace")[-2000:])
    return result.stdout


def verify_commit(folder, files, branch):
    require(local_git(folder, "symbolic-ref", "--short", "HEAD").decode().strip() == branch,
            "Transfer branch differs; do not checkout/reset it")
    head = local_git(folder, "rev-parse", "HEAD").decode().strip()
    require(len(local_git(folder, "rev-list", "--parents", "-n", "1", "HEAD").split()) == 1,
            "Transfer is not the expected independent root commit")
    tree = {}
    for row in local_git(folder, "ls-tree", "-r", "-z", "HEAD").split(b"\0"):
        if row:
            metadata, name = row.split(b"\t", 1)
            mode, kind, oid = metadata.decode().split()
            require(mode == "100644" and kind == "blob", "Unexpected transfer object type/mode")
            tree[name.decode()] = oid
    expected = {name: hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
                for name, raw in files.items()}
    require(tree == expected, "Transfer commit tree differs; do not amend or overwrite")
    require(not local_git(folder, "status", "--porcelain=v1", "--untracked-files=all").strip(),
            "Transfer worktree/index differs; preserve and inspect")
    return dict(commit=head, tree=local_git(folder, "rev-parse", "HEAD^{tree}").decode().strip())


def prepare_transfer(archive, destination, branch, summary):
    destination = Path(destination)
    require(not destination.resolve().is_relative_to(Path(REPO)), "Never write the canonical source repository")
    for p in (destination, *destination.parents):
        require(not p.is_symlink(), "Transfer directory is symlinked")
    files = payloads(archive, branch, summary)
    destination.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in destination.iterdir()}
    require(existing <= set(files) | {".git"}, "Unknown existing transfer files; preserve them")
    for name in existing - {".git"}:
        p = destination / name
        require(p.is_file() and not p.is_symlink() and p.read_bytes() == files[name],
                "Existing transfer payload differs; preserve it: " + name)
    metadata = destination / ".git"
    if metadata.exists():
        require(metadata.is_dir() and not metadata.is_symlink() and
                "transfer-owner.json" in existing, "Existing Git directory is not owned by this run")
        require(not any(p.is_symlink() for p in metadata.rglob("*")), "Symlink inside transfer Git directory")
        require(local_git(destination, "symbolic-ref", "--short", "HEAD").decode().strip() == branch,
                "Existing transfer branch differs")
    for name, raw in files.items():
        same_or_create(destination / name, raw)
    if not metadata.exists():
        local_git(destination, "init", "--template=", "-b", branch)
    origin = local_git(destination, "config", "--local", "--get", "remote.origin.url", missing_ok=True)
    require(origin is None or origin.decode().strip() == ORIGIN, "Existing transfer origin differs")
    for key in ("remote.origin.pushurl", "remote.origin.push"):
        require(local_git(destination, "config", "--local", "--get-all", key, missing_ok=True) is None,
                "Unexpected transfer push override: " + key)
    settings = {"remote.origin.url": ORIGIN, "push.default": "current", "commit.gpgsign": "false",
                "core.hooksPath": "/dev/null", "core.autocrlf": "false", "core.filemode": "false",
                "user.name": "Microduck Stage12 Evidence", "user.email": "stage12@local.invalid"}
    for key, value in settings.items():
        current = local_git(destination, "config", "--local", "--get", key, missing_ok=True)
        if current is None or current.decode().strip() != value:
            local_git(destination, "config", "--local", key, value)
    head = local_git(destination, "rev-parse", "--verify", "--quiet", "HEAD", missing_ok=True)
    if head is None:
        local_git(destination, "add", "--", *sorted(files))
        local_git(destination, "commit", "--no-gpg-sign", "-m", "Archive Stage12 local completion bounded native pilot evidence " + branch)
    identity = verify_commit(destination, files, branch)
    return dict(status="LOCAL_TRANSFER_COMMIT_READY_MANUAL_PUSH_PENDING", directory=str(destination),
                branch=branch, **identity, archive_sha256=summary["archive_sha256"],
                origin=ORIGIN, automatic_push=False, automatic_PR=False, automatic_merge=False,
                upload_command="git -C " + str(destination) + " push origin HEAD")
