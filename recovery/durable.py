"""Exclusive durable publication. Existing files and failed writes are retained."""
from pathlib import Path
import hashlib
import json
import os
import secrets


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def safe_parent(path):
    path = Path(path).absolute()
    for p in (path, *path.parents):
        require(not p.is_symlink(), "Symlink refused: " + str(p))
    require(path.parent.is_dir(), "Parent directory missing")
    return path


def publish(path, writer):
    """Write/fdatasync a unique temp; link atomically without replacing old data.

    Failed temp files and claim markers remain evidence. The parent must already
    exist. This targets the local Linux filesystem, not a network object store.
    """
    path = safe_parent(path)
    require(not path.exists(), "Existing output preserved: " + str(path))
    temp = path.with_name(path.name + ".partial-" + secrets.token_hex(8))
    with temp.open("xb") as f:
        writer(f)
        f.flush()
        os.fsync(f.fileno())
    os.link(temp, path, follow_symlinks=False)  # EEXIST is an error, never overwrite.
    sync_dir(path.parent)
    temp.unlink()
    sync_dir(path.parent)
    return path


def write_json(path, value):
    raw = encoded(value)
    return publish(path, lambda f: f.write(raw))


class UpdateLedger:
    """One process, append-only update attempts. Failed attempts are never retried."""
    def __init__(self, path, identity, max_updates, start_update=0):
        require(type(max_updates) is int and max_updates >= 0, "Invalid update budget")
        require(type(start_update) is int and start_update >= 0, "Invalid starting counter")
        self.path = safe_parent(path)
        self.identity, self.max_updates = identity, max_updates
        self.stream = self.path.open("xb", buffering=0)
        self.seq, self.head, self.pending = 0, "0" * 64, None
        self.completed, self.attempted = start_update, 0
        self.failed = False
        self.append("created", identity=identity, max_updates=max_updates, start_update=start_update)
        sync_dir(self.path.parent)

    def append(self, kind, **fields):
        require(not self.failed, "Poisoned update ledger; preserve, do not retry")
        row = dict(sequence=self.seq + 1, previous_sha256=self.head, kind=kind, **fields)
        raw = encoded(row)
        try:
            require(self.stream.write(raw) == len(raw), "Partial ledger append")
            os.fsync(self.stream.fileno())
        except BaseException:
            self.failed = True
            raise
        self.seq += 1
        self.head = digest(raw)
        return row

    def begin(self, transitions):
        require(self.pending is None and self.attempted < self.max_updates,
                "Pending attempt or exhausted update budget")
        require(type(transitions) is int and transitions > 0, "Invalid transition reservation")
        self.append("update_started", next_iteration=self.completed, transitions_reserved=transitions)
        self.attempted += 1
        self.pending = self.completed
        return self.pending

    def commit(self, receipt_path):
        require(self.pending == self.completed, "No pending update to commit")
        receipt_path = safe_parent(receipt_path)
        row = json.loads(receipt_path.read_bytes())
        require(row["identity"] == self.identity and row["ledger_begin_sha256"] == self.head and
                row["completed_updates"] == self.completed + 1 and
                row["next_iteration"] == self.completed + 1, "Checkpoint/ledger boundary mismatch")
        verify_checkpoint(receipt_path, self.identity)
        self.append("update_committed", completed_updates=self.completed + 1,
                    receipt=receipt_path.name, receipt_sha256=file_digest(receipt_path))
        self.completed += 1
        self.pending = None

    def close(self):
        self.stream.close()


def inspect_ledger(path):
    head, sequence, pending, attempted, completed = "0" * 64, 0, None, 0, None
    first = last = None
    incomplete_tail = False
    for raw in Path(path).read_bytes().splitlines(keepends=True):
        if not raw.endswith(b"\n"):
            incomplete_tail = True
            break
        row = json.loads(raw)
        require(encoded(row) == raw and row["sequence"] == sequence + 1 and
                row["previous_sha256"] == head, "Ledger sequence/hash chain differs")
        if sequence == 0:
            require(row["kind"] == "created", "Missing ledger identity")
            first = row
            completed = row["start_update"]
        elif row["kind"] == "update_started":
            require(pending is None and row["next_iteration"] == completed and
                    attempted < first["max_updates"], "Attempt order/budget differs")
            pending, attempted = completed, attempted + 1
        elif row["kind"] == "update_committed":
            require(pending == completed and row["completed_updates"] == completed + 1,
                    "Completion without matching attempt")
            completed += 1
            pending = None
        else:
            raise RuntimeError("Unknown ledger event")
        sequence += 1
        head, last = digest(raw), row
    require(first is not None, "Empty ledger")
    return dict(identity=first["identity"], completed_updates=completed, attempted_updates=attempted,
                pending=pending, head=head, last=last, incomplete_tail=incomplete_tail,
                resumable=pending is None and not incomplete_tail and attempted > 0,
                remaining_updates=first["max_updates"] - attempted)


def save_checkpoint(folder, ledger, serializer, counters):
    """Serializer writes bytes (future torch.save accepts this file object).

    The serialization hook runs only after an update_started event. A failed
    save leaves that attempt pending. No automatic retry, refund, or torch.load.
    """
    require(ledger.pending == ledger.completed and not ledger.failed, "No pending update")
    folder = Path(folder)
    n = ledger.completed + 1
    stem = "update_%06d" % n
    write_json(folder / (stem + ".started.json"), dict(ledger_begin_sha256=ledger.head, update=n))
    payload = publish(folder / (stem + ".pt"), serializer)
    receipt = dict(format="stage12-update-checkpoint-v1", identity=ledger.identity,
                   payload=payload.name, bytes=payload.stat().st_size, sha256=file_digest(payload),
                   completed_updates=n, next_iteration=n, last_completed_index=n - 1,
                   ledger_begin_sha256=ledger.head, counters=counters,
                   continuation="fresh-environment-reset-and-zero-RNN-at-update-boundary",
                   bitwise_mid_rollout_resume=False)
    p = write_json(folder / (stem + ".receipt.json"), receipt)
    ledger.commit(p)
    return p


def verify_checkpoint(receipt_path, identity):
    receipt_path = safe_parent(receipt_path)
    row = json.loads(receipt_path.read_bytes())
    require(row["format"] == "stage12-update-checkpoint-v1" and row["identity"] == identity,
            "Task/config/source/dependency identity differs")
    n = row["completed_updates"]
    require(type(n) is int and n > 0 and row["last_completed_index"] == n - 1 and
            row["next_iteration"] == n, "Ambiguous checkpoint iteration")
    require(row["payload"] == "update_%06d.pt" % n, "Unsafe checkpoint payload name")
    payload = safe_parent(receipt_path.parent / row["payload"])
    require(payload.is_file() and payload.stat().st_size == row["bytes"] and
            file_digest(payload) == row["sha256"], "Checkpoint payload corrupted")
    return row


def verify_resume(ledger_path, receipt_path, identity):
    state = inspect_ledger(ledger_path)
    row = verify_checkpoint(receipt_path, identity)
    require(state["resumable"] and state["identity"] == identity and
            state["last"]["kind"] == "update_committed" and
            state["last"]["receipt"] == Path(receipt_path).name and
            state["last"]["receipt_sha256"] == file_digest(receipt_path) and
            state["completed_updates"] == row["completed_updates"],
            "Only the final durable committed update can continue; no retry of a pending update")
    return row
