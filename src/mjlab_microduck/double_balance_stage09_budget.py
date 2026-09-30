"""Persistent Stage 09 cloud-time ledger and process watchdog (stdlib only)."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

from mjlab_microduck.double_balance_stage09_plan import budget, budget_decision


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def create_ledger(path, gpu, started_at, *, now=None):
    """started_at is the retained deployment start, not a claimed power-on time."""
    current = time.time() if now is None else now
    start = datetime.fromisoformat(started_at)
    if start.tzinfo is None:
        raise ValueError("Budget start must include a timezone")
    epoch = start.timestamp()
    if not math.isfinite(epoch) or epoch > current + 1e-6:
        raise ValueError("Budget start cannot be in the future")
    epoch = min(epoch, current)  # ISO datetime rounds to microseconds.
    value = {"schema_version": 1, "campaign_id": str(uuid.uuid4()), **budget(gpu),
             "charged_updates": {name: 0 for name in ("F0", "F1", "F2", "F3")},
             "started_at": start.astimezone(timezone.utc).isoformat(),
             "started_epoch": epoch, "last_observed_epoch": current,
             "elapsed_seconds": current - epoch,
             "accounting_note": "continuous wall time since retained deployment start; subsequent idle/restart/offline gaps also count conservatively; earlier power-on time is unknown"}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return value


class BudgetLedger:
    def __init__(self, path, *, wall=time.time, monotonic=time.monotonic):
        self.path = Path(path).resolve()
        self.wall, self.monotonic = wall, monotonic
        self._wall_start, self._mono_start = wall(), monotonic()
        self.identity = self.snapshot()["campaign_id"]

    def snapshot(self):
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(self.path.read_text())
            expected = budget(state["gpu"])
            for key, value in expected.items():
                if state[key] != value:
                    raise ValueError(f"Budget contract changed: {key}")
            if state["schema_version"] != 1:
                raise ValueError("Unknown budget schema")
            if hasattr(self, "identity") and state["campaign_id"] != self.identity:
                raise ValueError("Campaign ledger replaced during execution")
            now = max(self.wall(), self._wall_start + self.monotonic() - self._mono_start)
            values = (now, state["started_epoch"], state["last_observed_epoch"], state["elapsed_seconds"])
            if not all(math.isfinite(v) for v in values) or state["elapsed_seconds"] < 0:
                raise ValueError("Invalid budget clock")
            # Across processes a backward wall-clock jump must not grant extra time.
            if now + 1 < state["last_observed_epoch"]:
                raise ValueError("Cloud clock moved backwards; inspect the ledger")
            state["last_observed_epoch"] = max(now, state["last_observed_epoch"])
            state["elapsed_seconds"] = max(state["elapsed_seconds"], now - state["started_epoch"])
            atomic_json(self.path, state)
            return state

    def remaining(self, *, training):
        state = self.snapshot()
        hours = state["maximum_cloud_hours"]
        if training:
            hours -= state["finalization_reserve_hours"]
        return max(0., hours * 3600 - state["elapsed_seconds"])

    def admit(self, updates, seconds_per_update, overhead=0):
        state = self.snapshot()
        return budget_decision(state["gpu"], spent_seconds=state["elapsed_seconds"],
                               additional_updates=updates, seconds_per_update=seconds_per_update,
                               overhead_seconds=overhead)["admit"]

    def charge_update(self, profile):
        # Charge BEFORE optimization: a failed/incomplete update still consumes budget.
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(self.path.read_text())
            if state['campaign_id'] != self.identity:
                raise ValueError('Campaign identity changed')
            counts = state['charged_updates']
            if set(counts) != {'F0','F1','F2','F3'} or any(type(n) is not int or n < 0 for n in counts.values()):
                raise ValueError('Invalid charged update ledger')
            if profile not in counts or counts[profile] >= 500 or sum(counts.values()) >= 2000:
                raise ValueError('Approved PPO update cap exhausted; no automatic extension')
            counts[profile] += 1
            atomic_json(self.path, state)
            return counts[profile]

    @contextmanager
    def job_lock(self):
        with self.path.with_suffix(".job.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("Another Stage 09 job is using this campaign") from exc
            yield


def supervise(argv, ledger, *, training, grace_seconds=120, poll_seconds=1.,
              timeout_seconds=None):
    """Independent watchdog; graceful SIGTERM, then kill the whole process group.

    Normal training saves at an update boundary. A hung process may only retain
    its preceding validated checkpoint. The watchdog never powers off a machine.
    """
    with ledger.job_lock():
        remaining = ledger.remaining(training=training)
        if remaining <= 0:
            raise ValueError("No time remains for this job")
        limit = min(remaining, timeout_seconds) if timeout_seconds is not None else remaining
        soft_deadline = time.monotonic() + limit
        absolute_deadline = time.monotonic() + max(0., ledger.remaining(training=False) - 30)
        hard_deadline = min(soft_deadline + grace_seconds, absolute_deadline)
        stop_requested = False
        handlers = {}
        def request_stop(*_):
            nonlocal stop_requested
            stop_requested = True
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, request_stop)
        child = None
        reason = None
        try:
            env = dict(os.environ, STAGE09_SUPERVISOR_PID=str(os.getpid()))
            child = subprocess.Popen(argv, env=env, start_new_session=True)
            while child.poll() is None:
                now = time.monotonic()
                if reason is None and (stop_requested or now >= soft_deadline):
                    reason = "user_signal" if stop_requested else "budget_or_job_timeout"
                    os.killpg(child.pid, signal.SIGTERM)
                    hard_deadline = min(hard_deadline, now + grace_seconds)
                if now >= hard_deadline:
                    reason = "forced_exit_after_grace"
                    os.killpg(child.pid, signal.SIGKILL)
                    break
                time.sleep(poll_seconds)
            code = child.wait()
            ledger.snapshot()
            return {"exit_code": code, "watchdog_reason": reason, "child_exited": True,
                    "shutdown_performed": False}
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
