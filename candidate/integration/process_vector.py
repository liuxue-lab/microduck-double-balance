"""Process-isolated owner transport. No shared CUDA handles or global patches.

This is a candidate N=1/2 interface, not native nworld>1 batching. Every message
has a sequence and owner. A missing/failed reply poisons the whole vector; no
step resend, child replacement, transparent reset, or automatic retry exists.
"""
import time
import math
from functools import wraps
from .durable import require


def poison_on_error(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except BaseException:
            self.transport.stopped = True
            raise
    return call


class Fanout:
    def __init__(self, connections, timeout_seconds):
        require(1 <= len(connections) <= 2 and timeout_seconds > 0, "Invalid process vector scope")
        self.connections, self.timeout = list(connections), timeout_seconds
        self.sequence, self.stopped = 0, False
        self.last_sent = [0] * len(self.connections)
        self.last_reply = [0] * len(self.connections)
        self.last_ok = [True] * len(self.connections)
        self.last_operation = [None] * len(self.connections)
        self.uncertain = [False] * len(self.connections)

    def exchange(self, operation, payloads):
        require(not self.stopped, "Poisoned vector cannot retry")
        require(operation in ("step", "reset", "observe", "inspect", "close"), "Unknown operation")
        require(len(payloads) == len(self.connections), "Owner payload count differs")
        self.sequence += 1
        deadline = time.monotonic() + self.timeout
        try:
            # Send all commands before receiving; otherwise N=2 would serialize.
            for i, (conn, payload) in enumerate(zip(self.connections, payloads, strict=True)):
                self.uncertain[i] = True
                conn.send(dict(sequence=self.sequence, owner=i, operation=operation, payload=payload))
                self.last_sent[i] = self.sequence
                self.last_operation[i] = operation
                self.uncertain[i] = False
            rows = []
            for i, conn in enumerate(self.connections):
                remaining = max(0, deadline - time.monotonic())
                require(conn.poll(remaining), "Owner response timeout; attempted work is not refunded")
                row = conn.recv()
                require(row["sequence"] == self.sequence and row["owner"] == i, "Stale/wrong-owner response")
                self.last_reply[i] = self.sequence
                self.last_ok[i] = row["ok"] is True
                require(row["ok"] is True, "Owner failed: " + str(row.get("error")))
                rows.append(row["payload"])
            if operation == "close":
                self.stopped = True
            return rows
        except BaseException:
            self.stopped = True
            raise


    def close_after_failure(self, timeout_seconds=60):
        """Drain an outstanding reply, then send close only to an idle healthy owner.

        The poisoned vector remains poisoned. No step/reset/observe/inspect is
        resent and a successful sibling response never completes a failed batch.
        """
        require(self.stopped and 0 < timeout_seconds <= 60, "Close-only shutdown requires a poisoned vector")
        deadline = time.monotonic() + timeout_seconds
        rows = []
        for i, conn in enumerate(self.connections):
            row = dict(owner=i, close_requested=False, acknowledged=False)
            try:
                require(not self.uncertain[i], "Previous send outcome is uncertain")
                if self.last_reply[i] < self.last_sent[i]:
                    require(conn.poll(max(0, deadline-time.monotonic())), "Outstanding response did not finish before cleanup deadline")
                    reply = conn.recv()
                    require(reply.get("sequence") == self.last_sent[i] and reply.get("owner") == i,
                            "Outstanding response identity differs")
                    self.last_reply[i] = self.last_sent[i]
                    self.last_ok[i] = reply.get("ok") is True
                if not self.last_ok[i]:
                    row["state"] = "FAILED_OWNER_UNWINDS_ITS_SESSION"
                elif self.last_operation[i] == "close":
                    row.update(state="CLOSE_ALREADY_ACKNOWLEDGED", acknowledged=True)
                else:
                    seq = self.last_sent[i] + 1
                    self.uncertain[i] = True
                    conn.send(dict(sequence=seq, owner=i, operation="close", payload=None))
                    self.last_sent[i], self.last_operation[i] = seq, "close"
                    self.uncertain[i] = False
                    row["close_requested"] = True
                    require(conn.poll(max(0, deadline-time.monotonic())), "Close acknowledgment timed out")
                    reply = conn.recv()
                    require(reply.get("sequence") == seq and reply.get("owner") == i and reply.get("ok") is True,
                            "Close acknowledgment identity/status differs")
                    self.last_reply[i], self.last_ok[i] = seq, True
                    row.update(state="CLOSE_ACKNOWLEDGED_RECEIPT_PENDING", acknowledged=True)
            except (Exception, KeyboardInterrupt) as exc:
                row.update(state="CLOSE_UNCONFIRMED", error=repr(exc))
            rows.append(row)
        return rows


def tensor_wire(value):
    """CPU values only; pickled CUDA tensors must never cross process boundaries."""
    # TensorDict exposes detach/dtype/shape/tolist too. Classify containers first.
    # Its tolist() contains dictionaries, not tensor scalars (integration-v1 bug).
    if isinstance(value, dict) or hasattr(value, "items"):
        require(all(type(k) is str and k != "_tensor" for k in value.keys()),
                "IPC mapping keys must be strings without the reserved _tensor key")
        return {k: tensor_wire(v) for k, v in value.items()}
    if hasattr(value, "detach") and hasattr(value, "dtype"):
        require(str(value.dtype).split(".")[-1] in ("float32", "float64", "int64", "int32", "bool"),
                "Unsupported IPC dtype")
        return dict(_tensor=True, dtype=str(value.dtype).split(".")[-1],
                    shape=list(value.shape), values=value.detach().cpu().tolist())
    if isinstance(value, (tuple, list)):
        return [tensor_wire(v) for v in value]
    if value is None or type(value) in (bool, int, float, str):
        require(type(value) is not float or math.isfinite(value), "Nonfinite IPC scalar")
        return value
    raise RuntimeError("Unreviewed IPC value: " + type(value).__name__)


def wire_tensor(value, torch, device):
    if isinstance(value, dict) and "_tensor" in value:
        require(set(value) == {"_tensor", "dtype", "shape", "values"} and value["_tensor"] is True,
                "Invalid IPC tensor envelope")
        require(value["dtype"] in ("float32", "float64", "int64", "int32", "bool"), "Unsupported IPC dtype")
        shape = value["shape"]
        require(type(shape) is list and len(shape) <= 8 and
                all(type(n) is int and 0 <= n <= 65536 for n in shape) and math.prod(shape) <= 65536,
                "Invalid or oversized IPC tensor shape")
        def scalars(v, dims):
            if dims:
                require(type(v) is list and len(v) == dims[0], "IPC tensor nested shape differs")
                for child in v:
                    scalars(child, dims[1:])
            elif value["dtype"] == "bool":
                require(type(v) is bool, "IPC bool scalar differs")
            elif value["dtype"].startswith("int"):
                bits = int(value["dtype"][3:])
                require(type(v) is int and -(2**(bits-1)) <= v < 2**(bits-1), "IPC integer scalar differs")
            else:
                require(type(v) in (int, float) and math.isfinite(v), "IPC float scalar differs")
        scalars(value["values"], shape)
        t = torch.tensor(value["values"], dtype=getattr(torch, value["dtype"]), device=device)
        t = t.reshape(shape)  # Empty tensors lose trailing dimensions in tolist().
        require(bool(torch.isfinite(t).all()), "Nonfinite IPC tensor after dtype conversion")
        return t
    if isinstance(value, dict):
        require(all(type(k) is str for k in value), "Invalid IPC mapping key")
        return {k: wire_tensor(v, torch, device) for k, v in value.items()}
    if isinstance(value, list):
        return [wire_tensor(v, torch, device) for v in value]
    require(value is None or type(value) in (bool, int, float, str), "Unsupported IPC payload")
    require(type(value) is not float or math.isfinite(value), "Nonfinite IPC scalar")
    return value


class ProcessVector:
    """Uses already-started owner connections; process lifecycle belongs to caller."""
    def __init__(self, connections, torch, tensor_dict, device="cpu", timeout_seconds=3600):
        self.transport = Fanout(connections, timeout_seconds)
        self.torch, self.td, self.device = torch, tensor_dict, device
        self.num_envs, self.num_actions, self.max_episode_length = len(connections), 14, 500
        self.obs_buf = None

    def batch_observations(self, rows):
        require(all(set(r.keys()) == {"actor", "critic"} for r in rows), "IPC observation groups differ")
        values = {k: self.torch.cat([r[k] for r in rows], dim=0) for k in ("actor", "critic")}
        require(tuple(values["actor"].shape) == (self.num_envs, 61) and
                tuple(values["critic"].shape) == (self.num_envs, 85), "IPC observation batch differs")
        require(all(bool(self.torch.isfinite(v).all()) for v in values.values()), "Nonfinite IPC observations")
        return self.td(values, batch_size=[self.num_envs])

    @poison_on_error
    def get_observations(self):
        if self.obs_buf is None:
            rows = self.transport.exchange("observe", [None] * self.num_envs)
            self.obs_buf = self.batch_observations([wire_tensor(v, self.torch, self.device) for v in rows])
        require(not self.transport.stopped, "Vector unavailable")
        return self.obs_buf.clone()

    @poison_on_error
    def step(self, actions):
        require(tuple(actions.shape) == (self.num_envs, 14) and bool(self.torch.isfinite(actions).all()),
                "Bad vector action")
        rows = self.transport.exchange("step", [tensor_wire(actions[i:i+1]) for i in range(self.num_envs)])
        rows = [wire_tensor(v, self.torch, self.device) for v in rows]
        self.obs_buf = self.batch_observations([r[0] for r in rows])
        rewards = self.torch.cat([r[1] for r in rows])
        dones = self.torch.cat([r[2] for r in rows])
        transitions = [r[3]["stage12_transition"] for r in rows]
        merged = dict(terminal_observation=self.batch_observations([r["terminal_observation"] for r in transitions]),
                      bootstrap_applied=False)
        for key in ("terminated", "truncated", "bootstrap_mask"):
            merged[key] = self.torch.cat([r[key] for r in transitions])
        require(tuple(rewards.shape) == tuple(dones.shape) == (self.num_envs,) and
                bool(self.torch.isfinite(rewards).all()), "IPC reward/done batch differs")
        require(all(tuple(merged[k].shape) == (self.num_envs,) and merged[k].dtype == self.torch.bool
                    for k in ("terminated", "truncated", "bootstrap_mask")), "IPC flags differ")
        require(bool((dones == self.torch.logical_or(merged["terminated"], merged["truncated"]).long()).all()),
                "IPC done disagrees with terminal flags")
        require(bool((merged["bootstrap_mask"] == self.torch.logical_and(
            merged["truncated"], self.torch.logical_not(merged["terminated"]))).all()), "IPC bootstrap mask differs")
        # Per-owner extras retain their own keys and statistics; no accidental sum
        # of averages, mixed episode lengths, or silently dropped logger values.
        extras = dict(stage12_transition=merged, owner_extras=[r[3] for r in rows])
        return self.get_observations(), rewards, dones, extras

    @poison_on_error
    def inspect(self):
        """Synchronized read-only witnesses; no reset, observation compute or RNG."""
        return [wire_tensor(v, self.torch, self.device) for v in
                self.transport.exchange("inspect", [None] * self.num_envs)]

    @poison_on_error
    def reset(self, ids=None):
        ids = list(range(self.num_envs)) if ids is None else list(ids)
        require(all(type(i) is int for i in ids) and len(set(ids)) == len(ids) and
                all(0 <= i < self.num_envs for i in ids), "Invalid selective reset IDs")
        if not ids:
            return self.get_observations(), {}
        # Unselected owners only return their committed cache, never reset/RNG.
        rows = self.transport.exchange("reset", [i in ids for i in range(self.num_envs)])
        rows = [wire_tensor(v, self.torch, self.device) for v in rows]
        self.obs_buf = self.batch_observations([r[0] for r in rows])
        return self.get_observations(), dict(owner_extras=[r[1] for r in rows])


def owner_loop(connection, owner, session):
    """Runs inside an isolated process after native factory initialization."""
    expected = 1
    torch, bridge, journal = session["torch"], session["bridge"], session["journal"]
    while True:
        message = connection.recv()
        try:
            require(message["sequence"] == expected and message["owner"] == owner,
                    "Owner command order differs")
            operation, payload = message["operation"], message["payload"]
            if operation == "step":
                value = bridge.step(wire_tensor(payload, torch, bridge.device))
                journal.formal_steps += 1
                journal.checkpoint("public-step-completed")
            elif operation == "reset":
                require(type(payload) is bool, "Selective reset payload differs")
                value = bridge.reset() if payload else (bridge.get_observations(), {})
            elif operation == "observe":
                value = bridge.get_observations()
            elif operation == "inspect":
                require(payload is None and callable(session.get("inspect")), "Owner inspection unavailable")
                value = session["inspect"]()
            elif operation == "close":
                # Acknowledgment here is not the resource-cleanup receipt.
                value = dict(close_requested=True, cleanup_receipt_required=True)
            else:
                raise RuntimeError("Unknown owner command")
            connection.send(dict(sequence=expected, owner=owner, ok=True, payload=tensor_wire(value)))
            expected += 1
            if operation == "close":
                return
        except BaseException as exc:
            connection.send(dict(sequence=expected, owner=owner, ok=False, error=repr(exc)))
            raise
