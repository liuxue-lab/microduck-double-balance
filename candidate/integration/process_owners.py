"""Spawn/lifecycle candidate for independently owned native singleton processes.

Called by the bounded local campaign; never by the historical CPU-only probe.
Native plan admission is checked again in each child.
No process replacement or same-process reuse is supported after any failure.
"""
from pathlib import Path
import json
import multiprocessing
import os
import signal
import sys
import time
from .durable import require, write_json
from .native_factory import admission
from .process_vector import ProcessVector, owner_loop


def native_child(connection, logical_world, repo, out, plan, authorization):
    # Spawn (never fork) avoids sharing a CUDA context or global patched classes.
    root = Path(__file__).resolve().parents[1]
    out = Path(out)
    try:
        admission(plan, authorization)
        from integration.child_bootstrap import owner_bootstrap
        from integration.native_factory import owner_session
        with owner_bootstrap(root, out) as bootstrap:
            bootstrap['native_session_entered'] = True
            with owner_session(repo, out, plan, authorization) as session:
                connection.send(dict(sequence=0, owner=logical_world, ok=True, ready=True))
                owner_loop(connection, logical_world, session)
    except BaseException as exc:
        try:
            connection.send(dict(sequence=0, owner=logical_world, ok=False, error=repr(exc)))
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        connection.close()


class OwnerProcesses:
    def __init__(self, repo, output, plans, authorizations, torch, tensor_dict,
                 device="cpu", timeout_seconds=3600):
        require(len(plans) == len(authorizations) and 1 <= len(plans) <= 2, "N=1/2 owner roster required")
        for i, (plan, authorization) in enumerate(zip(plans, authorizations, strict=True)):
            require(plan["logical_world"] == i, "Logical owner order differs")
            admission(plan, authorization)
        self.output, self.processes, self.connections = Path(output), [], []
        self.output.mkdir(parents=False)
        write_json(self.output / "started-v1.json", dict(plans=plans, authorizations=authorizations, no_retry=True))
        self.vector, self.closed = None, False
        ctx = multiprocessing.get_context("spawn")
        try:
            for i, (plan, authorization) in enumerate(zip(plans, authorizations, strict=True)):
                parent, child = ctx.Pipe(duplex=True)
                process = ctx.Process(target=native_child,
                    args=(child, i, str(repo), str(self.output / ("owner-%d" % i)), plan, authorization), daemon=False)
                process.start()
                child.close()
                self.processes.append(process)
                self.connections.append(parent)
            deadline = time.monotonic() + timeout_seconds
            for i, conn in enumerate(self.connections):
                require(conn.poll(max(0, deadline-time.monotonic())), "Owner initialization timed out")
                r = conn.recv()
                require(r.get("sequence") == 0 and r.get("owner") == i and r.get("ok") is True and r.get("ready") is True,
                        "Owner initialization failed: " + str(r))
            self.vector = ProcessVector(self.connections, torch, tensor_dict, device, timeout_seconds)
        except BaseException as original:
            try:
                self.close()
            except BaseException as cleanup:
                original.add_note('Owner initialization cleanup: '+repr(cleanup))
            raise

    def close(self):
        if self.closed:
            return
        errors = []
        if self.vector is not None and not self.vector.transport.stopped:
            try:
                self.vector.transport.exchange("close", [None] * len(self.connections))
            except BaseException as exc:
                errors.append(repr(exc))
        close_only = []
        if self.vector is not None and self.vector.transport.stopped:
            close_only = self.vector.transport.close_after_failure(timeout_seconds=60)
        rows = []
        cleanup_deadline = time.monotonic() + 60
        for i, p in enumerate(self.processes):
            p.join(timeout=max(0, cleanup_deadline-time.monotonic()))
            interrupted = p.is_alive()
            if interrupted:
                # Preserve WAL uncertainty; never replace/retry this owner.
                os.kill(p.pid, signal.SIGINT)
                p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join(timeout=10)
            if p.is_alive():
                p.kill()
                p.join(timeout=10)
            receipt = self.output / ("owner-%d/native-report.json" % i)
            row = dict(owner=i, exitcode=p.exitcode, interrupted=interrupted,
                       report=str(receipt), report_exists=receipt.is_file())
            if receipt.is_file():
                try:
                    report = json.loads(receipt.read_bytes())
                    row.update(status=report["status"], native_ledger=report["native_ledger"])
                except (OSError, ValueError, KeyError) as exc:
                    row['receipt_error'] = repr(exc)
                    errors.append(repr(exc))
            row["clean"] = (not interrupted and p.exitcode == 0 and receipt.is_file()
                            and row.get("status") == "STAGE12_NATIVE_CANDIDATE_FINISHED_REVIEW_REQUIRED"
                            and row.get("native_ledger", {}).get("exact") is True)
            rows.append(row)
        for conn in self.connections:
            conn.close()
        self.closed = True
        write_json(self.output / "owner-closeout.json", dict(owners=rows, errors=errors,
                   all_clean=bool(rows) and all(r["clean"] for r in rows) and not errors,
                   close_only_shutdown=close_only,
                   no_retry=True, training_ready=False))
        require(rows and all(r["clean"] for r in rows) and not errors,
                "Owner shutdown incomplete; inspect WAL and closeout; never retry")

    def __enter__(self):
        require(self.vector is not None, "Owners unavailable")
        return self.vector

    def __exit__(self, exc_type, exc, tb):
        try:
            self.close()
        except BaseException as cleanup:
            if exc is None:
                raise
            exc.add_note('Owner cleanup: '+repr(cleanup))
