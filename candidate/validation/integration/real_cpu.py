"""Real Torch/TensorDict IPC regression; no policies, optimizers or physics.

Uses two actual spawn processes, the production owner_loop and ProcessVector.
Fake environments are explicitly labelled. All children are joined before exit.
"""
from pathlib import Path
import argparse
import json
import multiprocessing
import os
import sys
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))


def backend():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import torch
    from tensordict import TensorDict
    torch.set_num_threads(1)
    assert torch.__version__.split("+")[0] == "2.9.1"
    import tensordict
    assert tensordict.__version__ == "0.10.0"
    def forbidden(*args, **kwargs):
        raise RuntimeError("CPU regression forbids CUDA, checkpoint IO and model/optimizer construction")
    for obj, name in ((torch, "load"), (torch, "save"), (torch.cuda, "_lazy_init"),
                      (torch.jit, "load"), (torch.nn.Linear, "__init__"),
                      (torch.nn.LSTM, "__init__"), (torch.optim.Adam, "__init__")):
        setattr(obj, name, forbidden)
    return torch, TensorDict


def child(conn, owner, target, fail_second):
    from integration.process_vector import owner_loop
    from integration.transitions import TerminalBridge
    from validation.integration.cpu_cases import fake_env
    torch, td = backend()
    env = fake_env(torch, "TensorDict", td, owner == 0, owner == 1)
    attempts = 0
    original = env.step
    def step(action):
        nonlocal attempts
        attempts += 1
        if fail_second and owner == 1 and attempts == 2:
            raise RuntimeError("injected CPU owner failure; do not retry")
        return original(action)
    env.step = step
    journal = SimpleNamespace(formal_steps=0, checkpoint=lambda *args: None)
    report = dict(owner=owner, expected_failure=fail_second and owner == 1)
    try:
        bridge = TerminalBridge(env, torch, td, 8)
        conn.send(dict(sequence=0, owner=owner, ok=True))
        owner_loop(conn, owner, dict(torch=torch, bridge=bridge, journal=journal))
        report["closed"] = True
    except BaseException as exc:
        report["error"] = repr(exc)
        if not (fail_second and owner == 1 and "injected CPU owner failure" in str(exc)):
            raise
    finally:
        report.update(attempted_steps=attempts, completed_steps=journal.formal_steps,
                      calls=env.calls, cuda_initialized=torch.cuda.is_initialized(),
                      native=0, PPO=0, cloud_training=0)
        Path(target).write_text(json.dumps(report, sort_keys=True))
        conn.close()


def spawn_case(torch, td, out, fail_second=False):
    from integration.process_vector import ProcessVector
    ctx = multiprocessing.get_context("spawn")
    children, connections = [], []
    vector = None
    try:
        for owner in (0, 1):
            parent, endpoint = ctx.Pipe()
            proc = ctx.Process(target=child, args=(endpoint, owner, str(out / ("child-%d.json" % owner)), fail_second))
            proc.start()
            endpoint.close()
            children.append(proc)
            connections.append(parent)
        for owner, conn in enumerate(connections):
            assert conn.poll(60), "CPU child startup timed out"
            assert conn.recv() == dict(sequence=0, owner=owner, ok=True)
        vector = ProcessVector(connections, torch, td, timeout_seconds=30)
        before = vector.get_observations()
        obs, reward, done, extra = vector.step(torch.zeros(2, 14))
        assert reward.tolist() == [3, 3] and done.tolist() == [1, 1]
        assert extra["stage12_transition"]["bootstrap_mask"].tolist() == [False, True]
        assert (before["actor"] == -10).all() and (obs["actor"] == -10).all()
        if fail_second:
            try:
                vector.step(torch.zeros(2, 14))
                raise AssertionError("Failed owner accepted")
            except RuntimeError as exc:
                assert "injected CPU owner failure" in str(exc)
            sequence = vector.transport.sequence
            try:
                vector.step(torch.zeros(2, 14))
                raise AssertionError("Poisoned vector retried")
            except RuntimeError as exc:
                assert "Poisoned" in str(exc)
            assert vector.transport.sequence == sequence
            # CPU fixture cleanup only: never resend step; tell healthy owner to close.
            connections[0].send(dict(sequence=sequence+1, owner=0, operation="close", payload=None))
            assert connections[0].poll(10) and connections[0].recv()["ok"]
        else:
            vector.reset([0])
            vector.reset([])
            vector.step(torch.zeros(2, 14))
            vector.transport.exchange("close", [None, None])
        for proc in children:
            proc.join(30)
            assert not proc.is_alive() and proc.exitcode == 0
        rows = [json.loads((out / ("child-%d.json" % i)).read_text()) for i in (0, 1)]
        assert all(not r["cuda_initialized"] for r in rows)
        if fail_second:
            assert [r["attempted_steps"] for r in rows] == [2, 2]
            assert [r["completed_steps"] for r in rows] == [2, 1]
        else:
            assert [r["calls"]["reset"] for r in rows] == [4, 3]
            assert [r["completed_steps"] for r in rows] == [2, 2]
        return dict(owners=rows, actual_spawn=True, actual_Torch_TensorDict=True, native_worlds=0)
    finally:
        for conn in connections:
            conn.close()
        for proc in children:
            if proc.is_alive():
                proc.terminate()
                proc.join(10)
            if proc.is_alive():
                proc.kill()
                proc.join(10)


def wire_case(torch, td):
    from integration.process_vector import tensor_wire, wire_tensor
    original = td(dict(actor=torch.ones(1, 61), critic=torch.zeros(1, 85)), batch_size=[1])
    assert hasattr(original, "detach") and hasattr(original, "dtype") and hasattr(original, "tolist")
    payload = dict(obs=original, nested=[dict(x=torch.tensor(7, dtype=torch.int64)),
                    torch.empty((0, 3), dtype=torch.float32)], flag=torch.tensor([True, False]))
    decoded = wire_tensor(tensor_wire(payload), torch, "cpu")
    original["actor"].zero_()
    assert decoded["obs"]["actor"].sum() == 61
    assert decoded["nested"][0]["x"].shape == torch.Size([])
    assert decoded["nested"][1].shape == torch.Size([0, 3])
    assert decoded["flag"].tolist() == [True, False]
    for dtype in (torch.float32, torch.float64, torch.int32, torch.int64, torch.bool):
        value = torch.tensor([0, 1], dtype=dtype)
        assert torch.equal(value, wire_tensor(tensor_wire(value), torch, "cpu"))
    bad = [dict(_tensor=True, dtype="float32", shape=[1], values=[float("nan")]),
           dict(_tensor=True, dtype="int64", shape=[1], values=[2**63]),
           dict(_tensor=True, dtype="bool", shape=[1], values=[1]),
           dict(_tensor=True, dtype="float32", shape=[1, 2], values=[[1.0]]),
           dict(_tensor=True, dtype="float32", shape=[1], values=[1e100]),
           dict(_tensor=True, dtype="float32", shape=[65537], values=[]),
           dict(_tensor=False), {1: "invalid key"}, object()]
    for item in bad:
        try:
            wire_tensor(item, torch, "cpu")
        except (RuntimeError, ValueError, TypeError):
            continue
        raise AssertionError("Malformed IPC accepted: " + repr(item))
    return dict(actual_TensorDict_ambiguity_tested=True, malformed_cases=len(bad),
                scalar_and_empty_shapes=True, dtypes=5, detached_snapshot=True)


def lifecycle_child(conn, owner, repo, folder, plan, authorization):
    """Explicit fake native factory for exercising the REAL process supervisor."""
    from integration.process_vector import owner_loop
    from integration.transitions import TerminalBridge
    from validation.integration.cpu_cases import fake_env
    torch, td = backend()
    out = Path(folder)
    out.mkdir()
    env = fake_env(torch, 'TensorDict', td, False, owner == 1)
    bridge = TerminalBridge(env, torch, td, plan['max_resets'])
    journal = SimpleNamespace(formal_steps=0, checkpoint=lambda *args: None)
    def inspect():
        return dict(public_steps=journal.formal_steps, macros=journal.formal_steps*10,
                    resets=bridge.resets, counts=dict(integrations_completed=journal.formal_steps*1600,
                                                     integrations_attempted=journal.formal_steps*1600),
                    rng_sha256='synthetic-'+str(bridge.resets),
                    observation_sha256=str(bridge.get_observations()['actor'][0,0].item()),
                    episode_length=[0], memory=dict(scope='CPU fixture only'), fixture=True)
    try:
        conn.send(dict(sequence=0, owner=owner, ok=True, ready=True))
        owner_loop(conn, owner, dict(torch=torch, bridge=bridge, journal=journal, inspect=inspect))
        (out/'native-report.json').write_text(json.dumps(dict(
            status='STAGE12_NATIVE_CANDIDATE_FINISHED_REVIEW_REQUIRED',
            native_ledger=dict(exact=True), fixture=True, actual_native_integrations=0)))
    finally:
        conn.close()


def campaign_case(torch, td, out):
    from unittest.mock import patch
    from integration.local_campaign import measure_point
    from integration.planning import proposal
    from integration import process_owners
    rows = []
    with patch.object(process_owners, 'native_child', lifecycle_child):
        for point in proposal()['points']:
            rows.append(measure_point(point, out/point['name'], torch, td, 60))
    assert [p['owners'] for p in rows] == [1, 2]
    assert all(p['closeout']['all_clean'] for p in rows)
    assert [r['public_steps'] for r in rows[0]['final']] == [5]
    assert [r['public_steps'] for r in rows[1]['final']] == [3,3]
    return dict(actual_spawn=True, actual_supervisor_and_campaign=True,
                native_factory='FAKE_CPU_FIXTURE', actual_native_integrations=0,
                synthetic_counters_are_not_native_evidence=True)


def run(out):
    from validation.integration.cpu_cases import all_cases
    torch, td = backend()
    rows = all_cases(torch, td)
    def record(name, operation):
        try:
            rows.append(dict(name=name, passed=True, result=operation()))
        except BaseException as exc:
            rows.append(dict(name=name, passed=False, error=repr(exc), traceback=traceback.format_exc()))
    record("wire-real-tensors-and-malformed-inputs", lambda: wire_case(torch, td))
    for failure in (False, True):
        target = out / ("spawn-failure" if failure else "spawn-normal")
        target.mkdir()
        record(target.name, lambda f=failure, t=target: spawn_case(torch, td, t, f))
    target = out / 'campaign-fixture'
    target.mkdir()
    record('production-supervisor-and-campaign-CPU-fixture', lambda: campaign_case(torch, td, target))
    result = dict(passed=all(r["passed"] for r in rows), cases=rows,
                  torch_version=torch.__version__, tensordict_version="0.10.0",
                  backend="actual CPU tensors and spawn; simulated environments",
                  native_integrations=0, real_environments=0, model_constructions=0,
                  optimizer_constructions=0, torch_load=0, PPO=0, cloud_training=0,
                  cuda_initialized=torch.cuda.is_initialized())
    (out / "real-cpu-report.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir()
    raise SystemExit(0 if run(args.output)["passed"] else 1)
