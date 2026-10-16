from pathlib import Path
import ast
import copy
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from integration.durable import (UpdateLedger, save_checkpoint, inspect_ledger, verify_resume,
    verify_checkpoint, write_json, publish, file_digest, require)
from integration.planning import owner_plan, proposal, validate_owner, recurrent_batches, throughput_row
from integration.process_vector import Fanout
from integration.runner import construction_gate, restore_learning_rate, load_committed_update, run_updates


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.identity = dict(task="stage12", config="abc", source="def", dependencies="ghi", seed=1111003)
        self.ledger = UpdateLedger(self.folder / "updates.jsonl", self.identity, 2)
    def tearDown(self):
        self.ledger.close()
        self.tmp.cleanup()
    def commit(self):
        self.ledger.begin(24)
        return save_checkpoint(self.folder, self.ledger, lambda f: f.write(b"not-a-model"),
                               dict(native_attempted=100, native_completed=100, public_transitions=24))
    def test_two_updates_have_no_duplicate_iteration(self):
        first, second = self.commit(), self.commit()
        self.assertNotEqual(first, second)
        row = verify_resume(self.folder / "updates.jsonl", second, self.identity)
        self.assertEqual((row["last_completed_index"], row["next_iteration"]), (1, 2))
        self.assertEqual(inspect_ledger(self.folder / "updates.jsonl")["remaining_updates"], 0)
        with self.assertRaises(RuntimeError): self.ledger.begin(24)
    def test_old_checkpoint_cannot_be_latest(self):
        old = self.commit()
        self.commit()
        with self.assertRaises(RuntimeError): verify_resume(self.folder / "updates.jsonl", old, self.identity)
    def test_writer_failure_retains_started_and_pending(self):
        self.ledger.begin(24)
        def fail(f):
            f.write(b"partial")
            raise OSError("simulated disk failure")
        with self.assertRaises(OSError): save_checkpoint(self.folder, self.ledger, fail, {})
        self.assertTrue((self.folder / "update_000001.started.json").exists())
        self.assertTrue(list(self.folder.glob("*.partial-*")))
        self.assertFalse(list(self.folder.glob("*.receipt.json")))
        self.assertFalse(inspect_ledger(self.folder / "updates.jsonl")["resumable"])
        with self.assertRaises(RuntimeError): self.ledger.begin(24)
        with self.assertRaises(RuntimeError): save_checkpoint(self.folder, self.ledger, lambda f: f.write(b"retry"), {})
    def test_pending_rollout_blocks_previous_resume(self):
        old = self.commit()
        self.ledger.begin(24)
        with self.assertRaises(RuntimeError): verify_resume(self.folder / "updates.jsonl", old, self.identity)
    def test_payload_corruption_blocks_before_deserialization(self):
        receipt = self.commit()
        (self.folder / "update_000001.pt").write_bytes(b"bad")
        with self.assertRaises(RuntimeError): verify_checkpoint(receipt, self.identity)
    def test_identity_mismatch_blocks(self):
        receipt = self.commit()
        with self.assertRaises(RuntimeError): verify_checkpoint(receipt, dict(self.identity, config="different"))
    def test_ledger_tamper_rejected(self):
        self.commit()
        path = self.folder / "updates.jsonl"
        path.write_bytes(path.read_bytes().replace(b'"transitions_reserved":24', b'"transitions_reserved":25'))
        with self.assertRaises(RuntimeError): inspect_ledger(path)
    def test_partial_tail_is_not_clean(self):
        receipt = self.commit()
        with (self.folder / "updates.jsonl").open("ab") as f: f.write(b'{"interrupted"')
        row = inspect_ledger(self.folder / "updates.jsonl")
        self.assertTrue(row["incomplete_tail"])
        with self.assertRaises(RuntimeError): verify_resume(self.folder / "updates.jsonl", receipt, self.identity)
    def test_existing_output_not_overwritten(self):
        target = self.folder / "retained"
        target.write_bytes(b"old")
        with self.assertRaises(RuntimeError): publish(target, lambda f: f.write(b"new"))
        self.assertEqual(target.read_bytes(), b"old")
    def test_symlink_refused(self):
        path = self.folder / "alias"
        path.symlink_to(self.folder / "outside")
        with self.assertRaises(RuntimeError): write_json(path, {})


class PlanningTests(unittest.TestCase):
    def test_complete_proposal_is_unapproved(self):
        p = proposal()
        self.assertEqual(p["authorization"], "PENDING_USER_BUDGET")
        self.assertEqual(p["native_integrations_max"], 38720)
        self.assertEqual(p["environment_constructions"], 3)
        self.assertEqual((p["PPO"], p["cloud_training"], p["torch_load"]), (0, 0, 0))
        self.assertGreater(p["native_GPU_forward_max"], 110 * 1413)
    def test_caps_cover_worst_case_and_cleanup(self):
        p = owner_plan(24, 26)
        validate_owner(p)
        self.assertEqual(p["caps"]["integrations"], 84480)
        self.assertEqual(p["caps"]["GPU_reset_data"], 26)
        self.assertEqual(p["caps"]["Scene_write"], 266)
        self.assertEqual(p["graph_limits"]["graph_destroy"], 1)
        self.assertEqual(p["graph_limits"]["launch"], p["caps"]["GPU_forward"])
    def test_old_tier_and_capacity_unchanged(self):
        base = json.loads((ROOT / "baseline/g11/plan.json").read_bytes())
        p = owner_plan(3, 4)
        self.assertEqual(p["tiers"], base["tiers"])
        self.assertTrue(p["exact_replay"])
        self.assertEqual(p["contact_layout"], base["contact_layout"])
    def test_edited_cap_cannot_pass(self):
        p = owner_plan(3, 4)
        p["caps"]["integrations"] -= 1
        with self.assertRaises(RuntimeError): validate_owner(p)
    def test_recurrent_N1_and_N2(self):
        for n, b in ((1, 1), (2, 1), (2, 2)):
            self.assertEqual(recurrent_batches(n, b)["transitions_per_update"], n * 24)
        for n, b in ((1, 4), (2, 4), (2, 3), (0, 1), (True, 1), (4, 2)):
            with self.assertRaises(RuntimeError): recurrent_batches(n, b)
    def test_throughput_counts_transitions_not_microsteps(self):
        row = throughput_row(4, 2., 1000, 1001, 200, 1000)
        self.assertEqual(row["public_transitions_per_s"], 2.)
        self.assertEqual(row["native_integrations_per_s"], 500.)
        self.assertFalse(row["stable_rate_claim"])
        with self.assertRaises(RuntimeError): throughput_row(4, 0., 1000, 1000, 200, 1000)
    def test_native_denied_before_creation(self):
        from integration.native_factory import owner_session
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "must-not-exist"
            with self.assertRaises(RuntimeError):
                with owner_session("/unused", p, owner_plan(3, 4), {}): pass
            self.assertFalse(p.exists())
    def test_training_denied_before_models(self):
        with self.assertRaises(RuntimeError): construction_gate({}, {})
        release = dict(physical_acceptance=True, full_episode_and_reset_accepted=True,
                       throughput_measured=True, cloud_package_verified=True)
        with self.assertRaises(RuntimeError): construction_gate(release, dict(authorization="PENDING_USER_BUDGET"))
    def test_lr_scalar_restored(self):
        alg = SimpleNamespace(optimizer=SimpleNamespace(param_groups=[{"lr": .0003}]), learning_rate=1.)
        restore_learning_rate(alg)
        self.assertEqual(alg.learning_rate, .0003)
        alg.optimizer.param_groups.append({"lr": .0002})
        with self.assertRaises(RuntimeError): restore_learning_rate(alg)


class RunnerLoopTests(unittest.TestCase):
    def fixture(self, fail_update=False):
        from contextlib import nullcontext
        from validation.integration.protocol_tensor import Torch, TD
        counts = dict(actions=0, steps=0, transitions=0, updates=0, saves=0)
        obs = TD(dict(actor=Torch.zeros(1,61), critic=Torch.zeros(1,85)), batch_size=[1])
        class Env:
            num_envs, device = 1, 'cpu'
            def get_observations(self): return obs.clone()
            def step(self, actions):
                counts['steps'] += 1
                return obs.clone(), Torch.zeros(1), Torch.tensor([0]), dict(stage12_transition=dict(
                    terminal_observation=obs.clone(), bootstrap_mask=Torch.tensor([False]), bootstrap_applied=False))
        class Algorithm:
            num_mini_batches, rnd = 1, None
            critic = SimpleNamespace(is_recurrent=False)
            def train_mode(self): pass
            def act(self, values): counts['actions'] += 1; return Torch.zeros(1,14)
            def process_env_step(self, *a): counts['transitions'] += 1
            def compute_returns(self, values): pass
            def update(self):
                if fail_update: raise RuntimeError('simulated optimizer failure; no real optimizer')
                counts['updates'] += 1
                return {'fake_loss':0.}
            def save(self): return dict(fake_algorithm=True)
        def serializer(state, f):
            counts['saves'] += 1
            f.write(json.dumps(state).encode())
        t = SimpleNamespace(inference_mode=nullcontext, get_rng_state=lambda: [], save=serializer,
                            cuda=SimpleNamespace(is_initialized=lambda: False))
        return Env(), Algorithm(), t, counts
    def test_two_fake_updates_have_48_transitions_and_unique_checkpoints(self):
        env, alg, t, counts = self.fixture()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); ledger = UpdateLedger(d/'ledger.jsonl', {'task':'fixture'}, 2)
            try:
                completed = run_updates(env, alg, t, ledger, d, 2, 1, lambda: dict(public_transitions=counts['steps']))
                self.assertEqual(completed, 2)
                self.assertEqual(counts, dict(actions=48, steps=48, transitions=48, updates=2, saves=2))
                row = verify_resume(d/'ledger.jsonl', d/'update_000002.receipt.json', {'task':'fixture'})
                self.assertEqual(row['next_iteration'], 2)
            finally: ledger.close()
    def test_failed_fake_update_is_pending_and_not_checkpointed(self):
        env, alg, t, counts = self.fixture(True)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); ledger = UpdateLedger(d/'ledger.jsonl', {'task':'fixture'}, 1)
            try:
                with self.assertRaises(RuntimeError):
                    run_updates(env, alg, t, ledger, d, 1, 1, lambda: {})
                self.assertEqual(counts['transitions'], 24)
                self.assertEqual(counts['saves'], 0)
                self.assertFalse(inspect_ledger(d/'ledger.jsonl')['resumable'])
                with self.assertRaises(RuntimeError): ledger.begin(24)
            finally: ledger.close()


class TransportTests(unittest.TestCase):
    def connection(self, owner, response=None):
        class Conn:
            def send(self, row): self.sent = row
            def poll(self, timeout): return response != "timeout"
            def recv(self):
                return response or dict(sequence=self.sent["sequence"], owner=owner, ok=True, payload=owner)
        return Conn()
    def test_ordered_fanout(self):
        v = Fanout([self.connection(0), self.connection(1)], 1)
        self.assertEqual(v.exchange("observe", [None, None]), [0, 1])
    def test_timeout_no_resend(self):
        v = Fanout([self.connection(0, "timeout")], 1)
        with self.assertRaises(RuntimeError): v.exchange("step", [{}])
        with self.assertRaises(RuntimeError): v.exchange("step", [{}])
        self.assertEqual(v.sequence, 1)
    def test_wrong_owner_poisoned(self):
        v = Fanout([self.connection(9)], 1)
        with self.assertRaises(RuntimeError): v.exchange("step", [{}])
        self.assertTrue(v.stopped)
    def test_stale_sequence_rejected(self):
        v = Fanout([self.connection(0, dict(sequence=0, owner=0, ok=True, payload={}))], 1)
        with self.assertRaises(RuntimeError): v.exchange("step", [{}])
    def test_child_failure_no_retry(self):
        v = Fanout([self.connection(0, dict(sequence=1, owner=0, ok=False, error="native failed"))], 1)
        with self.assertRaises(RuntimeError): v.exchange("step", [{}])
        self.assertTrue(v.stopped)


class SourceTests(unittest.TestCase):
    def test_runtime_30_macros_reset_tick_and_nonzero_global_counter(self):
        import time
        from collections import defaultdict
        from validation.integration.protocol_tensor import Torch
        from integration.durable import digest
        source = ast.parse((ROOT / "integration/native_runtime.py").read_text())
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "BudgetedRuntime")
        counts = defaultdict(int)
        class Budget:
            def reserve(self, **cost): pass
            def run(self, name, fn): counts[name] += 1; return fn()
        class Driver:
            def __init__(self, bridge, engine, ticks, emit, limit): self.ticks, self.commits = ticks, 0
            def advance(self):
                self.commits += 1
                return dict(replayed=True, public_ticks=[self.ticks[0]+1])
        class Evidence:
            def __init__(self, *args): self.started, self.closed, self.stopped = True, False, False
            def finalize(self): self.closed = True
        class Array:
            def numpy(self): return b"unchanged-endpoint"
        data = SimpleNamespace(**{n: Array() for n in ('qpos','qvel','time','xpos','xquat','site_xpos','site_xmat','cvel','sensordata')})
        env = SimpleNamespace(_sim_step_counter=90, _g11_tick=0, _manual_reset_pending=Torch.tensor([False]),
            physics_dt=.002, action_manager=SimpleNamespace(apply_action=lambda: None),
            scene=SimpleNamespace(write_data_to_sim=lambda: None, update=lambda dt: None),
            metrics_manager=SimpleNamespace(compute_substep=lambda: None),
            sim=SimpleNamespace(wp_model=None, wp_data=data, data=SimpleNamespace(qpos=Torch.zeros(1), qvel=Torch.zeros(1),
                                xfrc_applied=Torch.zeros(1), qfrc_applied=Torch.zeros(1))))
        journal = SimpleNamespace(checkpoint=lambda label: None, macros=0)
        scope = dict(Path=Path, time=time, require=require, macro_cost=lambda: defaultdict(int),
                     save=lambda *a, **k: None, digest=digest, InterfaceEvidence=Evidence, EpisodeAdaptive=Driver)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), "runtime-candidate", "exec"), scope)
        modules = {
            'integration.owner_roster': SimpleNamespace(roots_for=lambda *a: ([], {})),
            'pilot_state': SimpleNamespace(GroundPairStateBridge=lambda *a: None),
            'native_rk4_bridge': SimpleNamespace(NativeRK4Bridge=lambda *a: SimpleNamespace()),
            'formal_task': SimpleNamespace(measured_surface=lambda e: None),
            'scene_evidence': SimpleNamespace(owner_state=None, save_arrays=None)}
        with tempfile.TemporaryDirectory() as d, patch.dict(sys.modules, modules):
            runtime = scope['BudgetedRuntime'](env, Torch, None, None, None, None,
                SimpleNamespace(journal=journal), Budget(), d,
                SimpleNamespace(archive=SimpleNamespace(sample_next=False)),
                SimpleNamespace(order=SimpleNamespace(sample_next=False)), None, None,
                max_macros=30, max_episode_ticks=5000)
            for _ in range(25): runtime.advance(env)
            self.assertEqual((env._g11_tick, env._sim_step_counter), (25,115))
            env._g11_tick = 0  # Native nominal reset keeps the global counter.
            for _ in range(5): runtime.advance(env)
            self.assertEqual((env._g11_tick, env._sim_step_counter), (5,120))
            self.assertEqual(counts['Action_apply'], 30)
            self.assertEqual(counts['Metrics_substep'], 30)
            self.assertEqual(len(runtime.results), 1)
            with self.assertRaises(RuntimeError): runtime.advance(env)
    def test_bundle_roundtrip_and_identity(self):
        from integration.source_bundle import build_zip
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            source = d / "sample.obj"
            source.write_bytes(b"geometry-example\n")
            rows = [dict(name="project/assets/sample.obj", source=str(source),
                         bytes=source.stat().st_size, sha256=file_digest(source))]
            result = build_zip(d / "input.zip", rows, dict(cloud_training=0))
            self.assertEqual(result["members"], 1)
            source.write_bytes(b"changed")
            with self.assertRaises(RuntimeError): build_zip(d / "bad.zip", rows, {})
            self.assertFalse((d / "bad.zip").exists())
    def test_bundle_path_and_duplicates_rejected(self):
        from integration.source_bundle import checked_relative, build_zip
        for path in ("../outside", "/outside", "a\\b", "folder/"):
            with self.assertRaises(RuntimeError): checked_relative(path)
        with self.assertRaises(RuntimeError): build_zip("never-created", [dict(name="a"), dict(name="a")], {})
    def test_episode_clock_crosses_20_without_changing_algorithm(self):
        source = ast.parse((ROOT / "integration/native_runtime.py").read_text())
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "EpisodeAdaptive")
        class Base:
            def __init__(self, bridge, engine, ticks, emit): self.ticks = ticks
        scope = dict(AdaptiveContinuation=Base, require=require)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), "clock-candidate", "exec"), scope)
        for tick in (0, 19, 20, 4999):
            obj = scope["EpisodeAdaptive"](None, None, (tick,), None, 5000)
            self.assertEqual(obj.ticks, (tick,))
        for tick in (-1, 5000, True):
            with self.assertRaises(RuntimeError): scope["EpisodeAdaptive"](None, None, (tick,), None, 5000)
    def test_numerical_helpers_ast_unchanged(self):
        def classes(path):
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)}
        a, b = classes(ROOT / "baseline/g11/formal_runtime.py"), classes(ROOT / "integration/native_runtime.py")
        for name in set(a) - {"Runtime"}:
            self.assertEqual(a[name], b[name], name)
    def test_owner_roster_only_adds_explicit_null_recorder_dispatch(self):
        def method(path):
            return next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "roots_for")
        old=method(ROOT/"baseline/g11/formal_owners.py")
        new=method(ROOT/"integration/owner_roster.py")
        class Normalize(ast.NodeTransformer):
            changed=0
            def visit_Call(self,node):
                if isinstance(node.func,ast.Name) and node.func.id=='recorder_terms':
                    self.changed+=1
                    return ast.Attribute(value=ast.Name(id='manager',ctx=ast.Load()),attr='_terms',ctx=ast.Load())
                return self.generic_visit(node)
        normalize=Normalize();normalized=normalize.visit(new)
        self.assertEqual(normalize.changed,1)
        self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(normalized,include_attributes=False))
        for name in ('native_factory.py','native_runtime.py','rng_policy.py'):
            source=(ROOT/'integration'/name).read_text()
            self.assertIn('from integration.owner_roster import roots_for',source)
            self.assertNotIn('from formal_owners import roots_for',source)
    def test_native_factory_compiles_without_importing_native(self):
        for p in (ROOT / "integration").glob("*.py"):
            compile(p.read_text(), str(p), "exec")
    def test_no_push_or_stock_train_in_new_modules(self):
        for p in (ROOT / "integration").glob("*.py"):
            tree = ast.parse(p.read_text())
            for n in ast.walk(tree):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                    self.assertNotEqual(n.func.attr, "learn", str(p))


def run_suite(stream):
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    return unittest.TextTestRunner(stream=stream, verbosity=2, failfast=False).run(suite)


if __name__ == "__main__":
    result = run_suite(sys.stdout)
    from validation.integration.cpu_cases import all_cases
    from validation.integration.protocol_tensor import Torch, TD
    rows = all_cases(Torch, TD)
    print(json.dumps(dict(scope="NUMPY_PROTOCOL_DOUBLES_NOT_REAL_TORCH", cases=rows), indent=2))
    raise SystemExit(0 if result.wasSuccessful() and all(r["passed"] for r in rows) else 1)
