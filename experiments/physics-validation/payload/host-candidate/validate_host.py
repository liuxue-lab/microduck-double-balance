"""Execute pinned Python methods with explicit NumPy-based runtime stand-ins.

These checks establish host control-flow contracts and reproduce source bugs.
They do NOT run Torch, mjlab, a real environment, GPU physics or checkpoints.
"""
import ast
import copy
import enum
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from checkpoint_candidate import MigrationError, compare_semantics, migrate_checkpoint_keys
from source_contracts import (ENV_CLASS, PINS, ROOT, build_step_candidate,
                              compile_method, load_method, patched_event_reset,
                              pinned_source)

NS = types.SimpleNamespace
CHECKS = []
EVIDENCE = {}


def check(fn):
    try:
        fn()
        CHECKS.append({"name": fn.__name__, "passed": True})
    except Exception as exc:
        CHECKS.append({"name": fn.__name__, "passed": False, "error": repr(exc)})
    return fn


def must_raise(kind, fn):
    try:
        fn()
    except kind as exc:
        return str(exc)
    raise AssertionError("Expected " + kind.__name__)


class Tensor(np.ndarray):
    def __new__(cls, data, dtype=None):
        return np.asarray(data, dtype=dtype).view(cls)
    def nonzero(self, as_tuple=False):
        indices = np.nonzero(np.asarray(self))
        return tuple(Tensor(i) for i in indices) if as_tuple else Tensor(np.column_stack(indices))
    def to(self, device):
        return self
    def cpu(self):
        return self


class TorchStandIn:
    Tensor = Tensor
    int64 = np.int64
    def __init__(self):
        self.draws = []
    def any(self, value):
        return bool(np.any(np.asarray(value)))
    def arange(self, count, dtype=None, device=None):
        return Tensor(np.arange(count, dtype=dtype))
    def rand(self, count, device=None):
        self.draws.append((count, device))
        return Tensor(np.full(count, 0.25))


class Recompute(enum.IntEnum):
    none = 0
    set_const = 1
    full = 2


TORCH = TorchStandIn()
GLOBALS = {"torch": TORCH, "RecomputeLevel": Recompute}
STEP = load_method("manager_based_rl_env.py", ENV_CLASS, "step", GLOBALS)
RESET = load_method("manager_based_rl_env.py", ENV_CLASS, "reset", GLOBALS)
RESET_IDX = load_method("manager_based_rl_env.py", ENV_CLASS, "_reset_idx", GLOBALS)
EVENT_APPLY = load_method("event_manager.py", "EventManager", "apply", GLOBALS)
EVENT_RESET = load_method("event_manager.py", "EventManager", "reset", GLOBALS)
PATCHED_RESET = patched_event_reset(GLOBALS)


def ids(value):
    return None if value is None else np.asarray(value).tolist()


def term(label, log, *, is_class=False, global_time=False, interval=(0.1, 0.1), recompute=0):
    def fire(env, env_ids, **kwargs):
        log.append(("fire", label, ids(env_ids)))
    fire.recompute = recompute
    if is_class:
        class ClassTerm:
            def __call__(self, *a, **k):
                return fire(*a, **k)
            def reset(self, env_ids=None):
                log.append(("class_reset", label, ids(env_ids)))
        callback = ClassTerm()
        callback.recompute = recompute
    else:
        callback = fire
    return NS(func=callback, is_global_time=global_time, interval_range_s=interval,
              params={}, min_step_count_between_reset=0)


def event_manager(terms, times, log):
    env = NS(num_envs=2, sim=NS(recompute_constants=lambda level: log.append(("recompute", int(level)))))
    return NS(_env=env, device="cpu",
              _mode_term_cfgs={"interval": terms},
              _mode_class_term_cfgs={"interval": [t for t in terms if hasattr(t.func, "reset")]},
              _interval_term_time_left=[Tensor(a, dtype=np.float64) for a in times])


@check
def reproduce_class_subset_resets_wrong_timer():
    log = []
    m = event_manager([term("function", log), term("class", log, is_class=True, interval=(0.2, 0.2))], [[0.01, 0.02], [0.03, 0.04]], log)
    EVENT_RESET(m, Tensor([0]))
    assert m._interval_term_time_left[0].tolist() == [0.2, 0.02]
    assert m._interval_term_time_left[1].tolist() == [0.03, 0.04]
    EVIDENCE["original_wrong_timer"] = [a.tolist() for a in m._interval_term_time_left]


@check
def reproduce_global_before_class_index_error():
    log = []
    m = event_manager([term("global", log, global_time=True), term("local_class", log, is_class=True)], [[0.01], [0.03, 0.04]], log)
    EVIDENCE["original_global_index_error"] = must_raise(IndexError, lambda: EVENT_RESET(m, Tensor([1])))


@check
def patch_resets_all_local_terms_at_their_actual_indices():
    log = []
    m = event_manager([term("function", log), term("class", log, is_class=True, interval=(0.2, 0.2))], [[0.01, 0.02], [0.03, 0.04]], log)
    owners = [id(a) for a in m._interval_term_time_left]
    PATCHED_RESET(m, Tensor([0]))
    assert [a.tolist() for a in m._interval_term_time_left] == [[0.1, 0.02], [0.2, 0.04]]
    assert [id(a) for a in m._interval_term_time_left] == owners
    assert log == [("class_reset", "class", [0])]
    EVIDENCE["patched_local_timers"] = [a.tolist() for a in m._interval_term_time_left]


@check
def patch_preserves_global_timer_and_unselected_world():
    log = []
    m = event_manager([term("global", log, global_time=True), term("local_class", log, is_class=True)], [[0.01], [0.03, 0.04]], log)
    before = m._interval_term_time_left[0].tobytes()
    PATCHED_RESET(m, Tensor([1]))
    assert m._interval_term_time_left[0].tobytes() == before
    assert m._interval_term_time_left[1].tolist() == [0.03, 0.1]


@check
def patch_handles_function_only_all_and_empty_resets():
    log = []
    m = event_manager([term("function", log)], [[0.01, 0.02]], log)
    original = m._interval_term_time_left[0].tobytes()
    PATCHED_RESET(m, Tensor([], dtype=np.int64))
    assert m._interval_term_time_left[0].tobytes() == original
    PATCHED_RESET(m, None)
    assert m._interval_term_time_left[0].tolist() == [0.1, 0.1]


@check
def real_apply_fires_global_once_local_selected_and_recomputes_once():
    log = []
    m = event_manager([term("global", log, global_time=True, recompute=1), term("local", log, recompute=2)], [[0.01], [0.01, 0.03]], log)
    EVENT_APPLY(m, "interval", dt=0.020)
    assert log == [("fire", "global", None), ("fire", "local", [0]), ("recompute", 2)]
    assert abs(float(m._interval_term_time_left[1][1]) - 0.01) < 1e-15
    EVIDENCE["interval_expiry_trace"] = log


@check
def real_apply_nonexpired_timers_do_not_fire_or_resample():
    log = []
    m = event_manager([term("global", log, global_time=True), term("local", log)], [[0.03], [0.03, 0.04]], log)
    n = len(TORCH.draws)
    EVENT_APPLY(m, "interval", dt=0.020)
    assert log == [] and len(TORCH.draws) == n
    np.testing.assert_allclose(m._interval_term_time_left[1], [0.01, 0.02], rtol=0, atol=1e-15)


@check
def real_apply_rejects_missing_dt_and_explicit_interval_ids():
    log = []
    m = event_manager([], [], log)
    must_raise(ValueError, lambda: EVENT_APPLY(m, "interval"))
    must_raise(ValueError, lambda: EVENT_APPLY(m, "interval", env_ids=Tensor([0]), dt=0.02))
    must_raise(ValueError, lambda: EVENT_APPLY(m, "step"))
    must_raise(ValueError, lambda: EVENT_APPLY(m, "reset"))


@check
def real_apply_reset_minimum_steps_and_first_trigger_are_per_world():
    log = []
    t = term("reset", log)
    t.min_step_count_between_reset = 5
    m = event_manager([], [], log)
    m._mode_term_cfgs["reset"] = [t]
    m._reset_term_last_triggered_step_id = [Tensor([0, 8], dtype=np.int64)]
    m._reset_term_last_triggered_once = [Tensor([False, True], dtype=bool)]
    EVENT_APPLY(m, "reset", env_ids=Tensor([0, 1]), global_env_step_count=10)
    assert log == [("fire", "reset", [0])]
    assert m._reset_term_last_triggered_step_id[0].tolist() == [10, 8]
    log.clear()
    EVENT_APPLY(m, "reset", env_ids=Tensor([0, 1]), global_env_step_count=13)
    assert log == [("fire", "reset", [1])]


class EnvironmentStandIn:
    def __init__(self, *, auto_reset=True, done=(False, False), timeouts=(False, False), events=True):
        self.cfg = NS(auto_reset=auto_reset, decimation=10)
        self.physics_dt, self.step_dt, self.num_envs, self.device = 0.002, 0.020, 2, "cpu"
        self._sim_step_counter, self.common_step_counter = 30, 11
        self.episode_length_buf = Tensor([4, 7], dtype=np.int64)
        self._manual_reset_pending = Tensor([False, False], dtype=bool)
        self.physical, self.derived = Tensor([0, 0]), Tensor([0, 0])
        self.trace, self.extras = [], {}
        self._reset_idx = types.MethodType(RESET_IDX, self)
        self.reset = types.MethodType(RESET, self)
        self.sim = NS(step=self.sim_step, forward=self.forward, sense=lambda: self.log("sense"), reset=self.sim_reset)
        self.scene = NS(write_data_to_sim=lambda: self.log("scene_write"), update=lambda dt: self.log("scene_update", dt), reset=lambda value: self.log("scene_reset", ids(value)))
        for name in ("action", "observation", "reward", "metrics", "curriculum", "command", "event", "termination"):
            manager = NS(reset=lambda value, n=name: self.manager_reset(n, value))
            setattr(self, name + "_manager", manager)
        self.action_manager.process_action = lambda action: self.log("process_action", ids(action))
        self.action_manager.apply_action = lambda: self.log("apply_action")
        self.termination_manager.terminated = Tensor(done, dtype=bool)
        self.termination_manager.time_outs = Tensor(timeouts, dtype=bool)
        self.termination_manager.compute = self.terminate
        self.reward_manager.compute = self.reward
        self.metrics_manager.compute_substep = lambda: self.log("metrics_substep")
        self.metrics_manager.compute = lambda: self.log("metrics")
        self.curriculum_manager.compute = lambda env_ids: self.log("curriculum", ids(env_ids))
        self.command_manager.compute = lambda dt: self.log("command", dt)
        self.event_manager.available_modes = ["reset", "step", "interval"] if events else []
        self.event_manager.apply = lambda **kwargs: self.log("event", {k: ids(v) if k == "env_ids" else v for k, v in kwargs.items()})
        self.observation_manager.compute = self.observation
        self.recorder_manager = NS(record_pre_reset=lambda value: self.log("record_pre_reset", ids(value)),
            record_post_reset=lambda value: self.log("record_post_reset", ids(value)), record_post_step=lambda: self.log("record_post_step"))
        self.seed = lambda value: self.log("seed", value)
    def log(self, event, *details):
        self.trace.append((event, *details))
    def sim_step(self):
        self.log("sim_step")
        self.derived[:] = self.physical
        self.physical[:] += 1
    def forward(self):
        self.log("forward")
        self.derived[:] = self.physical
    def sim_reset(self, value):
        self.log("sim_reset", ids(value))
        self.physical[value] = 0
    def manager_reset(self, name, value):
        self.log("manager_reset", name, ids(value))
        return {name: ids(value)}
    def terminate(self):
        self.log("termination", self.derived.tolist())
        return self.termination_manager.terminated | self.termination_manager.time_outs
    def reward(self, dt):
        self.log("reward", dt, self.derived.tolist())
        return self.derived.copy()
    def observation(self, update_history):
        self.log("observation", update_history, self.derived.tolist())
        return {"actor": self.derived.copy()}


ACTION = Tensor([[0.1], [0.2]])


def advance_standin(env, *, fresh=False):
    env._sim_step_counter += 1
    env.action_manager.apply_action()
    env.scene.write_data_to_sim()
    env.sim.step()
    if fresh:
        env.forward()
    env.scene.update(dt=env.physics_dt)
    env.metrics_manager.compute_substep()


@check
def outer_hook_preserves_real_step_control_flow_in_six_reset_cases():
    cases = [(True, (False, False), (False, False)), (True, (True, False), (False, False)),
             (True, (True, True), (False, False)), (True, (False, False), (False, True)),
             (False, (True, False), (False, False)), (False, (False, False), (False, False))]
    candidate = build_step_candidate(GLOBALS, advance_standin, derived_state_policy="accepted_endpoint")
    for auto, done, timeouts in cases:
        original = EnvironmentStandIn(auto_reset=auto, done=done, timeouts=timeouts)
        changed = EnvironmentStandIn(auto_reset=auto, done=done, timeouts=timeouts)
        before, after = STEP(original, ACTION), candidate(changed, ACTION)
        assert original.trace == changed.trace
        for a, b in zip(before[1:4], after[1:4]):
            np.testing.assert_array_equal(a, b)
        assert original._sim_step_counter == changed._sim_step_counter == 40
        assert original.common_step_counter == changed.common_step_counter == 12
        np.testing.assert_array_equal(original._manual_reset_pending, changed._manual_reset_pending)
    EVIDENCE["outer_control_flow_cases"] = len(cases)


@check
def outer_clock_actions_history_and_events_commit_at_the_correct_cadence():
    e = EnvironmentStandIn()
    STEP(e, ACTION)
    labels = [r[0] for r in e.trace]
    for name in ("apply_action", "scene_write", "sim_step", "scene_update", "metrics_substep"):
        assert labels.count(name) == 10
    for name in ("process_action", "termination", "reward", "metrics", "forward", "command", "sense", "observation", "record_post_step"):
        assert labels.count(name) == 1
    assert [row for row in e.trace if row[0] == "event"] == [
        ("event", {"mode": "step", "dt": 0.020}), ("event", {"mode": "interval", "dt": 0.020})]
    assert e.episode_length_buf.tolist() == [5, 8]
    assert labels.index("termination") < labels.index("reward") < labels.index("forward") < labels.index("command") < labels.index("sense") < labels.index("observation") < labels.index("record_post_step")


@check
def auto_reset_preserves_terminal_signals_and_resets_only_selected_world():
    e = EnvironmentStandIn(done=(True, False))
    result = STEP(e, ACTION)
    assert result[2].tolist() == [True, False] and result[3].tolist() == [False, False]
    assert result[0]["actor"].tolist() == [0, 10] and result[1].tolist() == [9, 9]
    assert e.episode_length_buf.tolist() == [0, 8]
    assert [r[1] for r in e.trace if r[0] == "sim_reset"] == [[0]]
    labels = [r[0] for r in e.trace]
    assert labels.index("record_pre_reset") < labels.index("sim_reset") < labels.index("observation") < labels.index("record_post_reset") < labels.index("record_post_step")


@check
def manual_reset_pending_rejects_before_actions_and_explicit_reset_recovers():
    e = EnvironmentStandIn(auto_reset=False, done=(False, True))
    STEP(e, ACTION)
    assert e._manual_reset_pending.tolist() == [False, True]
    e.trace.clear()
    must_raise(RuntimeError, lambda: STEP(e, ACTION))
    assert e.trace == []
    e.reset(env_ids=Tensor([1]))
    assert e._manual_reset_pending.tolist() == [False, False]
    assert e.physical.tolist() == [10, 0]
    assert e._sim_step_counter == 40 and e.common_step_counter == 12


@check
def real_reset_uses_zero_command_dt_and_ordered_manager_resets():
    e = EnvironmentStandIn()
    e.reset(seed=42)
    resets = [r[1] for r in e.trace if r[0] == "manager_reset"]
    assert resets == ["observation", "action", "reward", "metrics", "curriculum", "command", "event", "termination"]
    assert ("command", 0.0) in e.trace
    assert e.trace[0] == ("seed", 42) and e.trace[-1] == ("record_post_reset", [0, 1])
    assert e._sim_step_counter == 30 and e.common_step_counter == 11


@check
def source_step_without_optional_event_modes_calls_no_events():
    e = EnvironmentStandIn(events=False)
    STEP(e, ACTION)
    assert all(r[0] != "event" for r in e.trace)


@check
def adaptive_endpoint_freshness_is_a_measured_host_semantic_difference():
    original, changed = EnvironmentStandIn(), EnvironmentStandIn()
    candidate = build_step_candidate(GLOBALS, lambda e: advance_standin(e, fresh=True), derived_state_policy="accepted_endpoint")
    STEP(original, ACTION)
    candidate(changed, ACTION)
    old = next(r for r in original.trace if r[0] == "reward")
    new = next(r for r in changed.trace if r[0] == "reward")
    assert old[2] == [9, 9] and new[2] == [10, 10]
    EVIDENCE["reward_freshness_witness"] = {"units": "synthetic macro generation, not physical values", "stock": old[2], "accepted_endpoint": new[2]}


@check
def failed_macro_stops_before_reward_events_or_formal_counter_commit():
    e = EnvironmentStandIn()
    called = []
    def fail(env):
        called.append(env._sim_step_counter)
        raise RuntimeError("Synthetic macro failure")
    candidate = build_step_candidate(GLOBALS, fail, derived_state_policy="accepted_endpoint")
    must_raise(RuntimeError, lambda: candidate(e, ACTION))
    assert called == [30] and e.common_step_counter == 11 and e.episode_length_buf.tolist() == [4, 7]
    assert [r[0] for r in e.trace] == ["process_action"]


@check
def hook_rejects_missing_clock_ownership_and_undeclared_freshness():
    must_raise(ValueError, lambda: build_step_candidate(GLOBALS, advance_standin, derived_state_policy="stock_lag"))
    candidate = build_step_candidate(GLOBALS, lambda e: None, derived_state_policy="accepted_endpoint")
    must_raise(RuntimeError, lambda: candidate(EnvironmentStandIn(), ACTION))


def legacy():
    return {"model_state_dict": {"actor.0.weight": object(), "critic.0.weight": object(),
        "actor_obs_normalizer.mean": object(), "critic_obs_normalizer.mean": object(), "std": object()},
        "iter": 7, "infos": {"env_state": {"common_step_counter": 19}}, "optimizer_state_dict": {"slot": "preserved"}}


@check
def migration_preserves_tensor_identity_metadata_and_input_mapping():
    before = legacy()
    keyset = set(before["model_state_dict"])
    after, report = migrate_checkpoint_keys(before)
    assert set(before["model_state_dict"]) == keyset and "actor_state_dict" not in before
    assert after["actor_state_dict"]["mlp.0.weight"] is before["model_state_dict"]["actor.0.weight"]
    assert after["actor_state_dict"]["distribution.std_param"] is before["model_state_dict"]["std"]
    assert after["infos"] is before["infos"] and after["optimizer_state_dict"] is before["optimizer_state_dict"]
    assert report["semantic_compatibility"] == "UNVERIFIED"


@check
def migration_is_idempotent_for_current_format():
    once, _ = migrate_checkpoint_keys(legacy())
    twice, report = migrate_checkpoint_keys(once)
    assert once == twice and report["changes"] == []


@check
def migration_rejects_unknown_legacy_keys_without_partial_mutation():
    before = legacy()
    before["model_state_dict"]["extra_state"] = object()
    keys = set(before["model_state_dict"])
    must_raise(MigrationError, lambda: migrate_checkpoint_keys(before))
    assert set(before["model_state_dict"]) == keys and "actor_state_dict" not in before


@check
def migration_rejects_mixed_legacy_and_current_dictionaries():
    before = legacy()
    before["actor_state_dict"] = {"sentinel": object()}
    must_raise(MigrationError, lambda: migrate_checkpoint_keys(before))


@check
def migration_rejects_std_overwrite_and_ambiguous_parameterization():
    before, _ = migrate_checkpoint_keys(legacy())
    before["actor_state_dict"]["std"] = object()
    must_raise(MigrationError, lambda: migrate_checkpoint_keys(before))
    before = legacy()
    before["model_state_dict"]["log_std"] = object()
    must_raise(MigrationError, lambda: migrate_checkpoint_keys(before))


@check
def migration_changes_only_the_leading_namespace():
    before = legacy()
    value = object()
    before["model_state_dict"]["actor.nested.actor.weight"] = value
    after, _ = migrate_checkpoint_keys(before)
    assert after["actor_state_dict"]["mlp.nested.actor.weight"] is value


@check
def original_runner_demonstrates_lossy_and_overwriting_paths():
    def run(mapping):
        seen = []
        runner = NS(alg=NS(load=lambda data, load_cfg, strict: seen.append(data) or False), env=NS(unwrapped=NS(common_step_counter=0)))
        fn = load_method("runner.py", "MjlabOnPolicyRunner", "load", {"torch": NS(load=lambda *a, **k: mapping), "print": lambda *a: None})
        fn(runner, "synthetic-in-memory-only")
        return seen[0]
    first = legacy()
    first["model_state_dict"]["unknown_extra"] = "lost"
    migrated = run(first)
    assert "unknown_extra" not in migrated["actor_state_dict"] and "unknown_extra" not in migrated["critic_state_dict"]
    second = legacy()
    second["actor_state_dict"] = {"current_sentinel": "overwritten"}
    migrated = run(second)
    assert "current_sentinel" not in migrated["actor_state_dict"]
    EVIDENCE["original_runner_witness"] = {"unknown_legacy_key_dropped": True, "current_actor_dictionary_overwritten": True,
                                            "real_checkpoint_deserialization": False}


def manifest():
    obs = {"term_order": ["position", "velocity"], "terms": {"position": {"dim": 1, "unit": "m"}, "velocity": {"dim": 1, "unit": "m/s"}},
           "history_layout": [2, 2], "flatten_order": "time_then_feature", "normalizer": {"kind": "running_mean_std", "shape": [4], "epsilon": 1e-5}}
    return {"actor_observations": copy.deepcopy(obs), "critic_observations": copy.deepcopy(obs),
            "actions": {"joint_order": ["j0", "j1"], "scale": [1.0, 1.0], "offset": [0.0, 0.0], "clip": [-1.0, 1.0], "actuator_model": "synthetic-BAM", "delay": [0, 6]},
            "timing": {"physics_dt_s": 0.002, "decimation": 10, "step_dt_s": 0.02, "derived_state_policy": "accepted_endpoint"},
            "task": {"id": "synthetic-contract-only", "reward_terms": ["r"], "termination_terms": ["t"], "source_sha256": "0" * 64}}


@check
def semantic_manifest_requires_complete_fields_and_provenance():
    before = manifest()
    same = compare_semantics(before, copy.deepcopy(before))
    assert same["status"] == "MANIFESTS_MATCH_REQUIRE_PROVENANCE" and same["real_checkpoint_compatible"] is False
    missing = copy.deepcopy(before)
    del missing["actions"]["delay"]
    assert compare_semantics(before, missing)["status"] == "BLOCKED"


@check
def equal_dimensions_do_not_hide_observation_history_normalizer_or_action_changes():
    before = manifest()
    mutations = [lambda m: m["actor_observations"]["term_order"].reverse(),
                 lambda m: m["critic_observations"].update(flatten_order="feature_then_time"),
                 lambda m: m["actor_observations"]["normalizer"].update(epsilon=1e-4),
                 lambda m: m["actions"]["joint_order"].reverse(),
                 lambda m: m["actions"].update(delay=[0, 5]),
                 lambda m: m["timing"].update(derived_state_policy="stock_lag"),
                 lambda m: m["task"].update(reward_terms=["different_r"]) ]
    for mutate in mutations:
        other = copy.deepcopy(before)
        mutate(other)
        review = compare_semantics(before, other)
        assert review["status"] == "BLOCKED" and review["differences"]
    EVIDENCE["semantic_counterexamples"] = len(mutations)


@check
def original_formal_task_release_guards_remain_present_and_unmodified():
    cfg = pinned_source("microduck_head_contact_env_cfg.py")
    assert 'if static_audit_only is not True:' in cfg
    assert '"stage11_release_block": EventTermCfg(func=mdp.stage11_head_release_block, mode="startup")' in cfg
    assert 'cfg.terminations["top_ball_lost"].func = mdp.stage11_head_release_block' in cfg
    source = pinned_source("mdp.py")
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "stage11_head_release_block")
    guard = compile_method(node, {}, "original_release_guard.py")
    must_raise(RuntimeError, guard)


@check
def all_pinned_sources_are_exact_and_no_native_runtime_was_imported():
    for name in PINS:
        pinned_source(name)
    assert not set(("torch", "warp", "mujoco", "mujoco_warp", "mjlab", "rsl_rl")) & set(sys.modules)


result = {"status": "G11_HOST_CONTRACTS_PASS" if all(c["passed"] for c in CHECKS) else "G11_HOST_CONTRACTS_FAILED",
          "checks_passed": sum(c["passed"] for c in CHECKS), "checks_total": len(CHECKS),
          "checks": CHECKS, "evidence": EVIDENCE, "source_pins": PINS,
          "method_body_execution": "pinned source AST, with explicit NumPy-based stand-ins",
          "real_environment_executed": False, "GPU_acceptance": False, "checkpoint_loaded": False,
          "installed_sources_modified": False, "release_guards_changed": False,
          "native_integrations": 0, "PPO": 0, "cloud_training": 0,
          "canonical_completed": 667608, "canonical_attempted": 667609,
          "G10": "OPEN", "G11": "OPEN", "Stage11_gate_percent": 100 * 9 / 11}
(ROOT / "regression-review.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({k: result[k] for k in ("status", "checks_passed", "checks_total", "native_integrations")}))
for failure in CHECKS:
    if not failure["passed"]:
        print(json.dumps(failure))
raise SystemExit(0 if all(c["passed"] for c in CHECKS) else 1)
