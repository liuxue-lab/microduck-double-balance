"""Pinned source methods and a reviewable Stage11 outer-step candidate.

No mjlab, MuJoCo, Warp, Torch, checkpoint loader, or task constructor is imported.
The caller supplies an explicit runtime namespace and already-owned macro hook.
This module neither installs the hook nor removes the task's release guards.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PINS = json.loads((ROOT / "source-pins.json").read_text())
ENV_CLASS = "ManagerBasedRlEnv"
LOOP_BODY = """self._sim_step_counter += 1
self.action_manager.apply_action()
self.scene.write_data_to_sim()
self.sim.step()
self.scene.update(dt=self.physics_dt)
self.metrics_manager.compute_substep()
"""


def pinned_source(name):
    raw = (ROOT / "pinned-source" / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PINS[name]["sha256"]:
        raise RuntimeError("Pinned source changed: " + name)
    return raw.decode()


def method_node(source, owner, method):
    classes = [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == owner]
    if len(classes) != 1:
        raise RuntimeError("Source class is missing or ambiguous")
    nodes = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == method]
    if len(nodes) != 1:
        raise RuntimeError("Source method is missing or ambiguous")
    return copy.deepcopy(nodes[0])


def compile_method(node, namespace, filename):
    node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    ast.fix_missing_locations(module)
    scope = dict(namespace)
    exec(compile(module, filename, "exec"), scope)
    return scope[node.name]


def load_method(name, owner, method, namespace):
    return compile_method(method_node(pinned_source(name), owner, method), namespace, name)


def patched_event_reset(namespace):
    source = pinned_source("event_manager.py")
    old = 'for index, term_cfg in enumerate(self._mode_class_term_cfgs["interval"]):'
    new = 'for index, term_cfg in enumerate(self._mode_term_cfgs["interval"]):'
    if source.count(old) != 1:
        raise RuntimeError("Reset patch no longer applies exactly")
    return compile_method(method_node(source.replace(old, new), "EventManager", "reset"), namespace, "event_manager_reset_candidate.py")


def build_step_candidate(namespace, advance_macro, *, derived_state_policy):
    """Replace the complete six-statement substep body, leaving its loop intact.

    advance_macro(env) owns the counter, BAM/action application, scene write,
    accepted physical macro, scene update and substep metrics, exactly once.
    Accepted-endpoint freshness is an explicit semantic change from stock step.
    A real formal task, complete rollback owner roster, and GPU regression are
    still required before installing the returned function on an environment.
    """
    if derived_state_policy != "accepted_endpoint":
        raise ValueError("Explicit accepted_endpoint reward/termination semantics required")
    if not callable(advance_macro):
        raise TypeError("An owned physical macro callback is required")
    node = method_node(pinned_source("manager_based_rl_env.py"), ENV_CLASS, "step")
    loops = [n for n in ast.walk(node) if isinstance(n, ast.For) and ast.unparse(n.iter) == "range(self.cfg.decimation)"]
    expected = ast.dump(ast.parse(LOOP_BODY), include_attributes=False)
    if len(loops) != 1 or loops[0].orelse or ast.dump(ast.Module(body=loops[0].body, type_ignores=[]), include_attributes=False) != expected:
        raise RuntimeError("Pinned six-statement substep contract changed")

    def owned_advance(env):
        if env.physics_dt != 0.002 or env.cfg.decimation != 10 or env.step_dt != 0.020:
            raise RuntimeError("Formal 2 ms / 10 / 20 ms cadence changed")
        before = env._sim_step_counter
        result = advance_macro(env)
        if env._sim_step_counter != before + 1:
            raise RuntimeError("Macro callback must own exactly one public substep counter increment")
        return result

    loops[0].body = [ast.Expr(value=ast.Call(func=ast.Name(id="_stage11_advance_macro", ctx=ast.Load()), args=[ast.Name(id="self", ctx=ast.Load())], keywords=[]))]
    return compile_method(node, {**namespace, "_stage11_advance_macro": owned_advance}, "stage11_outer_step_candidate.py")
