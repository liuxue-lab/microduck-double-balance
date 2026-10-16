"""Run with the existing project interpreter; verify completed sources, then CPU config only."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata as meta
import json
import os
import platform
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stage12.contracts import REPO, require, checkpoint_policy, scaling_plan, EXECUTION_BUDGET


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def config_probe(out, font_fallback, git_probe):
    from stage12.headless import prepare as prepare_headless
    counts = {}
    guards = []
    def blocked(name):
        def refuse(*args, **kwargs):
            counts[name] = counts.get(name, 0) + 1
            raise RuntimeError("ZERO_BUDGET_BLOCKED: " + name)
        return refuse
    def patch(obj, name, label=None):
        if hasattr(obj, name):
            label = label or getattr(obj, "__name__", type(obj).__name__) + "." + name
            setattr(obj, name, blocked(label))
            guards.append(label)
    report = dict(status="STARTED", budget=EXECUTION_BUDGET, guard_attempts=counts,
                  guards_installed=guards, native_integrations=0, native_forward=0,
                  real_environment_constructed=False, model_constructed=False, PPO=0, cloud_training=0)
    try:
        report["headless_discovery"] = prepare_headless(ROOT, out)
        prior_headless = json.loads((ROOT / "prior-readiness-v4/headless-setup.json").read_bytes())
        require(report["headless_discovery"]["library_identity"] == prior_headless["library_identity"],
                "Bundled GLFW library differs from the v4 successful import")
        font_fallback.configure(ROOT)
        sys.path.insert(0, REPO + "/src")
        print("Stage12ConfigPhase=gitpython-read-only-import", flush=True)
        git_probe.configure(ROOT, out)
        git_probe.preload()
        print("Stage12ConfigPhase=torch", flush=True)
        import torch
        require(not torch.cuda.is_initialized(), "CUDA unexpectedly initialized")
        patch(torch, "load"); patch(torch, "save"); patch(torch.jit, "load")
        patch(torch.cuda, "_lazy_init")
        print("Stage12ConfigPhase=mujoco-and-glfw", flush=True)
        import mujoco
        for n in ("mj_step", "mj_step1", "mj_step2", "mj_Euler", "mj_RungeKutta", "mj_implicit",
                  "mj_forward", "mj_kinematics", "mj_resetData"):
            patch(mujoco, n)
        for n in ("from_xml_path", "from_xml_string", "from_binary_path"):
            patch(mujoco.MjModel, n, "MjModel." + n)
        patch(mujoco.MjSpec, "compile", "MjSpec.compile")
        patch(mujoco.Renderer, "__init__", "Renderer.__init__")
        import glfw
        for n in ("init", "create_window", "make_context_current"):
            patch(glfw, n)
        import mujoco.viewer
        for n in ("launch", "launch_passive", "launch_from_path"):
            patch(mujoco.viewer, n)
        print("Stage12ConfigPhase=warp-import-only", flush=True)
        import warp as wp
        import mujoco_warp as mw
        from mujoco_warp._src import forward as fw
        for obj in (mw, fw):
            for n in ("step", "step1", "step2", "euler", "implicit", "rungekutta4", "forward"):
                patch(obj, n)
        for n in ("init", "launch", "capture_begin", "capture_end", "capture_launch", "get_device"):
            patch(wp, n)
        # mjlab's import-time media setup otherwise probes ffmpeg with a process.
        # No video is produced here. Select its existing bundled binary by metadata
        # and suppress only discovery/probing, without changing installed files.
        media_dist = meta.distribution("imageio-ffmpeg")
        media_bins = [Path(media_dist.locate_file(f)) for f in media_dist.files or []
                      if str(f).startswith("imageio_ffmpeg/binaries/ffmpeg-") and not str(f).endswith((".py", ".pyc"))]
        require(len(media_bins) == 1 and media_bins[0].is_file(), "Bundled ffmpeg identity requires review")
        os.environ["IMAGEIO_FFMPEG_EXE"] = str(media_bins[0])
        report["media_binary_probe_suppressed_config_only"] = True
        print("Stage12ConfigPhase=mjlab-and-font-discovery", flush=True)
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.sim import Simulation
        from mjlab.scene import Scene
        from rsl_rl.algorithms import PPO
        from rsl_rl.runners import OnPolicyRunner
        for cls in (ManagerBasedRlEnv, Simulation, Scene, PPO, OnPolicyRunner):
            patch(cls, "__init__")
        base = ROOT / "baseline/g11"
        sys.path[:0] = [str(base), str(base / "accepted"), str(base / "host-candidate")]
        from checkpoint_review import config_manifest, describe
        from stage12.task_candidate import make_candidates, register_prepared
        from stage12.vec_adapter import make_wrapper_class
        from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
        from tensordict import TensorDict
        print("Stage12ConfigPhase=candidate-config", flush=True)
        original, candidate, agent = make_candidates()
        archived = json.loads((ROOT / "baseline/semantic-config-before.json").read_bytes())["target_head_config"]
        current = config_manifest(original)
        require(current == archived, "Config-only source no longer reproduces archived G11 semantics")
        report["registry"] = register_prepared(candidate, agent)
        from stage12.runner_contract import plan as runner_plan
        save(out / "runner-contract-plan.json", runner_plan(agent.algorithm.num_mini_batches))
        report["config_semantics_reproduced"] = True
        save(out / "prepared-env-config.json", dict(manifest=config_manifest(candidate),
             events=describe(candidate.events), metrics=describe(candidate.metrics),
             recorders=describe(candidate.recorders), seed=candidate.seed, auto_reset=candidate.auto_reset,
             note="Config-only candidate; physical runtime not released"))
        save(out / "prepared-agent-config.json", describe(agent))
        # These deterministic configuration results already passed on the v6
        # host. Preserve that baseline while repairing only the adapter contract.
        for name in ("prepared-env-config.json", "prepared-agent-config.json", "runner-contract-plan.json"):
            require(json.loads((out / name).read_bytes()) ==
                    json.loads((ROOT / "prior-readiness-v6" / name).read_bytes()),
                    "Prepared configuration differs from the accepted v6 CPU result: " + name)
        report["prepared_configuration_matches_v6"] = True
        print("Stage12ConfigPhase=cached-wrapper", flush=True)
        report["wrapper_contract"] = test_wrapper(torch, RslRlVecEnvWrapper, TensorDict, make_wrapper_class)
        require(not counts, "A forbidden API was attempted")
        require(not torch.cuda.is_initialized(), "CPU preflight initialized CUDA")
        report.update(status="CPU_CONFIGURATION_AND_ADAPTER_PASS", cuda_initialized=False,
                      training_ready=False, public_physics_step_called=False)
    except BaseException as exc:
        report.update(status="CPU_CONFIGURATION_BLOCKED", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        save(out / "configuration-probe.json", report)
    return report


def test_wrapper(torch, base, td, make_class):
    cases = {kind: test_wrapper_case(torch, base, td, make_class, kind)
             for kind in ("dict", "TensorDict")}
    return dict(status="PASS_FAKE_CPU_ENV_ONLY", cases=cases,
                native_env_tested=False, native_integrations=0, observation_recomputations=0)


def test_wrapper_case(torch, base, td, make_class, cache_kind):
    from types import SimpleNamespace
    class FakeEnv:
        num_envs, device, max_episode_length = 2, "cpu", 500
        cfg = SimpleNamespace(auto_reset=True, is_finite_horizon=False)
        action_manager = SimpleNamespace(total_action_dim=14)
        def __init__(self):
            self.unwrapped = self
            self.obs_buf = {"actor": torch.zeros(2, 61), "critic": torch.zeros(2, 85)}
            if cache_kind == "TensorDict":
                self.obs_buf = td(self.obs_buf, batch_size=[2])
            self.extras = {}
            self.resets = self.steps = self.computes = 0
            self.observation_manager = SimpleNamespace(compute=self.forbidden_compute)
        def forbidden_compute(self):
            self.computes += 1
            raise RuntimeError("Observation recomputation is forbidden")
        def reset(self):
            self.resets += 1
            for v in self.obs_buf.values():
                v.zero_()
            return self.obs_buf, self.extras
        def step(self, actions):
            self.steps += 1
            for v in self.obs_buf.values():
                v.add_(3)
            return self.obs_buf, torch.ones(2), torch.tensor([False, True]), torch.tensor([True, False]), {}
    env = FakeEnv()
    wrapper = make_class(base, td)(env, clip_actions=None)
    require(env.resets == 1 and wrapper.num_actions == 14, "Wrapper reset/action layout changed")
    a, b = wrapper.get_observations(), wrapper.get_observations()
    require(isinstance(a, td) and tuple(a.batch_size) == (2,) and set(a.keys()) == {"actor", "critic"},
            "Wrapper observation type, batch size or group keys differ")
    a["actor"].add_(9)
    require(bool((b["actor"] == 0).all()) and bool((env.obs_buf["actor"] == 0).all()),
            "Cached observation clone aliases source")
    obs, rew, dones, extras = wrapper.step(torch.zeros(2, 14))
    require(isinstance(obs, td) and tuple(obs.batch_size) == (2,), "Step did not retain TensorDict batch layout")
    require(dones.tolist() == [1, 1] and extras["time_outs"].tolist() == [True, False],
            "Termination/timeout bootstrap flags differ")
    require(bool((wrapper.get_observations()["actor"] == 3).all()) and env.computes == 0 and env.steps == 1,
            "New committed observations are stale or recomputed")
    wrapper.step(torch.zeros(2, 14))
    require(bool((obs["actor"] == 3).all()), "Next in-place step changed the previous transition snapshot")
    reset_obs, _ = wrapper.reset()
    require(isinstance(reset_obs, td) and tuple(reset_obs.batch_size) == (2,),
            "Reset did not retain TensorDict batch layout")
    require(bool((reset_obs["actor"] == 0).all()) and bool((obs["actor"] == 3).all()) and
            env.resets == 2 and env.steps == 2 and env.computes == 0, "Reset alias or duplicate base call")
    return dict(status="PASS_FAKE_CPU_ENV_ONLY", cache_kind=cache_kind,
                tests=["one wrapper reset", "cached repeated reads", "TensorDict type/batch/keys",
                "clone isolation", "fresh post-step cache", "terminated OR truncated", "timeout bootstrap flag",
                "previous transition survives in-place buffer reuse", "reset snapshots and single delegation"],
                native_env_tested=False, native_integrations=0, observation_recomputations=0)


def audit_boundary(out, font_fallback, git_probe=None):
    from stage12.git_import import is_null_device
    allowed = out.resolve()
    def hook(event, args):
        font_fallback.handle(event, args)
        if git_probe is not None and git_probe.handle(event, args):
            return
        if event in ("socket.connect", "socket.connect_ex", "socket.getaddrinfo", "subprocess.Popen", "os.system"):
            raise RuntimeError("Preflight forbids network/process side effects: " + event)
        if event == "open":
            path, mode, flags = args
            writing = isinstance(mode, str) and any(c in mode for c in "wax+")
            writing = writing or isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if writing and not isinstance(path, int):
                if is_null_device(path):
                    if git_probe is not None:
                        git_probe.report["null_device_write_opens"] += 1
                    return
                require(Path(path).resolve().is_relative_to(allowed), "Write outside new Stage12 run refused: " + str(path))
    sys.addaudithook(hook)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    require(out.is_dir(), "Output directory was not prepared by launcher")
    from stage12.optional_imports import FontconfigFallback
    from stage12.git_import import GitImportProbe
    font_fallback = FontconfigFallback()
    git_probe = GitImportProbe()
    audit_boundary(out, font_fallback, git_probe)
    report = dict(status="STARTED", budget=EXECUTION_BUDGET, training_ready=False)
    from stage12.host_sources import verify as verify_sources
    code, before = 1, None
    try:
        print("Stage12Phase=verify-completed-v2-v3-sources-no-recapture", flush=True)
        before = verify_sources(ROOT, out, "before")
        report["source_reuse"] = before
        save(out / "checkpoint-policy.json", checkpoint_policy())
        save(out / "scale-throughput-plan.json", scaling_plan())
        print("Stage12Phase=CPU-config-and-new-wrapper-contract", flush=True)
        config_probe(out, font_fallback, git_probe)
        report["status"] = "HOST_READINESS_PASS_TRAINING_BLOCKED"
        code = 0
    except BaseException as exc:
        report.update(status="HOST_READINESS_BLOCKED", error=repr(exc), traceback=traceback.format_exc())
    finally:
        try:
            font_fallback.verify_unchanged()
        except BaseException as exc:
            report.update(status="HOST_READINESS_BLOCKED", font_source_preservation_error=repr(exc))
            code = 1
        save(out / "font-discovery-policy.json", font_fallback.report)
        try:
            git_probe.verify_unchanged()
        except BaseException as exc:
            report.update(status="HOST_READINESS_BLOCKED", git_import_preservation_error=repr(exc))
            code = 1
        save(out / "git-import-probe.json", git_probe.report)
        if before is not None:
            try:
                after = verify_sources(ROOT, out, "after")
                require(before == after, "Source or retained evidence changed during the CPU probe")
                report["source_preservation"] = "PASS"
            except BaseException as exc:
                report.update(status="HOST_READINESS_BLOCKED", source_preservation="FAILED",
                              preservation_error=repr(exc))
                code = 1
        save(out / "host-report.json", report)
    print("Stage12HostReadiness=" + report["status"], flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
