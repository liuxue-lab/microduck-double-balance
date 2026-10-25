"""Protect new-stage progress and actual-capacity reporting; no simulator run."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.double_balance_stage07 import (
    HOLD_LEVELS, monitor_updates, progress_after_updates,
    reset_stage07_progress, summarize_iterations,
)


def test_new_stage_resets_ages_before_reset_and_preserves_model():
    model = torch.nn.Linear(2, 1)
    before = deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    actor_resets = []
    raw = SimpleNamespace(
        cfg=SimpleNamespace(actions={"ball_hold": SimpleNamespace(levels=HOLD_LEVELS)}),
        common_step_counter=168048, _sim_step_counter=1680480,
        episode_length_buf=torch.tensor([400, 1]),
        _basketball_state=SimpleNamespace(level=torch.tensor([3, 4]), hold=torch.tensor([.1, .03])),
    )
    runner = SimpleNamespace(current_learning_iteration=6999,
                             alg=SimpleNamespace(optimizer=optimizer, learning_rate=2e-4,
                                                 actor=SimpleNamespace(reset=lambda: actor_resets.append(1))))
    def reset():
        # Check what the real curriculum manager will see, before it computes.
        assert raw.common_step_counter == raw._sim_step_counter == 0
        assert raw.episode_length_buf.tolist() == [0, 0]
        assert raw._basketball_state.level.tolist() == [0, 0]
    reset_stage07_progress(runner, SimpleNamespace(unwrapped=raw, reset=reset))
    assert runner.current_learning_iteration == 0
    assert runner.alg.learning_rate == optimizer.param_groups[0]["lr"] == .001
    assert actor_resets == [1]
    assert raw._basketball_state.hold.tolist() == [1, 1]
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[key])


def test_progress_uses_completed_updates_not_last_loop_index():
    assert progress_after_updates(0) == {
        "completed_updates": 0, "last_completed_iteration": -1,
        "next_iteration": 0, "common_step_counter": 0,
    }
    progress = progress_after_updates(15)
    assert progress["last_completed_iteration"] == 14
    assert progress["next_iteration"] == 15
    assert progress["common_step_counter"] == 360
    for invalid in (-1, True, 1.5):
        with pytest.raises(ValueError):
            progress_after_updates(invalid)


def test_capacity_excludes_compilation_and_reports_transitions_not_vector_steps():
    rows = [{"iteration_seconds": t} for t in (200, 30, 20, 1, 2, 3)]
    result = summarize_iterations(rows, 3, 1024)
    assert result["mean_iteration_seconds"] == 2
    assert result["transitions_per_second"] == 12288
    assert result["projected_2000_updates_hours_excluding_setup_eval_save"] == pytest.approx(4000 / 3600)
    for value in (float("nan"), float("inf"), 0, -1):
        rows[-1]["iteration_seconds"] = value
        with pytest.raises(ValueError, match="duration"):
            summarize_iterations(rows, 3, 1024)


def test_capacity_rejects_insufficient_measurements():
    with pytest.raises(ValueError, match="three"):
        summarize_iterations([{"iteration_seconds": 1}] * 4, 3, 512)


def test_capacity_detects_nan_termination_hidden_by_autoreset_and_restores_hooks(tmp_path):
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.Adam([parameter])
    original_update = lambda: {}
    original_returns = lambda obs: None
    alg = SimpleNamespace(optimizer=optimizer, compute_returns=original_returns, update=original_update)
    obs = {"actor": torch.zeros(512, 61), "critic": torch.zeros(512, 85)}
    original_step = lambda actions: (obs, torch.zeros(512), torch.zeros(512), {})
    raw = SimpleNamespace(device="cpu", termination_manager=SimpleNamespace(
        get_term=lambda name: torch.ones(512, dtype=torch.bool)))
    env = SimpleNamespace(num_envs=512, unwrapped=raw, step=original_step)
    original_log = lambda **kwargs: None
    runner = SimpleNamespace(alg=alg, logger=SimpleNamespace(log=original_log))
    with pytest.raises(ValueError, match="nan_state"):
        with monitor_updates(runner, env, tmp_path, 15):
            env.step(torch.zeros(512, 14))
    assert env.step is original_step
    assert alg.update is original_update
    assert alg.compute_returns is original_returns
    assert runner.logger.log is original_log
    assert not optimizer._optimizer_step_pre_hooks


@pytest.mark.parametrize("exit_code,peak,status", [(0, 74000, "PASS"), (3, 1024, "FAIL")])
def test_parent_uses_whole_gpu_headroom_and_rejects_failed_child(tmp_path, monkeypatch, exit_code, peak, status):
    path = Path(__file__).resolve().parents[1] / "scripts/measure_stage07_capacity.py"
    spec = importlib.util.spec_from_file_location("stage07_parent", path)
    parent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent)
    samples = iter([{"total_mib": 80000, "used_mib": 1000, "utilization_percent": 0},
                    {"total_mib": 80000, "used_mib": peak, "utilization_percent": 80}])
    monkeypatch.setattr(parent, "gpu_sample", lambda: next(samples))
    monkeypatch.setattr(parent.time, "sleep", lambda duration: None)
    class Child:
        pid = 99999999
        returncode = None
        polls = 0
        def poll(self):
            self.polls += 1
            if self.polls > 1:
                self.returncode = exit_code
            return self.returncode
    output = tmp_path / "envs-512"
    output.mkdir()
    # A stale success report alone cannot override a nonzero process exit.
    (output / "report.json").write_text(json.dumps({"status": "PASS", "torch_peak_reserved_bytes": 1024,
                                                  "transitions_per_second": 10000}))
    monkeypatch.setattr(parent.subprocess, "Popen", lambda *args, **kwargs: Child())
    args = SimpleNamespace(checkpoint=tmp_path / "checkpoint.pt", expected_gpu="A800", case_timeout=900)
    result = parent.run_case(args, 512, tmp_path, parent.time.monotonic() + 60)
    assert result["status"] == status
    assert result["sampled_peak_gpu_used_mib"] == max(1000, peak)
    assert result["eligible_with_15_percent_headroom"] is False
