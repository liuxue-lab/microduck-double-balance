"""Resume, checkpoint and terminal-metric failure modes; no MuJoCo/PPO smoke."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.double_balance_stage07 import HOLD_LEVELS, monitor_updates
from mjlab_microduck.double_balance_training import (
    BASELINE_COMMIT, EVALUATION_PROTOCOL, MIGRATED_SHA256, checkpoint_metadata,
    evaluation_rank, restore_training_progress, save_checkpoint, terminal_episodes,
    validate_capacity, validate_training_checkpoint,
)


def fixture_state(completed=7):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    for _ in range(completed):
        optimizer.zero_grad()
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
    raw = SimpleNamespace(device="cpu", num_envs=6, cfg=SimpleNamespace(decimation=10),
                          common_step_counter=completed * 24, _sim_step_counter=completed * 240,
                          episode_length_buf=torch.full((6,), 150),
                          _basketball_state=SimpleNamespace(level=torch.arange(6),
                                                           hold=torch.tensor(HOLD_LEVELS)))
    def state_dict():
        return {"actor_state_dict": model.state_dict(), "critic_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()}
    alg = SimpleNamespace(optimizer=optimizer, learning_rate=2e-5, save=state_dict,
                          num_learning_epochs=1, num_mini_batches=1,
                          actor=SimpleNamespace(reset=lambda: None))
    runner = SimpleNamespace(alg=alg, current_learning_iteration=completed - 1)
    checkpoint = deepcopy(state_dict())
    checkpoint.update(iter=completed - 1, infos={"env_state": {"common_step_counter": completed * 24},
                      "stage07": checkpoint_metadata(runner, raw, completed, "test-commit")})
    return runner, raw, checkpoint


def test_resume_preserves_moments_lr_and_course_despite_age_zero_demotion():
    runner, raw, checkpoint = fixture_state()
    before = deepcopy(runner.alg.save())
    raw._basketball_state.level.zero_()
    raw._basketball_state.hold.fill_(1)
    def reset():
        assert raw.common_step_counter == 7 * 24
        assert not raw.episode_length_buf.any()
        # Ordinary reset logic would demote any loaded levels at this point.
        raw._basketball_state.level.sub_(1).clamp_(min=0)
        raw._basketball_state.hold.copy_(torch.tensor(HOLD_LEVELS)[raw._basketball_state.level])
    env = SimpleNamespace(unwrapped=raw, num_envs=6, reset=reset)
    assert restore_training_progress(runner, env, checkpoint) == 7
    assert runner.current_learning_iteration == 7  # NOT the saved last index 6.
    assert raw._sim_step_counter == 1680
    assert raw._basketball_state.level.tolist() == list(range(6))
    assert torch.equal(raw._basketball_state.hold, torch.tensor(HOLD_LEVELS))
    assert runner.alg.learning_rate == runner.alg.optimizer.param_groups[0]["lr"] == 2e-5
    for key, value in before["actor_state_dict"].items():
        assert torch.equal(value, runner.alg.save()["actor_state_dict"][key])
    for key, value in before["optimizer_state_dict"]["state"].items():
        assert torch.equal(value["exp_avg"], runner.alg.optimizer.state_dict()["state"][key]["exp_avg"])


@pytest.mark.parametrize("corruption", ["lr", "counter", "level", "hold", "lineage", "index"])
def test_resume_rejects_inconsistent_metadata(corruption):
    _, _, checkpoint = fixture_state()
    state = checkpoint["infos"]["stage07"]
    if corruption == "lr": state["learning_rate"] = 1e-3
    if corruption == "counter": state["common_step_counter"] += 24
    if corruption == "level": state["level"][0] = 6
    if corruption == "hold": state["hold"][0] = .5
    if corruption == "lineage": state["initialization_sha256"] = "smoke-checkpoint"
    if corruption == "index": checkpoint["iter"] += 1
    with pytest.raises(ValueError):
        validate_training_checkpoint(checkpoint, 6)


def test_initial_and_trained_checkpoints_save_with_completed_update_names(tmp_path):
    for completed in (0, 7):
        runner, raw, _ = fixture_state(completed)
        before_index = runner.current_learning_iteration
        def registered_save(path, infos):
            saved = deepcopy(runner.alg.save())
            saved.update(iter=runner.current_learning_iteration,
                         infos={**infos, "env_state": {"common_step_counter": raw.common_step_counter}})
            torch.save(saved, path)
        output = tmp_path / str(completed)
        output.mkdir()
        path = save_checkpoint(runner, SimpleNamespace(unwrapped=raw, num_envs=6),
                               output, completed, "test-commit", registered_save)
        assert path.name == f"update_{completed:06d}.pt"
        assert (output / "latest.pt").resolve() == path
        saved = torch.load(path, weights_only=False)
        assert saved["iter"] == completed - 1
        assert validate_training_checkpoint(saved)["next_iteration"] == completed
        assert runner.current_learning_iteration == before_index
        # Corruption must be detected rather than treated as an already-saved checkpoint.
        path.write_bytes(b"broken")
        with pytest.raises(ValueError, match="checksum"):
            save_checkpoint(runner, SimpleNamespace(unwrapped=raw, num_envs=6),
                            output, completed, "test-commit", registered_save)


def test_terminal_metrics_use_final_value_not_any_earlier_success_or_reset_state():
    names = ["double_balance_success", "double_balance_stable_fraction", "top_ball_center_error_m"]
    metrics = SimpleNamespace(active_terms=names, _step_count=torch.tensor([500, 500, 20, 10]),
                              _step_values=torch.tensor([[0., .8, .01], [1., 1., 0.],
                                                         [1., 1., 0.], [1., 1., 0.]]),
                              _episode_sums={names[0]: torch.tensor([100., 250., 1., 10.]),
                                             names[1]: torch.tensor([400., 500., 20., 10.]),
                                             names[2]: torch.tensor([5., 0., 0., 0.])})
    raw = SimpleNamespace(metrics_manager=metrics, step_dt=.02,
                          reset_terminated=torch.tensor([False, False, True, False]),
                          termination_manager=SimpleNamespace(active_terms=[]))
    episodes, ended = terminal_episodes(raw, torch.tensor([True, True, True, False]), torch.ones(4))
    assert ended.tolist() == [True, True, True, False]
    assert [e["success"] for e in episodes] == [False, True, False]
    assert episodes[0]["double_balance_stable_fraction"] == pytest.approx(.8)
    assert metrics._episode_sums[names[0]][0] == 100  # No reset or recompute.


def test_selection_prefers_earlier_better_model_over_later_regression():
    best = {"status": "PASS", "protocol": EVALUATION_PROTOCOL, "success_fraction": 1.,
            "mean_survival_seconds": 10., "mean_stable_fraction": .8, "mean_top_center_error_m": .005}
    later = {**best, "success_fraction": .5, "mean_stable_fraction": .9}
    assert evaluation_rank(best) > evaluation_rank(later)
    with pytest.raises(ValueError):
        evaluation_rank({**best, "status": "FAIL"})


def test_capacity_pass_alone_does_not_authorize_a_case_without_headroom():
    report = {"status": "PASS", "checkpoint_sha256": MIGRATED_SHA256, "matrix": [
        {"num_envs": 4096, "status": "PASS", "exit_code": 0,
         "eligible_with_15_percent_headroom": False, "sampled_peak_gpu_used_mib": 79000,
         "gpu_total_mib": 81920, "transitions_per_second": 20000}]}
    with pytest.raises(ValueError, match="headroom"):
        validate_capacity(report, 4096)


def test_resumed_monitor_uses_absolute_progress_and_callback_after_native_log(tmp_path, monkeypatch):
    runner, raw, _ = fixture_state()
    raw.step_dt = .02
    runner.alg.compute_returns = lambda obs: None
    runner.alg.update = lambda: {}
    order = []
    runner.logger = SimpleNamespace(log=lambda **kwargs: order.append("native"),
                                    ep_extras=[], rewbuffer=[], lenbuffer=[])
    env = SimpleNamespace(unwrapped=raw, num_envs=6, step=lambda actions: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    with monitor_updates(runner, env, tmp_path, 1, start_completed=7,
                         on_update=lambda row, evidence: order.append(row["completed_updates"])):
        raw.common_step_counter = 8 * 24
        runner.logger.log(it=7, collect_time=.2, learn_time=.1, loss_dict={"value": 1.})
    assert order == ["native", 8]
    row = json.loads((tmp_path / "iterations.jsonl").read_text())
    assert row["completed_updates"] == 8 and row["runner_iteration"] == 7
