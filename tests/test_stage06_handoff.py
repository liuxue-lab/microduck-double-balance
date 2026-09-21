"""Archive/render tests with synthetic evidence, never CUDA acceptance results."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "finalize_stage06", ROOT / "scripts/finalize_double_balance_stage06.py"
)
finalize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(finalize)


@pytest.fixture()
def evidence(tmp_path, monkeypatch):
    run = tmp_path / "synthetic-run"
    run.mkdir()
    model = run / "model_7003.pt"
    model.write_bytes(b"synthetic checkpoint for file-integrity testing only")
    source = tmp_path / "migration.pt"
    source.write_bytes(b"synthetic input for file-integrity testing only")
    monkeypatch.setattr(finalize, "MIGRATED_SHA256", finalize.sha256(source))
    report = {
        "status": "PASS", "mode": "smoke", "task_id": "Mjlab-DoubleBalance-MicroDuck",
        "git_head": finalize.IMPLEMENTATION_COMMIT, "num_envs": 64,
        "input_checkpoint": str(source), "input_sha256": finalize.MIGRATED_SHA256,
        "input_size_bytes": 3342581, "completed_iterations": 5,
        "final_iteration": 7003, "final_common_step_counter": 168168,
        "transitions": 7680, "formal_training_started": False,
        "formal_success_rate": None, "input_unmodified": True,
        "gpu": "SYNTHETIC TEST FIXTURE - NOT CUDA EVIDENCE",
        "created_utc": "2026-09-21T00:00:00+00:00", "smoke_seconds": 1.0,
        "versions": {"torch": "2.9.1"}, "actor_new_columns_max_abs": 0.01,
        "critic_new_columns_max_abs": 0.02, "peak_torch_allocated_bytes": 100,
        "peak_torch_reserved_bytes": 200, "automatic_onnx_files": [],
        "checkpoint": {"path": str(model), "size_bytes": model.stat().st_size,
                       "sha256": finalize.sha256(model)},
        "load": {"status": "PASS", "actor_dimension": 61, "critic_dimension": 85,
                 "action_dimension": 14, "iteration": 6999, "common_step_counter": 168048,
                 "optimizer_state_entries": 0, "learning_rate": 1e-3,
                 "runner_class": "synthetic.fixture", "inherited_ball_hold_levels": [1, 0]},
        "smoke": {"vector_steps": 120, "return_passes": 5, "optimizer_steps": 100,
                  "nan_terminations": 0, "iterations": [
                      {"update": i, "common_step_counter": 168048 + 24 * i,
                       "learning_rate": 1e-3, "losses": {"value": 0.1, "surrogate": -0.1}}
                      for i in range(1, 6)]},
        "reload": {"status": "PASS", "strict_model_and_optimizer_equality": True,
                   "recurrent_action_hidden_and_value_equality": True,
                   "scheduler_learning_rate": 1e-3},
        "cloud_preflight": {"configuration_status": "PASS",
                            "capacity_status": "NOT_MEASURED_ON_CLOUD_GPU",
                            "long_training_started": False,
                            "known_rollout_and_lstm_buffer_bytes": 277315584},
    }
    path = run / "acceptance.json"
    path.write_text(json.dumps(report))
    Path(str(run) + ".pytest.log").write_text(
        "254 passed, 2 skipped, 3 warnings in 19.95s\n")
    return path, report


def test_reads_original_bytes_and_renders_actual_checkpoint_metadata(evidence):
    path, _ = evidence
    report, raw, log, summary = finalize.read_evidence(path)
    assert raw == path.read_bytes()
    assert "254 passed" in summary
    template = (ROOT / "docs/templates/microduck-stage-06-handoff.md.in").read_text()
    handoff = finalize.render_handoff(template, report, path, summary, "fixture-tool-commit")
    assert report["checkpoint"]["sha256"] in handoff
    assert report["checkpoint"]["path"] in handoff
    assert "$checkpoint" not in handoff
    assert "NOT_MEASURED_ON_CLOUD_GPU" in handoff
    assert "microduck-local-ssh-push-protocol.md" in handoff
    assert "SYNTHETIC TEST FIXTURE" in handoff


@pytest.mark.parametrize("field,value", [
    ("mode", "load"), ("completed_iterations", 4),
    ("final_iteration", 7004), ("formal_training_started", True),
])
def test_rejects_pass_label_with_inconsistent_evidence(evidence, field, value):
    _, original = evidence
    report = deepcopy(original)
    report[field] = value
    with pytest.raises(ValueError, match="field mismatch"):
        finalize.validate_report(report)


def test_rejects_corrupted_checkpoint_even_when_report_says_pass(evidence):
    path, report = evidence
    checkpoint = Path(report["checkpoint"]["path"])
    payload = checkpoint.read_bytes()
    checkpoint.write_bytes(b"X" + payload[1:])
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        finalize.read_evidence(path)


def test_rejects_nonfinite_loss_and_failed_pytest_summary(evidence):
    _, report = evidence
    report["smoke"]["iterations"][0]["losses"]["value"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        finalize.validate_report(report)
    with pytest.raises(ValueError, match="did not pass"):
        finalize.pytest_summary("1 failed, 254 passed, 2 skipped in 20s")
