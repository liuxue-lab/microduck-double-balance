"""Archive a completed local Stage-06 run and render its factual handoff.

Standard library only. Reads existing evidence; never runs PPO, pytest, or a
network command. Git commits, annotated tags and SSH pushes remain explicit
commands in the user's local terminal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from string import Template
import subprocess


IMPLEMENTATION_COMMIT = "06ec7241766d4e6ef67b90580855344dfa9748d3"
MIGRATED_SHA256 = "548758afc546e11d8f3f446a4bcc846283a669ff20f4048da7a3a7bb1332b98e"
HANDOFF = "docs/handoffs/microduck-stage-06-runner-smoke-handoff.md"
EVIDENCE = "docs/audits/stage-06-cuda-acceptance.json"
PYTEST_LOG = "docs/audits/stage-06-local-pytest.txt"
PLAN = "docs/audits/stage-06-runner-and-smoke-plan.md"
PENDING = "状态：代码与 CPU 预检已完成；用户本机 CUDA 验收结果待回收。本文不是 Stage 06 完成交接，当前不创建 `stage-06-complete` 标签。"
COMPLETE = "状态：用户本机 CUDA smoke、保存重载与完整回归均已通过。实际证据和最终交接见 `stage-06-cuda-acceptance.json`、`stage-06-local-pytest.txt` 及 `../handoffs/microduck-stage-06-runner-smoke-handoff.md`。本文件保留实现前的预检记录。"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_finite(value):
    if isinstance(value, float):
        require(math.isfinite(value), "non-finite number in acceptance report")
    elif isinstance(value, dict):
        for child in value.values():
            require_finite(child)
    elif isinstance(value, list):
        for child in value:
            require_finite(child)


def pytest_summary(text):
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    summaries = [line.strip().strip("= ") for line in text.splitlines()
                 if re.search(r"\b\d+ passed\b", line)]
    require(bool(summaries), "pytest success summary missing")
    result = summaries[-1]
    require(not re.search(r"\b[1-9]\d* (?:failed|errors?)\b", result), "pytest did not pass")
    require(re.search(r"\b254 passed\b", result) is not None, "expected the accepted 254-test run")
    require(re.search(r"\b2 skipped\b", result) is not None, "unexpected skipped-test count")
    return result


def validate_report(report):
    require_finite(report)
    expected = {
        "status": "PASS", "mode": "smoke", "task_id": "Mjlab-DoubleBalance-MicroDuck",
        "git_head": IMPLEMENTATION_COMMIT, "num_envs": 64,
        "input_sha256": MIGRATED_SHA256, "input_size_bytes": 3342581,
        "completed_iterations": 5, "final_iteration": 7003,
        "final_common_step_counter": 168168, "transitions": 7680,
        "formal_training_started": False, "formal_success_rate": None,
        "input_unmodified": True,
    }
    for key, value in expected.items():
        require(report.get(key) == value, f"acceptance field mismatch: {key}")
    for key, value in {"status": "PASS", "actor_dimension": 61, "critic_dimension": 85,
                       "action_dimension": 14, "iteration": 6999,
                       "common_step_counter": 168048, "optimizer_state_entries": 0,
                       "learning_rate": 1e-3}.items():
        require(report["load"].get(key) == value, f"load field mismatch: {key}")
    for key, value in {"vector_steps": 120, "return_passes": 5,
                       "optimizer_steps": 100, "nan_terminations": 0}.items():
        require(report["smoke"].get(key) == value, f"smoke field mismatch: {key}")
    rows = report["smoke"]["iterations"]
    require(len(rows) == 5, "missing iteration evidence")
    for index, row in enumerate(rows, 1):
        require(row["update"] == index, "iteration order mismatch")
        require(row["common_step_counter"] == 168048 + index * 24, "step evidence mismatch")
        require(bool(row["losses"]) and row["learning_rate"] > 0, "invalid loss/LR evidence")
    reload = report["reload"]
    require(reload["status"] == "PASS", "reload did not pass")
    require(reload["strict_model_and_optimizer_equality"] is True, "reload state mismatch")
    require(reload["recurrent_action_hidden_and_value_equality"] is True, "reload inference mismatch")
    require(reload["scheduler_learning_rate"] == rows[-1]["learning_rate"], "restored LR mismatch")
    for name in ("actor_new_columns_max_abs", "critic_new_columns_max_abs"):
        require(report[name] > 0, f"new inputs did not learn: {name}")
    require(report["cloud_preflight"]["configuration_status"] == "PASS", "cloud config failed")
    require(report["cloud_preflight"]["capacity_status"] == "NOT_MEASURED_ON_CLOUD_GPU",
            "unexpected cloud capacity claim")
    require(report["cloud_preflight"]["long_training_started"] is False, "unexpected long run")


def read_evidence(path):
    raw_report = path.read_bytes()
    report = json.loads(raw_report)
    validate_report(report)
    checkpoint = Path(report["checkpoint"]["path"]).resolve()
    require(checkpoint.is_relative_to(path.parent.resolve()), "checkpoint outside accepted run")
    require(checkpoint.is_file(), "smoke checkpoint missing")
    require(checkpoint.stat().st_size == report["checkpoint"]["size_bytes"], "smoke size mismatch")
    require(sha256(checkpoint) == report["checkpoint"]["sha256"], "smoke SHA-256 mismatch")
    require(sha256(report["input_checkpoint"]) == MIGRATED_SHA256, "migration input changed")
    test_log = Path(str(path.parent) + ".pytest.log").read_bytes()
    summary = pytest_summary(test_log.decode("utf-8"))
    return report, raw_report, test_log, summary


def render_handoff(template, report, report_path, summary, tool_commit):
    checkpoint = report["checkpoint"]
    rows = report["smoke"]["iterations"]
    loss_names = sorted({key for row in rows for key in row["losses"]})
    loss_table = ["| PPO 更新 | runner iteration | 环境计数 | 学习率 | " + " | ".join(loss_names) + " |",
                  "|---|---|---|---|" + "---|" * len(loss_names)]
    for row in rows:
        cells = [str(row["update"]), str(6998 + row["update"]), str(row["common_step_counter"]),
                 f"{row['learning_rate']:.9g}"]
        cells += [f"{row['losses'][key]:.9g}" for key in loss_names]
        loss_table.append("| " + " | ".join(cells) + " |")
    versions = "\n".join(f"| {name} | {value} |" for name, value in report["versions"].items())
    return Template(template).substitute(
        created_utc=report["created_utc"], implementation_commit=report["git_head"],
        tool_commit=tool_commit, report_path=str(report_path), report_sha256=sha256(report_path),
        input_path=report["input_checkpoint"], input_sha256=report["input_sha256"],
        checkpoint_path=checkpoint["path"], checkpoint_size=checkpoint["size_bytes"],
        checkpoint_sha256=checkpoint["sha256"], gpu=report["gpu"], versions=versions,
        runner_class=report["load"]["runner_class"], pytest_summary=summary,
        elapsed=f"{report['smoke_seconds']:.3f}", loss_table="\n".join(loss_table),
        actor_new_columns=f"{report['actor_new_columns_max_abs']:.9g}",
        critic_new_columns=f"{report['critic_new_columns_max_abs']:.9g}",
        scheduler_lr=f"{report['reload']['scheduler_learning_rate']:.9g}",
        peak_allocated=report["peak_torch_allocated_bytes"],
        peak_reserved=report["peak_torch_reserved_bytes"],
        hold_levels=json.dumps(report["load"]["inherited_ball_hold_levels"]),
        onnx_files="\n".join("- `" + p + "`" for p in report["automatic_onnx_files"]) or "无。",
        buffer_bytes=report["cloud_preflight"]["known_rollout_and_lstm_buffer_bytes"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    def git(*argv):
        return subprocess.check_output(["git", *argv], cwd=repo, text=True).strip()

    require(git("branch", "--show-current") == "double-balance", "wrong branch")
    require(not git("status", "--porcelain"), "working tree must be clean before finalization")
    require(git("rev-parse", "stage-05-complete^{}") ==
            "c5b0aee66de10aa0c8aea0bdc85f1a917fada1df", "Stage 05 tag mismatch")
    subprocess.run(["git", "merge-base", "--is-ancestor", IMPLEMENTATION_COMMIT, "HEAD"],
                   cwd=repo, check=True)
    # Only documentation/finalization additions may follow the accepted runtime.
    require(not git("diff", IMPLEMENTATION_COMMIT, "HEAD", "--", "src", "pyproject.toml",
                    "uv.lock", "scripts/validate_double_balance_smoke.py"),
            "accepted runtime changed after the CUDA smoke")
    require(git("remote", "get-url", "--push", "origin") ==
            "git@github.com:liuxue-lab/microduck-double-balance.git", "origin is not the fixed SSH remote")
    report_path = args.report.resolve()
    report, raw_report, test_log, summary = read_evidence(report_path)
    tool_commit = git("log", "-1", "--format=%H", "--", "scripts/finalize_double_balance_stage06.py")
    template = (repo / "docs/templates/microduck-stage-06-handoff.md.in").read_text()
    handoff = render_handoff(template, report, report_path, summary, tool_commit)
    old_plan = (repo / PLAN).read_text()
    require(PENDING in old_plan or COMPLETE in old_plan, "unexpected audit plan status")
    updated_plan = old_plan.replace(PENDING, COMPLETE)
    # Prepare all payloads before any write. Existing evidence must agree exactly.
    payloads = {HANDOFF: handoff.encode("utf-8"), EVIDENCE: raw_report, PYTEST_LOG: test_log}
    for name, content in payloads.items():
        target = repo / name
        require(not target.exists() or target.read_bytes() == content,
                f"refusing to replace different existing evidence: {name}")
    for name, content in {**payloads, PLAN: updated_plan.encode("utf-8")}.items():
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    print("Stage06FinalDocsBegin=")
    print("CheckpointSHA256=" + report["checkpoint"]["sha256"])
    print("LocalPytestSummary=" + summary)
    print("FinalHandoff=" + str(repo / HANDOFF))
    print("ArchivedAcceptance=" + str(repo / EVIDENCE))
    print("Stage06FinalDocs=PASS")
    print("Stage06FinalDocsEnd=")


if __name__ == "__main__":
    main()
