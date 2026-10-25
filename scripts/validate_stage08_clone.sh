#!/usr/bin/env bash
# Preserve the cloned venv. No apt, pip, uv sync, or Stage 06 smoke.
set -Eeuo pipefail
stage08_head=${1:?Expected committed source required}
[[ "$stage08_head" =~ ^[0-9a-f]{40}$ ]]
stage08_root=/root/autodl-tmp/microduck-double-balance
stage08_artifacts="$stage08_root/artifacts/double-balance-stage08"
cd "$stage08_root/workspace"
test "$(git rev-parse HEAD)" = "$stage08_head"
test -z "$(git status --porcelain)"
test -f "$stage08_artifacts/budget.json"
test -x .venv/bin/python
mkdir -p "$stage08_artifacts/clone-validation"
if [ "${2:-}" != --worker ]; then
  if tmux has-session -t '=microduck-stage08-clone' 2>/dev/null; then
    printf 'Stage08CloneValidation=ALREADY_RUNNING\n'
  else
    printf -v stage08_command 'bash %q %q --worker' "$PWD/scripts/validate_stage08_clone.sh" "$stage08_head"
    tmux new-session -d -s microduck-stage08-clone "$stage08_command"
    printf 'Stage08CloneValidation=STARTED\n'
  fi
  printf 'Log=%s/clone-validation/latest.log\nFormalTrainingStarted=False\n' "$stage08_artifacts"
  exit 0
fi
exec 9>"$stage08_artifacts/setup/install.lock"
flock -n 9
stage08_log=$(mktemp "$stage08_artifacts/clone-validation/validate-XXXXXX.log")
ln -sfn "$stage08_log" "$stage08_artifacts/clone-validation/latest.log"
exec > >(tee -a "$stage08_log") 2>&1
trap 'stage08_exit=$?; printf "ExitCode=%s\nLog=%s\n" "$stage08_exit" "$stage08_log" > "$stage08_artifacts/clone-validation/status.txt"' EXIT
printf 'Stage08Environment=REUSE_CLONE\nFormalTrainingStarted=False\n'
export MUJOCO_GL=egl
export PYTHONPATH="$PWD/src"
export XDG_CACHE_HOME="$stage08_root/cache/xdg"
mkdir -p "$XDG_CACHE_HOME"
timeout --signal=TERM --kill-after=30s 300s .venv/bin/python -u - "$stage08_artifacts/budget.json" <<'PY'
import json, sys
import torch
from mjlab_microduck.double_balance_stage08_budget import BudgetLedger
from mjlab_microduck.double_balance_stage08_state import environment_preflight
ledger = BudgetLedger(sys.argv[1])
with ledger.job_lock():
    state = ledger.snapshot()
    assert state['gpu'] == '5090' and ledger.remaining(training=True) > 2100, 'Insufficient retained budget for validation'
    assert sys.version_info[:2] == (3, 12), 'Preserve the supported Python 3.12 clone'
    print('ClonePython=' + sys.version.splitlines()[0], flush=True)
    print('Stage08Source=' + environment_preflight('5090'), flush=True)
    x = torch.arange(16, device='cuda', dtype=torch.float32)
    assert (x * x).sum().item() == 1240
    torch.cuda.synchronize()
    print('Stage08TorchCuda=PASS', flush=True)
    print('Budget=' + json.dumps({'campaign_id': state['campaign_id'], 'started_at': state['started_at'],
          'elapsed_hours': ledger.snapshot()['elapsed_seconds'] / 3600,
          'remaining_training_hours': ledger.remaining(training=True) / 3600}), flush=True)
PY
# Real Warp/MuJoCo kernels, checkpoint restore and memory headroom are checked
# by the existing bounded D/E capacity path, not certified by metadata.
stage08_capacity="$stage08_artifacts/capacity-5090"
if [ -e "$stage08_capacity" ]; then
  # Preserve all paths referenced by the previous failure report/checkpoints.
  stage08_attempt=$(mktemp -d "$stage08_artifacts/capacity-retry-XXXXXX")
  stage08_capacity="$stage08_attempt/run"
fi
printf 'CapacitySummary=%s/capacity-summary.json\n' "$stage08_capacity"
timeout --signal=TERM --kill-after=30s 1950s .venv/bin/python -u \
  -m mjlab_microduck.double_balance_stage08 capacity --gpu 5090 \
  --ledger "$stage08_artifacts/budget.json" \
  --checkpoint "$stage08_artifacts/references/update_001000.pt" \
  --datasets "$stage08_artifacts/initial-states" \
  --output "$stage08_capacity"
printf 'CapacitySummary=%s/capacity-summary.json\n' "$stage08_capacity"
printf 'Stage08CloneValidation=PASS\nStage08Capacity=PASS\nFormalTrainingStarted=False\n'
