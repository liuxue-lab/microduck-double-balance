#!/usr/bin/env bash
# Reuse the transferred clean checkout and lockfile. Never starts PPO.
set -Eeuo pipefail
stage08_gpu=${1:?GPU required}
stage08_commit=${2:?Expected commit required}
stage08_started=${3:?Deployment budget start with timezone required}
case "$stage08_gpu" in A800|5090) ;; *) exit 2;; esac
[[ "$stage08_commit" =~ ^[0-9a-f]{40}$ ]]
stage08_root=/root/autodl-tmp/microduck-double-balance
stage08_repo="$stage08_root/workspace"
stage08_artifacts="$stage08_root/artifacts/double-balance-stage08"
test -d /root/autodl-tmp
test "$(uname -m)" = x86_64
cd "$stage08_repo"
source scripts/stage08_bootstrap.sh
stage08_python=$(stage08_resolve_python)
test "$(git rev-parse HEAD)" = "$stage08_commit"
test -z "$(git status --porcelain)"
mkdir -p "$stage08_artifacts/setup"

# Account from the automatically retained deployment start. Never reset on retry.
if [ ! -f "$stage08_artifacts/budget.json" ]; then
  PYTHONPATH=src "$stage08_python" -m mjlab_microduck.double_balance_stage08 init-budget \
    --gpu "$stage08_gpu" --started-at "$stage08_started" --ledger "$stage08_artifacts/budget.json"
else
  PYTHONPATH=src "$stage08_python" - "$stage08_artifacts/budget.json" "$stage08_gpu" <<'PY'
import sys
from mjlab_microduck.double_balance_stage08_budget import BudgetLedger
ledger=BudgetLedger(sys.argv[1]); state=ledger.snapshot()
assert state['gpu']==sys.argv[2], 'Existing budget uses another GPU'
assert ledger.remaining(training=True)>0, 'Training budget is exhausted'
print('Stage08Budget=REUSED')
PY
fi

if [ "${4:-}" != --worker ]; then
  if ! command -v tmux >/dev/null; then
    apt-get -o DPkg::Lock::Timeout=120 update -qq
    apt-get -o DPkg::Lock::Timeout=120 install -y -qq --no-install-recommends tmux
  fi
  if tmux has-session -t '=microduck-stage08-setup' 2>/dev/null; then
    printf 'Stage08Setup=ALREADY_RUNNING\n'
  else
    printf -v stage08_command 'bash %q %q %q %q --worker' "$stage08_repo/scripts/setup_stage08_autodl.sh" "$stage08_gpu" "$stage08_commit" "$stage08_started"
    tmux new-session -d -s microduck-stage08-setup "$stage08_command"
    printf 'Stage08Setup=STARTED\n'
  fi
  printf 'Log=%s/setup/latest.log\n' "$stage08_artifacts"
  exit 0
fi

exec 9>"$stage08_artifacts/setup/install.lock"
flock -n 9
stage08_log=$(mktemp "$stage08_artifacts/setup/install-XXXXXX.log")
ln -sfn "$stage08_log" "$stage08_artifacts/setup/latest.log"
exec > >(tee -a "$stage08_log") 2>&1
trap 'stage08_exit=$?; printf "ExitCode=%s\n" "$stage08_exit" > "$stage08_artifacts/setup/status.txt"' EXIT
export DEBIAN_FRONTEND=noninteractive
apt-get -o DPkg::Lock::Timeout=120 update -qq
apt-get -o DPkg::Lock::Timeout=120 install -y -qq --no-install-recommends \
  git ca-certificates curl xz-utils python3 python3-venv libegl1 libgl1 libglfw3 ffmpeg
mkdir -p "$stage08_root"/{tools,cache/uv,cache/xdg,cache/python,cache/tmp}
export UV_CACHE_DIR="$stage08_root/cache/uv"
export UV_PYTHON_INSTALL_DIR="$stage08_root/cache/python"
export XDG_CACHE_HOME="$stage08_root/cache/xdg"
export TMPDIR="$stage08_root/cache/tmp"
export UV_LINK_MODE=copy
export MUJOCO_GL=egl
if [ ! -x "$stage08_root/tools/bootstrap/bin/python" ]; then
  /usr/bin/python3 -m venv "$stage08_root/tools/bootstrap"
fi
"$stage08_root/tools/bootstrap/bin/python" -m pip install --no-cache-dir 'uv==0.12.15'
stage08_uv="$stage08_root/tools/bootstrap/bin/uv"
if [ -r /etc/network_turbo ]; then
  set +u
  source /etc/network_turbo >/dev/null
  set -u
fi
"$stage08_uv" sync --locked --python 3.12.14 --no-progress
.venv/bin/python scripts/preflight_stage08_cloud.py --gpu "$stage08_gpu" \
  --output "$stage08_artifacts/setup/preflight-$(date +%Y%m%dT%H%M%S).json"
.venv/bin/python - "$stage08_gpu" <<'PY'
import sys
from mjlab_microduck.double_balance_stage08_state import environment_preflight
print('Stage08Source='+environment_preflight(sys.argv[1]))
print('Stage08Dependencies=PASS')
print('Stage08CudaCapacity=PENDING_OR_REUSE_A800_EVIDENCE')
PY
.venv/bin/python -m pytest -q tests/test_stage08_plan.py tests/test_stage08_runtime.py \
  tests/test_stage08_budget.py tests/test_stage08_evaluation.py tests/test_stage08_review.py tests/test_stage08_tools.py
printf 'Stage08Setup=PASS\nFormalTrainingStarted=False\n'
