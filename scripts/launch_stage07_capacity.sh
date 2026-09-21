#!/usr/bin/env bash
# Cloud-only launcher. No formal training, package installation, or Stage 06 smoke.
set -Eeuo pipefail
stage07_root=/root/autodl-tmp/microduck-double-balance
stage07_repo="$stage07_root/workspace"
stage07_artifacts="$stage07_root/artifacts/double-balance-stage07"
stage07_checkpoint="$stage07_root/artifacts/double-balance-stage05/checkpoint.pt"
stage07_session=microduck-stage07-capacity
cd "$stage07_repo"

if [ "${1:-}" = --worker ]; then
  stage07_output=${2:?missing capacity output path}
  exec > >(tee -a "$stage07_output.console.log") 2>&1
  exec .venv/bin/python -u scripts/measure_stage07_capacity.py \
    "$stage07_checkpoint" --output "$stage07_output" --expected-gpu A800
fi

if tmux has-session -t "=$stage07_session" 2>/dev/null; then
  printf 'Stage07CapacityJob=ALREADY_RUNNING\n'
  exit 0
fi
test -f "$stage07_checkpoint"
.venv/bin/python - <<'PY'
import json
from pathlib import Path
reports = sorted(Path('../artifacts/stage07-setup').glob('clone-environment-*.json'))
report = json.loads((reports[-1] if reports else Path('../artifacts/stage07-setup/environment.json')).read_text())
assert report['status'] == 'PASS', 'Cloud environment validation has not passed'
assert 'A800' in report['gpu'], 'Expected the selected A800 cloud instance'
PY
mkdir -p "$stage07_artifacts"
if [ -e "$stage07_artifacts/latest-capacity" ] || [ -L "$stage07_artifacts/latest-capacity" ]; then
  printf 'Existing capacity run retained: %s\nReview it before starting another run.\n' \
    "$(readlink -f "$stage07_artifacts/latest-capacity")"
  exit 1
fi
stage07_output="$stage07_artifacts/capacity-$(date -u +%Y%m%dT%H%M%SZ)-$$"
ln -s "$stage07_output" "$stage07_artifacts/latest-capacity"
ln -sfn "$stage07_output.console.log" "$stage07_artifacts/latest-capacity.log"
printf -v stage07_command 'bash %q --worker %q' \
  "$stage07_repo/scripts/launch_stage07_capacity.sh" "$stage07_output"
tmux new-session -d -s "$stage07_session" "$stage07_command"
printf 'Stage07CapacityJob=STARTED\nOutput=%s\nLog=%s\n' \
  "$stage07_output" "$stage07_artifacts/latest-capacity.log"
