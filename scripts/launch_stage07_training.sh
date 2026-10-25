#!/usr/bin/env bash
# Cloud entry: reuse the installed environment and completed capacity report.
set -Eeuo pipefail
stage07_root=/root/autodl-tmp/microduck-double-balance
stage07_repo="$stage07_root/workspace"
stage07_artifacts="$stage07_root/artifacts/double-balance-stage07"
stage07_session=microduck-stage07-training
cd "$stage07_repo"
export PYTHONUNBUFFERED=1 MUJOCO_GL=egl
export XDG_CACHE_HOME="$stage07_root/cache/xdg" TMPDIR="$stage07_root/cache/tmp"
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"

if [ "${1:-}" = --worker ]; then
  shift
  stage07_output=${1:?missing output}
  shift
  exec > >(tee -a "$stage07_output.console.log") 2>&1
  exec .venv/bin/python -u -m mjlab_microduck.double_balance_training train \
    --output "$stage07_output" "$@"
fi

stage07_target=6000
stage07_resume=
while [ "$#" -gt 0 ]; do
  case "$1" in
    --target-updates) stage07_target=${2:?missing target}; shift 2 ;;
    --resume) stage07_resume=${2:?missing checkpoint}; shift 2 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ "$stage07_target" =~ ^[0-9]+$ ]] && [ "$stage07_target" -ge 1 ] && [ "$stage07_target" -le 16000 ]
if tmux has-session -t "=$stage07_session" 2>/dev/null; then
  printf 'Stage07TrainingJob=ALREADY_RUNNING\n'
  exit 0
fi
if tmux has-session -t '=microduck-stage07-capacity' 2>/dev/null; then
  printf 'Capacity job is still running.\n' >&2
  exit 1
fi
test -x .venv/bin/python
test -f "$stage07_artifacts/latest-capacity/capacity-summary.json"
if [ -e "$stage07_artifacts/latest-training" ] || [ -L "$stage07_artifacts/latest-training" ]; then
  if [ -z "$stage07_resume" ]; then
    printf 'Existing training retained. Inspect training.json, then use --resume with its checkpoint.\n' >&2
    exit 1
  fi
fi
stage07_args=(--num-envs 4096 --target-updates "$stage07_target")
if [ -n "$stage07_resume" ]; then
  test -f "$stage07_resume"
  stage07_args+=(--resume --checkpoint "$(readlink -f "$stage07_resume")")
fi
stage07_output="$stage07_artifacts/train-$(date -u +%Y%m%dT%H%M%SZ)-$$"
ln -sfn "$stage07_output" "$stage07_artifacts/latest-training"
ln -sfn "$stage07_output.console.log" "$stage07_artifacts/latest-training.log"
printf -v stage07_command '%q ' bash "$stage07_repo/scripts/launch_stage07_training.sh" \
  --worker "$stage07_output" "${stage07_args[@]}"
tmux new-session -d -s "$stage07_session" "$stage07_command"
printf 'Stage07TrainingJob=STARTED\nTargetCompletedUpdates=%s\nOutput=%s\nLog=%s\n' \
  "$stage07_target" "$stage07_output" "$stage07_artifacts/latest-training.log"
