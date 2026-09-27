#!/usr/bin/env bash
# Import the new bundle locally first; deploy with one SSH authentication.
set -Eeuo pipefail
stage07_repo=/home/lx/microduck-double-balance/workspace
stage07_downloads=/home/lx/下载
stage07_bundle=microduck-stage-07-from-stage-06.bundle
stage07_port=${1:-26497}
if [ "$#" -gt 0 ]; then shift; fi
[[ "$stage07_port" =~ ^[0-9]+$ ]] && [ "$stage07_port" -ge 1 ] && [ "$stage07_port" -le 65535 ]
cd "$stage07_repo"
test "$(git branch --show-current)" = double-balance
test -z "$(git status --porcelain)"
stage07_bundle_head=$(git bundle list-heads "$stage07_downloads/$stage07_bundle" refs/heads/double-balance)
test "${stage07_bundle_head%% *}" = "$(git rev-parse HEAD)"
stage07_remote=$(cat <<'REMOTE'
set -Eeuo pipefail
stage07_root=/root/autodl-tmp/microduck-double-balance
cd "$stage07_root/workspace"
test -x .venv/bin/python
test "$(git branch --show-current)" = double-balance
test -z "$(git status --porcelain)"
for stage07_session in microduck-stage07-training microduck-stage07-capacity; do
  if tmux has-session -t "=$stage07_session" 2>/dev/null; then
    printf 'Existing job still running: %s. Source update stopped.\n' "$stage07_session" >&2
    exit 1
  fi
done
stage07_incoming=$(mktemp -d /root/autodl-tmp/stage07-training-upload.XXXXXX)
tar --no-same-owner --no-same-permissions -xf - -C "$stage07_incoming"
git bundle verify "$stage07_incoming/microduck-stage-07-from-stage-06.bundle"
git fetch "$stage07_incoming/microduck-stage-07-from-stage-06.bundle" \
  refs/heads/double-balance:refs/remotes/stage07/double-balance
git merge --ff-only refs/remotes/stage07/double-balance
# Retained reports/caches/checkpoints are reused; no package installation or capacity rerun.
REMOTE
)
printf -v stage07_launch '%q ' bash scripts/launch_stage07_training.sh "$@"
stage07_remote+=$'\n'
stage07_remote+="$stage07_launch"
tar -cf - -C "$stage07_downloads" "$stage07_bundle" \
  | ssh -T -p "$stage07_port" -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    root@connect.bjb1.seetacloud.com "$stage07_remote"
