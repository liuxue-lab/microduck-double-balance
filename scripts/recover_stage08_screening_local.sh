#!/usr/bin/env bash
# Laptop: apply the committed delta and launch one detached recovery on the clone.
set -Eeuo pipefail
stage08_host=${1:?Usage: recover_stage08_screening_local.sh HOST PORT}
stage08_port=${2:?Port required}
[[ "$stage08_host" =~ ^[A-Za-z0-9.-]+$ ]]
[[ "$stage08_port" =~ ^[0-9]+$ ]] && ((stage08_port>=1 && stage08_port<=65535))
cd /home/lx/microduck-double-balance/workspace
test "$(git branch --show-current)" = double-balance
test -z "$(git status --porcelain)"
stage08_base=c63fa4a4a5a804875288972b38abdbeb57a7eb16
git merge-base --is-ancestor "$stage08_base" HEAD
stage08_head=$(git rev-parse HEAD)
stage08_local=/home/lx/microduck-double-balance/artifacts/double-balance-stage08
mkdir -p "$stage08_local"
stage08_attempt=$(mktemp -d "$stage08_local/screening-recovery-XXXXXX")
stage08_id=$(basename "$stage08_attempt")
git bundle create "$stage08_attempt/delta.bundle" refs/heads/double-balance "^$stage08_base"
stage08_sha=$(sha256sum "$stage08_attempt/delta.bundle" | cut -d ' ' -f1)
stage08_root=/root/autodl-tmp/microduck-double-balance
stage08_remote="$stage08_root/incoming/$stage08_id"
stage08_common=(-o ControlMaster=auto -o ControlPersist=12h
  -o "ControlPath=$HOME/.ssh/microduck-stage08-%C"
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ConnectTimeout=15)
ssh "${stage08_common[@]}" -p "$stage08_port" "root@$stage08_host" "mkdir -p '$stage08_remote'"
scp "${stage08_common[@]}" -P "$stage08_port" "$stage08_attempt/delta.bundle" \
  "root@$stage08_host:$stage08_remote/delta.bundle"
ssh "${stage08_common[@]}" -p "$stage08_port" "root@$stage08_host" \
  "bash -s -- '$stage08_id' '$stage08_head' '$stage08_sha' '$stage08_base'" <<'REMOTE'
set -Eeuo pipefail
stage08_id=$1 stage08_head=$2 stage08_sha=$3 stage08_base=$4
stage08_root=/root/autodl-tmp/microduck-double-balance
stage08_artifacts="$stage08_root/artifacts/double-balance-stage08"
stage08_attempt="$stage08_artifacts/$stage08_id"
cd "$stage08_root/workspace"
test "$(git branch --show-current)" = double-balance
test -z "$(git status --porcelain)"
test -x .venv/bin/python
command -v tmux >/dev/null
for stage08_session in microduck-stage08-screening microduck-stage08-recovery microduck-stage08-clone; do
  if tmux has-session -t "=$stage08_session" 2>/dev/null; then
    printf 'Existing session: %s. Inspect it before deployment.\n' "$stage08_session" >&2
    exit 1
  fi
done
exec 8>"$stage08_artifacts/budget.job.lock"
exec 9>"$stage08_artifacts/setup/install.lock"
flock -n 8
flock -n 9
stage08_actual=$(git rev-parse HEAD)
test "$stage08_actual" = "$stage08_base" || test "$stage08_actual" = "$stage08_head"
stage08_gpu_pids=$(timeout 15s nvidia-smi --query-compute-apps=pid --format=csv,noheader)
test -z "$stage08_gpu_pids"
printf '%s  %s\n' "$stage08_sha" "$stage08_root/incoming/$stage08_id/delta.bundle" | sha256sum -c -
git bundle verify "$stage08_root/incoming/$stage08_id/delta.bundle"
git fetch "$stage08_root/incoming/$stage08_id/delta.bundle" \
  refs/heads/double-balance:refs/remotes/stage08/recovery
git merge --ff-only refs/remotes/stage08/recovery
test "$(git rev-parse HEAD)" = "$stage08_head"
test -z "$(git status --porcelain)"
mkdir "$stage08_attempt"
ln -sfn "$stage08_attempt/recovery.log" "$stage08_artifacts/screening-recovery-latest.log"
ln -sfn "$stage08_attempt/recovery.json" "$stage08_artifacts/screening-recovery-latest.json"
export MUJOCO_GL=egl PYTHONPATH="$PWD/src" XDG_CACHE_HOME="$stage08_root/cache/xdg"
printf -v stage08_command 'cd %q && env MUJOCO_GL=egl PYTHONPATH=%q XDG_CACHE_HOME=%q %q -u %q --attempt %q >%q 2>&1' \
  "$PWD" "$PYTHONPATH" "$XDG_CACHE_HOME" "$PWD/.venv/bin/python" \
  "$PWD/scripts/recover_stage08_screening.py" "$stage08_attempt" "$stage08_attempt/recovery.log"
flock -u 8
flock -u 9
tmux new-session -d -s microduck-stage08-recovery "$stage08_command"
printf 'Stage08Recovery=LAUNCHED\nA_AdditionalPPOUpdates=0\n'
printf 'Log=%s/screening-recovery-latest.log\n' "$stage08_artifacts"
REMOTE
