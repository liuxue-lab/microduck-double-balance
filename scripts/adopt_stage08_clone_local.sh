#!/usr/bin/env bash
# Laptop entry: archive failed setup, update clone by Git delta, reuse its venv.
set -Eeuo pipefail
if [ "${1:-}" = --clone-only ] || [ "${1:-}" = --retry-capacity ]; then
  # The old address is only the key of the retained LAPTOP start record.
  # This mode makes no SSH/SCP connection to the old instance.
  stage08_clone_host=${2:?Clone host required}
  stage08_clone_port=${3:?Clone port required}
  stage08_clone_mode=--old-unavailable
  if [ "$1" = --retry-capacity ]; then stage08_clone_mode=--retry-capacity; fi
  set -- connect.bjb2.seetacloud.com 45743 "$stage08_clone_host" "$stage08_clone_port" "$stage08_clone_mode"
fi
stage08_old_host=${1:?Usage: adopt_stage08_clone_local.sh OLD_HOST OLD_PORT NEW_HOST NEW_PORT [--old-unavailable]}
stage08_old_port=${2:?Old port required}
stage08_host=${3:?Clone host required}
stage08_port=${4:?Clone port required}
stage08_mode=${5:-}
case "$stage08_mode" in ''|--old-unavailable|--retry-capacity) ;; *) exit 2;; esac
for stage08_value in "$stage08_old_host" "$stage08_host"; do
  [[ "$stage08_value" =~ ^[A-Za-z0-9.-]+$ ]]
done
for stage08_value in "$stage08_old_port" "$stage08_port"; do
  [[ "$stage08_value" =~ ^[0-9]+$ ]] && ((stage08_value>=1 && stage08_value<=65535))
done
test "$stage08_old_host:$stage08_old_port" != "$stage08_host:$stage08_port"
cd /home/lx/microduck-double-balance/workspace
test "$(git branch --show-current)" = double-balance
test -z "$(git status --porcelain)"
stage08_base=2d10af503945f26f1d7364f19e4152eb540fb4d2
if [ "$stage08_mode" = --retry-capacity ]; then
  stage08_base=b8e991c49e9ede7bfc9b2d0725632fbcfc268b81
fi
git merge-base --is-ancestor "$stage08_base" HEAD
stage08_head=$(git rev-parse HEAD)
stage08_root=/root/autodl-tmp/microduck-double-balance
stage08_local=/home/lx/microduck-double-balance/artifacts/double-balance-stage08
mkdir -p "$stage08_local" /home/lx/下载
stage08_evidence=$(mktemp -d "$stage08_local/clone-migration-XXXXXX")
stage08_id=$(basename "$stage08_evidence")
stage08_archive="/home/lx/下载/microduck-stage08-$stage08_id.tar.gz"
stage08_common=(-o ControlMaster=auto -o ControlPersist=12h
  -o "ControlPath=$HOME/.ssh/microduck-stage08-%C"
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ConnectTimeout=15)
stage08_old="root@$stage08_old_host"
stage08_target="root@$stage08_host"
stage08_remote="$stage08_root/incoming/$stage08_id"

stage08_archive_sha=unused
if [ "$stage08_mode" != --retry-capacity ]; then
  if [ "$stage08_mode" = --old-unavailable ]; then
    # Never contact the unavailable host. Persist one clearly labelled replacement
    # identity, while charging from the original local deployment timestamp.
    stage08_archive_sha=$(python3 scripts/migrate_stage08_setup.py offline-pack \
      --root "$stage08_local/offline-recovery-$stage08_old_host-$stage08_old_port" \
      --original-start "$stage08_local/deployment-start-$stage08_old_host-$stage08_old_port-5090.txt" \
      --archive "$stage08_archive")
  else
    stage08_archive_sha=$(ssh "${stage08_common[@]}" -p "$stage08_old_port" "$stage08_old" \
      "/usr/bin/python3 - pack --archive '$stage08_remote/setup-evidence.tar.gz'" \
      < scripts/migrate_stage08_setup.py)
    scp "${stage08_common[@]}" -P "$stage08_old_port" \
      "$stage08_old:$stage08_remote/setup-evidence.tar.gz" "$stage08_archive"
  fi
  [[ "$stage08_archive_sha" =~ ^[0-9a-f]{64}$ ]]
  printf '%s  %s\n' "$stage08_archive_sha" "$stage08_archive" | sha256sum -c -
  python3 scripts/migrate_stage08_setup.py verify --archive "$stage08_archive" \
    --sha256 "$stage08_archive_sha" --output "$stage08_evidence/verified-old-setup"
  if [ "$stage08_mode" = --old-unavailable ]; then
    printf 'OldInstanceEvidence=UNAVAILABLE\nStage08BudgetClock=LOCAL_ORIGINAL_RETAINED\n'
  else
    printf 'OldInstanceEvidence=RETURNED_AND_VERIFIED\n'
    printf 'Reminder=旧实例 %s:%s 的必要安装证据已回传校验；没有其他任务时可以关机。请保留新实例 %s:%s。\n' \
      "$stage08_old_host" "$stage08_old_port" "$stage08_host" "$stage08_port"
  fi
fi

git bundle create "$stage08_evidence/clone-delta.bundle" refs/heads/double-balance \
  refs/tags/stage-06-complete refs/tags/stage-07-complete "^$stage08_base"
stage08_delta_sha=$(sha256sum "$stage08_evidence/clone-delta.bundle" | cut -d ' ' -f1)
ssh "${stage08_common[@]}" -p "$stage08_port" "$stage08_target" "mkdir -p '$stage08_remote'"
scp "${stage08_common[@]}" -P "$stage08_port" "$stage08_evidence/clone-delta.bundle" \
  "$stage08_target:$stage08_remote/clone-delta.bundle"
if [ "$stage08_mode" != --retry-capacity ]; then
  scp "${stage08_common[@]}" -P "$stage08_port" "$stage08_archive" \
    "$stage08_target:$stage08_remote/setup-evidence.tar.gz"
fi
ssh "${stage08_common[@]}" -p "$stage08_port" "$stage08_target" \
  "bash -s -- '$stage08_id' '$stage08_head' '$stage08_delta_sha' '$stage08_archive_sha' '$stage08_mode'" <<'REMOTE'
set -Eeuo pipefail
stage08_id=$1 stage08_head=$2 stage08_delta_sha=$3 stage08_archive_sha=$4
stage08_mode=$5
stage08_root=/root/autodl-tmp/microduck-double-balance
stage08_artifacts="$stage08_root/artifacts/double-balance-stage08"
stage08_incoming="$stage08_root/incoming/$stage08_id"
cd "$stage08_root/workspace"
test "$(git branch --show-current)" = double-balance
test -z "$(git status --porcelain)"
test -x .venv/bin/python
command -v tmux >/dev/null
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
mkdir -p "$stage08_artifacts/setup"
exec 8>"$stage08_artifacts/budget.job.lock"
exec 9>"$stage08_artifacts/setup/install.lock"
flock -n 8
flock -n 9
if tmux has-session -t '=microduck-stage08-clone' 2>/dev/null; then
  printf 'Clone validation is already running.\n' >&2; exit 1
fi
printf '%s  %s\n' "$stage08_delta_sha" "$stage08_incoming/clone-delta.bundle" | sha256sum -c -
git bundle verify "$stage08_incoming/clone-delta.bundle"
git fetch "$stage08_incoming/clone-delta.bundle" \
  refs/heads/double-balance:refs/remotes/stage08/clone \
  refs/tags/stage-06-complete:refs/tags/stage-06-complete \
  refs/tags/stage-07-complete:refs/tags/stage-07-complete
git merge --ff-only refs/remotes/stage08/clone
test "$(git rev-parse HEAD)" = "$stage08_head"
test -z "$(git status --porcelain)"
if [ "$stage08_mode" = --retry-capacity ]; then
  test -f "$stage08_artifacts/budget.json"
  printf 'Stage08Budget=EXISTING_CLOUD_LEDGER_REUSED\n'
else
  .venv/bin/python scripts/migrate_stage08_setup.py adopt \
    --archive "$stage08_incoming/setup-evidence.tar.gz" --sha256 "$stage08_archive_sha" \
    --output "$stage08_artifacts/migrations/$stage08_id"
fi
flock -u 8
flock -u 9
bash scripts/validate_stage08_clone.sh "$stage08_head"
REMOTE
printf 'Stage08Clone=DELTA_TRANSFERRED\nFormalTrainingStarted=False\n'
