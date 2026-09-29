#!/usr/bin/env bash
# Laptop-only recovery for an already transferred Stage 08 deployment.
# Send only the code delta; preserve checkpoints, cache and budget. Never run PPO.
set -Eeuo pipefail
stage08_host=${1:?Usage: recover_stage08_setup_local.sh HOST PORT}
stage08_port=${2:?Port required}
[[ "$stage08_host" =~ ^[A-Za-z0-9.-]+$ ]]
[[ "$stage08_port" =~ ^[0-9]+$ ]] && ((stage08_port>=1 && stage08_port<=65535))
cd /home/lx/microduck-double-balance/workspace
test "$(git branch --show-current)" = double-balance
test -z "$(git status --porcelain)"
stage08_base=d78d1e6e9fd334d85bc8de0082a29007220a1b39
git merge-base --is-ancestor "$stage08_base" HEAD
stage08_head=$(git rev-parse HEAD)
test "$stage08_head" != "$stage08_base"
stage08_temp=$(mktemp -d)
trap 'rm -rf -- "$stage08_temp"' EXIT
stage08_control="$HOME/.ssh/microduck-stage08-%C"
stage08_connection=(-o ControlMaster=auto -o ControlPersist=12h
  -o "ControlPath=$stage08_control" -o ServerAliveInterval=30
  -o ServerAliveCountMax=3 -o ConnectTimeout=15)
stage08_target="root@$stage08_host"
stage08_root=/root/autodl-tmp/microduck-double-balance
git bundle create "$stage08_temp/recovery.bundle" refs/heads/double-balance "^$stage08_base"
stage08_sha=$(sha256sum "$stage08_temp/recovery.bundle" | cut -d ' ' -f1)
ssh "${stage08_connection[@]}" -p "$stage08_port" "$stage08_target" \
  "test -d '$stage08_root/artifacts/double-balance-stage08/setup'"
scp "${stage08_connection[@]}" -P "$stage08_port" "$stage08_temp/recovery.bundle" \
  "$stage08_target:$stage08_root/artifacts/double-balance-stage08/setup/recovery.bundle"
ssh "${stage08_connection[@]}" -p "$stage08_port" "$stage08_target" \
  "bash -s -- '$stage08_head' '$stage08_sha' '$stage08_base'" <<'REMOTE'
set -Eeuo pipefail
stage08_head=$1
stage08_sha=$2
stage08_base=$3
stage08_root=/root/autodl-tmp/microduck-double-balance
stage08_repo="$stage08_root/workspace"
stage08_artifacts="$stage08_root/artifacts/double-balance-stage08"
stage08_bundle="$stage08_artifacts/setup/recovery.bundle"
cd "$stage08_repo"
test "$(git branch --show-current)" = double-balance
test -z "$(git status --porcelain)"
test "$(sha256sum "$stage08_bundle" | cut -d ' ' -f1)" = "$stage08_sha"
git merge-base --is-ancestor "$stage08_base" HEAD
test -f "$stage08_artifacts/budget.json"
test -x .venv/bin/python
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
# Hold the campaign lock throughout recovery; never interrupt an active job.
exec 8>"$stage08_artifacts/budget.job.lock"
flock -n 8
git bundle verify "$stage08_bundle"
git fetch "$stage08_bundle" refs/heads/double-balance:refs/remotes/stage08/recovery
test "$(git rev-parse refs/remotes/stage08/recovery)" = "$stage08_head"
git merge-base --is-ancestor HEAD "$stage08_head"
.venv/bin/python - <<'PY'
import os
from pathlib import Path
import signal

repo = Path('/root/autodl-tmp/microduck-double-balance/workspace')
bootstrap = repo.parent / 'tools/bootstrap'
worker = os.fsencode(repo / 'scripts/setup_stage08_autodl.sh')
# /proc may be mounted from an outer PID namespace. Signal the PID visible to us.
own_status = Path('/proc/self/status').read_text().splitlines()
namespace_depth = len(next(line.split()[1:] for line in own_status if line.startswith('NSpid:')))

def is_installer(args):
    if len(args) < 2 or args[1] != b'sync':
        return False
    executable = Path(os.fsdecode(args[0]))
    return executable.name == 'uv' and executable.resolve().is_relative_to(bootstrap)

matches = []
for p in Path('/proc').glob('[0-9]*'):
    try:
        args = (p / 'cmdline').read_bytes().split(b'\0')
        if not is_installer(args) or (p / 'cwd').resolve() != repo:
            continue
        status = (p / 'status').read_text().splitlines()
        parent = next(line.split()[1] for line in status if line.startswith('PPid:'))
        parent_args = (Path('/proc') / parent / 'cmdline').read_bytes().split(b'\0')
        if worker not in parent_args or b'--worker' not in parent_args:
            raise SystemExit('Unrecognized installer parent; no process was stopped')
        namespace_pids = next(line.split()[1:] for line in status if line.startswith('NSpid:'))
        if len(namespace_pids) < namespace_depth:
            raise SystemExit('Installer is outside this PID namespace; no process was stopped')
        matches.append((int(p.name), int(namespace_pids[namespace_depth - 1])))
    except (FileNotFoundError, ProcessLookupError):
        continue
if len(matches) > 1:
    raise SystemExit('Multiple setup installers found; no process was stopped')
for proc_pid, pid in matches:
    try:
        handle = os.pidfd_open(pid)
        try:
            args = Path(f'/proc/{proc_pid}/cmdline').read_bytes().split(b'\0')
            if not is_installer(args):
                raise SystemExit('Installer changed; no process was stopped')
            signal.pidfd_send_signal(handle, signal.SIGTERM)
        finally:
            os.close(handle)
        print(f'Stage08PreviousInstaller=STOP_REQUESTED PID={pid}')
    except (FileNotFoundError, ProcessLookupError):
        print('Stage08PreviousInstaller=ALREADY_EXITED')
if not matches:
    print('Stage08PreviousInstaller=NOT_DOWNLOADING; waiting for setup lock')
PY
# A signal to uv makes the old set -e worker exit. Wait for its log/lock to close.
exec 9>"$stage08_artifacts/setup/install.lock"
if ! flock -w 45 9; then
  printf 'Setup is still active; recovery stopped without changing the checkout.\n' >&2
  exit 1
fi
for stage08_attempt in {1..20}; do
  if ! tmux has-session -t '=microduck-stage08-setup' 2>/dev/null; then break; fi
  sleep 0.25
done
if tmux has-session -t '=microduck-stage08-setup' 2>/dev/null; then
  printf 'Old setup session is still active; inspect it before retrying.\n' >&2
  exit 1
fi
git merge --ff-only refs/remotes/stage08/recovery
test -z "$(git status --porcelain)"
stage08_started=$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["started_at"])' "$stage08_artifacts/budget.json")
stage08_gpu=$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["gpu"])' "$stage08_artifacts/budget.json")
flock -u 9
exec 9>&-
flock -u 8
exec 8>&-
bash scripts/setup_stage08_autodl.sh "$stage08_gpu" "$stage08_head" "$stage08_started"
printf 'Stage08Recovery=STARTED\nFormalTrainingStarted=False\n'
REMOTE
