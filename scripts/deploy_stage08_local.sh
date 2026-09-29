#!/usr/bin/env bash
# Run on the Ubuntu laptop. Transfers committed code and the two known models.
# No GitHub push, Stage 06 smoke or formal training is performed.
set -Eeuo pipefail
stage08_host=${1:?Usage: deploy_stage08_local.sh HOST PORT GPU POWER_ON_ISO}
stage08_port=${2:?Port required}
stage08_gpu=${3:?GPU required}
stage08_started=${4:?Actual power-on time with timezone required}
[[ "$stage08_host" =~ ^[A-Za-z0-9.-]+$ ]]
[[ "$stage08_port" =~ ^[0-9]+$ ]] && ((stage08_port>=1 && stage08_port<=65535))
case "$stage08_gpu" in A800|5090) ;; *) exit 2;; esac
stage08_repo=/home/lx/microduck-double-balance/workspace
stage08_root=/root/autodl-tmp/microduck-double-balance
cd "$stage08_repo"
test "$(git branch --show-current)" = double-balance
test -z "$(git status --porcelain)"
git merge-base --is-ancestor 1fe5f54dc467205b846a807f2bd494c46f3e2c40 HEAD
stage08_head=$(git rev-parse HEAD)
stage08_temp=$(mktemp -d)
chmod 700 "$stage08_temp"
stage08_ssh=(-p "$stage08_port" -o ControlMaster=auto -o ControlPersist=600 -o "ControlPath=$stage08_temp/ssh")
stage08_scp=(-P "$stage08_port" -o ControlMaster=auto -o ControlPersist=600 -o "ControlPath=$stage08_temp/ssh")
stage08_target="root@$stage08_host"
stage08_cleanup() {
  ssh "${stage08_ssh[@]}" -O exit "$stage08_target" >/dev/null 2>&1 || true
  rm -rf -- "$stage08_temp"
}
trap stage08_cleanup EXIT
stage08_local_artifacts=/home/lx/microduck-double-balance/artifacts/double-balance-stage08
mkdir -p "$stage08_local_artifacts"
stage08_preflight="$stage08_local_artifacts/preflight-$(date +%Y%m%dT%H%M%S).json"
printf 'PreflightReport=%s\n' "$stage08_preflight"
ssh "${stage08_ssh[@]}" "$stage08_target" "python3 - --gpu '$stage08_gpu'" \
  < scripts/preflight_stage08_cloud.py > "$stage08_preflight"
git bundle create "$stage08_temp/source.bundle" refs/heads/double-balance \
  refs/tags/stage-06-complete refs/tags/stage-07-complete
ssh "${stage08_ssh[@]}" "$stage08_target" 'test -d /root/autodl-tmp && mkdir -p /root/autodl-tmp/microduck-double-balance/incoming/stage08'
scp "${stage08_scp[@]}" "$stage08_temp/source.bundle" "$stage08_target:$stage08_root/incoming/stage08/source.bundle"
ssh "${stage08_ssh[@]}" "$stage08_target" 'bash -s' <<'REMOTE'
set -Eeuo pipefail
stage08_root=/root/autodl-tmp/microduck-double-balance
stage08_repo="$stage08_root/workspace"
stage08_bundle="$stage08_root/incoming/stage08/source.bundle"
stage08_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [ -n "$stage08_pids" ]; then
  printf 'GPU has active compute processes; inspect them before deployment.\n' >&2
  exit 1
fi
if [ ! -d "$stage08_repo/.git" ]; then
  test ! -e "$stage08_repo"
  git clone --branch double-balance "$stage08_bundle" "$stage08_repo"
else
  cd "$stage08_repo"
  test -z "$(git status --porcelain)"
  git fetch "$stage08_bundle" refs/heads/double-balance:refs/remotes/stage08/deploy \
    refs/tags/stage-06-complete:refs/tags/stage-06-complete \
    refs/tags/stage-07-complete:refs/tags/stage-07-complete
  git switch double-balance
  git merge --ff-only refs/remotes/stage08/deploy
fi
mkdir -p "$stage08_root/artifacts/double-balance-stage08/references"
python3 - <<'PY'
import hashlib,shutil
from pathlib import Path
root=Path('/root/autodl-tmp/microduck-double-balance/artifacts')
known={'001000':'c667f96607b68383047f23956ba58805920434465245ef8e32d1148b17fc65b7',
       '006000':'a5ad0aedb500555b649c5d4b11aded833cbc0f932c1fff1a747e1492534c495c'}
for update,digest in known.items():
    name='update_'+update+'.pt'; destination=root/'double-balance-stage08/references'/name
    if destination.exists(): continue
    for source in (root/'double-balance-stage07').rglob(name):
        if hashlib.sha256(source.read_bytes()).hexdigest()==digest:
            shutil.copyfile(source,destination); break
PY
REMOTE
for stage08_update in 001000 006000; do
  stage08_checkpoint=$(python3 - "$stage08_update" <<'PY'
import hashlib,sys
from pathlib import Path
expected={'001000':'c667f96607b68383047f23956ba58805920434465245ef8e32d1148b17fc65b7',
          '006000':'a5ad0aedb500555b649c5d4b11aded833cbc0f932c1fff1a747e1492534c495c'}
root=Path('/home/lx/microduck-double-balance/artifacts/double-balance-stage07')
for path in sorted(root.rglob('update_'+sys.argv[1]+'.pt')):
    h=hashlib.sha256(path.read_bytes()).hexdigest()
    if h==expected[sys.argv[1]]:
        print(path); break
else: raise SystemExit('The archived Stage 07 checkpoint is missing or has a different hash')
PY
)
  stage08_destination="$stage08_root/artifacts/double-balance-stage08/references/update_$stage08_update.pt"
  stage08_local_sha=$(sha256sum "$stage08_checkpoint" | cut -d ' ' -f1)
  stage08_remote_sha=$(ssh "${stage08_ssh[@]}" "$stage08_target" "if test -f '$stage08_destination'; then sha256sum '$stage08_destination'; fi" | cut -d ' ' -f1)
  if [ -n "$stage08_remote_sha" ]; then
    test "$stage08_remote_sha" = "$stage08_local_sha"
  else
    scp "${stage08_scp[@]}" "$stage08_checkpoint" "$stage08_target:$stage08_destination"
    stage08_remote_sha=$(ssh "${stage08_ssh[@]}" "$stage08_target" "sha256sum '$stage08_destination'" | cut -d ' ' -f1)
    test "$stage08_remote_sha" = "$stage08_local_sha"
  fi
done
printf -v stage08_command 'bash %q %q %q %q' "$stage08_root/workspace/scripts/setup_stage08_autodl.sh" "$stage08_gpu" "$stage08_head" "$stage08_started"
ssh "${stage08_ssh[@]}" "$stage08_target" "$stage08_command"
printf 'Stage08Deployment=TRANSFERRED\nFormalTrainingStarted=False\n'
