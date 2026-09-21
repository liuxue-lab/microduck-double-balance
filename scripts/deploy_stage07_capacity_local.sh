#!/usr/bin/env bash
# Run on lxlab after importing the Stage 07 bundle. One SSH authentication.
set -Eeuo pipefail
stage07_repo=/home/lx/microduck-double-balance/workspace
stage07_downloads=/home/lx/下载
stage07_bundle=microduck-stage-07-from-stage-06.bundle
stage07_source=/home/lx/microduck-double-balance/artifacts/double-balance-stage05/checkpoint.pt
stage07_hash=548758afc546e11d8f3f446a4bcc846283a669ff20f4048da7a3a7bb1332b98e
stage07_port=${1:-26497}
[[ "$stage07_port" =~ ^[0-9]+$ ]] && [ "$stage07_port" -ge 1 ] && [ "$stage07_port" -le 65535 ]
cd "$stage07_repo"
test "$(git branch --show-current)" = double-balance
test -z "$(git status --porcelain)"
test -f "$stage07_downloads/$stage07_bundle"
test -f "$stage07_source"
stage07_actual=$(sha256sum "$stage07_source")
test "${stage07_actual%% *}" = "$stage07_hash"
stage07_bundle_head=$(git bundle list-heads "$stage07_downloads/$stage07_bundle" refs/heads/double-balance)
test "${stage07_bundle_head%% *}" = "$(git rev-parse HEAD)"

stage07_remote=$(cat <<'REMOTE'
set -Eeuo pipefail
stage07_root=/root/autodl-tmp/microduck-double-balance
# Reuse the cloned environment. A missing data disk must not trigger downloads.
if [ ! -d "$stage07_root/workspace/.git" ] || [ ! -x "$stage07_root/workspace/.venv/bin/python" ]; then
  printf 'Stage07ClonedEnvironment=MISSING\nExpectedProject=%s/workspace\n' "$stage07_root" >&2
  printf 'Copy the previous A800 data disk into this instance, including workspace/.venv and cache. No installation was started.\n' >&2
  exit 42
fi
cd "$stage07_root/workspace"
.venv/bin/python - <<'PY'
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import torch

expected = {'mjlab': '1.3.0', 'mujoco': '3.10.0', 'mujoco-warp': '3.8.1',
            'warp-lang': '1.12.0', 'rsl-rl-lib': '5.0.1'}
actual = {name: version(name) for name in expected}
assert actual == expected, actual
assert sys.version_info[:2] == (3, 12), sys.version
assert torch.__version__.split('+')[0] == '2.9.1', torch.__version__
assert torch.version.cuda == '12.8', torch.version.cuda
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert 'A800' in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
assert torch.cuda.get_device_properties(0).total_memory >= 75 * 1024**3, 'Expected full A800 80GB allocation'
import mjlab_microduck.tasks

report = {'status': 'PASS', 'hostname': socket.gethostname(),
          'created_utc': datetime.now(timezone.utc).isoformat(),
          'source': 'current cloned A800: direct version and CUDA inspection; no packages installed',
          'python': sys.version, 'versions': actual, 'torch': torch.__version__,
          'torch_cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
          'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
          'nvidia_smi': subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                                '--format=csv,noheader'], text=True).strip(),
          'cpu_affinity_count': len(os.sched_getaffinity(0)),
          'data_disk_free_bytes': shutil.disk_usage('/root/autodl-tmp').free,
          'cgroup_limits': {name: Path('/sys/fs/cgroup', name).read_text().strip()
                            for name in ('cpu.max', 'memory.max', 'cpuset.cpus.effective',
                                         'cpu/cpu.cfs_quota_us', 'cpu/cpu.cfs_period_us',
                                         'memory/memory.limit_in_bytes')
                            if Path('/sys/fs/cgroup', name).is_file()},
          'formal_training_started': False}
out = Path('../artifacts/stage07-setup')
out.mkdir(parents=True, exist_ok=True)
path = out / ('clone-environment-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
print('Stage07ClonedEnvironment=PASS', flush=True)
PY
stage07_incoming=$(mktemp -d /root/autodl-tmp/stage07-upload.XXXXXX)
tar --no-same-owner --no-same-permissions -xf - -C "$stage07_incoming"
stage07_hash=548758afc546e11d8f3f446a4bcc846283a669ff20f4048da7a3a7bb1332b98e
stage07_actual=$(sha256sum "$stage07_incoming/checkpoint.pt")
test "${stage07_actual%% *}" = "$stage07_hash"
cd "$stage07_root/workspace"
test "$(git branch --show-current)" = double-balance
test -z "$(git status --porcelain)"
git bundle verify "$stage07_incoming/microduck-stage-07-from-stage-06.bundle"
git fetch "$stage07_incoming/microduck-stage-07-from-stage-06.bundle" \
  refs/heads/double-balance:refs/remotes/stage07/double-balance
git merge --ff-only refs/remotes/stage07/double-balance
mkdir -p "$stage07_root/artifacts/double-balance-stage05"
stage07_target="$stage07_root/artifacts/double-balance-stage05/checkpoint.pt"
if [ ! -e "$stage07_target" ]; then
  cp -n "$stage07_incoming/checkpoint.pt" "$stage07_target"
fi
stage07_actual=$(sha256sum "$stage07_target")
test "${stage07_actual%% *}" = "$stage07_hash"
printf 'Stage05CloudCheckpoint=PASS\n'
bash scripts/launch_stage07_capacity.sh
REMOTE
)
tar -cf - -C "$stage07_downloads" "$stage07_bundle" \
  -C "$(dirname "$stage07_source")" checkpoint.pt \
  | ssh -T -p "$stage07_port" -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    root@connect.bjb1.seetacloud.com "$stage07_remote"
