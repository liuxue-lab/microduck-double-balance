#!/usr/bin/env bash
# Run inside AutoDL. Installation survives SSH disconnects; never starts PPO.
set -Eeuo pipefail

stage07_expected_gpu=${1:?Usage: bash scripts/setup_stage07_autodl.sh A800}
case "$stage07_expected_gpu" in
  A800|5090) ;;
  *) printf 'Unsupported target GPU: %s\n' "$stage07_expected_gpu" >&2; exit 2 ;;
esac
stage07_root=/root/autodl-tmp/microduck-double-balance
stage07_logs="$stage07_root/artifacts/stage07-setup"
stage07_session=microduck-stage07-setup
stage07_script=$(readlink -f "$0")
stage06_commit=e3f26e3cbf633c70c75d4d073dbeb8c47c4f5bd1
test -d /root/autodl-tmp
test "$(uname -m)" = x86_64
mkdir -p "$stage07_logs"
export DEBIAN_FRONTEND=noninteractive

if [ "${2:-}" != --worker ]; then
  if ! command -v tmux >/dev/null; then
    apt-get -o DPkg::Lock::Timeout=120 update -qq
    apt-get -o DPkg::Lock::Timeout=120 install -y -qq --no-install-recommends tmux
  fi
  if tmux has-session -t "=$stage07_session" 2>/dev/null; then
    printf 'Stage07SetupSession=ALREADY_RUNNING\n'
  else
    printf -v stage07_worker 'bash %q %q --worker' "$stage07_script" "$stage07_expected_gpu"
    tmux new-session -d -s "$stage07_session" "$stage07_worker"
    printf 'Stage07SetupSession=STARTED\n'
  fi
  printf 'Log=%s/latest.log\nStatus=%s/status.txt\n' "$stage07_logs" "$stage07_logs"
  exit 0
fi

exec 9>"$stage07_logs/install.lock"
flock -n 9 || { printf 'Another Stage 07 installation holds the lock.\n' >&2; exit 1; }
stage07_log=$(mktemp "$stage07_logs/install-XXXXXX.log")
ln -sfn "$stage07_log" "$stage07_logs/latest.log"
exec > >(tee -a "$stage07_log") 2>&1
stage07_step=starting
stage07_finish() {
  local stage07_exit=$?
  trap - EXIT
  if [ "$stage07_exit" -eq 0 ]; then
    printf 'PASS\n' > "$stage07_logs/status.txt"
    printf 'Stage07CloudSetup=PASS\n'
  else
    printf 'FAILED step=%s exit=%s\n' "$stage07_step" "$stage07_exit" > "$stage07_logs/status.txt"
    printf 'Stage07CloudSetup=FAILED step=%s exit=%s\n' "$stage07_step" "$stage07_exit"
  fi
  printf 'Stage07CloudSetupEnd=\n'
  exit "$stage07_exit"
}
trap stage07_finish EXIT
printf 'RUNNING\n' > "$stage07_logs/status.txt"
printf 'Stage07CloudSetupBegin=\n'
date -Is
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

export STAGE07_EXPECTED_GPU="$stage07_expected_gpu"
export GIT_TERMINAL_PROMPT=0
mkdir -p "$stage07_root"/{tools,cache/uv,cache/xdg,cache/python,cache/tmp}
export UV_CACHE_DIR="$stage07_root/cache/uv"
export UV_PYTHON_INSTALL_DIR="$stage07_root/cache/python"
export XDG_CACHE_HOME="$stage07_root/cache/xdg"
export TMPDIR="$stage07_root/cache/tmp"
export UV_LINK_MODE=copy

stage07_step=system_dependencies
apt-get -o DPkg::Lock::Timeout=120 update -qq
apt-get -o DPkg::Lock::Timeout=120 install -y -qq --no-install-recommends \
  git ca-certificates curl xz-utils python3 python3-venv libegl1 libgl1 libglfw3 ffmpeg

# Noninteractive SSH need not initialize Conda or provide a `python` command.
# Bootstrap uv from the distribution interpreter, then select Python 3.12 below.
stage07_step=uv_bootstrap
if [ ! -x "$stage07_root/tools/bootstrap/bin/python" ]; then
  /usr/bin/python3 -m venv "$stage07_root/tools/bootstrap"
fi
"$stage07_root/tools/bootstrap/bin/python" -m pip install --no-cache-dir 'uv==0.12.15'
stage07_uv="$stage07_root/tools/bootstrap/bin/uv"

# Provider-supported access to GitHub/BAM and Python distribution downloads.
if [ -r /etc/network_turbo ]; then
  stage07_step=provider_network_helper
  set +u
  source /etc/network_turbo >/dev/null
  set -u
  stage07_no_proxy="${no_proxy:-${NO_PROXY:-}}"
  export no_proxy="${stage07_no_proxy:+$stage07_no_proxy,}pypi.org,files.pythonhosted.org,mirrors.aliyun.com,pypi.tuna.tsinghua.edu.cn"
  export NO_PROXY="$no_proxy"
fi

stage07_step=repository_clone
if [ ! -d "$stage07_root/workspace/.git" ]; then
  if [ -e "$stage07_root/workspace" ]; then
    printf 'Existing incomplete workspace retained; inspect it before retrying.\n' >&2
    exit 1
  fi
  timeout 600 git clone --single-branch --branch double-balance \
    https://github.com/liuxue-lab/microduck-double-balance.git \
    "$stage07_root/workspace"
fi
cd "$stage07_root/workspace"
stage07_step=source_baseline
test "$(git rev-parse HEAD)" = "$stage06_commit"
test "$(git rev-parse 'stage-06-complete^{}')" = "$stage06_commit"
test -z "$(git status --porcelain)"
printf 'SourceBaseline=PASS commit=%s\n' "$stage06_commit"

stage07_step=python_312
stage07_python=
for stage07_candidate in /root/miniconda3/bin/python /root/miniforge3/bin/python \
  /usr/local/bin/python3.12 /usr/bin/python3.12; do
  if [ -x "$stage07_candidate" ] && "$stage07_candidate" -c \
    'import sys; sys.exit(sys.version_info[:2] != (3, 12))'; then
    stage07_python="$stage07_candidate"
    break
  fi
done
if [ -z "$stage07_python" ]; then
  "$stage07_uv" python install 3.12.14 --no-progress
  stage07_python=3.12.14
fi
printf 'SelectedPython=%s\n' "$stage07_python"
stage07_step=locked_environment
"$stage07_uv" sync --locked --python "$stage07_python" --no-progress

stage07_step=environment_validation
.venv/bin/python - <<'PY'
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import torch

expected = {'mjlab': '1.3.0', 'mujoco': '3.10.0', 'mujoco-warp': '3.8.1',
            'warp-lang': '1.12.0', 'rsl-rl-lib': '5.0.1'}
actual = {name: version(name) for name in expected}
assert actual == expected, actual
assert torch.__version__.split('+')[0] == '2.9.1', torch.__version__
assert torch.version.cuda == '12.8', torch.version.cuda
assert sys.version_info[:2] == (3, 12), sys.version
assert torch.cuda.is_available(), 'CUDA unavailable'
assert torch.cuda.device_count() == 1, 'Expected one allocated GPU'
assert os.environ['STAGE07_EXPECTED_GPU'] in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
x = torch.ones((64, 64), device='cuda')
assert torch.equal(x @ x, torch.full_like(x, 64)), 'CUDA matrix check failed'
torch.cuda.synchronize()
import mjlab_microduck.tasks  # noqa: E402; verifies the registered task imports

quota_paths = ['memory.max', 'cpu.max', 'cpuset.cpus.effective',
               'memory/memory.limit_in_bytes', 'cpu/cpu.cfs_quota_us',
               'cpu/cpu.cfs_period_us']
quotas = {p: Path('/sys/fs/cgroup', p).read_text().strip()
          for p in quota_paths if Path('/sys/fs/cgroup', p).is_file()}
report = {'status': 'PASS', 'created_utc': datetime.now(timezone.utc).isoformat(),
          'hostname': socket.gethostname(),
          'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
          'python': sys.version, 'versions': actual, 'torch': torch.__version__,
          'torch_cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
          'expected_gpu': os.environ['STAGE07_EXPECTED_GPU'],
          'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
          'cuda_architectures': torch.cuda.get_arch_list(), 'cgroup_limits': quotas,
          'cpu_affinity_count': len(os.sched_getaffinity(0)),
          'cuda_matrix_check': 'PASS', 'registered_task_import': 'PASS',
          'capacity_4096': 'NOT_MEASURED', 'formal_training_started': False}
out = Path('../artifacts/stage07-setup/environment.json')
out.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
print('EnvironmentReport=' + str(out.resolve()))
PY
stage07_step=completed
