"""Render commands with the actual committed transfer identity; never execute."""
import re
from integration.durable import require


def render(receipt):
    branch, commit = receipt['branch'], receipt['commit']
    require(re.fullmatch(r'stage12-local-completion-v5-transfer-[0-9]{8}t[0-9]{6}z-[0-9a-f]{6}',branch) and
            re.fullmatch(r'[0-9a-f]{40}',commit), 'Cloud transfer identity differs')
    return ('''#!/usr/bin/env bash
set -euo pipefail
# Run on the manually started, separately budgeted cloud machine only.
# Set these three variables to the explicitly agreed environment-check budget.
: "${STAGE12_GPU_HOURS:?Set the agreed cloud GPU hours first}"
: "${STAGE12_SPEND_LIMIT:?Set the agreed total spend ceiling first}"
: "${STAGE12_CURRENCY:?Set the agreed currency first}"
stage12_cloud_root="$HOME/microduck-stage12-COMMITSHORT"
test ! -e "$stage12_cloud_root"
mkdir "$stage12_cloud_root"
git clone --single-branch --branch BRANCH git@github.com:liuxue-lab/microduck-double-balance.git "$stage12_cloud_root/transfer"
test "$(git -C "$stage12_cloud_root/transfer" rev-parse HEAD)" = COMMIT
python3 -I -S "$stage12_cloud_root/transfer/cloud-stage12.py" --transfer "$stage12_cloud_root/transfer" --expected-commit COMMIT --output "$stage12_cloud_root/verified"
uv sync --locked --python 3.12.14 --project "$stage12_cloud_root/verified/inputs/project"
"$stage12_cloud_root/verified/inputs/project/.venv/bin/python" -I -B "$stage12_cloud_root/verified/program/cloud-stage12.py" --verify-environment "$stage12_cloud_root/verified" --gpu-hours "$STAGE12_GPU_HOURS" --spend-limit "$STAGE12_SPEND_LIMIT" --currency "$STAGE12_CURRENCY"
# Stop here. Native cloud acceptance/PPO/model construction are not authorized.
# Keep cloud-environment-report.json and started-environment-check.json.
# After all authorized cloud work and verified return, shut the GPU down manually.
'''.replace('COMMITSHORT',commit[:12]).replace('BRANCH',branch).replace('COMMIT',commit))
