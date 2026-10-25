#!/usr/bin/env bash
# Run on the laptop after the final bundle is imported. No network or training.
set -Eeuo pipefail
cd /home/lx/microduck-double-balance/workspace
test "$(git branch --show-current)" = double-balance
test -z "$(git status --porcelain)"
test "$(git remote get-url origin)" = git@github.com:liuxue-lab/microduck-double-balance.git
test "$(git rev-parse 'stage-06-complete^{}')" = e3f26e3cbf633c70c75d4d073dbeb8c47c4f5bd1
git merge-base --is-ancestor stage-06-complete HEAD
test -z "$(git diff --name-only stage-06-complete -- src/mjlab_microduck/tasks src/mjlab_microduck/robot src/mjlab_microduck/actuator pyproject.toml uv.lock)"
python3 scripts/verify_stage07_local_archive.py

if git show-ref --verify --quiet refs/tags/stage-07-complete; then
  test "$(git cat-file -t stage-07-complete)" = tag
  test "$(git rev-parse 'stage-07-complete^{}')" = "$(git rev-parse HEAD)"
else
  git tag -a stage-07-complete -F docs/audits/stage-07-tag-message.txt
fi
printf 'Stage07LocalTag=PASS\n'
printf 'LocalCommit=%s\n' "$(git rev-parse HEAD)"
printf 'Next: git push origin double-balance\n'
printf 'Then: git push origin stage-07-complete\n'
