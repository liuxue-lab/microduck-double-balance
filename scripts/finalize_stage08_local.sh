#!/usr/bin/env bash
# Run on the laptop after importing the final bundle. No network or GPU work.
set -Eeuo pipefail
cd /home/lx/microduck-double-balance/workspace
test "$(git branch --show-current)" = double-balance
test -z "$(git status --porcelain)"
test "$(git remote get-url origin)" = git@github.com:liuxue-lab/microduck-double-balance.git
test "$(git rev-parse 'stage-07-complete^{}')" = 1fe5f54dc467205b846a807f2bd494c46f3e2c40
git merge-base --is-ancestor stage-07-complete HEAD
git merge-base --is-ancestor 61060f0684aabb460727a0d226b8d17225ee7954 HEAD
test -z "$(git diff --name-only stage-07-complete -- src/mjlab_microduck/tasks src/mjlab_microduck/robot src/mjlab_microduck/actuator pyproject.toml uv.lock src/mjlab_microduck/double_balance_training.py src/mjlab_microduck/double_balance_stage07.py src/mjlab_microduck/double_balance_smoke.py)"

if git show-ref --verify --quiet refs/tags/stage-08-complete; then
  test "$(git cat-file -t stage-08-complete)" = tag
  test "$(git rev-parse 'stage-08-complete^{}')" = "$(git rev-parse HEAD)"
fi

python3 scripts/verify_stage08_local_archive.py

if ! git show-ref --verify --quiet refs/tags/stage-08-complete; then
  git tag -a stage-08-complete -F docs/audits/stage-08-tag-message.txt
fi
printf 'Stage08TechnicalWork=COMPLETE\n'
printf 'Stage08Policy=STRICT_SUCCESSES_OBSERVED_WITH_RESIDUAL_FAILURES\n'
printf 'Stage08HeadPosture=OBSERVED_UNRESOLVED\n'
printf 'Stage08LocalTag=PASS\n'
printf 'LocalCommit=%s\n' "$(git rev-parse HEAD)"
printf 'Next: git push origin double-balance\n'
printf 'Then: git push origin stage-08-complete\n'
