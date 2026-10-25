#!/usr/bin/env bash
# Shared shell-only bootstrap. Discovery never activates Conda or imports CUDA.
stage08_resolve_python() {
  local candidate resolved
  for candidate in python3 python /usr/bin/python3 /usr/local/bin/python3 \
    /root/miniconda3/bin/python3 /root/miniconda3/bin/python \
    /opt/conda/bin/python3 /opt/conda/bin/python; do
    command -v "$candidate" >/dev/null 2>&1 || continue
    if resolved=$("$candidate" -I -c \
      'import sys; assert sys.version_info >= (3, 10); print(sys.executable)' \
      </dev/null 2>/dev/null); then
      [[ "$resolved" =~ ^/[A-Za-z0-9_./+-]+$ ]] || continue
      printf '%s\n' "$resolved"
      return 0
    fi
  done
  printf 'Stage08Bootstrap=NO_USABLE_PYTHON\nNeed Python >=3.10 in PATH or a supported system/Conda location.\nPATH=%s\n' "$PATH" >&2
  return 127
}

stage08_deployment_start() {
  local record=$1 requested=${2:-} started
  if [[ ! -s "$record" ]]; then
    started=${requested:-$(date -Iseconds)}
    (umask 077; set -o noclobber; printf '%s\n' "$started" > "$record") || return
  fi
  IFS= read -r started < "$record" || return
  test -n "$started" || return
  printf '%s\n' "$started"
}

if [[ ${1:-} == --resolve-python ]]; then
  stage08_resolve_python
fi
