import hashlib
import json
import os
from pathlib import Path

BASE = Path('/root/autodl-tmp/microduck-double-balance')
SLOT = BASE/'artifacts/double-balance-stage12/native-training-v2'
SOURCE = BASE/'artifacts/double-balance-stage12/finite-trial-v1/cloud-20261020-fp64-one-shot-v1/inputs/project'
PYTHON = BASE/'workspace/.venv/bin/python'

def require(ok, text):
    if not ok:
        raise RuntimeError(text)

def read(path):
    return json.loads(Path(path).read_bytes())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def write(path, value, exclusive=False):
    path = Path(path)
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n').encode()
    if exclusive:
        with path.open('xb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
    else:
        tmp = path.with_name(path.name+'.tmp-'+str(os.getpid()))
        with tmp.open('xb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)

def identity(pid):
    try:
        s = Path('/proc')/str(pid)/'stat'
        row = s.read_text().rsplit(')', 1)[1].split()
        return dict(pid=int(pid), start=row[19], state=row[0], group=int(row[2]))
    except FileNotFoundError:
        return None

def alive(pin):
    if not pin:
        return False
    current = identity(pin['pid'])
    return bool(current and current['start']==pin['start'] and current['state'] not in ('Z','X'))

def reserve(budget, now):
    require(now < budget['deadline_utc'], 'Campaign time budget exhausted')
    require(not budget['pending'], 'An earlier attempt remains uncommitted')
    require(budget['attempted_updates'] < budget['max_updates'], 'Campaign PPO budget exhausted')
    budget['attempted_updates'] += 1
    budget['pending'] = True

def commit(budget):
    require(budget['pending'], 'PPO completion without reservation')
    budget['completed_updates'] += 1
    budget['pending'] = False

def abandon_partial_for_explicit_resume(budget):
    # A spent attempt is never refunded; only its in-flight flag is closed.
    if budget['pending']:
        budget['interrupted_attempts'] += 1
        budget['pending'] = False

LEGACY_SLOT = BASE/'artifacts/double-balance-stage12/native-training-v1'
PRIMARY_SHA = '86d55c3703c18fcf4817db49c6e4dcf839b3f5195d68ae2c559db7e222bded97'
PRIMARY_REL = 'artifacts/double-balance-stage08/returned/5d998fc322345102/evidence-20260930T093645-8955/extension/E-20260929/segment-001/checkpoints/update_004000.pt'
MAX_CHECKPOINT_BYTES = 512*1024**2

def has_campaign(slot):
    return any((slot/n).exists() for n in ('started.json','budget.json')) or any(
        alive(read(slot/n)) for n in ('supervisor.json','child.json') if (slot/n).exists())

def validation_remaining(slot):
    if (slot/'smoke-started.json').exists():
        return 0  # One attempt only, including failed/interrupted attempts.
    return 1
