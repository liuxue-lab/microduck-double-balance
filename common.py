"""Stage12 cloud execution v1: standard-library identities, durable steps and transport."""
from pathlib import Path, PurePosixPath
import base64
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import time
import zipfile

ORIGIN = 'git@github.com:liuxue-lab/microduck-double-balance.git'
BASELINE = {'completed': 704664, 'attempted': 704665}
GPU_UUID = 'GPU-f231d519-f45e-6111-fc31-33dbb8b98f8f'
SOURCE_SHA = '24701df9e56face7980a1a2e0544dfebc41a4cf1b84ec4912be795d1a35a43b2'
INVENTORY_COMMIT = '530adc361e93d013c04d5a80176819f03150d1d9'
LOCAL_N2_COMMIT = 'd506be014253a2bb0439ec03b3282497a33f2e2b'
# One conservative, immutable campaign deadline: confirmation time + 24 hours.
# Includes installation, idle waiting and interruptions. Never reset on resume.
CAMPAIGN_DEADLINE = 1792209194.0  # 2026-10-17T03:53:14Z
RETURN_RESERVE_SECONDS = 1800
GIB = 1024**3
HOST_KEY = 'github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl\n'
HOST_FINGERPRINT = 'SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def encoded(x):
    return (json.dumps(x, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024**2), b''):
            h.update(b)
    return h.hexdigest()


def info(path):
    p = Path(path)
    return {'bytes': p.stat().st_size, 'sha256': sha(p)}


def read(path):
    return json.loads(Path(path).read_bytes())


def publish(path, raw):
    path = Path(path)
    require(path.parent.is_dir() and not any(p.is_symlink() for p in (path,*path.parents)), 'Unsafe output '+str(path))
    tmp = path.with_name(path.name+'.partial-'+secrets.token_hex(6))
    with tmp.open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    os.link(tmp,path)
    fd = os.open(path.parent,os.O_RDONLY); os.fsync(fd); os.close(fd)
    tmp.unlink()
    return path


def write(path, value):
    return publish(path, encoded(value))


def same_or_new(path, raw):
    path = Path(path)
    if path.exists():
        require(path.is_file() and path.read_bytes()==raw, 'Existing bytes differ; preserved: '+str(path))
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        publish(path,raw)


def command(argv, cwd=None, env=None, timeout=120):
    if env is None:
        env={k:v for k,v in os.environ.items() if not k.startswith('GIT_')}
        env.update(GIT_OPTIONAL_LOCKS='0',GIT_TERMINAL_PROMPT='0')
    r = subprocess.run([str(x) for x in argv],cwd=cwd,env=env,capture_output=True,text=True,timeout=timeout)
    require(r.returncode==0, 'Command failed '+repr([str(x) for x in argv])+': '+r.stderr[-6000:])
    return r.stdout.strip()


def safe_name(name):
    p = PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts and '\\' not in name and str(p)==name and name not in ('','.'), 'Unsafe archive path '+name)
    return name


def manifest(root, exclude=()):
    root = Path(root)
    rows = {}
    for p in sorted(root.rglob('*')):
        require(not p.is_symlink(), 'Symlink in package '+str(p))
        if p.is_file():
            n=p.relative_to(root).as_posix()
            if '.git' in p.relative_to(root).parts or '__pycache__' in p.parts or n in exclude:
                continue
            rows[n] = info(p)
    return rows


def verify(root, index='PACKAGE.json'):
    root = Path(root)
    expected = read(root/index)
    require(manifest(root,(index,))==expected, 'Package roster/hash differs')
    return sha(root/index)


def extract_source(archive, dest):
    require(sha(archive)==SOURCE_SHA, 'Frozen source archive hash differs')
    dest = Path(dest); dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        names=z.namelist(); require(len(names)==len(set(names)),'Duplicate ZIP member')
        m=json.loads(z.read('SOURCE-BUNDLE-MANIFEST.json'))
        require(len(m['files'])==410 and set(names)==set(m['files'])|{'SOURCE-BUNDLE-MANIFEST.json'},'Source roster differs')
        for n in names:
            safe_name(n); zi=z.getinfo(n)
            require((zi.external_attr>>16)&0o170000 != 0o120000,'ZIP symlink')
            b=z.read(n)
            if n in m['files']:
                require({'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}==m['files'][n],'Source bytes differ '+n)
            same_or_new(dest/n,b)
    return m


def native_env():
    e={k:v for k,v in os.environ.items() if not k.startswith(('GIT_','PYTHON','UV_'))}
    e.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             WANDB_MODE='disabled',WANDB_DISABLED='true',HF_HUB_OFFLINE='1',MUJOCO_GL='disable',MPLBACKEND='Agg')
    return e


def phase_state(folder, name):
    folder=Path(folder)
    if (folder/(name+'.done.json')).is_file():
        return 'DONE'
    if (folder/(name+'.started.json')).exists():
        return 'STARTED_NO_COMPLETE_RECEIPT'
    return 'NOT_STARTED'


def run_once(folder, name, action):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    state=phase_state(folder,name)
    if state=='DONE':
        return read(folder/(name+'.done.json'))['result']
    require(state=='NOT_STARTED',name+' already started without completion; no automatic retry')
    require(time.time()<CAMPAIGN_DEADLINE-RETURN_RESERVE_SECONDS,'Original 24-hour deadline reached; no budget reset')
    write(folder/(name+'.started.json'),dict(utc_epoch=time.time(),deadline=CAMPAIGN_DEADLINE,no_retry=True,PPO=0,cloud_training=0))
    try:
        result=action()
    except BaseException as exc:
        write(folder/(name+'.failed.json'),dict(error=repr(exc),utc_epoch=time.time(),no_retry=True))
        raise
    write(folder/(name+'.done.json'),dict(utc_epoch=time.time(),result=result))
    return result


def preparation_once(folder, name, action, retry=False):
    """Continue reversible installation only on explicit --resume.

    Old started/failed receipts are retained. Native phases never use this.
    The same immutable total deadline applies to every preparation attempt.
    """
    folder=Path(folder)
    names=[name]+[name+'-attempt-%02d'%i for i in range(2,21)]
    for current in names:
        state=phase_state(folder,current)
        if state=='DONE':return read(folder/(current+'.done.json'))['result']
        if state=='NOT_STARTED':
            return run_once(folder,current,lambda:action(current))
        require(retry,'Interrupted preparation retained; --resume may continue setup only: '+current)
    raise RuntimeError('Preparation attempt ceiling exhausted')


def prepare_git(folder, branch, identity, ssh_command=None):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    if not (folder/'.git').exists():
        command(['git','init','--initial-branch',branch,folder])
        command(['git','-C',folder,'remote','add','origin',ORIGIN])
    require(command(['git','-C',folder,'remote','get-url','origin'])==ORIGIN,'SSH origin differs')
    require(command(['git','-C',folder,'branch','--show-current'])==branch,'Transfer branch differs')
    for k,v in [('user.name',identity['name']),('user.email',identity['email']),('core.hooksPath','/dev/null'),('commit.gpgsign','false')]:
        command(['git','-C',folder,'config','--local',k,v])
    if ssh_command:
        command(['git','-C',folder,'config','--local','core.sshCommand',ssh_command])


def commit_git(folder, message):
    command(['git','-C',folder,'add','--all'])
    if command(['git','-C',folder,'status','--porcelain=v1']):
        command(['git','-C',folder,'commit','-m',message],timeout=300)
    return command(['git','-C',folder,'rev-parse','HEAD'])
