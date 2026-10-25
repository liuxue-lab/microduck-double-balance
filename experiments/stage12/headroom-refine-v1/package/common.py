"""Pinned paths and read-only preflight for the approved headroom campaign."""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess

from budget import require, MAX_UPDATES, MAX_SECONDS, TARGETS, EVAL_SEEDS

BASE=Path('/root/autodl-tmp/microduck-double-balance')
ARTIFACTS=BASE/'artifacts/double-balance-stage12'
SLOT=ARTIFACTS/'native-headroom-refine-v1'
TRAIN=ARTIFACTS/'native-training-v2'
SOURCE=ARTIFACTS/'finite-trial-v1/cloud-20261020-fp64-one-shot-v1/inputs/project'
PYTHON=BASE/'workspace/.venv/bin/python'
HERE=Path(__file__).resolve().parent
MODEL=ARTIFACTS/'native-speed-refine-v1/run/checkpoints/refine_000500.pt'
MODEL_SHA='51821a28880aba74b80e6d5a388dfd7a82098bd3dca04a1e8947b4b2a8d5539a'
MODEL_BYTES=10036828
ORIGIN_ID='340f064e33b61ed8968724084ebab8dc35563cb2239506ab20b1d01056581c10'
TASK_SHA='b6909b9cff864dcfce8203062a75282571ca2c7ef5d66b2aa58c05c4626a55af'
PINS_SHA='08c781088ecb4b2023c593ff8ffa45061bf6a57fad5902fcf79b50155daa969c'
REWARD_SOURCE_SHA='0683e8de9d1c1534bd336c943af64db59d0aa76498a33c3aa49fa5d00464ac1d'
VERSIONS={'mjlab':'1.3.0','mujoco':'3.10.0','mujoco-warp':'3.8.1',
          'warp-lang':'1.12.0','rsl-rl-lib':'5.0.1'}
NUM_ENVS=4096
LEARNING_RATE=1e-5
TRAIN_SEED=2026102301
SMOKE_SEED=2026102300
CODE_PATH='experiments/stage12/headroom-refine-v1'
ORIGIN_SSH='git@github.com:liuxue-lab/microduck-double-balance.git'


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def write(path,value,exclusive=False):
    path=Path(path)
    data=(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n').encode()
    tmp=path if exclusive else path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with tmp.open('xb') as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    if not exclusive:
        os.replace(tmp,path)
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def identity(pid):
    try:
        fields=(Path('/proc')/str(int(pid))/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=int(pid),start=fields[19],state=fields[0],group=int(fields[2]))
    except FileNotFoundError:
        return None


def alive(pin):
    current=identity(pin['pid']) if pin else None
    return bool(current and current['start']==pin['start'] and current['state'] not in ('Z','X'))


def verify_package():
    manifest=read(HERE/'manifest.json')
    calculated=hashlib.sha256(json.dumps(manifest['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    require(calculated==manifest['identity'],'Package manifest identity differs')
    for name,entry in manifest['files'].items():
        p=HERE/name
        require(Path(name).name==name and p.is_file() and not p.is_symlink(),'Invalid package file')
        require(p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],'Package source differs: '+name)
    return manifest['identity']


def verify_runtime():
    actual={name:importlib.metadata.version(name) for name in VERSIONS}
    require(actual==VERSIONS,'Pinned installed runtime differs; no package installation or upgrade')
    return dict(actual, torch=importlib.metadata.version('torch'))


def verify_inputs():
    require(MODEL.is_file() and MODEL.stat().st_size==MODEL_BYTES and sha(MODEL)==MODEL_SHA,
            'Selected Stage12 speed500 model missing or changed; preserve, do not substitute a trajectory model')
    require(sha(HERE/'native_task.py')==TASK_SHA and sha(HERE/'speed_task.py')==REWARD_SOURCE_SHA,
            'Original Stage12 task or selected speed reward changed')
    require(sha(HERE/'source-pins.json')==PINS_SHA,'Original frozen-source pins changed')
    pins=read(HERE/'source-pins.json')
    require(len(pins)==365,'Expected the original 365 source pins')
    for rel,pin in pins.items():
        p=SOURCE/rel
        require(p.is_file() and p.stat().st_size==pin['bytes'] and sha(p)==pin['sha256'],
                'Frozen project source differs: '+rel)
    return dict(checkpoint_sha256=MODEL_SHA,task_sha256=TASK_SHA,
                source_pins_sha256=PINS_SHA,source_files_verified=len(pins))


def protected_records():
    paths={MODEL}
    for root in (ARTIFACTS,BASE/'artifacts/double-balance-stage13',BASE/'artifacts/double-balance-stage14'):
        if not root.is_dir():continue
        for pattern in ('**/budget.json','**/selection.json','**/started.json','**/exit.json','**/run/latest.json'):
            paths.update(p for p in root.glob(pattern) if p.is_file() and not p.is_relative_to(SLOT))
    return {str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(paths)}


def verify_protected(records):
    for name,pin in records.items():
        p=Path(name)
        require(p.is_file() and p.stat().st_size==pin['bytes'] and sha(p)==pin['sha256'],
                'Protected original record changed: '+name)
    return True


def preflight():
    require(PYTHON.is_file(),'Existing cloud project Python missing')
    require(str(Path(os.sys.executable).parent)==str(PYTHON.parent), 'Use the existing project Python')
    package=verify_package();runtime=verify_runtime();inputs=verify_inputs()
    require(importlib.util.find_spec('onnx') is not None,
            'The existing ONNX smoke-export dependency is missing; no PPO started, no installer run')
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    require(not active,'Another GPU compute process is active; do not launch a second task: '+active)
    return dict(package_identity=package,runtime=runtime,inputs=inputs,
                protected_records=protected_records())


def input_checkpoint():
    latest=SLOT/'run/latest.json'
    if latest.is_file():return read(latest)
    return dict(path=str(MODEL),sha256=MODEL_SHA,bytes=MODEL_BYTES,
                completed_updates=0,policy_updates=5000,attempted_updates=0)
