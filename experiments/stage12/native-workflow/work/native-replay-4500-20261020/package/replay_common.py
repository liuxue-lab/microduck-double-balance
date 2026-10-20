import hashlib
import json
import os
from pathlib import Path

PROJECT=Path.home()/'microduck-double-balance/workspace'
SELECTED=Path.home()/'microduck-double-balance/artifacts/double-balance-stage12/native-selected-4500'
SLOT=SELECTED.parent/'native-replay-4500-v1'
MODEL=SELECTED/'checkpoints/update_004500.pt'
MODEL_SHA='df03202acf010680b63d3277ec879f63e2d4808e5d1286653923a314f125ac6a'
TASK_SHA='b6909b9cff864dcfce8203062a75282571ca2c7ef5d66b2aa58c05c4626a55af'
PINS_SHA='08c781088ecb4b2023c593ff8ffa45061bf6a57fad5902fcf79b50155daa969c'
ORIGIN_ID='dd6d1e1e697bcaa348dbdd92dd8c7b30ea31ae9c32599338106859ae7657523a'
PYTHON=PROJECT/'.venv/bin/python'
SEEDS=(20261021,20261022)
MAX_SECONDS=900
HORIZON=500

def need(ok,message):
    if not ok:raise RuntimeError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_bytes())

def write(path,value,exclusive=False):
    path=Path(path);data=(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n').encode()
    out=path if exclusive else path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with out.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    if not exclusive:os.replace(out,path)

def identity(pid):
    try:
        row=(Path('/proc')/str(int(pid))/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=int(pid),start=row[19],state=row[0],group=int(row[2]))
    except FileNotFoundError:return None

def alive(pin):
    now=identity(pin['pid']) if pin else None
    return bool(now and now['start']==pin['start'] and now['state'] not in ('Z','X'))

def verify_inputs():
    need(MODEL.is_file() and not MODEL.is_symlink() and MODEL.stat().st_size==10036060 and sha(MODEL)==MODEL_SHA,'Selected 4500 checkpoint differs')
    task=SELECTED/'task/native_task.py';pins=SELECTED/'task/source-pins.json'
    need(sha(task)==TASK_SHA and sha(pins)==PINS_SHA,'Frozen task or source pins differ')
    source=read(pins)
    for rel,entry in source.items():
        relative=Path(rel);need(not relative.is_absolute() and '..' not in relative.parts,'Invalid source pin path')
        p=PROJECT/relative
        need(p.is_file() and p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],'Local source differs: '+rel)
    return dict(checkpoint_sha256=MODEL_SHA,task_sha256=TASK_SHA,source_pins_sha256=PINS_SHA,source_files_verified=len(source))
