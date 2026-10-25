"""Install only the new native-training slot; never alter earlier attempts."""
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zlib

BASE=Path('/root/autodl-tmp/microduck-double-balance')
SLOT=BASE/'artifacts/double-balance-stage12/native-training-v2'
PYTHON=BASE/'workspace/.venv/bin/python'

def need(ok,reason):
    if not ok:raise RuntimeError(reason)

def main():
    packed=sys.stdin.buffer.read(1024*1024+1)
    need(len(packed)<=1024*1024,'Unexpected package size')
    need(hashlib.sha256(packed).hexdigest()==sys.argv[1],'Package transmission mismatch')
    payload=json.loads(zlib.decompress(packed));files=payload['files'];cfg=payload['config']
    decoded={name:base64.b85decode(value.encode()) for name,value in files.items()}
    manifest=json.loads(decoded['manifest.json'])
    need(manifest['identity']==cfg['package_identity'],'Package/config mismatch')
    for name,data in decoded.items():
        need(Path(name).name==name and name not in ('.','..'),'Unsafe package name')
        if name!='manifest.json':
            pin=manifest['files'][name]
            need(len(data)==pin['bytes'] and hashlib.sha256(data).hexdigest()==pin['sha256'],'Package content mismatch')
    need(set(manifest['files'])==set(decoded)-{'manifest.json'},'Manifest completeness')
    ns={'__name__':'readonly_probe'}
    exec(compile(decoded['common.py']+b'\n'+decoded['probe.py'].replace(b'from common import SLOT, LEGACY_SLOT, read, alive, has_campaign, validation_remaining',b''),'<campaign-probe>','exec'),ns)
    prior=ns['discover']()
    if prior.get('campaign_exists'):
        print('MD12_NATIVE_STATE='+json.dumps(prior),flush=True);return
    need(SLOT.parent.is_dir() and PYTHON.is_file(),'Existing Stage12 cloud runtime not found')
    SLOT.mkdir(exist_ok=True)
    need(not SLOT.is_symlink(),'Unexpected new slot symlink')
    target=SLOT/'package'
    if target.exists():
        need(not target.is_symlink(),'Unexpected package symlink')
        for name,data in decoded.items():
            need((target/name).is_file() and (target/name).read_bytes()==data,'Existing formal package differs; preserve it')
    else:
        stage=SLOT/('package-incoming-'+str(os.getpid()));stage.mkdir()
        for name,data in decoded.items():
            with (stage/name).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        try:os.rename(stage,target)
        except FileExistsError:
            for name,data in decoded.items():need((target/name).read_bytes()==data,'Concurrent package differs')
            shutil.rmtree(stage)  # Only this invocation's identical staging files.
    path=SLOT/'config.json'
    if path.exists():need(json.loads(path.read_bytes())==cfg,'Existing campaign configuration differs')
    else:
        with path.open('x') as f:json.dump(cfg,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    os.execv(str(PYTHON),[str(PYTHON),'-B',str(target/'agent.py'),'prepare'])

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        print('MD12_NATIVE_ERROR='+json.dumps({'error':repr(exc)}),flush=True)
        raise SystemExit(2)
