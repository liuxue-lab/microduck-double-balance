"""Install an immutable small evaluation package and start once."""
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import zlib

SLOT=Path('/root/autodl-tmp/microduck-double-balance/artifacts/double-balance-stage12/native-evaluation-v1')

def main():
    data=json.loads(zlib.decompress(sys.stdin.buffer.read()))
    files={name:base64.b64decode(value) for name,value in data['files'].items()}
    for name in files:
        if Path(name).name!=name:raise ValueError('Invalid package filename')
    manifest=json.loads(files['manifest.json'])
    body={k:v for k,v in files.items() if k!='manifest.json'}
    digest=hashlib.sha256(json.dumps({k:hashlib.sha256(v).hexdigest() for k,v in sorted(body.items())},sort_keys=True).encode()).hexdigest()
    if digest!=manifest['identity'] or digest!=sys.argv[1]:raise ValueError('Evaluation package identity differs')
    if not SLOT.parent.is_dir():raise ValueError('Existing cloud project not found')
    SLOT.mkdir(exist_ok=True)
    if SLOT.is_symlink():raise ValueError('Unexpected slot symlink')
    target=SLOT/'package'
    if target.exists():
        if target.is_symlink() or any((target/k).read_bytes()!=v for k,v in files.items()):
            raise ValueError('Existing evaluation package differs; preserve it')
    else:
        stage=SLOT/('package-incoming-'+str(os.getpid()));stage.mkdir()
        for name,value in files.items():
            with (stage/name).open('xb') as f:f.write(value);f.flush();os.fsync(f.fileno())
        try:stage.rename(target)
        except OSError:
            if not target.is_dir() or any((target/k).read_bytes()!=v for k,v in files.items()):raise
            # Only remove this invocation's verified identical staging files.
            for name in files:(stage/name).unlink()
            stage.rmdir()
    os.execv(sys.executable,[sys.executable,'-B',str(target/'agent.py'),'start'])

if __name__=='__main__':main()
