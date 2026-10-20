"""Read fixed completed artifacts. No remote filesystem writes or GPU imports."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path('/root/autodl-tmp/microduck-double-balance/artifacts/double-balance-stage12')
FILES=@@FILES@@
CHUNK=1024*1024

def need(ok,message):
    if not ok:raise RuntimeError(message)

def digest(path,length=None):
    h=hashlib.sha256()
    with path.open('rb') as f:
        remaining=path.stat().st_size if length is None else length
        while remaining:
            data=f.read(min(CHUNK,remaining));need(bool(data),'Unexpected EOF')
            h.update(data);remaining-=len(data)
    return h.hexdigest()

def locate(key):
    need(key in FILES,'Unknown artifact key')
    item=FILES[key];path=ROOT/item['remote']
    need(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(ROOT.resolve()),'Missing or unexpected artifact path')
    size=path.stat().st_size
    need(0<size<=item['max_bytes'],'Unexpected artifact size')
    if item.get('bytes') is not None:need(size==item['bytes'],'Checkpoint size changed')
    return path,size

def main(argv):
    operation=argv[0]
    if operation=='inventory':
        result={}
        for key,item in FILES.items():
            path,size=locate(key);actual=digest(path)
            need(actual==item['sha256'],'Original artifact SHA256 differs: '+key)
            result[key]=dict(bytes=size,sha256=actual,remote=str(path))
        print(json.dumps(result,sort_keys=True));return
    need(operation in ('prefix','chunk'),'Unsupported operation')
    key=argv[1];path,size=locate(key)
    offset=int(argv[2]);need(0<=offset<=size,'Invalid offset')
    if operation=='prefix':
        print(json.dumps(dict(bytes=offset,sha256=digest(path,offset))));return
    count=int(argv[3]);need(0<count<=CHUNK and offset+count<=size,'Invalid chunk range')
    with path.open('rb') as f:f.seek(offset);data=f.read(count)
    need(len(data)==count,'Short artifact read')
    sys.stdout.buffer.write(('MD12_DATA %d %s\n'%(len(data),hashlib.sha256(data).hexdigest())).encode()+data)
    sys.stdout.buffer.flush()

if __name__=='__main__':
    try:main(sys.argv[1:])
    except Exception as exc:
        print('MD12_READONLY_ERROR='+json.dumps(dict(error=repr(exc))),file=sys.stderr,flush=True)
        raise SystemExit(2)
