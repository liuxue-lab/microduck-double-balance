"""Full native evidence, bounded ZIP attachments, idempotent publication."""
from pathlib import Path
import hashlib
import json
import zipfile
from .audit import read, verify_file
from .durable import require, file_digest, publish
from .transfer import same_or_create, json_bytes, verify_return

MAX_GROUP = 24*1024**2


def excluded(path):
    return any(p.endswith('runtime-scratch') or p in ('kernel-cache','__pycache__') for p in path.parts)


def index_native(out):
    native = Path(out)/'native'
    rows = []
    if native.exists():
        for p in sorted(native.rglob('*')):
            if not p.is_file() or excluded(p.relative_to(native)): continue
            require(not p.is_symlink(), 'Native evidence link refused')
            rows.append(dict(path=str(p.relative_to(native)),bytes=p.stat().st_size,sha256=file_digest(p)))
    same_or_create(Path(out)/'native-evidence-index.json',json_bytes(dict(files=rows,
        total_bytes=sum(r['bytes'] for r in rows),all_noncache_native_files_included=True)))
    return rows


def checked_zip(target, sources, manifest_name):
    target = Path(target)
    expected = {n:dict(bytes=p.stat().st_size,sha256=file_digest(p)) for n,p in sources.items()}
    def writer(f):
        with zipfile.ZipFile(f,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for n,p in sources.items(): z.write(p,n)
            z.writestr(manifest_name,json_bytes(expected))
    if not target.exists(): publish(target,writer)
    require(target.stat().st_size <= 49*1024**2, 'ZIP exceeds 49 MiB')
    with zipfile.ZipFile(target) as z:
        require(len(z.namelist())==len(expected)+1 and set(z.namelist())==set(expected)|{manifest_name}, 'Archive roster changed')
        require(json.loads(z.read(manifest_name))==expected, 'Archive manifest changed')
        for n,r in expected.items():
            b=z.read(n)
            require(len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'], 'Archive member changed: '+n)
    return dict(file=target.name,bytes=target.stat().st_size,sha256=file_digest(target),
                members=len(expected),expanded_bytes=sum(r['bytes'] for r in expected.values()))


def package(out, root, base):
    out,root,base = Path(out),Path(root),Path(base)
    rows=index_native(out)
    groups=[]; group=[]; size=0
    for row in rows:
        require(row['bytes']<=MAX_GROUP,'Single evidence file exceeds partition bound; preserved')
        if group and size+row['bytes']>MAX_GROUP:
            groups.append(group); group=[]; size=0
        group.append(row); size+=row['bytes']
    if group: groups.append(group)
    archives=[]
    for i,group in enumerate(groups):
        sources={r['path']:verify_file(out/'native',r) for r in group}
        archives.append(checked_zip(out/('raw-evidence-%03d.zip'%i),sources,'RAW-MANIFEST.json'))
        print('Stage12RawArchive=%d/%d'%(i+1,len(groups)),flush=True)
    same_or_create(out/'raw-evidence-manifest.json',json_bytes(dict(archives=archives,
        files=len(rows),expanded_bytes=sum(r['bytes'] for r in rows),per_zip_limit_bytes=49*1024**2)))
    # The main ZIP has the whole runnable package, top-level receipts, and key
    # native reports. Every non-cache native byte is also in verified attachments.
    sources={}
    for p in sorted(root.rglob('*')):
        if p.is_file():
            require(not p.is_symlink(),'Program link refused')
            sources['program/'+str(p.relative_to(root))]=p
    for p in sorted(out.iterdir()):
        if p.is_file() and p.suffix!='.zip' and '.partial-' not in p.name:
            sources['evidence/'+p.name]=p
    important={'host-report.json','measurement.json','owner-closeout.json','native-report.json',
               'macro-failure.json','macro-result.json','progress.json','bootstrap-report.json'}
    for r in rows:
        if Path(r['path']).name in important:
            sources['evidence/native/'+r['path']]=out/'native'/r['path']
    require(sum(p.stat().st_size for p in sources.values())<=49*1024**2,'Main archive too large; retained')
    target=base/('stage12-n2-return-'+out.name+'.zip')
    checked_zip(target,sources,'RETURN-MANIFEST.json')
    checked=verify_return(target)
    same_or_create(base/'return-receipt.json',json_bytes(dict(run=str(out),archive=str(target),
        bytes=checked['bytes'],sha256=checked['sha256'],members=checked['members'])))
    return checked
