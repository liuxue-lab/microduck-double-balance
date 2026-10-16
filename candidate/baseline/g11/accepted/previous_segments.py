"""Immutable predecessor identity, including its failed consumer and all costs."""
from pathlib import Path,PurePosixPath
import hashlib,json,zipfile
COMMIT='6e0294ebee75b4d857064e96de9900423010f973'
RUN_ID='20261015T060419Z-8ab075'
BUNDLE='fdc7b29d1e081da8c0450252bac40dedb5e894de24f5dafb2257e039f3230b66'
ARCHIVE='4556cc3189d312cff1390fabbe04652e45508e38f9910d6b450284cdfb3089c2'
def need(ok,msg):
    if not ok:raise RuntimeError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def verify(root,current_git):
    base=Path(root)/'g10-rk4-segments-rebuild';run=base/RUN_ID
    mark=base/'started-v1.json';m=json.loads(mark.read_bytes())
    need(m['run']==str(run) and m['bundle_sha256']==BUNDLE,'Previous segments marker differs')
    archive=run/'g10-rk4-segments-rebuild-return.zip'
    need(not archive.is_symlink() and archive.stat().st_size==55556771 and sha(archive.read_bytes())==ARCHIVE,'Previous segments ZIP differs')
    with zipfile.ZipFile(archive) as z:
        inv=json.loads(z.read('evidence-inventory.json'));names=z.namelist()
        need(len(names)==len(set(names))==2462 and set(names)==set(inv)|{'evidence-inventory.json'},'Previous segments roster differs')
        for n,row in inv.items():
            p=PurePosixPath(n);local=run/n;raw=z.read(n)
            need(not p.is_absolute() and '..' not in p.parts and '\\' not in n and not local.is_symlink() and local.resolve().is_relative_to(run.resolve()),'Unsafe predecessor path')
            need(row==dict(bytes=len(raw),sha256=sha(raw)) and local.read_bytes()==raw,'Previous segments bytes differ: '+n)
        need((run/'evidence-inventory.json').read_bytes()==z.read('evidence-inventory.json'),'Old inventory changed')
        j=lambda n:json.loads(z.read(n))
        s=j('run-summary.json');finish=j('execution-finished.json')
        need((s['canonical_completed'],s['canonical_attempted'])==(666200,666201),'Historical count differs')
        need(s['counts']['integrations_completed']==s['counts']['integrations_attempted']==704 and s['counts']['GPU_forward_completed']==2830,'Old physical ledger differs')
        need(j('git-before.json')==j('git-after.json')==current_git,'Git state differs from returned segments')
        need(finish['process_has_exited'] and finish['exit_code']==2 and not finish['timed_out'] and not finish['interrupted'],'Old failed run exit differs')
        p=j('probe/world0-producer/run-review.json');c=j('probe/world0-consumer/run-review.json')
        need(p['status']=='G10_SEGMENTS_REBUILD_MEASURED_REVIEW_REQUIRED' and p['completed_macros']==['eager','graph'],'Reusable producer incomplete')
        need(c['status']=='G10_SEGMENTS_REBUILD_STOPPED_REVIEW_REQUIRED' and c['counts']['integrations_attempted']==0 and c['error']=="RuntimeError('Post-rebuild sense changed restored logical numeric state')",'Old failure differs')
    return dict(commit=COMMIT,run=str(run),archive_sha256=ARCHIVE,marker_sha256=sha(mark.read_bytes()),members=2462,
                preserved_failed_consumer=True,reused_macros=2,native_calls=0,
                canonical_completed=666200,canonical_attempted=666201,prior_counts=s['counts'])
