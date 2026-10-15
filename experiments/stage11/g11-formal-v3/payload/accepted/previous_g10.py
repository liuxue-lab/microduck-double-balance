"""Re-use the audited G10 return without replaying any physics."""
from pathlib import Path,PurePosixPath
import hashlib,json,zipfile
def need(ok,msg):
    if not ok:raise RuntimeError(msg)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def verify(root,current_git):
    base=Path(root)/'g10-native-graph-lifecycle';run=base/'20261015T051930Z-c3ad2b'
    marker=base/'started-v1.json';mark=json.loads(marker.read_bytes())
    need(mark['run']==str(run) and mark['bundle_sha256']=='adc520dc5c581cc2872090fec1d4e88bc219dba2d9f47c9edb0fe20d265730c2','Prior G10 started identity differs')
    archive=run/'g10-native-graph-lifecycle-return.zip'
    need(not archive.is_symlink() and sha(archive.read_bytes())=='ff73e707b82ef581a746bf020ae8699969f36e804cbd324f41e6cada376f2065','Audited G10 archive differs')
    with zipfile.ZipFile(archive) as z:
        inv=json.loads(z.read('evidence-inventory.json'));names=z.namelist()
        need(len(names)==len(set(names))==187 and set(names)==set(inv)|{'evidence-inventory.json'},'G10 archive roster differs')
        for n,row in inv.items():
            p=PurePosixPath(n);raw=z.read(n);local=run/n
            need(not p.is_absolute() and '..' not in p.parts and not local.is_symlink(),'Unsafe prior G10 member')
            need(row==dict(bytes=len(raw),sha256=sha(raw)) and local.read_bytes()==raw,'Prior G10 bytes changed: '+n)
        summary=json.loads(z.read('run-summary.json'))
        need((summary['canonical_completed'],summary['canonical_attempted'])==(665496,665497),'Prior G10 ledger differs')
        need(json.loads(z.read('git-before.json'))==json.loads(z.read('git-after.json'))==current_git,'Project state differs from G10')
        need(json.loads(z.read('execution-finished.json'))==dict(process_has_exited=True,exit_code=0,timed_out=False,interrupted=False,launch_error=None),'Prior G10 did not finish')
    return dict(commit='009fb207f123b4ac27aeca3c07d2713a2d97ebb7',archive_sha256=sha(archive.read_bytes()),marker_sha256=sha(marker.read_bytes()),all_187_members_preserved=True,native_calls=0)
