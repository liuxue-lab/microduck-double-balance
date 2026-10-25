"""Read-only preservation of the exact independently reviewed latest preflight."""
from pathlib import Path,PurePosixPath
import hashlib,json,zipfile
RUN_ID='20261015T030941Z-1cf0b0'
ARCHIVE_SHA='58101ee60c608e3eed013f2b5766cfdbc6c7c285852ad5b30926a0ce78612797'
BUNDLE_SHA='00bddaec2e08e68839867c039d10388687efef30187b97b693d6e1106e2f5740'
LAUNCHER_SHA='1b5ef7d645682507e612287d73ce426e3706a73625294c81669b56c667bba6b6'
COMMIT='6fd29f8037802d68f314187a84537d31c76f9316'
def require(ok,message):
    if not ok:raise RuntimeError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()

def verify(root,current_git,expected_program,expected_summary):
    base=Path(root)/'integration-contract-preflight';run=base/RUN_ID
    marker_path=base/'started-v1.json';marker=json.loads(marker_path.read_bytes())
    require(marker['run']==str(run) and marker['bundle_sha256']==BUNDLE_SHA,'Previous preflight started identity differs')
    archive=run/'integration-contract-preflight-return.zip'
    require(not archive.is_symlink() and archive.stat().st_size==1313970 and sha(archive.read_bytes())==ARCHIVE_SHA,'Reviewed preflight archive differs')
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();inv=json.loads(z.read('evidence-inventory.json'))
        require(len(names)==len(set(names))==257 and set(names)==set(inv)|{'evidence-inventory.json'},'Reviewed preflight roster differs')
        for name,row in inv.items():
            posix=PurePosixPath(name);path=run/name
            require(not posix.is_absolute() and '..' not in posix.parts and '\\' not in name and path.resolve().is_relative_to(run.resolve()) and not path.is_symlink(),'Unsafe predecessor path')
            raw=z.read(name)
            require(row==dict(bytes=len(raw),sha256=sha(raw)) and path.read_bytes()==raw,'Reviewed preflight bytes changed: '+name)
        require((run/'evidence-inventory.json').read_bytes()==z.read('evidence-inventory.json'),'Preflight inventory changed')
        require(sha(z.read('launcher.py'))==LAUNCHER_SHA,'Preflight launcher changed')
        program={n.removeprefix('program/'):dict(bytes=len(z.read(n)),sha256=sha(z.read(n))) for n in names if n.startswith('program/')}
        require(program==expected_program,'Reviewed preflight program differs')
        load=lambda n:json.loads(z.read(n))
        require(load('run-summary.json')==expected_summary,'Preflight summary differs')
        require(load('git-before.json')==load('git-after.json')==current_git,'Project changed after reviewed preflight')
        require(load('execution-finished.json')==dict(process_has_exited=True,exit_code=0,timed_out=False,interrupted=False,launch_error=None),'Preflight did not finish')
        report=load('probe/run-review.json');phase=load('probe/phase-progress.json');roster=load('probe/source-roster.json')
        require(report['native_calls']==0 and not any(report['counts'].values()),'Preflight native count differs')
        require(phase['completed_units']==phase['total_units']==8 and report['host_checks_passed']==report['host_checks_total']==12,'Preflight checks incomplete')
        require(len(roster)==218 and sum(r['previously_pinned'] for r in roster)==133,'Preflight source coverage differs')
        require(expected_summary['canonical_completed']==662799 and expected_summary['canonical_attempted']==662800,'Cumulative history differs')
        require(expected_summary['G09']=='OPEN' and expected_summary['stage11_complete'] is False,'Invalid acceptance')
    return dict(status='INDEPENDENTLY_VERIFIED_PREFLIGHT_BYTES_PRESERVED',run=str(run),commit=COMMIT,archive_sha256=ARCHIVE_SHA,
        marker_sha256=sha(marker_path.read_bytes()),archive_members=257,sources=218,host_checks='12/12',native_calls=0,summary=expected_summary)
