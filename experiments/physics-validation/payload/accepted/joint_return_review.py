"""Read-only preservation of all bytes in the reviewed joint return."""
from pathlib import Path,PurePosixPath
import hashlib,json,zipfile
RUN_ID='20261015T040045Z-77fe51'
ARCHIVE_SHA='90cfa8c975934045344aa208b30644a007c2527aefcd82078afc44f2e3c97ffa'
BUNDLE_SHA='1a8fd8d0c63da65974d0531be00adaa7e1fcdc2d6d71de904f56fcb7b40f80f2'
LAUNCHER_SHA='86c3cd74c852023622b5da2c7f743f8ea91aacdb7c267cc6e00c4c2b76a91de8'
COMMIT='085388e8f92255b3f885c8faf2ab7486ffe79589'
def require(ok,message):
    if not ok:raise RuntimeError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def verify(root,current_git,expected_program,expected_summary):
    base=Path(root)/'post-reset-joint-closure';run=base/RUN_ID
    marker_path=base/'started-v1.json';marker=json.loads(marker_path.read_bytes())
    require(marker['run']==str(run) and marker['bundle_sha256']==BUNDLE_SHA,'Previous joint started identity differs')
    archive=run/'post-reset-joint-closure-return.zip'
    require(not archive.is_symlink() and archive.stat().st_size==138934734 and sha(archive.read_bytes())==ARCHIVE_SHA,'Reviewed joint archive differs')
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();inv=json.loads(z.read('evidence-inventory.json'))
        require(len(names)==len(set(names))==21827 and set(names)==set(inv)|{'evidence-inventory.json'},'Reviewed joint roster differs')
        for name,row in inv.items():
            posix=PurePosixPath(name);path=run/name
            require(not posix.is_absolute() and '..' not in posix.parts and '\\' not in name and path.resolve().is_relative_to(run.resolve()) and not path.is_symlink(),'Unsafe predecessor path')
            raw=z.read(name)
            require(row==dict(bytes=len(raw),sha256=sha(raw)) and path.read_bytes()==raw,'Reviewed joint bytes changed: '+name)
        require((run/'evidence-inventory.json').read_bytes()==z.read('evidence-inventory.json'),'Joint inventory changed')
        require(sha(z.read('launcher.py'))==LAUNCHER_SHA,'Joint launcher changed')
        program={n.removeprefix('program/'):dict(bytes=len(z.read(n)),sha256=sha(z.read(n))) for n in names if n.startswith('program/')}
        require(program==expected_program,'Reviewed joint program differs')
        load=lambda n:json.loads(z.read(n))
        require(load('run-summary.json')==expected_summary,'Joint summary differs')
        require(load('git-before.json')==load('git-after.json')==current_git,'Project changed after reviewed joint return')
        require(load('execution-finished.json')==dict(process_has_exited=True,exit_code=0,timed_out=False,interrupted=False,launch_error=None),'Joint run did not finish')
        require(expected_summary['canonical_completed']==665487 and expected_summary['canonical_attempted']==665488,'Cumulative history differs')
        require(expected_summary['G09']=='OPEN' and expected_summary['stage11_complete'] is False,'Producer report was rewritten')
        require(load('probe/run-review.json')['status']=='JOINT_CLOSURE_MEASURED_REVIEW_REQUIRED','Joint producer result differs')
    return dict(status='INDEPENDENTLY_REVIEWED_JOINT_BYTES_PRESERVED',run=str(run),commit=COMMIT,archive_sha256=ARCHIVE_SHA,
        marker_sha256=sha(marker_path.read_bytes()),archive_members=21827,native_calls=0,summary=expected_summary)
