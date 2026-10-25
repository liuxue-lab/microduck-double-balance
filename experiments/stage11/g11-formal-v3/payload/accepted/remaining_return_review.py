"""Preserve the exact archive already independently audited; no native imports."""
from pathlib import Path, PurePosixPath
import hashlib, json, zipfile
RUN_ID='20261015T021357Z-af3aef'
ARCHIVE_SHA='66fd31da6d6d00cd54d331c9452e5330e847e47a9d4611a5770fb657709f5a2d'
BUNDLE_SHA='d8177e6c9b27249c407345a9ba7957ddfbb705f7935fbf67e0d679f9b3b9500a'
LAUNCHER_SHA='d865c1473d6b2e346db39592f674305f80afab269adb3960fc4c49880d2e2812'
COMMIT='74ba13b52e4f5cb82f959b5feea6312fe0c37275'
def require(ok,message):
    if not ok:raise RuntimeError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()

def verify(root,current_git,expected_program,expected_summary):
    base=Path(root)/'post-reset-world1-remaining-regression';run=base/RUN_ID
    marker_path=base/'started-v1.json';marker=json.loads(marker_path.read_bytes())
    require(marker['run']==str(run) and marker['bundle_sha256']==BUNDLE_SHA,'Previous started identity differs; preserve it')
    archive=run/'post-reset-world1-remaining-regression-return.zip'
    require(not archive.is_symlink() and archive.stat().st_size==66564955 and sha(archive.read_bytes())==ARCHIVE_SHA,'Reviewed archive identity differs')
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();inv=json.loads(z.read('evidence-inventory.json'))
        require(len(names)==len(set(names))==8330 and set(names)==set(inv)|{'evidence-inventory.json'},'Reviewed archive roster differs')
        for name,row in inv.items():
            posix=PurePosixPath(name);path=run/name
            require(not posix.is_absolute() and '..' not in posix.parts and '\\' not in name and path.resolve().is_relative_to(run.resolve()) and not path.is_symlink(),'Unsafe archived evidence path')
            raw=z.read(name)
            require(row==dict(bytes=len(raw),sha256=sha(raw)) and path.read_bytes()==raw,'Reviewed local/archive evidence changed: '+name)
        require((run/'evidence-inventory.json').read_bytes()==z.read('evidence-inventory.json'),'Previous inventory changed')
        require(sha(z.read('launcher.py'))==LAUNCHER_SHA,'Previous launcher changed')
        program={n.removeprefix('program/'):dict(bytes=len(z.read(n)),sha256=sha(z.read(n))) for n in names if n.startswith('program/')}
        require(program==expected_program,'Reviewed program changed')
        load=lambda name:json.loads(z.read(name))
        require(load('run-summary.json')==expected_summary,'Reviewed summary differs')
        require(load('git-before.json')==load('git-after.json')==current_git,'Project changed since reviewed run')
        require(load('execution-finished.json')==dict(process_has_exited=True,exit_code=0,timed_out=False,interrupted=False,launch_error=None),'Previous execution incomplete')
        review=load('probe/run-review.json');owner=review['owner_reports']['1'];phase=load('probe/phase-progress.json')
        require(review['owner_exit_codes']=={'1':0} and review['owners_exited'] and review['joint_publications']==[],'Previous execution scope changed')
        require(phase['completed_units']==phase['total_units']==11 and phase['passed_branches']==7,'Previous branches incomplete')
        rows=owner['continuation_cases'];checks=[v['passed'] for row in rows for k,v in row.items() if k.startswith('check_')]
        require(len(rows)==7 and len(checks)==31 and all(v is True for v in checks),'Previously verified comparisons changed')
        require(owner['original_material_Data_Model_owners_RNG_restored'] and owner['contact_ordering']['graph_closed'],'Previous restoration missing')
        require(review['counts']['integrations_completed']==1376 and review['counts']['GPU_forward_completed']==5533,'Previous native counts differ')
        require(expected_summary['canonical_completed']==662799 and expected_summary['canonical_attempted']==662800,'Cumulative history differs')
        require(expected_summary['G09']=='OPEN' and expected_summary['stage11_complete'] is False,'Invalid global acceptance')
        proc=Path('/proc')/str(owner['pid'])/'cmdline'
        if proc.is_file():require(str(run).encode() not in proc.read_bytes(),'Previous owner still active')
    return dict(status='INDEPENDENTLY_VERIFIED_REMAINING_BYTES_PRESERVED',run=str(run),commit=COMMIT,archive_sha256=ARCHIVE_SHA,
        marker_sha256=sha(marker_path.read_bytes()),archive_members=8330,native_calls=0,latest_branches='7/7',latest_comparisons='31/31',
        corrected_world1_branches='9/9',corrected_world1_comparisons='41/41',scoped_corrected_world1_acceptance=True,
        full_end_to_end_G09_acceptance=False,summary=expected_summary)
