#!/usr/bin/env python3
"""After a successful local check, commit only Stage 09 files and export a full bundle."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

REPO=Path('/home/lx/microduck-double-balance/workspace')
DOWNLOADS=Path('/home/lx/下载')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()


def main():
    pointer=json.loads((REPO.parent/'artifacts/double-balance-stage09/preparation/latest.json').read_text())
    report_path=Path(pointer['report']);report=json.loads(report_path.read_text())
    manifest=REPO/'docs/audits/stage-09-runtime-hashes.json'
    if report['status']!='ZERO_PPO_RUNTIME_CHECKS_COMPLETE_REVIEW_PENDING' or report['runtime_manifest_sha256']!=sha(manifest):
        raise ValueError('Matching successful zero-PPO checks are required')
    from mjlab_microduck.double_balance_stage09_state import verify_source
    verify_source(REPO)
    if git('branch','--show-current')!='double-balance':raise ValueError('Use double-balance')
    allowed=json.loads((REPO/'docs/audits/stage-09-preparation-files.json').read_text())['files']
    # Never stage or commit unrelated user work.
    staged=set(git('diff','--cached','--name-only').splitlines())
    changed=set(git('diff','--name-only').splitlines())|set(git('ls-files','--others','--exclude-standard').splitlines())
    if not (staged|changed)<=set(allowed):raise ValueError('Unrelated changes present; keep them and inspect before export')
    if changed or staged:
        subprocess.run(['git','add','--',*allowed],cwd=REPO,check=True)
        subprocess.run(['git','commit','-m','feat: prepare approved Stage 09 head posture pilot'],cwd=REPO,check=True)
    verify_source(REPO,clean=True)
    output=DOWNLOADS/'microduck-stage09-cloud-inputs'
    output.mkdir(parents=True,exist_ok=False)
    subprocess.run(['git','bundle','create',str(output/'source.bundle'),'double-balance'],cwd=REPO,check=True)
    shutil.copy2(report_path,output/'local-preflight.json')
    local_eval=json.loads((Path(pointer['directory'])/'nominal/evaluation.json').read_text())
    shutil.copy2(local_eval['checkpoint'],output/'primary.pt')
    datasets=Path(local_eval['initial_states']['path']).parent
    (output/'initial-states').mkdir()
    for protocol in ('nominal','dev','test-8201','test-8202','test-8203'):
        for suffix in ('.pt','.json'):
            shutil.copy2(datasets/(protocol+suffix),output/'initial-states'/(protocol+suffix))
    shutil.copy2(REPO/'scripts/stage09_cloud_launch.py',output/'launch.py')
    files={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}
    (output/'inputs.json').write_text(json.dumps(dict(files=files,git_head=git('rev-parse','HEAD'),
              runtime_manifest_sha256=sha(manifest),primary_sha256=report['source_sha256']),indent=2)+'\n')
    archive=output.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:tar.add(output,arcname=output.name)
    print('Stage09LocalCodeCommit='+git('rev-parse','HEAD'))
    print('Stage09CloudInputs='+str(archive))
    print('Stage09CloudInputsSHA256='+sha(archive))
    print('No tag, push, cloud connection or PPO was performed by this export.')


if __name__=='__main__':main()
