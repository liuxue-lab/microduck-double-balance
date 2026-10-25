#!/usr/bin/env python3
"""Run only laptop tests/inference and package evidence. Does not commit or use cloud."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid

REPO=Path('/home/lx/microduck-double-balance/workspace')
CLOUD=Path('/root/autodl-tmp/microduck-double-balance/artifacts/double-balance-stage08')
PRIMARY='86d55c3703c18fcf4817db49c6e4dcf839b3f5195d68ae2c559db7e222bded97'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if Path(__file__).resolve().parents[1]!=REPO:raise ValueError('Use the canonical laptop checkout')
    subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_stage09*.py','-v'],cwd=REPO,check=True)
    receipt=json.loads((REPO/'docs/audits/stage-08-return-verified.json').read_text())
    root=Path(receipt['extracted_directory'])
    if not root.is_dir():raise ValueError('Verified Stage 08 return directory is missing')
    def mapped(value):
        path=(root/Path(value).relative_to(CLOUD)).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():raise ValueError('Missing returned evidence: '+str(path))
        return path
    final=json.loads((REPO/'docs/audits/stage-08-final-evaluation-review.json').read_text())
    primary=next(m for m in final['models'] if m['identity']['sha256']==PRIMARY)
    checkpoint=mapped(primary['identity']['path'])
    if sha(checkpoint)!=PRIMARY:raise ValueError('Primary checkpoint checksum differs')
    primary_reports={}
    for protocol in ('nominal','dev','test-8201','test-8202','test-8203'):
        reference=primary['protocols'][protocol]['report_reference']
        report_path=mapped(reference['path'])
        if sha(report_path)!=reference['sha256']:raise ValueError('Frozen cloud report differs: '+protocol)
        old=json.loads(report_path.read_text());dataset=mapped(old['initial_states']['path'])
        if sha(dataset)!=old['initial_states']['sha256']:raise ValueError('Frozen initial states differ: '+protocol)
        primary_reports[protocol]=dict(path=str(report_path),sha256=reference['sha256'])
        if protocol=='nominal':datasets=dataset.parent
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    folder=REPO.parent/'artifacts/double-balance-stage09/preparation'/stamp
    inputs=folder.parent/(stamp+'-inputs');inputs.mkdir(parents=True,exist_ok=False)
    selection=json.loads(json.dumps(final['selection']))
    for item in selection['checkpoints']:item['path']=str(mapped(item['path']))
    for item in selection['development_reports']:
        path=mapped(item['path'])
        if sha(path)!=item['sha256']:raise ValueError('Original selection evidence differs')
        item['path']=str(path)
    selection['path_mapping_only']=True;selection['original_selection_sha256']=final['selection_sha256']
    selection_path=inputs/'old-selection-local-paths.json';selection_path.write_text(json.dumps(selection,indent=2)+'\n')
    jobs=[]
    for protocol in ('test-8201','test-8202','test-8203'):
        jobs.append(dict(protocol=protocol,name='old-regression-'+protocol,checkpoint=str(checkpoint),
                         selection=str(selection_path),video=219 if protocol=='test-8202' else None,
                         cloud_report=primary_reports[protocol]))
    for model in final['models']:
        identity=model['identity']
        if identity.get('profile')=='E' and identity['sha256']!=PRIMARY:
            cp=mapped(identity['path'])
            if sha(cp)!=identity['sha256']:raise ValueError('Other E checkpoint differs')
            reference=model['protocols']['dev']['report_reference'];cloud=mapped(reference['path'])
            if sha(cloud)!=reference['sha256']:raise ValueError('Other E cloud report differs')
            jobs.append(dict(protocol='dev',name='other-E-'+str(identity['seed']),checkpoint=str(cp),video=0,
                             cloud_report=dict(path=str(cloud),sha256=reference['sha256'])))
    jobs_path=inputs/'extra-jobs.json'
    jobs_path.write_text(json.dumps(dict(jobs=jobs,primary_reports=primary_reports),indent=2)+'\n')
    command=[sys.executable,'-u','-m','mjlab_microduck.double_balance_stage09','local-check',
             '--checkpoint',str(checkpoint),'--datasets',str(datasets),'--output',str(folder),'--extra-jobs',str(jobs_path)]
    subprocess.run(command,cwd=REPO,check=True)
    pointer=folder.parent/'latest.json'
    pointer.write_text(json.dumps(dict(directory=str(folder),report=str(folder/'local-preflight.json'),
                                      review_zip=str(folder.with_suffix('.zip'))),indent=2)+'\n')


if __name__=='__main__':main()
