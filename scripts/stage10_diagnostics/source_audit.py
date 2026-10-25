#!/usr/bin/env python3
"""Collect installed simulation source only: no framework import, env or GPU call."""
from pathlib import Path
from datetime import datetime, timezone
from importlib import metadata
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import uuid
import zipfile

REPO=Path('/home/lx/microduck-double-balance/workspace')
ART=REPO.parent/'artifacts/double-balance-stage10'
BASE='00e34c2038771c5d4ad49fe45dff828c0232e60c'
RUNTIME_MANIFEST='b61850afbd6fa0deccb3a23a098a910aae74bc0e9b01cca2ee0634c2d1f6ecfd'
BATCH='20261001T141949Z-37e84aba'
PROVENANCE={'plan.json': 'd71fb83db07bd976000ba05b485cfe1184cc12685290019478ca56748bb87806', '02-initialization/attempt-8181eb75/result.json': '7ea47cefc7adeca23ab142359362d143b0993aa0e44fe4ee23d14cd73da428e7', '03-initialization/attempt-41fdfd96/result.json': '82ab0d0bd16a8434c5628b3a648274a790c1e52862973e1f799cec5009050b8a', '02-initialization/attempt-8181eb75/dependency-sources.json': '3b51c54ef894b9e62e83e6f01f15d59a26d2a3790252310beae1bc86e817428a', '02-initialization/attempt-8181eb75/capture-schema.json': '5693a119cb9e39dc124823124c3e4a4efbdfd1bd0437304f9384ba651050ffeb', 'initialization-recovery-v1/run.json': 'dc0d132c2b3b951ae2f93ee8c63b8ea996ae7dca1795fd761585a349cfcca666', 'initialization-recovery-v1/comparison.json': '8973ebd0e4f31ead7cf256c7236e94b0329ae676a231b09261ccbccc52060d5c'}
PACKAGES=(('mujoco-warp','mujoco_warp','3.8.1'),('better-actuator-models','bam',None),
          ('mjlab','mjlab','1.3.0'),('warp-lang','warp','1.12.0'))
MAX_SOURCE_BYTES=24*1024**2
MAX_SOURCE_FILES=1000

def require(ok,message):
    if not ok:raise RuntimeError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def verify_source():
    def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args],text=True).strip()
    require(git('rev-parse','--show-toplevel')==str(REPO),'Unexpected local repository')
    require(git('branch','--show-current')=='double-balance' and git('rev-parse','HEAD')==BASE,'Source branch/HEAD changed')
    require(not git('diff','HEAD','--name-only'),'Tracked source edits found; preserve and review')
    manifest=REPO/'docs/audits/stage-09-runtime-hashes.json'
    require(sha(manifest)==RUNTIME_MANIFEST,'Frozen source manifest changed')
    rows=read(manifest);require(len(rows)==101,'Frozen file count changed')
    for rel,digest in rows.items():
        p=Path(rel);require(not p.is_absolute() and '..' not in p.parts,'Unsafe frozen path')
        require(sha(REPO/p)==digest,'Frozen source changed: '+rel)
    return {'source_head':BASE,'frozen_files':101}

def selected(relative,package):
    p=Path(str(relative))
    if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0]!=package:return False
    if p.suffix!='.py' or 'tests' in p.parts or p.name.endswith('_test.py') or p.name.startswith('test_'):return False
    if package=='mujoco_warp':return len(p.parts)==2 or p.parts[1]=='_src'
    if package=='bam':return True
    if package=='mjlab':return len(p.parts)>2 and p.parts[1] in ('sim','envs','entity','scene','managers','actuator')
    if package=='warp':return str(p) in ('warp/__init__.py','warp/config.py','warp/_src/config.py','warp/_src/context.py','warp/_src/codegen.py','warp/_src/build.py')
    return False

def collect_package(dist_name,package,expected,output,budget):
    dist=metadata.distribution(dist_name)
    item={'distribution':dist_name,'import_package':package,'version':dist.version,
          'expected_version':expected,'version_matches':expected is None or dist.version==expected,
          'framework_imported':False,'files':[],'gaps':[]}
    paths={}
    for relative in dist.files or []:
        if selected(relative,package):paths[str(relative)]=Path(dist.locate_file(relative))
    if not paths:
        # Bounded fallback for installations without a populated RECORD.
        package_root=Path(dist.locate_file(package))
        if package_root.is_dir():
            for p in package_root.rglob('*.py'):
                relative=str(Path(package)/p.relative_to(package_root))
                if selected(relative,package):paths[relative]=p
    for relative,source in sorted(paths.items()):
        require(source.is_file(),'Missing installed source: '+str(source))
        size=source.stat().st_size
        require(size<=4*1024**2,'Unexpectedly large source file: '+relative)
        require(budget['bytes']+size<=MAX_SOURCE_BYTES and budget['files']<MAX_SOURCE_FILES,'Source collection cap reached')
        data=source.read_bytes();digest=hashlib.sha256(data).hexdigest()
        require(sha(source)==digest,'Installed source changed during read: '+relative)
        target=output/'installed'/dist_name/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write(data)
        item['files'].append({'path':str(target.relative_to(output)),
                              'installed_path':str(source.resolve()),'sha256':digest,'size_bytes':len(data)})
        budget['bytes']+=len(data);budget['files']+=1
    if not paths:item['gaps'].append('No readable Python source in distribution RECORD or its package directory')
    if package=='mujoco_warp':
        for required in ('forward.py','smooth.py','solver.py','constraint.py','types.py'):
            if not any(Path(p).name==required for p in paths):item['gaps'].append('Missing '+required)
    return item

def package_output(folder):
    rows=[{'path':str(p.relative_to(folder)),'sha256':sha(p),'size_bytes':p.stat().st_size}
          for p in sorted(folder.rglob('*')) if p.is_file()]
    write(folder/'files.json',rows)
    archive=folder.parent/(folder.name+'-review.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(folder.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(folder))
    require(archive.stat().st_size<49*1024**2,'Source review ZIP exceeds upload limit')
    print('Stage10ReviewZIP='+str(archive),flush=True)
    print('Stage10ReviewSHA256='+sha(archive),flush=True)
    print('Stage10ReviewParts=1',flush=True)

def main():
    require(len(sys.argv)==1,'Run this source collector without arguments')
    python=REPO/'.venv/bin/python'
    require(python.is_file(),'Existing project venv missing; no dependency installation')
    if Path(sys.prefix).resolve()!=(REPO/'.venv').resolve():
        os.execv(str(python),[str(python),'-u',str(Path(__file__).resolve())])
    verify_source()
    with (ART/'local-diagnostic.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another Stage 10 local diagnostic is active')
        home=ART/'source-audit';home.mkdir(exist_ok=True)
        folder=home/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
        folder.mkdir(exist_ok=False)
        print('Stage10Output='+str(folder),flush=True)
        report={'status':'RUNNING','source_head':BASE,'packages':[],
                'environments_constructed':0,'cuda_initialized':False,'new_ppo_updates':0,'rollout_steps':0,
                'cloud_contacted':False,'cloud_status':'OFF_CONFIRMED_BY_USER','stage10_complete':False,
                'task_physics_rewards_acceptance_changed':False,'provenance_batch':BATCH}
        try:
            (folder/'collector.py').write_bytes(Path(__file__).read_bytes())
            batch=ART/'initialization'/BATCH
            for relative,digest in PROVENANCE.items():
                source=batch/relative
                require(sha(source)==digest,'Reviewed C evidence changed: '+relative)
                target=folder/'prior-evidence'/relative;target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(source.read_bytes())
            budget={'bytes':0,'files':0}
            for dist_name,package,expected in PACKAGES:
                try:report['packages'].append(collect_package(dist_name,package,expected,folder,budget))
                except metadata.PackageNotFoundError:
                    report['packages'].append({'distribution':dist_name,'gaps':['Distribution metadata unavailable']})
            # Only file and distribution metadata reads; no import of these frameworks.
            require(not any(n in sys.modules for n in ('torch','warp','mujoco','mujoco_warp','mjlab','bam')),
                    'A framework was unexpectedly imported')
            for package in report['packages']:
                for row in package.get('files',[]):
                    require(sha(row['installed_path'])==row['sha256'],'Installed source changed before finish')
            report['final_source_check']=verify_source()
            report['source_budget_used']=budget
            gap=any(p.get('gaps') or not p.get('version_matches',False) for p in report['packages'])
            report['status']='SOURCE_EVIDENCE_WITH_GAPS_REVIEW_REQUIRED' if gap else 'SOURCE_EVIDENCE_COLLECTED_REVIEW_REQUIRED'
        except BaseException as exc:
            report.update(status='STOPPED_REVIEW_REQUIRED',error=type(exc).__name__+': '+str(exc))
            raise
        finally:
            report['finished_utc']=datetime.now(timezone.utc).isoformat()
            write(folder/'result.json',report)
            package_output(folder)
            print('Stage10Batch='+report['status']+'; stage not complete; zero environments; zero PPO',flush=True)

if __name__=='__main__':
    try:main()
    except (Exception,KeyboardInterrupt) as exc:
        print('Stage10Stopped='+type(exc).__name__+': '+str(exc),flush=True)
        raise SystemExit(1)
