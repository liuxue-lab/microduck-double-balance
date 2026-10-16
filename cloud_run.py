"""Cloud preparation, sequential native gates and complete return preparation.

Execution requires the local launcher to create and pin a dedicated Git commit.
No automatic Git push, PPO, checkpoint load, instance creation or power action.
"""
from pathlib import Path
import argparse
import base64
import datetime
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
import zipfile
import io
import gzip

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))
from common import *
from native_stage import POINTS,point_plan,reconcile


def old_identity():
    repo=Path('/root/autodl-tmp/microduck-double-balance/workspace')
    return dict(head=command(['git','-C',repo,'rev-parse','HEAD']),
                status=command(['git','-C',repo,'status','--porcelain=v1']),
                pyproject=info(repo/'pyproject.toml'),lock=info(repo/'uv.lock'),
                python_venv=info(repo/'.venv/pyvenv.cfg'))


def host_gate():
    require(sys.platform=='linux' and os.getuid()==0,'Run on the supplied root cloud account')
    require(Path('/root/autodl-tmp').is_dir(),'Expected cloud data disk')
    gpu=command(['nvidia-smi','--query-gpu=uuid,name,driver_version,memory.total','--format=csv,noheader,nounits'])
    require(len(gpu.splitlines())==1 and GPU_UUID in gpu and 'RTX 5090' in gpu,'Inventoried GPU identity differs')
    processes=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'])
    require(not processes.strip(),'Existing GPU process present; preserve it and stop')
    return dict(gpu=gpu,processes=processes,affinity_cpu_count=len(os.sched_getaffinity(0)),
                cpu_cgroup_max=(Path('/sys/fs/cgroup/cpu.max').read_text().strip() if Path('/sys/fs/cgroup/cpu.max').exists() else None),
                memory_cgroup_max=(Path('/sys/fs/cgroup/memory.max').read_text().strip() if Path('/sys/fs/cgroup/memory.max').exists() else None),
                disk={p:shutil.disk_usage(p)._asdict() for p in ('/','/root/autodl-tmp')})


def logged(argv, work, label, env, timeout):
    logfile=work/'logs'/(label+'.log')
    require(not logfile.exists(),'Existing command log retained: '+label)
    limit=min(time.time()+timeout,CAMPAIGN_DEADLINE-RETURN_RESERVE_SECONDS)
    require(limit>time.time(),'Campaign deadline exhausted before '+label)
    with logfile.open('xb') as log:
        p=subprocess.Popen([str(x) for x in argv],cwd=work,env=env,stdin=subprocess.DEVNULL,
                           stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(work/'phases'/(label+'.process.json'),dict(pid=p.pid,started_epoch=time.time(),deadline=limit,argv=[str(x) for x in argv]))
        heartbeat=0
        while p.poll() is None:
            if time.time()>=limit:
                os.killpg(p.pid,signal.SIGINT)
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGTERM)
                    try:p.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
                raise RuntimeError('Command time bound reached; attempts retained: '+label)
            if time.monotonic()-heartbeat>=15:
                print('Stage12CloudPhase='+label+'; elapsed process running; log='+str(logfile),flush=True)
                heartbeat=time.monotonic()
            time.sleep(1)
    require(p.returncode==0,'Command failed; inspect '+str(logfile)+'; returncode='+str(p.returncode))
    return dict(log=str(logfile),log_sha256=sha(logfile),returncode=p.returncode)


def setup(work, runtime, retry=False):
    phases=work/'phases'
    source=preparation_once(phases,'source',lambda label:dict(files=len(extract_source(ROOT/'cloud-inputs.zip',work/'inputs')['files']),sha256=SOURCE_SHA),retry)
    project=work/'inputs/project'
    env=dict(os.environ,UV_PYTHON_INSTALL_DIR=str(runtime/'python'),UV_PROJECT_ENVIRONMENT=str(runtime/'venv'),
             UV_CACHE_DIR=str(runtime/'cache'),UV_LINK_MODE='hardlink',UV_COMPILE_BYTECODE='0',
             PYTHONDONTWRITEBYTECODE='1',GIT_TERMINAL_PROMPT='0')
    # All newly installed runtime data resides on the mostly empty system disk.
    # The retained old environment and data disk are never removed or modified.
    def install_uv(label):
        # A partial tool install can be repaired in a new target while keeping
        # all previous attempted files. It never replaces the retained venv.
        require(shutil.disk_usage(runtime).free>=24*GIB,'Need 24 GiB free on system disk before fresh locked runtime installation')
        tool=runtime/label
        r=logged(['/root/miniconda3/bin/python','-m','pip','install','--disable-pip-version-check','--no-cache-dir',
                  '--target',tool,'uv==0.13.1'],work,label,env,600)
        found=[p for p in (tool/'bin/uv',tool/'uv/uv') if p.is_file() and os.access(p,os.X_OK)]
        require(len(found)==1,'Pinned uv executable not found')
        require(command([found[0],'--version']).startswith('uv 0.13.1'),'Pinned uv version differs')
        return dict(**r,executable=str(found[0]),sha256=sha(found[0]))
    uv=preparation_once(phases,'uv-tool',install_uv,retry)
    require(sha(uv['executable'])==uv['sha256'],'uv binary changed')
    def sync(label):
        before={n:info(project/n) for n in ('uv.lock','pyproject.toml')}
        r=logged([uv['executable'],'sync','--locked','--no-dev','--python','3.12.14','--project',project],work,label,env,5400)
        require(before=={n:info(project/n) for n in before},'uv changed locked source')
        return dict(**r,lock=before,python=str(runtime/'venv/bin/python'))
    installed=preparation_once(phases,'locked-runtime',sync,retry)
    python=Path(installed['python']); require(python.is_file(),'Completed runtime interpreter missing')
    checkenv=native_env()
    def verify_runtime(label):
        out=work/'runtime-checks'/label;out.mkdir(parents=True)
        return logged([python,'-I','-B',ROOT/'runtime_check.py','--repo',project,'--out',out],work,label,checkenv,600)
    checked=preparation_once(phases,'runtime-check',verify_runtime,retry)
    return dict(python=str(python),project=str(project),source=source,uv=uv,locked_install=installed,runtime_check=checked)


def feasibility(point, prior, work, now=None):
    now=time.time() if now is None else now
    n=len(point['owners']); steps=point['public_steps_per_owner']; reasons=[]
    if steps==3:
        estimate_seconds=3600
        projected_bytes=n*steps*10*64*1024**2
    else:
        key=('n1-' if n==1 else 'n2-')+('rollout' if steps==500 else 'pilot')
        require(key in prior,'Required throughput point missing: '+key)
        prev=prior[key]
        require(prev['status']=='NATIVE_POINT_PASS','Predecessor not accepted')
        estimate_seconds=600+prev['maximum_step_seconds']*steps*1.75
        per_transition=prev['evidence_bytes']/max(1,prev['public_steps_per_owner']*n)
        # Space gate estimates four retained copies (raw, ZIP/ASCII and Git),
        # then the unchanged live 8 GiB guard stops before disk exhaustion.
        projected_bytes=int(per_transition*n*steps*4)
    seconds_left=CAMPAIGN_DEADLINE-RETURN_RESERVE_SECONDS-now
    free=shutil.disk_usage(work).free
    if estimate_seconds>seconds_left:reasons.append('REMAINING_TOTAL_TIME_INSUFFICIENT')
    if projected_bytes+8*GIB>free:reasons.append('EVIDENCE_AND_RETURN_DISK_INSUFFICIENT')
    return dict(admitted=not reasons,reasons=reasons,estimated_seconds=estimate_seconds,
                estimated_disk_bytes=projected_bytes,actual_free_bytes=free,remaining_native_seconds=seconds_left,
                estimates_are_not_guarantees=True,no_threshold_change=True,no_evidence_discard=True)


def execute(work, runtime, retry_setup=False):
    require(time.time()<CAMPAIGN_DEADLINE-RETURN_RESERVE_SECONDS,'Fixed 24-hour campaign deadline exhausted')
    if not (work/'identity-before.json').exists():
        host=host_gate(); old=old_identity()
        expected=read(ROOT/'inventory/cloud-inventory.json')['existing_projects'][1]
        require(old['head']==expected['git_head']['stdout'].strip() and not old['status'],'Retained source checkout differs from inventory')
        write(work/'identity-before.json',dict(old=old,host=host,package_sha256=verify(ROOT)))
    else:
        require(read(work/'identity-before.json')['old']==old_identity(),'Retained source changed; stop')
        host_gate()
    setup_result=setup(work,runtime,retry_setup)
    prior={}; skipped=[]
    env=native_env()
    for name,_,_ in POINTS:
        point=point_plan(name)
        state=phase_state(work/'phases',name)
        if state=='DONE':
            r=read(work/'reports'/(name+'.json'))
            require(info(work/'reports'/(name+'.json'))==read(work/'phases'/(name+'.done.json'))['result']['report_identity'],'Completed report changed')
            prior[name]=r
            if r['status']!='NATIVE_POINT_PASS':break
            continue
        require(state=='NOT_STARTED','Native point incomplete; preserve its WAL; no replay: '+name)
        decision=feasibility(point,prior,work)
        admission=work/'reports'/(name+'-admission.json')
        if admission.exists():
            # No native phase started: keep the old estimate and record the
            # current (smaller) remaining budget before continuing.
            admission=work/'reports'/(name+'-admission-'+str(time.time_ns())+'.json')
        write(admission,decision)
        if not decision['admitted']:
            skipped.append(dict(point=name,**decision));break
        def native_action():
            stage_deadline=min(CAMPAIGN_DEADLINE-RETURN_RESERVE_SECONDS,time.time()+max(3600,decision['estimated_seconds']))
            newenv=dict(env,MD12_NATIVE_DEADLINE=str(stage_deadline))
            logged([setup_result['python'],'-I','-B',ROOT/'native_stage.py','--repo',setup_result['project'],
                    '--work',work,'--point',name,'--deadline',str(stage_deadline)],work,name,newenv,
                   max(1,stage_deadline-time.time()+120))
            r=read(work/'reports'/(name+'.json'))
            require(not r.get('errors') and r['accounting']['all_ledgers_exact'],'Native report incomplete')
            return dict(report_identity=info(work/'reports'/(name+'.json')),status=r['status'])
        run_once(work/'phases',name,native_action)
        prior[name]=read(work/'reports'/(name+'.json'))
        if prior[name]['status']!='NATIVE_POINT_PASS':break
    return dict(setup=setup_result,points=[dict(name=n,status=r['status']) for n,r in prior.items()],skipped=skipped)


def campaign_report(work, result=None, error=None):
    points=[]; totals={k:0 for k in ('attempted_lower','attempted_upper','completed_lower','completed_upper')}
    for name,_,_ in POINTS:
        p=work/'reports'/(name+'.json')
        if p.exists():
            r=read(p)
        elif (work/'native'/name).exists():
            r=dict(status='INTERRUPTED_PRESERVE',point=name,accounting=reconcile(work/'native'/name,point_plan(name)))
        else:continue
        for k in totals:totals[k]+=r['accounting']['native_added'][k]
        points.append({k:v for k,v in r.items() if k not in ('final','closeout','review')})
    preservation=False
    if (work/'identity-before.json').exists():
        preservation=read(work/'identity-before.json')['old']==old_identity()
    full=any(p.get('full_10s_episode_proven') for p in points)
    passes=sum(x['status']=='NATIVE_POINT_PASS' for x in points)
    return dict(status='CLOUD_NATIVE_ACCEPTANCE_COMPLETE_PPO_BUDGET_PENDING' if full and passes==5 and preservation and not error
                else 'CLOUD_READINESS_INCOMPLETE_PRESERVE',error=error,execution=result,points=points,
                completed_native_points=passes,planned_native_points=5,full_10s_episode_proven=full,
                canonical_before=BASELINE,native_added=totals,
                cumulative_completed=[BASELINE['completed']+totals['completed_lower'],BASELINE['completed']+totals['completed_upper']],
                cumulative_attempted=[BASELINE['attempted']+totals['attempted_lower'],BASELINE['attempted']+totals['attempted_upper']],
                PPO=0,cloud_training=0,torch_load=0,old_local_batches_rerun=0,old_checkpoint='PRESERVED_NO_LOAD',
                training_ready=False,source_preservation=preservation,deadline=CAMPAIGN_DEADLINE,
                finished_epoch=time.time(),campaign_elapsed_seconds=time.time()-read(work/'started.json')['started_epoch'],
                instance_power_state='UNCONFIRMED_AFTER_INVENTORY',automatic_power_actions=0,automatic_push=False,
                outstanding=['Bounded actual PPO/checkpoint save and restore need explicit update budget',
                             'Formal training remains unreleased','N>2 and native batched nworld>1 unreleased'],
                source_commit=command(['git','-C',ROOT,'rev-parse','HEAD']))


def pack_evidence(work, transfer):
    # Only new campaign evidence. Runtime wheels/caches and old cloud archives are
    # excluded explicitly; original Stage09 full traces remain at their old paths.
    roots=['phases','reports','logs','native','runtime-checks']
    files=[]
    for name in roots:
        p=work/name
        if p.exists():
            for f in sorted(p.rglob('*')):
                require(not f.is_symlink(),'Linked evidence refused')
                if f.is_file():files.append(f)
    for n in ('started.json','identity-before.json','campaign-report.json'):
        if (work/n).is_file():files.append(work/n)
    records={f.relative_to(work).as_posix():info(f) for f in files}
    archive_dir=work/'return-archives';archive_dir.mkdir(exist_ok=True)
    # WAL files can exceed 49 MiB. Split their BYTES into indexed 8 MiB
    # segments; reassemble and verify the original file hash on the laptop.
    # No WAL record is removed, truncated, compressed in place or regenerated.
    members={};groups=[];group=[];size=0
    for file_index,f in enumerate(files):
        total=f.stat().st_size
        for j,offset in enumerate(range(0,max(1,total),8*1024**2)):
            length=min(8*1024**2,total-offset)
            name='segments/%06d/%06d.bin'%(file_index,j)
            if size+length>40*1024**2 and group:
                groups.append(group);group=[];size=0
            group.append((f,offset,length,name));size+=length
    if group:groups.append(group)
    archives=[]
    for i,group in enumerate(groups):
        target=archive_dir/('evidence-%04d.zip'%i)
        if not target.exists():
            tmp=target.with_name(target.name+'.partial-'+str(time.time_ns()))
            with zipfile.ZipFile(tmp,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
                for f,offset,length,name in group:
                    with f.open('rb') as stream:stream.seek(offset);raw=stream.read(length)
                    require(len(raw)==length,'Evidence changed while packaging')
                    zi=zipfile.ZipInfo(name,date_time=(2026,10,16,0,0,0));zi.compress_type=zipfile.ZIP_DEFLATED
                    z.writestr(zi,raw)
            require(tmp.stat().st_size<49*1024**2,'49 MiB archive bound exceeded')
            os.link(tmp,target);tmp.unlink()
        with zipfile.ZipFile(target) as z:
            require(set(z.namelist())=={name for f,offset,length,name in group},'Archive roster differs')
            for f,offset,length,n in group:
                with f.open('rb') as stream:stream.seek(offset);expected=stream.read(length)
                raw=z.read(n);require(raw==expected,'Archived evidence differs')
                members[n]=dict(path=f.relative_to(work).as_posix(),offset=offset,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        parts=[]
        with target.open('rb') as f:
            j=0
            for raw in iter(lambda:f.read(384*1024),b''):
                name='raw/%s.part-%04d.b64'%(target.stem,j);b=base64.b64encode(raw)
                same_or_new(transfer/name,b)
                parts.append(dict(path=name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()));j+=1
        archives.append(dict(filename=target.name,**info(target),parts=parts))
    require(records=={f.relative_to(work).as_posix():info(f) for f in files},'Evidence changed while packaging; no return completion')
    index=dict(files=records,members=members,file_count=len(records),raw_bytes=sum(r['bytes'] for r in records.values()),archives=archives,
               included='All new native boundary/microstep evidence, WAL, profiles, setup logs and receipts',
               excluded=['Reproducible new package/source copy already in input commit','Installed wheels/interpreter and compiler caches',
                         'Live runner/resume supervisor console; command output logs and durable reports are included',
                         'Historical Stage09 full traces remain cloud-only; unchanged, not newly backed up'],
               native_inner_raw_scope='Inherited G11 scope: raw microstep qpos/qvel/time and full macro boundaries; inner full Data not newly captured',
               all_new_native_files_included=True)
    # A full episode can have >100k evidence files. Keep its JSON index out of
    # the GitHub large-text API path as well: gzip, hash, then <=512 KiB ASCII.
    index_raw=encoded(index);compressed=gzip.compress(index_raw,mtime=0);parts=[]
    for j,offset in enumerate(range(0,len(compressed),384*1024)):
        raw=base64.b64encode(compressed[offset:offset+384*1024]);name='index/part-%04d.b64'%j
        same_or_new(transfer/name,raw)
        parts.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    storage=dict(format='stage12-evidence-index-gzip-parts-v1',json_bytes=len(index_raw),
                 json_sha256=hashlib.sha256(index_raw).hexdigest(),gzip_bytes=len(compressed),
                 gzip_sha256=hashlib.sha256(compressed).hexdigest(),file_count=len(records),
                 raw_bytes=index['raw_bytes'],archive_count=len(archives),parts=parts)
    same_or_new(transfer/'evidence-index-storage.json',encoded(storage))
    return index


def ensure_no_live_workers(work):
    candidates=[]
    for p in (work/'phases').glob('*.process.json'):
        r=read(p);candidates.append((r['pid'],' '.join(r.get('argv',[]))))
    for p in (work/'native').glob('*/owner-*-bootstrap/runtime-paths.json'):
        r=read(p);candidates.append((r['pid'],'native-owner'))
    for pid,label in candidates:
        cmd=Path('/proc')/str(pid)/'cmdline'
        if cmd.exists():
            raw=cmd.read_bytes()
            require(not raw or b'python' not in raw and b'uv' not in raw,
                    'A recorded worker PID is still active; use --status and do not package/restart: '+str(pid)+' '+label)


def package(work, config):
    receipt=work/'return-receipt.json'
    if receipt.exists():return read(receipt)
    require((work/'campaign-report.json').exists(),'No finalized report; use --finalize for an interrupted run')
    transfer=work/'return-transfer';branch='stage12-cloud-execution-v1-return-'+config['run_id'].lower()
    prepare_git(transfer,branch,config['identity'],config['github_ssh_command'])
    pack_evidence(work,transfer)
    for n in ('campaign-report.json','started.json'):
        same_or_new(transfer/n,(work/n).read_bytes())
    same_or_new(transfer/'input-identity.json',encoded(dict(commit=config['commit'],package_sha256=config['package_sha256'],
                 inventory_commit=INVENTORY_COMMIT,local_N2_commit=LOCAL_N2_COMMIT)))
    same_or_new(transfer/'README.txt',b'Stage12 new cloud evidence. No PPO, push, PR, merge or power action. Do not merge this branch.\n')
    same_or_new(transfer/'MANIFEST.json',encoded(manifest(transfer,('MANIFEST.json',))))
    verify(transfer,'MANIFEST.json')
    commit=commit_git(transfer,'Stage12 cloud bounded acceptance return; PPO0; retain all attempts')
    row=dict(directory=str(transfer),branch=branch,commit=commit,manual_push='git -C '+str(transfer)+' push origin HEAD')
    write(receipt,row);return row


def status(work):
    r=dict(work=str(work),PPO=0,cloud_training=0,deadline=CAMPAIGN_DEADLINE)
    r['phases']={p.name:read(p) for p in sorted((work/'phases').glob('*.done.json'))}
    r['started_without_matching_done']=[p.name for p in sorted((work/'phases').glob('*.started.json'))
        if not p.with_name(p.name.replace('.started.json','.done.json')).exists()]
    failed=sorted((work/'reports').glob('setup-interrupted-*.json'))
    if failed:r['last_setup_interruption']=read(failed[-1])
    for n in ('campaign-report.json','return-receipt.json'):
        if (work/n).exists():r[n]=read(work/n)
    if (work/'started.json').exists():
        started=read(work/'started.json')
        end=r.get('campaign-report.json',{}).get('finished_epoch',time.time())
        r['campaign_elapsed_seconds']=end-started['started_epoch']
        r['remaining_native_seconds']=max(0,CAMPAIGN_DEADLINE-RETURN_RESERVE_SECONDS-time.time())
    r['progress']=[]
    for p in sorted((work/'native').glob('*/owner-*/progress.json')):
        try:
            x=read(p);r['progress'].append(dict(path=str(p),phase=x['phase'],elapsed_s=x['elapsed_s'],
                macros=x['macro_completed'],public_steps=x['formal_steps_completed'],
                integrations_completed=x['counts'].get('integrations_completed',0),
                integrations_attempted=x['counts'].get('integrations_attempted',0)))
        except (ValueError,KeyError):pass
    print(json.dumps(r,ensure_ascii=False,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('action',choices=['run','resume','status','package-only','finalize'])
    a=p.parse_args();config=read(a.config);work=Path(config['work']);runtime=Path(config['runtime'])
    if a.action=='status':status(work);return 0
    work.mkdir(parents=True,exist_ok=True);runtime.mkdir(parents=True,exist_ok=True)
    lock=(work/'process.lock').open('a+b')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ensure_no_live_workers(work)
    for n in ('phases','reports','logs','native'):(work/n).mkdir(exist_ok=True)
    require(command(['git','-C',ROOT,'rev-parse','HEAD'])==config['commit'],'Pinned input commit differs')
    require(verify(ROOT)==config['package_sha256'],'Package differs from pinned local receipt')
    if not (work/'started.json').exists():
        write(work/'started.json',dict(config=config,started_epoch=time.time(),deadline=CAMPAIGN_DEADLINE,canonical_before=BASELINE,
              max_integrations=sum(point_plan(n)['max_integrations'] for n,_,_ in POINTS),PPO=0,cloud_training=0,no_retry=True))
    if a.action=='package-only':
        print(json.dumps(package(work,config),indent=2));return 0
    if (work/'campaign-report.json').exists():
        print(json.dumps(package(work,config),indent=2));status(work);return 0
    result=None;error=None
    if a.action in ('run','resume'):
        try:result=execute(work,runtime,retry_setup=a.action=='resume')
        except BaseException as exc:
            error=dict(error=repr(exc),traceback=traceback.format_exc())
    else:error=dict(error='EXPLICIT_FINALIZE_INTERRUPTED_CAMPAIGN_NO_NATIVE_RESTART')
    # Reversible setup failures do not freeze the campaign as a native attempt.
    # A user can --resume installation from its cache, keeping every old receipt.
    native_claimed=any((work/'phases'/(n+'.started.json')).exists() for n,_,_ in POINTS)
    if error and not native_claimed and a.action!='finalize':
        snap=work/'reports'/('setup-interrupted-'+str(time.time_ns())+'.json')
        write(snap,campaign_report(work,result,error))
        print('Stage12Cloud=SETUP_INTERRUPTED_NATIVE0; use --resume for setup continuation, or --finalize to stop and package',flush=True)
        print('Stage12SetupReport='+str(snap),flush=True)
        return 1
    write(work/'campaign-report.json',campaign_report(work,result,error))
    try:print(json.dumps(package(work,config),indent=2),flush=True)
    finally:
        print('Stage12CloudTrainingReady=False; PPO=0; cloud_training=0; manual push only',flush=True)
        print('Complete manual Git SSH return and local SHA verification, then shut GPU down manually; power state unconfirmed.',flush=True)
    return int(bool(error))


if __name__=='__main__':
    raise SystemExit(main())
