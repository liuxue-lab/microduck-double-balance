"""Read-only inventory on the user's existing cloud host. Standard library only."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

REFERENCE = {'torch':'2.9.1', 'numpy':'2.4.1', 'mujoco':'3.10.0',
             'mujoco-warp':'3.8.1', 'warp-lang':'1.12.0', 'mjlab':'1.3.0',
             'mjlab-microduck':'0.1.0', 'rsl-rl-lib':'5.0.1',
             'tensordict':'0.10.0', 'better-actuator-models':'1.0.1'}
ORIGIN = 'git@github.com:liuxue-lab/microduck-double-balance.git'


def command(argv, timeout=15, extra_env=None):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0', GIT_TERMINAL_PROMPT='0')
    env.update(extra_env or {})
    try:
        p = subprocess.run(argv, capture_output=True, text=True, check=False,
                           timeout=timeout, env=env)
        return {'returncode':p.returncode, 'stdout':p.stdout[:65536],
                'stderr':p.stderr[:8192], 'truncated':len(p.stdout)>65536}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {'returncode':None, 'stdout':'', 'stderr':str(e), 'truncated':False}


def main():
    report = {'status':'CLOUD_READONLY_INVENTORY_ONLY',
              'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'hostname':platform.node(), 'platform':platform.platform(),
              'machine':platform.machine(), 'cpu_count':os.cpu_count(),
              'PPO':0,'cloud_training':0,'native_integrations':0,'native_forward':0,
              'torch_load':0,'installs':0,'training_ready':False,
              'budget_hours':24,'power_actions':0,'automatic_push':False}
    mem = Path('/proc/meminfo')
    report['memory'] = [x for x in mem.read_text().splitlines()
                        if x.startswith(('MemTotal:','MemAvailable:','SwapTotal:'))] if mem.exists() else []
    report['disk'] = []
    for name in ['/','/root/autodl-tmp','/tmp']:
        p = Path(name)
        if p.exists():
            u = shutil.disk_usage(p)
            report['disk'].append({'path':name,'total_bytes':u.total,'used_bytes':u.used,'free_bytes':u.free})
    report['gpu'] = command(['nvidia-smi', '--query-gpu=index,name,uuid,memory.total,memory.used,driver_version,utilization.gpu',
                             '--format=csv,noheader,nounits'])
    report['gpu_processes'] = command(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                                       '--format=csv,noheader,nounits'])
    report['uv'] = command(['uv','--version'])
    report['git_version'] = command(['git','--version'])
    projects = [Path('/root/autodl-tmp/microduck-double-balance'),
                Path('/root/autodl-tmp/microduck-double-balance/workspace'),
                Path('/root/microduck-double-balance')]
    report['existing_projects'] = []
    for p in projects:
        if not p.is_dir():
            continue
        row = {'path':str(p), 'git_head':command(['git','-C',str(p),'rev-parse','HEAD']),
               'git_status':command(['git','-C',str(p),'status','--porcelain=v1'])}
        for name in ['pyproject.toml','uv.lock']:
            f=p/name
            if f.is_file():
                row[name]={'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
        report['existing_projects'].append(row)
    # Metadata queries do not import torch/mujoco/warp or create a CUDA context.
    probe = '''import importlib.metadata as m, json, sys
names = %r
rows=[]
for name,reference in names.items():
    try:
        d=m.distribution(name); version=d.version; location=str(d.locate_file(''))
    except m.PackageNotFoundError:
        version=location=None
    rows.append(dict(name=name,observed=version,reference=reference,matches=version==reference,location=location))
print(json.dumps(dict(python=sys.version,executable=sys.executable,prefix=sys.prefix,packages=rows,CUDA_imported=False)))
''' % REFERENCE
    interpreters = [sys.executable] + [str(p/'.venv/bin/python') for p in projects]
    interpreters += ['/root/miniconda3/bin/python','/root/miniconda3/envs/microduck/bin/python']
    report['python_environments'] = []
    seen = set()
    for candidate in interpreters:
        if candidate in seen or not Path(candidate).is_file():
            continue
        seen.add(candidate)
        r = command([candidate,'-I','-B','-c',probe],timeout=20)
        entry={'requested_interpreter':candidate, 'result':r}
        if r['returncode']==0:
            try: entry['metadata']=json.loads(r['stdout'])
            except ValueError: entry['metadata_parse_failed']=True
        report['python_environments'].append(entry)
    ssh = 'ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=8'
    report['github_ssh_default'] = command(['git','ls-remote',ORIGIN,'refs/heads/double-balance'],
                                           timeout=25,extra_env={'GIT_SSH_COMMAND':ssh})
    old_key=Path('/root/.ssh/id_ed25519_microduck_stage09')
    report['historical_key_present']=old_key.is_file()
    if report['github_ssh_default']['returncode']!=0 and old_key.is_file():
        # Reuse the previously documented identity; never read or return key bytes.
        report['github_ssh_existing_stage09_identity']=command(
            ['git','ls-remote',ORIGIN,'refs/heads/double-balance'],timeout=25,
            extra_env={'GIT_SSH_COMMAND':ssh+' -o IdentitiesOnly=yes -i /root/.ssh/id_ed25519_microduck_stage09'})
    report['interpretation']=[
        'This inventory is not a cloud environment or native acceptance PASS.',
        'Image labels are not actual venv versions; use the metadata rows.',
        'PPO and cloud training remain zero; no old checkpoint was loaded.',
        'No existing project, environment, archive or started marker was changed.',
        'No GPU provisioning, push, PR, merge, installation, rollout or native forward.']
    print('STAGE12_INVENTORY_JSON_BEGIN')
    print(json.dumps(report,ensure_ascii=False,sort_keys=True))
    print('STAGE12_INVENTORY_JSON_END')


if __name__=='__main__':
    main()
