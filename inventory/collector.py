#!/usr/bin/env python3
"""One SSH metadata inventory and local transfer commit; user pushes manually."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

REMOTE_PROBE = '"""Read-only inventory on the user\'s existing cloud host. Standard library only."""\nimport datetime\nimport hashlib\nimport json\nimport os\nfrom pathlib import Path\nimport platform\nimport shutil\nimport subprocess\nimport sys\n\nREFERENCE = {\'torch\':\'2.9.1\', \'numpy\':\'2.4.1\', \'mujoco\':\'3.10.0\',\n             \'mujoco-warp\':\'3.8.1\', \'warp-lang\':\'1.12.0\', \'mjlab\':\'1.3.0\',\n             \'mjlab-microduck\':\'0.1.0\', \'rsl-rl-lib\':\'5.0.1\',\n             \'tensordict\':\'0.10.0\', \'better-actuator-models\':\'1.0.1\'}\nORIGIN = \'git@github.com:liuxue-lab/microduck-double-balance.git\'\n\n\ndef command(argv, timeout=15, extra_env=None):\n    env = dict(os.environ, GIT_OPTIONAL_LOCKS=\'0\', GIT_TERMINAL_PROMPT=\'0\')\n    env.update(extra_env or {})\n    try:\n        p = subprocess.run(argv, capture_output=True, text=True, check=False,\n                           timeout=timeout, env=env)\n        return {\'returncode\':p.returncode, \'stdout\':p.stdout[:65536],\n                \'stderr\':p.stderr[:8192], \'truncated\':len(p.stdout)>65536}\n    except (OSError, subprocess.TimeoutExpired) as e:\n        return {\'returncode\':None, \'stdout\':\'\', \'stderr\':str(e), \'truncated\':False}\n\n\ndef main():\n    report = {\'status\':\'CLOUD_READONLY_INVENTORY_ONLY\',\n              \'utc\':datetime.datetime.now(datetime.timezone.utc).isoformat(),\n              \'hostname\':platform.node(), \'platform\':platform.platform(),\n              \'machine\':platform.machine(), \'cpu_count\':os.cpu_count(),\n              \'PPO\':0,\'cloud_training\':0,\'native_integrations\':0,\'native_forward\':0,\n              \'torch_load\':0,\'installs\':0,\'training_ready\':False,\n              \'budget_hours\':24,\'power_actions\':0,\'automatic_push\':False}\n    mem = Path(\'/proc/meminfo\')\n    report[\'memory\'] = [x for x in mem.read_text().splitlines()\n                        if x.startswith((\'MemTotal:\',\'MemAvailable:\',\'SwapTotal:\'))] if mem.exists() else []\n    report[\'disk\'] = []\n    for name in [\'/\',\'/root/autodl-tmp\',\'/tmp\']:\n        p = Path(name)\n        if p.exists():\n            u = shutil.disk_usage(p)\n            report[\'disk\'].append({\'path\':name,\'total_bytes\':u.total,\'used_bytes\':u.used,\'free_bytes\':u.free})\n    report[\'gpu\'] = command([\'nvidia-smi\', \'--query-gpu=index,name,uuid,memory.total,memory.used,driver_version,utilization.gpu\',\n                             \'--format=csv,noheader,nounits\'])\n    report[\'gpu_processes\'] = command([\'nvidia-smi\', \'--query-compute-apps=pid,process_name,used_gpu_memory\',\n                                       \'--format=csv,noheader,nounits\'])\n    report[\'uv\'] = command([\'uv\',\'--version\'])\n    report[\'git_version\'] = command([\'git\',\'--version\'])\n    projects = [Path(\'/root/autodl-tmp/microduck-double-balance\'),\n                Path(\'/root/autodl-tmp/microduck-double-balance/workspace\'),\n                Path(\'/root/microduck-double-balance\')]\n    report[\'existing_projects\'] = []\n    for p in projects:\n        if not p.is_dir():\n            continue\n        row = {\'path\':str(p), \'git_head\':command([\'git\',\'-C\',str(p),\'rev-parse\',\'HEAD\']),\n               \'git_status\':command([\'git\',\'-C\',str(p),\'status\',\'--porcelain=v1\'])}\n        for name in [\'pyproject.toml\',\'uv.lock\']:\n            f=p/name\n            if f.is_file():\n                row[name]={\'bytes\':f.stat().st_size,\'sha256\':hashlib.sha256(f.read_bytes()).hexdigest()}\n        report[\'existing_projects\'].append(row)\n    # Metadata queries do not import torch/mujoco/warp or create a CUDA context.\n    probe = \'\'\'import importlib.metadata as m, json, sys\nnames = %r\nrows=[]\nfor name,reference in names.items():\n    try:\n        d=m.distribution(name); version=d.version; location=str(d.locate_file(\'\'))\n    except m.PackageNotFoundError:\n        version=location=None\n    rows.append(dict(name=name,observed=version,reference=reference,matches=version==reference,location=location))\nprint(json.dumps(dict(python=sys.version,executable=sys.executable,prefix=sys.prefix,packages=rows,CUDA_imported=False)))\n\'\'\' % REFERENCE\n    interpreters = [sys.executable] + [str(p/\'.venv/bin/python\') for p in projects]\n    interpreters += [\'/root/miniconda3/bin/python\',\'/root/miniconda3/envs/microduck/bin/python\']\n    report[\'python_environments\'] = []\n    seen = set()\n    for candidate in interpreters:\n        if candidate in seen or not Path(candidate).is_file():\n            continue\n        seen.add(candidate)\n        r = command([candidate,\'-I\',\'-B\',\'-c\',probe],timeout=20)\n        entry={\'requested_interpreter\':candidate, \'result\':r}\n        if r[\'returncode\']==0:\n            try: entry[\'metadata\']=json.loads(r[\'stdout\'])\n            except ValueError: entry[\'metadata_parse_failed\']=True\n        report[\'python_environments\'].append(entry)\n    ssh = \'ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=8\'\n    report[\'github_ssh_default\'] = command([\'git\',\'ls-remote\',ORIGIN,\'refs/heads/double-balance\'],\n                                           timeout=25,extra_env={\'GIT_SSH_COMMAND\':ssh})\n    old_key=Path(\'/root/.ssh/id_ed25519_microduck_stage09\')\n    report[\'historical_key_present\']=old_key.is_file()\n    if report[\'github_ssh_default\'][\'returncode\']!=0 and old_key.is_file():\n        # Reuse the previously documented identity; never read or return key bytes.\n        report[\'github_ssh_existing_stage09_identity\']=command(\n            [\'git\',\'ls-remote\',ORIGIN,\'refs/heads/double-balance\'],timeout=25,\n            extra_env={\'GIT_SSH_COMMAND\':ssh+\' -o IdentitiesOnly=yes -i /root/.ssh/id_ed25519_microduck_stage09\'})\n    report[\'interpretation\']=[\n        \'This inventory is not a cloud environment or native acceptance PASS.\',\n        \'Image labels are not actual venv versions; use the metadata rows.\',\n        \'PPO and cloud training remain zero; no old checkpoint was loaded.\',\n        \'No existing project, environment, archive or started marker was changed.\',\n        \'No GPU provisioning, push, PR, merge, installation, rollout or native forward.\']\n    print(\'STAGE12_INVENTORY_JSON_BEGIN\')\n    print(json.dumps(report,ensure_ascii=False,sort_keys=True))\n    print(\'STAGE12_INVENTORY_JSON_END\')\n\n\nif __name__==\'__main__\':\n    main()\n'
BUDGET = {'format': 'microduck-stage12-cloud-budget-v1', 'user_confirmation': {'at': '2026-10-16T11:53:14+08:00', 'text': '保留 24小时', 'context': 'Old RTX5090 instance retained; reply to question asking allowed hours this round'}, 'instance': {'gpu_model': 'RTX 5090', 'reuse_existing': True, 'retained_confirmed_by_user': True, 'power_state': 'RUNNING_OBSERVED_USER_SCREENSHOT_20261016T115457+0800', 'current_SSH': 'ssh -p 46171 root@connect.bjb2.seetacloud.com', 'current_VRAM_and_driver': {'VRAM_GB': 32, 'driver': 'READ_ON_HOST'}, 'screenshot': {'GPU': 'RTX5090 x1', 'vCPU': 25, 'CPU': 'Intel Xeon Platinum 8470Q', 'RAM_GB': 90, 'system_disk_GB': 30, 'data_disk_GB': 50, 'data_disk_used_percent': 48.37, 'image_label': 'PyTorch 2.8.0; Python3.12; Ubuntu22.04; CUDA12.8', 'image_label_is_actual_venv_identity': False, 'price_CNY_per_hour': 2.88, 'insufficient_balance_for_24_hours': True}}, 'time_budget': {'maximum_hours': 24, 'maximum_seconds': 86400, 'scope': 'This cloud readiness campaign total; not per process, per attempt, or per restart', 'started': False, 'deadline': None, 'unused_time_need_not_be_consumed': True, 'instance_billing_clock_is_separate_from_script_clock': True, 'execution_started': False, 'instance_running_already': True, 'instance_power_on_time': 'UNKNOWN; deduct any already billed time from agreed total', 'calculated_GPU_charge_CNY_at_screenshot_rate_for_24h': 69.12}, 'authorization': {'cloud_readiness_time_confirmed': True, 'PPO_updates': 0, 'cloud_training_updates': 0, 'automatic_instance_start': False, 'automatic_instance_stop': False, 'automatic_git_push': False, 'PR': False, 'merge': False, 'native_operation_caps': 'TO_BE_FIXED_IN_EXECUTION_PLAN_BEFORE_NATIVE_START'}, 'canonical_baseline': {'completed': 704664, 'attempted': 704665}, 'local_N2_fixed_commit': 'd506be014253a2bb0439ec03b3282497a33f2e2b', 'local_N2': 'PASS_60_OF_60_INDEPENDENTLY_VERIFIED_NO_RERUN', 'local_followup_native_budget': 0, 'reuse': {'cloud_inputs_sha256': '24701df9e56face7980a1a2e0544dfebc41a4cf1b84ec4912be795d1a35a43b2', 'cloud_inputs_bytes': 16181045, 'cloud_inputs_files': 410, 'source_recapture_required': False, 'source_cloud_image_installed_and_verified': False, 'old_checkpoint': 'PRESERVE_READ_ONLY_NO_LOAD', 'archives_and_started': 'PRESERVE'}, 'execution_order': ['Read-only current host/runtime/disk inventory', 'Prepare separate Stage12 directory/environment with pinned dependencies and source hashes', 'Cloud-specific numerical/full-episode candidate with fixed operation and total-time caps', 'Measure sustained N1/N2 throughput and resource cost within remaining total budget', 'Return complete evidence via existing Git SSH transfer workflow; user manually shuts down when verified'], 'not_released': ['Existing v7 native factory remains local RTX5060-only', 'Native full 10-second episode not passed', 'Sustained cloud throughput not measured', 'PPO/update/checkpoint recovery acceptance not authorized or completed', 'Formal training not released'], 'time_allocation_rule': 'Stop early on a failed acceptance or exhausted operation/time/disk bound; reserve time for cleanup and return. No automatic retry, no reset of consumed time or attempts.', 'next_user_input': 'Run the prepared read-only collection on local host; manual Git SSH push as printed. No further SSH/address clarification needed.', 'shutdown_reminder': 'Remind promptly after necessary cloud work and verified return; never infer or automate actual shutdown', 'training_ready': False, 'cloud_started': True, 'new_native_integrations': 0, 'PPO': 0, 'cloud_training': 0, 'prior_cloud_plan': {'PPO_budget': 0, 'accepted_local_commit': 'd506be014253a2bb0439ec03b3282497a33f2e2b', 'canonical_baseline': {'attempted': 704665, 'completed': 704664}, 'checkpoint': {'direct_resume': False, 'new_policy': 'fresh actor LSTM61->256 + MLP512/256/128->14; critic85->512/256/128->1; fresh normalizers and Adam', 'old_disposition': 'PRESERVE_READ_ONLY_NO_LOAD', 'old_sha256': '86d55c3703c18fcf4817db49c6e4dcf839b3f5195d68ae2c559db7e222bded97', 'pending_attempt': 'retain consumed counts; no automatic retry or budget refund', 'resume_boundary': 'last committed update only; fresh environment/RNN boundary; never claim bitwise mid-rollout recovery', 'save_boundary': 'completed update plus durable attempted/committed budget ledger and SHA receipt'}, 'cloud_SSH': None, 'cloud_budget_approval_required_by': 'User explicit instruction: PPO=0; cloud training=0; budget separately defined', 'cloud_environment_complete': False, 'cloud_native_integration_budget': 0, 'cloud_power_state': 'UNCONFIRMED', 'cloud_training_budget': 0, 'current_v7_factory_scope': 'local RTX5060 only; cloud candidate release required', 'local_N1': 'Historical short lifecycle PASS, reused; high-tier path not certified by N1', 'local_N2': 'PASS: v7 60/60 macros independently reviewed; 64/128/256 exercised; 512 unexercised', 'local_followup_native_budget': 0, 'maximum_integrations_worksheet_only': {'24_step_rollout_N1': 360960, '24_step_rollout_N2': 721920, '500_step_episode_N1': 7520000, '500_step_episode_N2': 15040000, 'macro_with_replay': 1504, 'public_step_per_owner': 15040}, 'next_budget_fields': ['GPU model', 'maximum wall hours', 'maximum cost', 'native integration cap', 'PPO update cap', 'checkpoint/save/return scope'], 'next_information_once': ['GPU model and VRAM', 'current SSH user@host and port; no secrets', 'maximum wall time and cost/currency', 'bounded native scope and separately explicit PPO update cap; default 0'], 'no_more_local_gpu_roundtrip_required': True, 'order': ['Pin and verify approved source/assets/runtime/GPU; do not reuse a historical SSH port', 'Native numerical acceptance for the new finite-refinement candidate and full 10-second episode', 'Profile end-to-end synchronized N=1 and N=2 independent owners including audit/IPC/reset; measure memory', 'Approve a separate performance implementation if the audit runtime is too slow; preserve acceptance criteria', 'Fresh actor/critic/normalizers/Adam, 24-step rollout, bounded PPO update and update-boundary checkpoint recovery acceptance', 'Only after those gates pass: explicitly budget a formal training run'], 'requires_local_result': 'N2_NATIVE_PASS_CLOUD_BUDGET_PENDING', 'requires_local_result_satisfied': True, 'scaling': {'N4096_or_B4_released': False, 'implemented_independent_owners': [1, 2], 'native_nworld_per_owner': 1, 'recurrent_minibatches_N1': [1], 'recurrent_minibatches_N2': [1, 2], 'rollout_steps': 24}, 'shutdown': 'After necessary cloud work and verified returns, remind user to shut down manually; confirmation is separate; never start or stop the instance automatically.', 'status': 'NOT_AUTHORIZED_NOT_STARTED', 'training_ready': False}, 'cloud_started_evidence': 'User screenshot shows existing instance running; no assistant power action', 'assistant_direct_access': {'status': 'DNS_UNAVAILABLE', 'cloud_host_fault_inferred': False, 'fallback': 'User-local SSH collection and Git transfer'}}
ORIGIN = 'git@github.com:liuxue-lab/microduck-double-balance.git'
HOST = 'root@connect.bjb2.seetacloud.com'
PORT = '46171'
REPO = Path('/home/lx/microduck-double-balance/workspace')
BASE = Path('/home/lx/microduck-double-balance/artifacts/double-balance-stage12/cloud-inventory-v2')


def require(ok, message):
    if not ok: raise RuntimeError(message)


def encoded(value):
    return (json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+'\n').encode()


def write_new(path, data):
    raw=data if isinstance(data,bytes) else encoded(data)
    with path.open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())


def git(repo, *args):
    env={k:v for k,v in os.environ.items() if not k.startswith('GIT_')}
    env['GIT_OPTIONAL_LOCKS']='0'
    p=subprocess.run(['git','-c','core.hooksPath=/dev/null','-C',str(repo),*args],
                     capture_output=True,text=True,env=env,timeout=30)
    require(p.returncode==0,'git '+args[0]+': '+p.stderr.strip())
    return p.stdout.strip()


def collect_remote():
    argv=['ssh','-T','-o','ConnectTimeout=15','-o','ServerAliveInterval=20',
          '-o','ServerAliveCountMax=3','-p',PORT,HOST,'python3','-I','-B','-']
    print('Stage12CloudPhase=read-only-host-inventory; PPO=0; native=0',flush=True)
    # SSH uses the user's existing keys/config or their local terminal password
    # prompt. No password/private-key value is read by this collector.
    p=subprocess.run(argv,input=REMOTE_PROBE,text=True,stdout=subprocess.PIPE,
                     stderr=None,timeout=300)
    raw=p.stdout
    require(p.returncode==0,'SSH inventory exited '+str(p.returncode)+'; preserve output and do not start training')
    require(len(raw.encode())<=2*1024**2,'Inventory unexpectedly large')
    begin='STAGE12_INVENTORY_JSON_BEGIN\n'; end='\nSTAGE12_INVENTORY_JSON_END'
    require(raw.count(begin)==1 and raw.count(end)==1,'Cloud JSON delimiters missing or duplicated')
    result=json.loads(raw.split(begin,1)[1].split(end,1)[0])
    for k in ('PPO','cloud_training','native_integrations','native_forward','torch_load','installs','power_actions'):
        require(result[k]==0,'Inventory boundary differs: '+k)
    require(result['training_ready'] is False,'Inventory cannot release training')
    return result


def package(repo,base,run):
    receipt=run/'transfer-receipt.json'
    if receipt.exists():
        row=json.loads(receipt.read_bytes())
        require(git(Path(row['directory']),'rev-parse','HEAD')==row['commit'],'Transfer HEAD changed; preserve')
        require(not git(Path(row['directory']),'status','--porcelain=v1'),'Transfer files changed; preserve')
        return row
    require((run/'cloud-inventory.json').is_file(),'No completed inventory report; never re-query through --package-only')
    report=json.loads((run/'cloud-inventory.json').read_bytes())
    transfer=base/'transfer'
    branch='stage12-cloud-inventory-v2-transfer-'+run.name.lower()
    artifacts={'cloud-inventory.json':(run/'cloud-inventory.json').read_bytes(),
               'cloud-budget.json':encoded(BUDGET),
               'collector.py':Path(__file__).read_bytes(),
               'remote-probe.py':REMOTE_PROBE.encode(),
               'started.json':(run/'started.json').read_bytes(),
               'transfer-summary.json':encoded({'status':report.get('status'),'run':str(run),
                    'native_added':0,'PPO':0,'cloud_training':0,'GPU_hours_upper':24,
                    'canonical':{'completed':704664,'attempted':704665},
                    'fixed_local_N2_commit':'d506be014253a2bb0439ec03b3282497a33f2e2b',
                    'automatic_push':False,'training_ready':False}),
               'README.txt':('Stage12 cloud read-only inventory. No native run, installation, PPO or power action.\n'
                    'Manual Git SSH push only; never merge this transfer branch.\n'
                    'Review cloud-inventory.json and cloud-budget.json. All transferred files are listed in MANIFEST.json.\n').encode()}
    artifacts['MANIFEST.json']=encoded({name:{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
                                      for name,raw in artifacts.items()})
    if not transfer.exists(): transfer.mkdir()
    require(transfer.is_dir() and not transfer.is_symlink(),'Unsafe transfer directory')
    for name,raw in artifacts.items():
        p=transfer/name
        if p.exists(): require(p.is_file() and not p.is_symlink() and p.read_bytes()==raw,'Existing transfer file differs: '+name)
        else: write_new(p,raw)
    if not (transfer/'.git').exists():
        git(transfer,'init','--initial-branch',branch)
    require(git(transfer,'symbolic-ref','--short','HEAD')==branch,'Transfer branch differs')
    user_name=git(repo,'config','--get','user.name')
    user_email=git(repo,'config','--get','user.email')
    require(user_name and user_email,'Existing Git user identity required')
    git(transfer,'config','user.name',user_name);git(transfer,'config','user.email',user_email)
    git(transfer,'config','commit.gpgsign','false')
    remotes=git(transfer,'remote').splitlines()
    if 'origin' not in remotes: git(transfer,'remote','add','origin',ORIGIN)
    require(git(transfer,'remote','get-url','origin')==ORIGIN,'SSH remote differs')
    git(transfer,'add','--',*sorted(artifacts))
    require(set(git(transfer,'ls-files').splitlines())==set(artifacts),'Unexpected tracked transfer files; preserve and stop')
    staged=git(transfer,'diff','--cached','--name-only')
    if staged: git(transfer,'commit','-m','Stage12 read-only cloud inventory; retained RTX5090; 24h ceiling; PPO0')
    row={'directory':str(transfer),'branch':branch,'commit':git(transfer,'rev-parse','HEAD'),
         'manual_push':'git -C '+shlex.quote(str(transfer))+' push origin HEAD'}
    write_new(receipt,row)
    return row


def print_status(run):
    report=run/'cloud-inventory.json'
    print('Stage12Run='+str(run))
    if report.exists():
        d=json.loads(report.read_bytes())
        print('Stage12CloudInventory='+str(d.get('status')))
    receipt=run/'transfer-receipt.json'
    if receipt.exists():
        r=json.loads(receipt.read_bytes())
        print('Stage12TransferCommit='+r['commit'])
        print('Stage12TransferBranch='+r['branch'])
        print('Stage12UploadCommand='+r['manual_push'])
    else:
        print('Stage12Transfer=NOT_READY; use --package-only only if cloud-inventory.json exists')
    print('NativeAdded=0; Cumulative=704664 completed / 704665 attempted; PPO=0; cloud_training=0')
    print('CloudPower=USER_SCREENSHOT_RUNNING; no automatic shutdown; inventory is not training acceptance')
    if report.exists() and json.loads(report.read_bytes()).get('status')=='CLOUD_READONLY_INVENTORY_ONLY':
        print('本次只读检查已返回本机。等待下一份云端执行包期间，请手动关机并保留实例/数据；不要释放实例。')
        print('关机后如GPU被其他用户占用，重新开机可能需要等待；是否保持占卡由你决定，计入24小时总上限。')


def execute(repo,base,action):
    require(repo.is_dir(),'Canonical local source checkout missing')
    require(not base.is_symlink(),'Artifacts base is a symlink')
    if action!='collect': require(base.is_dir(),'No prior collection; use --collect')
    base.mkdir(parents=True,exist_ok=True)
    for parent in base.parents: require(not parent.is_symlink(),'Artifacts ancestor is a symlink')
    pointer=base/'RUN.json'
    if pointer.exists():
        run=Path(json.loads(pointer.read_bytes())['run'])
        require(run.parent==base and run.is_dir() and not run.is_symlink(),'Run pointer differs')
        if action=='collect':
            print('Stage12Reuse=EXISTING_INVENTORY; no repeated SSH collection')
        elif action=='package-only': package(repo,base,run)
        print_status(run)
        return 0
    require(action=='collect','No prior collection; use --collect')
    run=base/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6])
    run.mkdir()
    write_new(pointer,{'run':str(run)})
    before={'head':git(repo,'rev-parse','HEAD'),'status':git(repo,'status','--porcelain=v1')}
    write_new(run/'started.json',{'kind':'READONLY_METADATA_COLLECTION','ssh':{'host':HOST,'port':PORT},
               'local_before':before,'native_budget':0,'PPO':0,'cloud_training':0,'no_automatic_requery':True})
    try:
        result=collect_remote()
    except (OSError,RuntimeError,ValueError,KeyError,subprocess.SubprocessError,KeyboardInterrupt) as exc:
        result={'status':'SSH_INVENTORY_FAILED_PRESERVE','error':repr(exc),
                'PPO':0,'cloud_training':0,'native_integrations':0,'training_ready':False}
    after={'head':git(repo,'rev-parse','HEAD'),'status':git(repo,'status','--porcelain=v1')}
    result['local_identity_unchanged']=before==after
    result['budget_hours_upper']=24
    write_new(run/'cloud-inventory.json',result)
    package(repo,base,run)
    print_status(run)
    return int(result['status']!='CLOUD_READONLY_INVENTORY_ONLY' or before!=after)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    a=p.add_mutually_exclusive_group(required=True)
    a.add_argument('--collect',action='store_true')
    a.add_argument('--status',action='store_true')
    a.add_argument('--package-only',action='store_true')
    args=p.parse_args()
    action='collect' if args.collect else 'package-only' if args.package_only else 'status'
    return execute(REPO,BASE,action)


if __name__=='__main__':
    try: raise SystemExit(main())
    except (OSError,RuntimeError,ValueError,subprocess.SubprocessError) as exc:
        print('Stage12CloudInventory=STOP_PRESERVE: '+str(exc),file=sys.stderr)
        raise SystemExit(1)
