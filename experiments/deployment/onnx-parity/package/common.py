"""Pinned local candidates and original inputs. Never modify training artifacts."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from budget import require,MAX_SECONDS,MAX_EVALUATIONS

PROJECT=Path.home()/'microduck-double-balance/workspace'
BASE=PROJECT.parent
SLOT=BASE/'artifacts/double-balance-stage12/native-push10-final6h-local-eval-v1'
PYTHON=PROJECT/'.venv/bin/python'
SOURCE=PROJECT
HERE=Path(__file__).resolve().parent
RETURN_COMMIT='0cbee30e5a83904371076ca09a8cfb4afd1fa3f4'
RETURN_ROOT=BASE/'artifacts/double-balance-stage12/native-push10-final6h-v1-client/returned'/RETURN_COMMIT
INPUTS=RETURN_ROOT/'files'
TRAINING_PACKAGE_ID='1b79efbd0afbbcdaacde1bd94014c2ecbe257d4479c877b6fc2b649914a61b39'
MODEL_SHA='ca2a82acbe939169367272ffb50242bbfc1b4a39c06ed186bfc4cf3009ed4130'
START_POLICY_UPDATES=6010
LEARNING_RATE=3e-5
TASK_SHA='b6909b9cff864dcfce8203062a75282571ca2c7ef5d66b2aa58c05c4626a55af'
PINS_SHA='08c781088ecb4b2023c593ff8ffa45061bf6a57fad5902fcf79b50155daa969c'
VERSIONS={'mjlab':'1.3.0','mujoco':'3.10.0','mujoco-warp':'3.8.1',
          'warp-lang':'1.12.0','rsl-rl-lib':'5.0.1','torch':'2.9.1'}
CANDIDATES=(('candidate-01',1553),('candidate-02',3428),('candidate-final',5304))
CLOUD_SLOT='/root/autodl-tmp/microduck-double-balance/artifacts/double-balance-stage12/native-push10-final6h-v1'


def read(path):return json.loads(Path(path).read_bytes())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def safe(root,relative):
    relative=Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts,'Unsafe relative path')
    target=Path(root);require(not target.is_symlink(),'Unexpected symlink root')
    for part in relative.parts:
        target=target/part;require(not target.is_symlink(),'Unexpected symlink: '+str(target))
    return target


def write(path,value,exclusive=False):
    path=Path(path);require(not path.is_symlink(),'Unexpected output symlink')
    data=(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n').encode()
    fd,temp=tempfile.mkstemp(prefix='.eval-write-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        if exclusive:os.link(temp,path)
        else:os.replace(temp,path)
        directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        if os.path.exists(temp):os.unlink(temp)


def identity(pid):
    try:
        fields=(Path('/proc')/str(int(pid))/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=int(pid),start=fields[19],state=fields[0],group=int(fields[2]))
    except FileNotFoundError:return None


def alive(pin):
    current=identity(pin['pid']) if pin else None
    return bool(current and current['start']==pin['start'] and current['state'] not in ('Z','X'))


def verify_package():
    manifest=read(HERE/'manifest.json')
    digest=hashlib.sha256(json.dumps(manifest['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    require(digest==manifest['identity'],'Local evaluation package identity differs')
    for name,pin in manifest['files'].items():
        require(Path(name).name==name,'Invalid package member')
        p=safe(HERE,name)
        require(p.is_file() and p.stat().st_size==pin['bytes'] and sha(p)==pin['sha256'],
                'Local evaluation code differs: '+name)
    return digest


def verify_runtime():
    actual={name:importlib.metadata.version(name) for name in VERSIONS}
    require(actual==VERSIONS,'Existing project runtime differs; no automatic installation or upgrade')
    return actual


def matches(path,pin):
    return path.is_file() and not path.is_symlink() and path.stat().st_size==pin['bytes'] and sha(path)==pin['sha256']


def verify_inputs():
    pins=read(HERE/'return-pins.json')
    # The verified return controller persisted the parsed index using indent=2.
    # Its receipt pins the original Git blob; pin the exact local serialization
    # separately instead of rewriting either historical file to pass a check.
    require(sha(safe(RETURN_ROOT,'return-index.json'))==pins['local_index_sha256'],'Returned local index changed')
    index=read(RETURN_ROOT/'return-index.json')
    require(index['source_commit']==pins['training_commit'] and index['package_identity']==TRAINING_PACKAGE_ID
            and index['branch']==pins['return_branch'],'Returned campaign identity differs')
    verified=read(safe(RETURN_ROOT,'return-verified.json'))
    require(verified['status']=='LOCAL_RETURN_VERIFIED' and verified['commit']==RETURN_COMMIT
            and verified['index_sha256']==pins['index_sha256'] and verified['verified_files']==400,
            'Expected this exact verified local return; do not use an earlier model')
    for name,pin in pins['files'].items():
        require(index['inventory']['included'].get(name)==pin and matches(safe(INPUTS,name),pin),
                'Returned training evidence/model differs: '+name)
    budget=read(INPUTS/'budget.json');result=read(INPUTS/'run/session-004/result.json')
    require(budget['completed_updates']==5304 and budget['attempted_updates']==5305
            and budget['phase_completed']=={'smoke':5,'formal':5299}
            and budget['evaluations']=={} and budget['pending']['attempt']==5305,
            'Frozen training ledger differs; it must not be resumed or reset')
    require(result['status']=='TRAINING_PHASE_COMPLETE' and result['reason']=='train_window_complete'
            and result['committed_snapshot_saved'] is True and result['completed_updates']==5304,
            'Missing the final complete saved training state')
    require(sha(HERE/'native_task.py')==TASK_SHA and sha(HERE/'source-pins.json')==PINS_SHA,
            'Original task/physics source pins changed')
    source=read(HERE/'source-pins.json');require(len(source)==365,'Expected 365 original source pins')
    for name,pin in source.items():
        require(matches(safe(PROJECT,name),pin),'本机冻结源码与训练版本不符：'+name+'；保留已有修改，不覆盖。')
    return dict(return_commit=RETURN_COMMIT,index_sha256=pins['index_sha256'],
        training_package_identity=TRAINING_PACKAGE_ID,old_completed_updates=5304,
        old_attempted_updates=5305,source_files_verified=365,training_ledger_preserved=True,
        candidate_sha256=[cp['sha256'] for cp in candidate_inputs()])


def candidate_inputs():
    pins=read(HERE/'return-pins.json')['files'];result=[]
    for label,updates in CANDIDATES:
        relative='run/checkpoints/final6h_%06d.pt'%updates
        original=safe(INPUTS,'run/candidates/'+label+'.json')
        cp=read(original)
        require(cp['candidate_label']==label and cp['completed_updates']==updates
                and cp['policy_updates']==START_POLICY_UPDATES+updates
                and cp['package_identity']==TRAINING_PACKAGE_ID
                and cp['path']==CLOUD_SLOT+'/'+relative
                and cp['sha256']==pins[relative]['sha256'] and cp['bytes']==pins[relative]['bytes'],
                'Candidate identity differs: '+label)
        result.append(dict(cp,path=str(safe(INPUTS,relative)),cloud_checkpoint_path=cp['path'],
            source_candidate_receipt=str(original),source_candidate_receipt_sha256=sha(original)))
    return result


def verify_checkpoint_metadata(saved,checkpoint):
    meta=saved['infos']['stage12_final6h']
    require(meta['package_identity']==TRAINING_PACKAGE_ID
            and meta['origin_checkpoint_sha256']==MODEL_SHA
            and meta['policy_updates']==checkpoint['policy_updates']
            and meta['campaign_completed_updates']==checkpoint['completed_updates']
            and meta['committed_snapshot'] is True and meta['num_envs']==4096
            and meta['lower_assistance']==0.,'Frozen candidate checkpoint metadata differs')
    return meta


def clean_environment():
    env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME','WORLD_SIZE','RANK','LOCAL_RANK')
         and not k.startswith('MICRODUCK_')}
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
        WANDB_MODE='disabled',HF_HUB_OFFLINE='1')
    return env


def preflight():
    require(sys.platform=='linux' and PYTHON.is_file(),'Use the existing local Linux project environment')
    package=verify_package();versions=verify_runtime();inputs=verify_inputs()
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    require(not active,'Another local GPU compute process is active; preserve: '+active)
    return dict(package_identity=package,versions=versions,inputs=inputs,
                GPU_initialized=False,physics_steps=0,PPO_updates=0,cloud_connections=0)


need=require
