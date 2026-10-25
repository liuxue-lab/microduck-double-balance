import hashlib
import json
import os
from pathlib import Path

PROJECT=Path.home()/'microduck-double-balance/workspace'
SELECTED=Path.home()/'microduck-double-balance/artifacts/double-balance-stage12/native-selected-speed500-v1'
SLOT=SELECTED.parent/'native-replay-speed500-v1'
MODEL=SELECTED/'checkpoints/refine_000500.pt'
MODEL_SHA='51821a28880aba74b80e6d5a388dfd7a82098bd3dca04a1e8947b4b2a8d5539a'
TASK_SHA='b6909b9cff864dcfce8203062a75282571ca2c7ef5d66b2aa58c05c4626a55af'
PINS_SHA='08c781088ecb4b2023c593ff8ffa45061bf6a57fad5902fcf79b50155daa969c'
ORIGIN_ID='dd6d1e1e697bcaa348dbdd92dd8c7b30ea31ae9c32599338106859ae7657523a'
PYTHON=PROJECT/'.venv/bin/python'
SEEDS=(20261021,20261022)
MAX_SECONDS=900
HORIZON=500

def need(ok,message):
    if not ok:raise RuntimeError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_bytes())

def write(path,value,exclusive=False):
    path=Path(path);data=(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n').encode()
    out=path if exclusive else path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with out.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    if not exclusive:os.replace(out,path)

def identity(pid):
    try:
        row=(Path('/proc')/str(int(pid))/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=int(pid),start=row[19],state=row[0],group=int(row[2]))
    except FileNotFoundError:return None

def alive(pin):
    now=identity(pin['pid']) if pin else None
    return bool(now and now['start']==pin['start'] and now['state'] not in ('Z','X'))

def verify_inputs():
    need(MODEL.is_file() and not MODEL.is_symlink() and MODEL.stat().st_size==10036828 and sha(MODEL)==MODEL_SHA,'Selected speed-refine +500 checkpoint differs')
    task=SELECTED/'task/native_task.py';pins=SELECTED/'task/source-pins.json'
    need(sha(task)==TASK_SHA and sha(pins)==PINS_SHA,'Frozen task or source pins differ')
    source=read(pins)
    for rel,entry in source.items():
        relative=Path(rel);need(not relative.is_absolute() and '..' not in relative.parts,'Invalid source pin path')
        p=PROJECT/relative
        need(p.is_file() and p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],'Local source differs: '+rel)
    return dict(checkpoint_sha256=MODEL_SHA,task_sha256=TASK_SHA,source_pins_sha256=PINS_SHA,source_files_verified=len(source))

TRAIN_ID='340f064e33b61ed8968724084ebab8dc35563cb2239506ab20b1d01056581c10'
ORIGIN_MODEL_SHA='df03202acf010680b63d3277ec879f63e2d4808e5d1286653923a314f125ac6a'
REWARD_SPEC={'changed_term': 'double_balance', 'consecutive_success_bonus': False, 'evaluation_uses_original_task': True, 'formula': 'original position/height/upright factors * exp(-upper_speed^2/0.08^2) * exp(-lower_xy_speed^2/0.15^2)', 'lower_horizontal_speed_std_m_s': 0.15, 'observation_and_physics_unchanged': True, 'original_upper_speed_std_m_s': 0.12, 'other_rewards_unchanged': True, 'smooth_dense_reward': True, 'upper_speed_std_m_s': 0.08, 'version': 'speed_alignment_v1', 'weight': 4.0}

def verify_checkpoint_metadata(saved):
    for key in ("stage12_native","stage12_refine"):
        meta=saved["infos"][key]
        need(meta["package_identity"]==TRAIN_ID and meta["origin_package_identity"]==ORIGIN_ID and meta["origin_checkpoint_sha256"]==ORIGIN_MODEL_SHA,"Checkpoint lineage differs")
        need(meta["policy_updates"]==5000 and meta["refine_updates"]==500 and meta["campaign_attempted_updates"]==500,"Checkpoint counters differ")
        need(meta["num_envs"]==4096 and meta["lower_assistance"]==0. and meta["reward_spec"]==REWARD_SPEC,"Checkpoint training metadata differs")
    return dict(policy_updates=5000,refine_updates=500,training_package_identity=TRAIN_ID,reward_spec=REWARD_SPEC)
