"""Focused offline checks. No real model, Torch, MuJoCo, CUDA or simulation."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import types
from unittest.mock import patch
import zlib

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent;passed=[]
def check(name,condition):
    if not condition:raise AssertionError(name)
    passed.append(name)
def source_node(text,name):
    return ast.get_source_segment(text,next(n for n in ast.parse(text).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name))
def load_controller():
    module=types.ModuleType('controller_test')
    exec(compile((ROOT/'replay_baseline.py').read_text(),'<controller>','exec'),module.__dict__)
    return module

controller=load_controller()
check('embedded_package_matches_sources',all((HERE/'package'/name).read_text()==text for name,text in controller.PACKAGE.items()))
for name,text in controller.PACKAGE.items():ast.parse(text)
check('all_embedded_sources_parse',True)
original=(HERE.parent/'candidate-evaluation/package/evaluate.py').read_text()
observer=(HERE/'package/observer.py').read_text()
check('observer_is_exact_completed_evaluation_copy',source_node(original,'Observer')==source_node(observer,'Observer'))
check('policy_fingerprint_is_exact_completed_evaluation_copy',source_node(original,'fingerprint')==source_node(observer,'fingerprint'))
record=(HERE/'package/record.py').read_text();tree=ast.parse(record)
calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
check('no_training_or_backward_calls',not any(isinstance(n.func,ast.Attribute) and n.func.attr in ('learn','backward','update_normalization','update_distribution') for n in calls))
check('no_network_or_installer_imports',not any(name in ''.join(controller.PACKAGE.values()) for name in ('import requests','import socket','pip install','uv pip','ssh -p','urlopen')))
check('fixed_horizon_and_no_auto_reset','for step in range(1,501)' in record and 'cfg.auto_reset=False' in record and 'make_cfg(1,seed)' in record)
check('fixed_seed_and_total_budget',controller.SEEDS==(20261021,20261022) and controller.HORIZON==500 and controller.MAX_SECONDS==900)
check('no_action_clipping_or_smoothing','RslRlVecEnvWrapper(raw)' in record and 'clip_actions=' not in record and 'stochastic_output=False' in record)
check('frozen_policy_and_normalizer_verified','not policy.training and not policy.obs_normalizer.training' in record and 'fingerprint(policy.state_dict())==before' in record)
check('no_precision_physics_change','make_cfg(1,seed)' in record and 'set_default_dtype' not in record and 'cfg.sim.' not in record)
viewer={}
exec((HERE/'reference/viewer_config.py').read_text(),viewer)
view=viewer['ViewerConfig'](origin_type=viewer['ViewerConfig'].OriginType.WORLD,width=640,height=480,max_extra_envs=0,enable_reflections=False,lookat=(0.,0.,.4),distance=.30,azimuth=45.,elevation=-12.)
check('camera_options_accept_fixed_mjlab_version',view.width==640 and view.max_extra_envs==0)
offscreen=(HERE/'reference/offscreen_renderer.py').read_text()
check('rendering_api_copies_gpu_state_without_simulation_step',all(n in offscreen for n in ('def initialize(', 'def update(', 'def render(', 'def close(')) and 'mujoco.mj_step' not in offscreen and 'data.qpos[env_idx].cpu().numpy()' in offscreen)
report=ROOT/'records/baseline-checkpoint-retrieval-20261020T094131.404863Z-10781.json'
r=json.loads(report.read_bytes())
check('returned_report_sha_matches_user_receipt',hashlib.sha256(report.read_bytes()).hexdigest()=='a81e45fb2f69706a13b222bafc2fa6f964aeb59a1766347b839f2bfdd3ae4948')
check('model_task_pins_match_returned_report',controller.MODEL_SHA==r['files']['model']['sha256'] and controller.TASK_SHA==r['files']['native_task.py']['sha256'] and controller.PINS_SHA==r['files']['source-pins.json']['sha256'])
check('all_365_local_sources_reported_identical',r['local_replay_inventory']['matching_files']==365 and r['local_replay_inventory']['different_count']==r['local_replay_inventory']['missing_count']==0)

with tempfile.TemporaryDirectory(prefix='md12-replay-offline-') as root:
    root=Path(root)
    def setup(name,worker):
        c=load_controller();c.SLOT=root/name;c.PROJECT=root;c.PYTHON=Path(sys.executable)
        c.verify_inputs=lambda:{'offline_fixture':True};c.PACKAGE={'record.py':worker};c.MAX_SECONDS=30
        c.identity=lambda pid:dict(pid=pid,group=pid,start='offline-process-fixture',state='S')
        c.alive=lambda pin:False
        return c
    success='''from pathlib import Path
import json
p=Path(__file__).resolve().parents[1]
(p/'fixture-video.mp4').write_bytes(b'offline fixture; not a real video')
(p/'progress.json').write_text(json.dumps({'status':'LOCAL_VIDEO_COMPLETE','completed_clips':2}))
(p/'result.json').write_text(json.dumps({'status':'LOCAL_VIDEO_COMPLETE','clips':[],'added_PPO_updates':0}))
'''
    c=setup('success',success)
    with contextlib.redirect_stdout(io.StringIO()):code=c.launch()
    check('controller_collects_success_and_worker_exits',code==0 and not c.snapshot()['worker_alive'] and c.snapshot()['finished']['worker_exit_code']==0)
    marker=(c.SLOT/'started.json').read_bytes();artifact=(c.SLOT/'fixture-video.mp4').read_bytes()
    with patch.object(c.subprocess,'Popen',side_effect=AssertionError('Duplicate launch')),contextlib.redirect_stdout(io.StringIO()):code=c.launch()
    check('repeat_record_never_relaunches_or_extends_budget',code==0 and marker==(c.SLOT/'started.json').read_bytes() and artifact==(c.SLOT/'fixture-video.mp4').read_bytes())
    with patch.object(c.subprocess,'Popen',side_effect=AssertionError('Watch launched worker')),contextlib.redirect_stdout(io.StringIO()):code=c.watch()
    check('watch_only_reads_worker_state',code==0 and marker==(c.SLOT/'started.json').read_bytes())
    failure=setup('failure',"print('specific EGL fixture failure',flush=True)\nraise SystemExit(2)\n")
    with contextlib.redirect_stdout(io.StringIO()):code=failure.launch()
    result=json.loads(next((failure.SLOT/'reports').glob('*.json')).read_bytes())
    check('failed_worker_returns_preserved_report_with_log_tail',code==2 and result['status']=='LOCAL_VIDEO_ERROR_PRESERVE' and 'specific EGL fixture failure' in result['log_tails']['worker.log'])
    marker=(failure.SLOT/'started.json').read_bytes()
    with patch.object(failure.subprocess,'Popen',side_effect=AssertionError('Failure retried')),contextlib.redirect_stdout(io.StringIO()):code=failure.launch()
    check('failure_cannot_automatically_restart',code==2 and marker==(failure.SLOT/'started.json').read_bytes())
    class WaitingProcess:
        pid=123456789
        returncode=None
        def poll(self):return self.returncode
        def wait(self,timeout=None):
            assert self.returncode is not None
            return self.returncode
    def signal_fixture(process,pgid,signum):
        assert pgid==process.pid
        process.returncode=-signum
    limit=setup('deadline','# offline process fixture; no executable task\n');limit.MAX_SECONDS=0
    waiting=WaitingProcess();limit.alive=lambda pin:waiting.returncode is None
    with patch.object(limit.subprocess,'Popen',return_value=waiting),patch.object(limit.os,'killpg',side_effect=lambda g,s:signal_fixture(waiting,g,s)),contextlib.redirect_stdout(io.StringIO()):code=limit.launch()
    check('original_deadline_stops_only_owned_worker_fixture',code==2 and waiting.returncode==-15 and not limit.snapshot()['worker_alive'])
    interrupted=setup('interrupted','import time\ntime.sleep(60)\n')
    original_snapshot=interrupted.snapshot;once=[True]
    def interrupt_once():
        if once[0]:once[0]=False;raise KeyboardInterrupt()
        return original_snapshot()
    interrupted.snapshot=interrupt_once
    waiting=WaitingProcess();interrupted.alive=lambda pin:waiting.returncode is None
    with patch.object(interrupted.subprocess,'Popen',return_value=waiting),patch.object(interrupted.os,'killpg',side_effect=lambda g,s:signal_fixture(waiting,g,s)),contextlib.redirect_stdout(io.StringIO()):code=interrupted.launch()
    check('ctrl_c_stops_worker_fixture_and_retains_marker',code==2 and waiting.returncode==-15 and not original_snapshot()['worker_alive'] and (interrupted.SLOT/'started.json').is_file())
    # Import helpers only: the worker's main is never invoked in this process.
    common=types.ModuleType('replay_common');exec(controller.PACKAGE['replay_common.py'],common.__dict__)
    observer_module=types.ModuleType('observer')
    with patch.dict(sys.modules,{'replay_common':common}):exec(observer,observer_module.__dict__)
    worker_module=types.ModuleType('record_fixture')
    with patch.dict(sys.modules,{'replay_common':common,'observer':observer_module}):exec(record,worker_module.__dict__)
    class Row:
        def __init__(self,data):self.data=data
        def tobytes(self):return self.data
    class Frame:
        shape=(2,2,3);dtype='uint8'
        def __getitem__(self,y):return Row(bytes([y,1,2,3,4,5]))
    png=root/'frame.png';worker_module.png(png,Frame())
    data=png.read_bytes();offset=8;pixels=[]
    while offset<len(data):
        size=struct.unpack('!I',data[offset:offset+4])[0];kind=data[offset+4:offset+8];body=data[offset+8:offset+8+size]
        assert zlib.crc32(kind+body)&0xffffffff==struct.unpack('!I',data[offset+8+size:offset+12+size])[0]
        if kind==b'IDAT':pixels.append(body)
        offset+=12+size
    check('png_frames_valid_crc_and_rgb_rows',data.startswith(b'\x89PNG\r\n\x1a\n') and zlib.decompress(b''.join(pixels))==b'\x00\x00\x01\x02\x03\x04\x05\x00\x01\x01\x02\x03\x04\x05')
    worker_module.SLOT=root
    worker_module.page([dict(seed=1,video=dict(path='4500-seed-1-whole-head.mp4'),initial_frame='4500-seed-1-initial.png',episode=dict(steps=500,longest_stable_steps=300,success=True,terminated=False))])
    check('video_index_uses_local_relative_media_and_scope_notice','src="4500-seed-1-whole-head.mp4"' in (root/'index.html').read_text() and '不是云端128环境轨迹的精确重放' in (root/'index.html').read_text())
    with patch.object(worker_module.shutil,'which',return_value=None),patch.object(worker_module.importlib.util,'find_spec',return_value=None):
        try:worker_module.find_encoder();raise AssertionError('Missing ffmpeg was accepted')
        except RuntimeError as exc:check('missing_encoder_fails_without_installation','nothing installed and no simulation started' in str(exc))

output=dict(status='OFFLINE_CHECKS_PASS',checks=len(passed),passed=passed,real_model_loaded=False,torch_imported=False,mujoco_imported=False,
            real_simulation_steps=0,PPO_updates=0,actual_5060_video_run='PENDING_USER_RUN',limits='No GPU, EGL rendering, checkpoint compatibility or physical success claim is made by these offline checks.')
(HERE/'offline-review.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(output,ensure_ascii=False))
