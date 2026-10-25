#!/usr/bin/env python3
"""Stage 10 batch A: unchanged Stage 09 evaluator, six local replays, zero PPO.

Run from the laptop with its existing .venv; no installs, network, source edits,
Git writes, cloud/power operations, or metric/physics changes. Raw outputs stay
outside the source checkout. This is a diagnostic batch, not stage completion.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import uuid
import zipfile

REPO = Path('/home/lx/microduck-double-balance/workspace')
BASE = '00e34c2038771c5d4ad49fe45dff828c0232e60c'
PRIMARY = '86d55c3703c18fcf4817db49c6e4dcf839b3f5195d68ae2c559db7e222bded97'
MANIFEST = 'b61850afbd6fa0deccb3a23a098a910aae74bc0e9b01cca2ee0634c2d1f6ecfd'
INITIAL = '253123669696a953830caeaf587e1300e023c12293d608e38e655a38cdc00f58'
ART = REPO.parent / 'artifacts/double-balance-stage10'
VIDEO09 = REPO.parent / 'artifacts/double-balance-stage09/video-review/20260930T210037Z-5a95cdc3'
MODULE = 'mjlab_microduck.double_balance_stage09'
FIELDS = ('policy_action', 'joint_pos_rad', 'joint_vel_rad_s', 'actuator_force',
          'trunk_quat_wxyz', 'jaw_quat_wxyz', 'tray_quat_wxyz',
          'bam_position_target', 'bam_joint_position', 'bam_joint_velocity',
          'bam_duty_applied', 'bam_motor_torque', 'metric_values')
SAMPLE_ENVS = (0, 43, 99)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    try:
        temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def utc():
    return datetime.now(timezone.utc).isoformat()


def checked(path, checksum):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing or linked input: ' + str(path))
    require(sha(path) == checksum, 'SHA256 differs: ' + str(path))
    return path


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True).strip()


def verify_source():
    require(REPO.is_dir(), 'Canonical laptop repository is missing: ' + str(REPO))
    require(git('rev-parse', '--show-toplevel') == str(REPO), 'Unexpected repository root')
    require(git('branch', '--show-current') == 'double-balance', 'Expected double-balance branch')
    require(git('rev-parse', 'HEAD') == BASE, 'Source HEAD changed; preserve it and return the message for review')
    require(not git('diff', 'HEAD', '--name-only'), 'Tracked local edits detected; preserve them and return the message')
    manifest = checked(REPO / 'docs/audits/stage-09-runtime-hashes.json', MANIFEST)
    rows = read(manifest)
    require(len(rows) == 101, 'Expected 101 frozen runtime files')
    for relative, checksum in rows.items():
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts, 'Unsafe manifest path')
        checked(REPO / relative, checksum)
    return {'source_head': BASE, 'frozen_files_verified': 101, 'runtime_manifest_sha256': MANIFEST}


def job_specs():
    # Interleave the two conditions to reduce a simple run-order confound.
    return [{'name': f'{2*i-1:02d}-headless-{i}', 'video_env_id': None, 'repeat': i}
            if j == 0 else {'name': f'{2*i:02d}-video99-{i}', 'video_env_id': 99, 'repeat': i}
            for i in (1, 2, 3) for j in (0, 1)]


def resolve_inputs():
    receipt = read(REPO / 'docs/audits/stage-08-return-verified.json')
    checkpoint = Path(receipt['extracted_directory']) / 'extension/E-20260929/segment-001/checkpoints/update_004000.pt'
    checked(checkpoint, PRIMARY)
    # Reuse exactly the Stage 09 local replay dataset, not a new random draw.
    dataset = checked(VIDEO09 / 'initial-states/dev.pt', INITIAL)
    sidecar = dataset.with_suffix('.json')
    metadata = read(sidecar)
    require(metadata['sha256'] == INITIAL and len(metadata['state_ids']) == 128,
            'Stage 09 dev dataset receipt differs')
    return checkpoint, dataset, sidecar


def evaluation_command(python, checkpoint, output, datasets, video_env_id):
    result = [str(python), '-u', '-m', MODULE, 'evaluate', '--local', '--protocol', 'dev',
              '--checkpoint', str(checkpoint), '--datasets', str(datasets), '--output', str(output)]
    if video_env_id is not None:
        result.extend(['--video-env-id', str(video_env_id)])
    return result


def supervised(command, log, timeout_s=900):
    """Supervise only the local child launched here; never manage other processes."""
    print('Stage10Command=' + shlex.join(command), flush=True)
    log = Path(log)
    started = time.monotonic()
    with log.open('xb') as stream:
        child = subprocess.Popen(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
        try:
            while True:
                try:
                    code = child.wait(timeout=15)
                    break
                except subprocess.TimeoutExpired:
                    elapsed = time.monotonic() - started
                    print(f'Stage10Heartbeat=local PID {child.pid}; {elapsed:.0f}s; log={log}', flush=True)
                    require(elapsed < timeout_s, 'Local job exceeded its 15-minute limit; stopping this child only')
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise
    require(code == 0, f'Local child exited {code}; see {log}')


def inventory(folder):
    return [{'path': str(p.relative_to(folder)), 'sha256': sha(p), 'size_bytes': p.stat().st_size}
            for p in sorted(Path(folder).rglob('*')) if p.is_file()]


def verify_inventory(folder, rows):
    for row in rows:
        rel = Path(row['path'])
        require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe receipt path')
        p = checked(Path(folder) / rel, row['sha256'])
        require(p.stat().st_size == row['size_bytes'], 'File size differs: ' + str(p))


def validate_evaluation(report, video_env_id):
    require(report['status'] == 'PASS' and report['protocol'] == 'dev', 'Incomplete or wrong evaluation')
    require(report['checkpoint_sha256'] == PRIMARY and report['git_head'] == BASE, 'Evaluation provenance differs')
    require(report['initial_states']['sha256'] == INITIAL, 'Evaluation initial states differ')
    require(report['num_envs'] == 128 and report['horizon_seconds'] == 10. and report['assistance'] == 0., 'Task differs')
    require(report['video_env_id'] == video_env_id and not report.get('reward_probe_only'), 'Evaluation mode differs')
    require(report['episodes_count'] == len(report['episodes']) == 128, 'Missing episodes')
    require(sorted(e['env_id'] for e in report['episodes']) == list(range(128)), 'Duplicate or missing environment IDs')
    require(sum(bool(e['success']) for e in report['episodes']) == report['successes'], 'Success count differs')


def first_difference(a, b, active, times, tolerance=0.):
    """Find first recorded difference; invalid/ended samples are excluded."""
    import numpy as np
    require(a.shape == b.shape and a.shape[:2] == active.shape, 'Trace shape mismatch')
    require(a.dtype == b.dtype, 'Trace dtype mismatch')
    require(np.isfinite(a[active]).all() and np.isfinite(b[active]).all(), 'Nonfinite live diagnostic data')
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    error = np.max(delta, axis=tuple(range(2, a.ndim))) if a.ndim > 2 else delta
    bad = (error > tolerance) & active
    if not bad.any():
        return None
    step = int(np.flatnonzero(bad.any(axis=1))[0])
    ids = np.flatnonzero(bad[step])
    return {'sample_index': step, 'time_s': float(times[step]), 'env_ids': ids.tolist(),
            'max_abs_at_first_sample': float(error[step, ids].max()),
            'max_abs_over_common_active_samples': float(error[active].max())}


def trailing_count(flags):
    result = 0
    for flag in reversed(flags):
        if not flag:
            break
        result += 1
    return result


def label_comparison(left, right):
    a = {e['env_id']: e for e in left['episodes']}
    b = {e['env_id']: e for e in right['episodes']}
    require(set(a) == set(b) and len(a) == len(left['episodes']) == len(right['episodes']), 'Episode IDs differ')
    require(all(a[i]['initial_state_id'] == b[i]['initial_state_id'] for i in a), 'Per-episode initial IDs differ')
    differences = [{'env_id': i, 'left_success': bool(a[i]['success']), 'right_success': bool(b[i]['success']),
                    'left_official_timer_s': a[i]['official_timer_seconds'],
                    'right_official_timer_s': b[i]['official_timer_seconds']}
                   for i in sorted(a) if bool(a[i]['success']) != bool(b[i]['success'])]
    return {'left_successes': sum(bool(x['success']) for x in a.values()),
            'right_successes': sum(bool(x['success']) for x in b.values()),
            'label_disagreements': len(differences), 'different_episodes': differences}


def runtime_probe(output):
    import inspect
    from importlib.metadata import version
    import platform
    import torch
    from mjlab_microduck.double_balance_stage09_state import preflight
    from mjlab_microduck.double_balance_training import VERSIONS
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.sim import Simulation
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab_microduck import double_balance_stage09_evaluation as evaluator
    require(Path(evaluator.__file__).resolve() == REPO/'src/mjlab_microduck/double_balance_stage09_evaluation.py',
            'The existing environment imports a different source checkout')
    head = preflight(cloud=False)
    data = {'status': 'LOCAL_RUNTIME_PREFLIGHT_PASS', 'source_head': head, 'python': sys.version,
            'executable': sys.executable, 'platform': platform.platform(),
            'versions': {**{k: version(k) for k in VERSIONS}, 'torch': torch.__version__},
            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
            'num_threads': torch.get_num_threads(),
            # Read the PyTorch 2.9 precision API used by mjlab. Never write
            # backend settings or access legacy allow_tf32 getters here.
            'backend': backend_snapshot(torch),
            'environment': {k: os.environ.get(k) for k in ('MUJOCO_GL', 'PYTHONHASHSEED',
                            'CUBLAS_WORKSPACE_CONFIG', 'CUDA_LAUNCH_BLOCKING')},
            'new_ppo_updates': 0, 'sources': {}}
    for cls in (ManagerBasedRlEnv, Simulation, RslRlVecEnvWrapper):
        for method in ('step', 'reset', 'forward', 'get_observations'):
            obj = getattr(cls, method, None)
            if obj is not None:
                try:
                    data['sources'][cls.__name__ + '.' + method] = inspect.getsource(obj)
                except (OSError, TypeError):
                    data['sources'][cls.__name__ + '.' + method] = 'SOURCE_UNAVAILABLE'
    write(output, data)


def backend_snapshot(torch):
    """Read current settings without switching precision or configuring backends."""
    return {'precision_api': 'fp32_precision',
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'global_fp32_precision': torch.backends.fp32_precision,
            'matmul_fp32_precision': torch.backends.cuda.matmul.fp32_precision,
            'cudnn_fp32_precision': torch.backends.cudnn.fp32_precision,
            'cudnn_conv_fp32_precision': torch.backends.cudnn.conv.fp32_precision,
            'cudnn_rnn_fp32_precision': torch.backends.cudnn.rnn.fp32_precision}


def contact_inventory(output):
    """Compile the EXISTING scene. No stepping, contact enabling, or body removal."""
    import numpy as np
    import mujoco
    from mjlab.scene import Scene
    from mjlab_microduck.tasks import microduck_double_balance_env_cfg as cfgmod
    cfg = cfgmod.make_microduck_double_balance_env_cfg(play=True, blind=True, history=1, physics_dt=.002)
    cfg.scene.num_envs = 1
    model = Scene(cfg.scene, device='cpu').compile()
    data = mujoco.MjData(model)
    require(model.nkey > 0, 'Compiled scene lacks HOME keyframe')
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    top = model.geom('top_ball/' + cfgmod.TOP_BALL_GEOM_NAME).id
    geoms, mask_allowed, shell_assets = [], [], {}
    for i in range(model.ngeom):
        mesh_id = int(model.geom_dataid[i]) if int(model.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_MESH) else -1
        mesh_name = model.mesh(mesh_id).name if mesh_id >= 0 else None
        ct, ca = int(model.geom_contype[i]), int(model.geom_conaffinity[i])
        allowed = i != top and bool((ct & int(model.geom_conaffinity[top])) or (int(model.geom_contype[top]) & ca))
        if allowed:
            mask_allowed.append(i)
        row = {'id': i, 'name': model.geom(i).name, 'body': model.body(int(model.geom_bodyid[i])).name,
               'type': int(model.geom_type[i]), 'mesh': mesh_name, 'contype': ct, 'conaffinity': ca,
               'condim': int(model.geom_condim[i]), 'friction': model.geom_friction[i].tolist(),
               'solref': model.geom_solref[i].tolist(), 'solimp': model.geom_solimp[i].tolist(),
               'priority': int(model.geom_priority[i]), 'top_ball_mask_compatible': allowed}
        if mesh_name and mesh_name.split('/')[-1] == 'top_head_shell':
            start, count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
            vertices = model.mesh_vert[start:start+count].copy()
            world = vertices @ data.geom_xmat[i].reshape(3, 3).T + data.geom_xpos[i]
            row['home_world_aabb_m'] = [world.min(axis=0).tolist(), world.max(axis=0).tolist()]
            row['aabb_is_not_support_surface'] = True
            shell_assets[f'geom_{i}_mesh_vertices'] = vertices
            shell_assets[f'geom_{i}_world_vertices'] = world
            start, count = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
            shell_assets[f'geom_{i}_mesh_faces'] = model.mesh_face[start:start+count].copy()
        geoms.append(row)
    require(shell_assets, 'No top_head_shell mesh found; review compiled naming before continuing')
    require(all(not g['top_ball_mask_compatible'] for g in geoms if g['mesh'] and g['mesh'].split('/')[-1]=='top_head_shell'),
            'Unexpected existing upper-ball/head-shell mask compatibility')
    bodies = [{'id': i, 'name': model.body(i).name, 'parent_id': int(model.body_parentid[i]),
               'mass_kg': float(model.body_mass[i]), 'ipos_m': model.body_ipos[i].tolist(),
               'iquat_wxyz': model.body_iquat[i].tolist(), 'principal_inertia_kg_m2': model.body_inertia[i].tolist()}
              for i in range(model.nbody)]
    pairs = [{'geom1': int(model.pair_geom1[i]), 'geom2': int(model.pair_geom2[i])} for i in range(model.npair)]
    output = Path(output)
    np.savez_compressed(output.with_suffix('.npz'), **shell_assets)
    write(output, {'status': 'EXISTING_MODEL_INVENTORY_COMPLETE_MIGRATION_NOT_IMPLEMENTED',
                   'geoms': geoms, 'mask_compatible_geom_ids': mask_allowed, 'explicit_contact_pairs': pairs,
                   'bodies': bodies, 'scene_total_mass_kg': float(model.body_mass.sum()),
                   'tray': {'mass_kg': cfgmod.TRAY_MASS, 'size_m': cfgmod.TRAY_FULL_SIZE,
                            'local_inertia_kg_m2': cfgmod.TRAY_INERTIA, 'site': cfgmod.TRAY_SITE_NAME},
                   'mesh_npz_sha256': sha(output.with_suffix('.npz')), 'simulation_steps': 0, 'new_ppo_updates': 0,
                   'limits': ['Mask compatibility is not proof of realized dynamic contact.',
                              'Compiled mesh vertices/AABB are not an audited usable support surface or a contact hull validation.',
                              'Friction, curvature, penetration and inertial migration require a separate approved contact probe.',
                              'Only HOME was inspected. No tray removal or head-shell collision was enabled.']})


def analyze(folder):
    import numpy as np
    folder = Path(folder)
    plan = read(folder / 'plan.json')
    review = folder / 'review'
    review.mkdir(exist_ok=True)
    loaded, summaries = [], []
    for spec in plan['jobs']:
        receipt = read(folder / spec['name'] / 'completed.json')
        run = folder / spec['name'] / receipt['attempt']
        verify_inventory(run, receipt['files'])
        report = read(run / 'evaluation.json')
        validate_evaluation(report, spec['video_env_id'])
        dest = review / spec['name']
        dest.mkdir(exist_ok=True)
        with np.load(run / 'head-trace.npz', allow_pickle=False) as z:
            times, active, cols = z['time_s'].copy(), z['active_first_episode'].copy(), z['metric_columns'].copy()
            require(active.shape[1] == 128 and len(times) == len(active), 'Unexpected trace dimensions')
            require(np.allclose(times, np.arange(1,len(times)+1)*.02, rtol=0, atol=1e-12), 'Trace clock differs')
            fields = {k: z[k].copy() for k in FIELDS}
            # Keep all-env initial 0.2 s and three full selected trajectories for review.
            early, selected = {}, {}
            for k in z.files:
                value = z[k]
                if value.ndim >= 2 and value.shape[:2] == active.shape:
                    early[k] = value[:10].copy()
                    selected[k] = value[:, SAMPLE_ENVS].copy()
                elif k == 'time_s':
                    early[k] = value[:10].copy(); selected[k] = value.copy()
                else:
                    early[k] = value.copy(); selected[k] = value.copy()
            selected['review_env_ids'] = np.asarray(SAMPLE_ENVS)
            np.savez_compressed(dest / 'early-all-envs.npz', **early)
            np.savez_compressed(dest / 'selected-envs.npz', **selected)
            np.savez_compressed(dest / 'all-env-metrics.npz', values=fields['metric_values'],
                                active_first_episode=active, time_s=times, columns=cols)
        with np.load(run / 'head-initial.npz', allow_pickle=False) as z:
            initial = {k: z[k].copy() for k in z.files}
        timer_cases = []
        for episode in report['episodes']:
            env = episode['env_id']
            rows = fields['metric_values'][active[:, env], env]
            require(len(rows) == episode['steps'], 'Episode length and trace differ')
            count = trailing_count(rows[:,list(cols).index('cached_stable')] > 0)
            shadow = count >= 250 and not episode['terminated']
            if shadow != bool(episode['success']):
                timer_cases.append({'env_id': env, 'official_success': bool(episode['success']),
                                    'shadow_count_success': shadow, 'terminal_stable_samples': count,
                                    'official_timer_s': episode['official_timer_seconds']})
        write(dest / 'timer-shadow.json', {'diagnostic_only': True, 'old_labels_changed': False, 'cases': timer_cases})
        for p in run.rglob('*'):
            if p.is_file() and p.suffix in ('.json', '.yaml', '.mp4', '.png'):
                target = dest / p.relative_to(run)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, target)
        shutil.copy2(run / 'head-initial.npz', dest / 'head-initial.npz')
        write(dest / 'raw-local-files.json', {'directory': str(run), 'files': receipt['files'],
                                            'full_traces_retained_on_laptop': True})
        loaded.append((spec, report, fields, active, times, initial))
        summaries.append({'name': spec['name'], 'successes': report['successes'], 'episodes': 128,
                          'video_env_id': spec['video_env_id'], 'physical_terminations': sum(e['terminated'] for e in report['episodes']),
                          'timer_shadow_disagreements': len(timer_cases)})
    pairs = []
    for left, right in itertools.combinations(loaded, 2):
        sa, ra, a, aa, ta, ia = left
        sb, rb, b, ab, tb, ib = right
        require(np.array_equal(ta,tb) and set(ia)==set(ib), 'Trace clock or initial schema differs')
        active = aa & ab
        pair = {'left': sa['name'], 'right': sb['name'],
                'same_render_condition': sa['video_env_id']==sb['video_env_id'], **label_comparison(ra,rb),
                'initial_recorded_fields_different': [k for k in ia if not np.array_equal(ia[k],ib[k])],
                'first_recorded_differences': {k: first_difference(a[k],b[k],active,ta) for k in FIELDS},
                'diagnostic_abs_tolerances_not_acceptance': {str(t): {k: first_difference(a[k],b[k],active,ta,t)
                     for k in ('policy_action','joint_pos_rad','joint_vel_rad_s','actuator_force')}
                     for t in (1e-7,1e-5)}}
        pairs.append(pair)
    result = {'status': 'REPLAY_BATCH_COMPLETE_REVIEW_REQUIRED', 'source_head': BASE, 'new_ppo_updates': 0,
              'jobs': summaries, 'pairs': pairs, 'stage10_complete': False, 'old_results_replaced': False,
              'causality': 'NOT_ESTABLISHED', 'video_visual_review': 'PENDING',
              'scope': 'Six repeated development replays; not independent benchmark episodes. No pooled success rate.',
              'measurement_limits': ['Existing 50Hz telemetry only; no new hooks or solver/backend setting changes.',
                  'First recorded difference is not necessarily the first physical substep difference.',
                  'Body poses and joint coordinates use the original documented metric-phase sampling.',
                  'Observation, LSTM state, full integrator/solver state and RNG are not captured by this batch.',
                  'Headless-versus-video differences are descriptive; six runs do not establish a rendering cause.']}
    write(folder / 'comparison.json', result)
    return result


def package_review(folder):
    folder = Path(folder); review = folder / 'review'; review.mkdir(exist_ok=True)
    for p in folder.iterdir():
        if p.is_file() and p.suffix in ('.json','.log','.npz'):
            shutil.copy2(p, review / p.name)
    for p in folder.glob('*/*.log'):
        dest = review / p.parent.name / p.name
        dest.parent.mkdir(exist_ok=True); shutil.copy2(p,dest)
    rows = [r for r in inventory(review) if r['path'] != 'review-files.json']
    write(review / 'review-files.json', rows)
    archives, batch, total = [], [], 0
    # Conservative UNCOMPRESSED limit leaves room for ZIP metadata/expansion.
    for p in sorted(review.rglob('*')):
        if not p.is_file():
            continue
        size = p.stat().st_size
        require(size < 48*1024**2, 'Unexpected large review item; full raw files stay local: '+str(p))
        if batch and total + size > 48*1024**2:
            archives.append(batch); batch=[]; total=0
        batch.append(p); total += size
    if batch:
        archives.append(batch)
    outputs=[]
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6]
    for i,batch in enumerate(archives,1):
        out=folder/f'review-{stamp}-part{i:02d}.zip'
        with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in batch:z.write(p,p.relative_to(review))
        require(out.stat().st_size<=49*1024**2,'Review archive exceeds 49 MiB')
        outputs.append({'path':str(out),'sha256':sha(out),'size_bytes':out.stat().st_size})
    for row in outputs:
        print('Stage10ReviewZIP='+row['path'],flush=True)
        print('Stage10ReviewSHA256='+row['sha256'],flush=True)
    return outputs


def run(resume=None):
    import fcntl
    source=verify_source()
    checkpoint,dataset,sidecar=resolve_inputs()
    require(shutil.disk_usage(REPO.parent).free>=3*1024**3,'Need at least 3 GiB free for retained raw diagnostics')
    ART.mkdir(parents=True,exist_ok=True)
    with (ART/'local-diagnostic.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('A Stage 10 diagnostic controller is already running')
        if resume:
            folder=Path(resume).resolve()
            require(folder.is_relative_to(ART.resolve()) and folder.is_dir(),'Resume must name an existing Stage 10 output directory')
            plan=read(folder/'plan.json')
            require(plan['script_sha256']==sha(__file__) and plan['source_head']==BASE and plan['jobs']==job_specs(),
                    'Resume plan/source/tool differs; preserve prior evidence')
            require(plan['checkpoint']==str(checkpoint) and plan['checkpoint_sha256']==PRIMARY,'Resume checkpoint differs')
            checked(folder/'initial-states/dev.pt',INITIAL)
            checked(folder/'initial-states/dev.json',plan['dataset_receipt_sha256'])
        else:
            folder=ART/'replay'/ (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
            (folder/'initial-states').mkdir(parents=True,exist_ok=False)
            shutil.copy2(dataset,folder/'initial-states/dev.pt');shutil.copy2(sidecar,folder/'initial-states/dev.json')
            plan={**source,'schema_version':1,'created_utc':utc(),'script_sha256':sha(__file__),
                  'checkpoint':str(checkpoint),'checkpoint_sha256':PRIMARY,'dataset_sha256':INITIAL,
                  'dataset_receipt_sha256':sha(sidecar),'jobs':job_specs(),'new_ppo_updates':0,
                  'cloud_gpu_hours':0,'manual_cloud_shutdown_status':'OFF_CONFIRMED_BY_USER',
                  'cloud_data_must_be_preserved':True,'stage10_complete':False}
            write(folder/'plan.json',plan)
        print('Stage10Output='+str(folder),flush=True)
        print('Stage10Resume='+shlex.join(['python3',str(Path(__file__).resolve()),'--resume',str(folder)]),flush=True)
        state={'status':'RUNNING','output_directory':str(folder),'new_ppo_updates':0,'cloud_contacted':False,
               'stage06_smoke_run':False,'source_edited':False,'stage10_complete':False,'completed_jobs':[]}
        write(folder/'run.json',state)
        try:
            probe_output=folder/('runtime-resume-'+uuid.uuid4().hex[:6]+'.json') if (folder/'runtime.json').exists() else folder/'runtime.json'
            supervised([sys.executable,'-u',str(Path(__file__).resolve()),'--probe',str(probe_output)],
                       folder/('probe-'+uuid.uuid4().hex[:6]+'.log'))
            old_runtime,new_runtime=read(folder/'runtime.json'),read(probe_output)
            for key in ('status','source_head','python','executable','versions','cuda','gpu','num_threads','backend','environment'):
                require(old_runtime[key]==new_runtime[key], 'Runtime changed during resume: '+key)
            if not (folder/'contact-inventory.json').exists():
                supervised([sys.executable,'-u',str(Path(__file__).resolve()),'--contact',str(folder/'contact-inventory.json')],
                           folder/('contact-'+uuid.uuid4().hex[:6]+'.log'))
            for spec in plan['jobs']:
                parent=folder/spec['name'];parent.mkdir(exist_ok=True)
                receipt_path=parent/'completed.json'
                if receipt_path.exists():
                    receipt=read(receipt_path);run_dir=parent/receipt['attempt']
                    require(Path(receipt['attempt']).name==receipt['attempt'],'Unsafe attempt path')
                    verify_inventory(run_dir,receipt['files'])
                    validate_evaluation(read(run_dir/'evaluation.json'),spec['video_env_id'])
                    print('Stage10ReuseVerified='+spec['name'],flush=True)
                else:
                    run_dir=parent/('attempt-'+uuid.uuid4().hex[:8])
                    command=evaluation_command(sys.executable,checkpoint,run_dir,folder/'initial-states',spec['video_env_id'])
                    write(parent/(run_dir.name+'-command.json'),{'command':command,'created_utc':utc()})
                    supervised(command,run_dir.with_suffix('.log'))
                    report=read(run_dir/'evaluation.json');validate_evaluation(report,spec['video_env_id'])
                    require((run_dir/'head-trace.npz').is_file(),'Missing full head trace')
                    write(receipt_path,{'attempt':run_dir.name,'files':inventory(run_dir),'new_ppo_updates':0})
                    print(f'Stage10Replay={spec["name"]}; successes={report["successes"]}/128; zero PPO',flush=True)
                state['completed_jobs'].append(spec['name']);write(folder/'run.json',state)
            result=analyze(folder)
            verify_source();checked(checkpoint,PRIMARY)
            state.update(status=result['status'],finished_utc=utc(),video_visual_review='PENDING')
            write(folder/'run.json',state)
            package_review(folder)
            print('Stage10Batch=COMPLETE_REVIEW_REQUIRED; stage not complete; upload all printed review ZIP parts',flush=True)
        except BaseException as exc:
            state.update(status='STOPPED',error=type(exc).__name__+': '+str(exc),finished_utc=utc())
            write(folder/'run.json',state)
            try: package_review(folder)
            except Exception as package_error: print('Stage10PartialPackageError='+str(package_error),flush=True)
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--resume',type=Path)
    group.add_argument('--probe',type=Path,help=argparse.SUPPRESS)
    group.add_argument('--contact',type=Path,help=argparse.SUPPRESS)
    group.add_argument('--analyze',type=Path,help='Rebuild comparison/review from completed local jobs only')
    args=parser.parse_args()
    require(REPO.is_dir(),'Run on the laptop with the existing project checkout')
    python=REPO/'.venv/bin/python'
    require(python.is_file(),'Existing project .venv is missing; no dependencies were installed')
    if Path(sys.prefix).resolve()!=(REPO/'.venv').resolve():
        os.execv(str(python),[str(python),'-u',str(Path(__file__).resolve()),*sys.argv[1:]])
    os.environ.setdefault('MUJOCO_GL','egl')
    os.environ.setdefault('MPLBACKEND','Agg')
    # Existing editable source installation is verified by the frozen evaluator.
    os.chdir(REPO)
    if args.probe: verify_source();runtime_probe(args.probe)
    elif args.contact: verify_source();contact_inventory(args.contact)
    elif args.analyze:
        require(args.analyze.resolve().is_relative_to(ART.resolve()),'Analyze only Stage 10 local artifacts')
        verify_source();analyze(args.analyze);package_review(args.analyze)
    else: run(args.resume)


if __name__=='__main__':
    try: main()
    except (Exception,KeyboardInterrupt) as exc:
        print('Stage10Stopped='+type(exc).__name__+': '+str(exc),flush=True)
        raise SystemExit(1)
