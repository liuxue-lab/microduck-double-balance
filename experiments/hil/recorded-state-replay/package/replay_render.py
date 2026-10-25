"""Offline frame generation; no Torch, ROS, ONNX runtime or stepping calls."""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from replay_common import (CASES, HERE, SOURCE_RUN, SOURCE_SHA, input_path, need,
                           package_identity, read, verify_source, write)
from replay_evidence import frame_steps, load_states, restore_visual_state, scene_ids, timing_rows
from replay_media import Video, Views, encoder_path, png


def runtime():
    import mujoco
    need(mujoco.__version__ == '3.10.0', 'Replay needs the existing MuJoCo 3.10.0 environment')
    # An accidental integration call is a failure, even though it cannot affect old files.
    def forbidden(*args, **kwargs):
        raise RuntimeError('Integration is forbidden during recorded-state replay')
    for key in ('mj_step', 'mj_step1', 'mj_step2'):
        setattr(mujoco, key, forbidden)
    return mujoco


def probe(model_path):
    mujoco = runtime()
    model = mujoco.MjModel.from_binary_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_camlight(model, data)
    views = Views(mujoco, model, scene_ids(model), width=160, height=120)
    try:
        frame = views.frame(data)
        need(frame.shape == (120, 320, 3), 'Renderer probe shape differs')
    finally:
        views.close()
    print(json.dumps(dict(status='RENDER_BACKEND_READY', backend=os.environ.get('MUJOCO_GL'),
                          mujoco=mujoco.__version__, nq=model.nq, nv=model.nv, nu=model.nu)), flush=True)


def render(output):
    output = Path(output)
    started = read(output/'started.json')
    def boundary():
        need(time.time() < started['deadline_utc'], 'Replay time budget reached; outputs preserved')
        need(not (output/'STOP.json').exists(), 'Replay stop requested; outputs preserved')
    def stop(*args):
        raise KeyboardInterrupt('Replay interrupted')
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)
    result = dict(schema='stage12-recorded-hil-replay-v1', status='REPLAY_RUNNING',
        source_report_sha256=SOURCE_SHA, source_run=str(SOURCE_RUN),
        new_control_ticks=0, new_physics_steps=0, policy_inferences=0, Orin_requests=0,
        PPO_updates=0, cloud_connections=0, motor_io=False, automatic_retry=False,
        source_hil_realtime_acceptance=True, new_realtime_acceptance_performed=False,
        backend=os.environ.get('MUJOCO_GL'), clips=[], package_identity=package_identity(HERE))
    def progress(**changes):
        boundary()
        write(output/'progress.json', dict(status='REPLAY_RUNNING', elapsed_seconds=time.time()-started['started_utc'],
              maximum_frames=6000, **changes))
    try:
        progress(phase='verify_original_data', completed_frames=0)
        source = verify_source()
        encoder = encoder_path()
        mujoco = runtime()
        for number, case in enumerate(source['simulator']['cases'], 1):
            need(case['case_id'] == CASES[number-1], 'Case order differs')
            prefix = 'case-%02d/'%number
            timings, statistics = timing_rows(input_path(source, prefix+'timing.csv'), case['timing'])
            model = mujoco.MjModel.from_binary_path(str(input_path(source, prefix+'native-model.mjb')))
            need((model.nq, model.nv, model.nu) == (35, 32, 14) and abs(model.opt.timestep-.005) < 1e-12,
                 'Recorded model dimensions or timestep differ')
            states = load_states(input_path(source, prefix+'state-data.npz'), model)
            data = mujoco.MjData(model)
            ids = scene_ids(model)
            views = writer = None
            name = case['case_id']
            snapshots = {}
            try:
                views = Views(mujoco, model, ids)
                restore_visual_state(mujoco, model, data, states, 0)
                initial = views.frame(data)
                snapshots['initial'] = str(output/(name+'-initial.png'))
                png(snapshots['initial'], initial)
                writer = Video(encoder, output/(name+'-recorded-hil.mp4'), 1280, 480)
                for index in frame_steps(states):
                    boundary()
                    restore_visual_state(mujoco, model, data, states, index)
                    frame = views.frame(data)
                    writer.add(frame)
                    if index in (1500, 3000):
                        label = 'middle' if index == 1500 else 'final'
                        snapshots[label] = str(output/(name+'-'+label+'.png'))
                        png(snapshots[label], frame)
                    if index == 1 or index % 50 == 0:
                        progress(phase='render_saved_states', case_id=name, case_frame=index,
                                 completed_frames=(number-1)*3000+index,
                                 percent=100*((number-1)*3000+index)/6000)
                progress(phase='verify_video_decode', case_id=name, completed_frames=number*3000)
                video = writer.finish(3000)
            finally:
                if writer is not None:
                    writer.close()
                if views is not None:
                    views.close()
            selected = ('step', 'simulation_time_s', 'release_to_finish_ms', 'environment_ms',
                        'ros_roundtrip_ms', 'reference_inference_ms', 'orin_inference_ms')
            clip = dict(case_id=name, video=video, snapshots=snapshots,
                timing_statistics=statistics, timing_csv_recomputed=True, state_samples=3001,
                frame_control_steps=dict(first=1, last=3000, count=3000), initial_state_saved_separately=True,
                all_formal_samples_rendered=True, cameras='whole body and head, tracking the recorded positions',
                perturbation_pulses=case['trial']['applied_disturbance_pulses'],
                plot=[{key:row[key] for key in selected} for row in timings])
            write(output/(name+'-clip.json'), {k:v for k,v in clip.items() if k != 'plot'})
            result['clips'].append(clip)
            del states, data, model, views, writer
            gc.collect()
        progress(phase='recheck_original_hashes', completed_frames=6000)
        need(verify_source() == source, 'Original HIL records changed during replay')
        from replay_page import page
        index = page(output, source, result['clips'])
        for clip in result['clips']:
            clip.pop('plot')
        result.update(status='HIL_REPLAY_COMPLETE', original_inputs_unchanged=True,
                      completed_frames=6000, index=str(index), finished_utc=time.time())
        write(output/'result.json', result)
        return 0
    except BaseException as exc:
        result.update(status='HIL_REPLAY_FAILED_PRESERVE', error=repr(exc),
                      traceback=traceback.format_exc()[-12000:], finished_utc=time.time())
        for clip in result['clips']:
            clip.pop('plot', None)
        write(output/'result.json', result)
        print('REPLAY_ERROR='+repr(exc), flush=True)
        return 130 if isinstance(exc, KeyboardInterrupt) else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('probe', 'render'))
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    if args.operation == 'probe':
        probe(args.path)
    else:
        raise SystemExit(render(args.path))
