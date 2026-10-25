"""Real MuJoCo rendering and encoding, and corrupted evidence rejection."""
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE/'package'))
os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np
import mujoco
from replay_common import input_path, sha
from replay_evidence import (frame_steps, load_states, restore_visual_state, scene_ids, timing_rows)
from replay_media import Views, Video, encoder_path, png
from replay_ops import candidates


def fixture_model():
    joints = ''.join('<body name="robot/servo%d" pos="%s 0 0"><joint name="j%d"/><geom size=".008" mass=".01"/></body>'
                     %(i, (i-6.5)*.01, i) for i in range(14))
    motors = ''.join('<motor joint="j%d"/>'%i for i in range(14))
    xml = '''<mujoco><option timestep=".005"/><visual><global offwidth="640" offheight="480"/></visual>
    <worldbody><light pos="0 0 2"/><geom type="plane" size="2 2 .01" rgba=".15 .2 .25 1"/>
    <body name="robot/trunk_base" pos="0 0 .4"><freejoint/><geom type="box" size=".06 .04 .10" rgba=".2 .6 .8 1"/>
    <site name="robot/stage11_head_material_frame" pos="0 0 .12"/>'''+joints+'''</body>
    <body name="ball/ball" pos="0 0 .12"><freejoint/><geom size=".12" rgba="1 .4 .1 1"/></body>
    <body name="top_ball/ball" pos="0 0 .54"><freejoint/><geom size=".02" rgba="1 .9 .1 1"/></body>
    </worldbody><actuator>'''+motors+'</actuator></mujoco>'
    return mujoco.MjModel.from_xml_string(xml)


def fixture_states(model):
    data = mujoco.MjData(model)
    result = dict(qpos=np.tile(data.qpos, (3001, 1)), qvel=np.zeros((3001, model.nv)),
                  ctrl=np.zeros((3001, model.nu)), control_step=np.arange(3001),
                  simulation_time_s=np.arange(3001)*.02)
    result['qpos'][:, 0] = np.sin(np.arange(3001)*.02)*.025
    return result


def fixture_timing(path):
    keys = ('step', 'simulation_time_s', 'release_ns', 'started_ns', 'deadline_ns', 'finished_ns',
            'release_to_finish_ms', 'deadline_overrun_ms', 'environment_ms', 'ros_roundtrip_ms')
    rows = []
    for index in range(1, 3001):
        release = 10**12+(index-1)*20_000_000
        rows.append(dict(step=index, simulation_time_s=index*.02, release_ns=release,
                         started_ns=release+100, deadline_ns=release+20_000_000,
                         finished_ns=release+10_000_000, release_to_finish_ms=10.,
                         deadline_overrun_ms=0., environment_ms=7., ros_roundtrip_ms=2.))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    expected = dict(deadline_misses=0, all_steps=dict(release_to_finish_ms=dict(
        count=3000, mean=10., p95=10., p99=10., max=10., over_20ms=0)))
    return rows, expected


class RecordedEvidence(unittest.TestCase):
    def test_every_sample_restores_without_stepping(self):
        model = fixture_model(); states = fixture_states(model); data = mujoco.MjData(model)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'states.npz'; np.savez_compressed(path, **states)
            saved = sha(path); recorded = load_states(path, model)
            ids = scene_ids(model)
            with patch.object(mujoco, 'mj_step', side_effect=AssertionError('no stepping')):
                for index in range(3001):
                    restore_visual_state(mujoco, model, data, recorded, index)
                    np.testing.assert_array_equal(data.xpos[ids['trunk']], recorded['qpos'][index, :3])
                    self.assertEqual(data.time, index*.02)
            self.assertEqual(list(frame_steps(recorded)), list(range(1, 3001)))
            self.assertEqual(sha(path), saved)
            self.assertFalse(recorded['qpos'].flags.writeable)

    def test_nonfinite_state_and_time_gap_rejected(self):
        model = fixture_model()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'state.npz'
            for failure in ('nan', 'time', 'shape'):
                values = fixture_states(model)
                if failure == 'nan': values['ctrl'][2, 0] = np.nan
                if failure == 'time': values['simulation_time_s'][123] += .02
                if failure == 'shape': values['qpos'] = values['qpos'][:-1]
                np.savez_compressed(path, **values)
                with self.assertRaises(RuntimeError): load_states(path, model)

    def test_all_deadlines_recomputed_and_late_first_tick_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'timing.csv'; rows, expected = fixture_timing(path)
            parsed, summary = timing_rows(path, expected)
            self.assertEqual(len(parsed), 3000); self.assertEqual(summary['max'], 10.)
            original = path.read_text()
            path.write_text(original.replace('1000010000000', '1000021000000', 1))
            with self.assertRaises(RuntimeError): timing_rows(path, expected)
            path.write_text(original.replace('1000020000000', '1000020000001', 1))
            with self.assertRaises(RuntimeError): timing_rows(path, expected)

    def test_export_checksum_and_symlink_protection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); path = root/'state.npz'; path.write_bytes(b'recorded')
            report = dict(exported_data={'state.npz':dict(bytes=8, sha256=sha(path))})
            self.assertEqual(input_path(report, 'state.npz', root), path)
            path.write_bytes(b'changed!')
            with self.assertRaises(RuntimeError): input_path(report, 'state.npz', root)
            path.unlink(); (root/'other').write_bytes(b'recorded'); path.symlink_to(root/'other')
            with self.assertRaises(RuntimeError): input_path(report, 'state.npz', root)

    def test_backend_choice_respects_explicit_setting(self):
        self.assertEqual(candidates({}), ['egl', 'osmesa'])
        self.assertEqual(candidates({'DISPLAY':':0'}), ['egl', 'glfw', 'osmesa'])
        self.assertEqual(candidates({'MUJOCO_GL':'osmesa'}), ['osmesa'])
        with self.assertRaises(RuntimeError): candidates({'MUJOCO_GL':'disabled'})


class RealMedia(unittest.TestCase):
    def test_native_mjb_dual_view_and_complete_video_decode(self):
        model = fixture_model(); states = fixture_states(model)
        self.assertEqual((model.nq, model.nv, model.nu), (35, 32, 14))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); binary = root/'model.mjb'
            mujoco.mj_saveModel(model, str(binary), None)
            model = mujoco.MjModel.from_binary_path(str(binary)); before = sha(binary)
            data = mujoco.MjData(model); views = Views(mujoco, model, scene_ids(model), width=320, height=240)
            writer = Video(encoder_path(), root/'test.mp4', 640, 240)
            try:
                with patch.object(mujoco, 'mj_step', side_effect=AssertionError('no stepping')):
                    for index in range(1, 26):
                        restore_visual_state(mujoco, model, data, states, index)
                        frame = views.frame(data)
                        self.assertGreater(float(frame.std()), 10)
                        writer.add(frame)
                    png(root/'last.png', frame)
                receipt = writer.finish(25)
                self.assertTrue(receipt['full_decode_verified']); self.assertEqual(receipt['decoded_frames'], 25)
                self.assertEqual(receipt['duration_seconds'], .5); self.assertEqual(sha(binary), before)
                keep = os.environ.get('MICRODUCK_REPLAY_TEST_MEDIA')
                if keep:
                    import shutil
                    Path(keep).mkdir(parents=True, exist_ok=True)
                    for name in ('last.png', 'test.mp4'): shutil.copyfile(root/name, Path(keep)/name)
            finally:
                writer.close(); views.close()

    def test_interrupted_encoder_exits_and_preserves_partial_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'partial.mp4'; video = Video(encoder_path(), path, 64, 64)
            video.add(np.zeros((64, 64, 3), dtype=np.uint8)); video.close()
            self.assertIsNotNone(video.process.poll())
            self.assertTrue(path.with_suffix('.ffmpeg.log').is_file())


if __name__ == '__main__': unittest.main(verbosity=2)
