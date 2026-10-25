"""Exercise backpressure, timing, native rendering and evidence failure paths."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE/'package'))
os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np
import mujoco
import live_stream as stream
import live_support as support
from live_viewer import LiveScene, Seen


def model_fixture():
    joints = ''.join('<body name="robot/servo%d"><joint name="j%d"/><geom size=".008" mass=".01"/></body>' % (i,i) for i in range(14))
    motors = ''.join('<motor joint="j%d"/>' % i for i in range(14))
    return mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".005"/>
    <visual><global offwidth="640" offheight="480"/></visual><worldbody>
    <light pos="0 0 2"/><geom type="plane" size="2 2 .01" rgba=".15 .2 .25 1"/>
    <body name="robot/trunk_base" pos="0 0 .4"><freejoint/><geom type="box" size=".06 .04 .10" rgba=".2 .6 .8 1"/>
    <site name="robot/stage11_head_material_frame" pos="0 0 .12"/>'''+joints+'''</body>
    <body name="ball/ball" pos="0 0 .12"><freejoint/><geom size=".12" rgba="1 .4 .1 1"/></body>
    <body name="top_ball/ball" pos="0 0 .54"><freejoint/><geom size=".02" rgba="1 .9 .1 1"/></body>
    </worldbody><actuator>'''+motors+'</actuator></mujoco>')


def sample_row(model=None):
    data = mujoco.MjData(model or model_fixture())
    return np.concatenate((data.qpos, data.qvel, data.ctrl))


class Stream(unittest.TestCase):
    def test_atomic_roundtrip_latest_and_immutable_state(self):
        writer, reader = stream.channel()
        try:
            timing = SimpleNamespace(rows=[])
            publisher = stream.Publisher(writer.fileno(), 1, timing, np)
            row = sample_row(); row.setflags(write=False)
            before = row.tobytes()
            for step in range(4): publisher.publish(row, step)
            packet = stream.latest(reader, np)
            self.assertEqual(packet['step'], 3)
            np.testing.assert_array_equal(packet['row'], row)
            self.assertEqual(row.tobytes(), before)
            self.assertFalse(packet['row'].flags.writeable)
            self.assertIsNone(stream.latest(reader, np))
        finally:
            writer.close(); reader.close()

    def test_stalled_receiver_never_blocks_control_and_recovers(self):
        writer, reader = stream.channel()
        try:
            publisher = stream.Publisher(writer.fileno(), 1, SimpleNamespace(rows=[]), np)
            row = sample_row(); before = row.tobytes()
            start = time.monotonic()
            for i in range(20000): publisher.publish(row, i % 3001)
            self.assertLess(time.monotonic()-start, 3.)
            self.assertGreater(publisher.skipped, 0)
            self.assertEqual(publisher.sent+publisher.skipped, 20000)
            self.assertEqual(publisher.errors, 0)
            stream.latest(reader, np)
            publisher.publish(row, 3000)
            self.assertEqual(stream.latest(reader, np)['step'], 3000)
            self.assertEqual(row.tobytes(), before)
        finally:
            writer.close(); reader.close()

    def test_closed_window_does_not_throw_or_modify_states(self):
        writer, reader = stream.channel(); reader.close()
        try:
            publisher = stream.Publisher(writer.fileno(), 1, SimpleNamespace(rows=[]), np)
            row = sample_row(); before = row.tobytes()
            for i in range(3001): publisher.publish(row, i)
            self.assertTrue(publisher.disabled)
            self.assertEqual(publisher.errors, 1)
            self.assertEqual(publisher.skipped, 3001)
            self.assertEqual(row.tobytes(), before)
        finally: writer.close()

    def test_invalid_packets_are_rejected(self):
        writer, reader = stream.channel()
        try:
            publisher = stream.Publisher(writer.fileno(), 1, SimpleNamespace(rows=[]), np)
            publisher.publish(sample_row(), 1)
            raw = reader.recv(stream.SIZE)
            for corrupt in (raw[:-1], raw+b'x', b'INVALID!'+raw[8:]):
                with self.assertRaises(RuntimeError): stream.decode(corrupt, np)
            corrupt = bytearray(raw)
            np.ndarray((81,),dtype='<f8',buffer=corrupt,offset=stream.HEADER.size)[0] = float('nan')
            with self.assertRaises(RuntimeError): stream.decode(corrupt,np)
        finally: writer.close(); reader.close()

    def test_publication_cost_stays_inside_first_deadline(self):
        from telemetry import States, Timing, FIELDS
        model = model_fixture(); data = mujoco.MjData(model)
        binary = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
        mujoco.mj_saveModel(model,None,binary)
        writer, reader = stream.channel()
        try:
            with tempfile.TemporaryDirectory() as folder:
                case = Path(folder)/'case-01'; case.mkdir()
                timing = Timing(case)
                raw = SimpleNamespace(sim=SimpleNamespace(mj_data=data,native_model=model))
                with patch.dict(os.environ, NATIVE_LIVE_FD=str(writer.fileno())), patch.object(support,'MODEL_SHA',hashlib.sha256(binary).hexdigest()):
                    states = States(case,raw,None,[],timing)
                    states.capture()
                real = states.publisher.publish
                def delayed(*args): time.sleep(.025); return real(*args)
                with patch.object(states.publisher,'publish',side_effect=delayed):
                    timing.begin(); start=timing.clock(); states.capture()
                    timing.add(state_capture_ms=(timing.clock()-start)/1e6)
                    for name in FIELDS:
                        if name not in timing.current: timing.add(**{name:0.})
                    timing.end()
                self.assertGreaterEqual(timing.rows[0]['state_capture_ms'],25.)
                self.assertEqual(timing.summary()['deadline_misses'],1)
                self.assertEqual(timing.summary()['dropped_ticks'],0)
        finally: writer.close(); reader.close()


class Visuals(unittest.TestCase):
    def test_native_scene_and_egl_render_without_physics(self):
        model = model_fixture(); visual = LiveScene(mujoco,np,model)
        row = sample_row(model)
        renderer = mujoco.Renderer(model,width=640,height=480)
        try:
            with patch.object(mujoco,'mj_step',side_effect=AssertionError('no integration')), patch.object(mujoco,'mj_forward',side_effect=AssertionError('no dynamics')):
                for i in range(3001):
                    row[0] = np.sin(i*.02)*.03
                    visual.apply(dict(row=row,sim_time=i*.02))
                    np.testing.assert_array_equal(visual.data.qpos,row[:35])
                    np.testing.assert_array_equal(visual.data.qvel,row[35:67])
                    np.testing.assert_array_equal(visual.data.ctrl,row[67:])
                for mode in (1,2):
                    visual.reset_camera(mode); visual.update()
                    renderer.update_scene(visual.data,camera=visual.camera)
                    pixels = renderer.render()
                    self.assertGreater(float(pixels.std()),10.)
                    if os.environ.get('MICRODUCK_TEST_MEDIA'):
                        from PIL import Image
                        destination=Path(os.environ['MICRODUCK_TEST_MEDIA']); destination.mkdir(parents=True,exist_ok=True)
                        Image.fromarray(pixels).save(destination/('synthetic-view%d.png'%mode))
        finally: renderer.close()

    def test_display_frame_coalescing_and_order(self):
        seen=Seen()
        for step in (1,3,3,5,3000): seen.observe(dict(case=1,step=step),3.)
        self.assertEqual(seen.frames,4)
        self.assertEqual(seen.cases['1']['last_step'],3000)
        self.assertEqual(seen.cases['1']['display_coalesced_samples'],2996)
        with self.assertRaises(RuntimeError): seen.observe(dict(case=1,step=100),2.)

    def test_visual_and_realtime_acceptance_remain_separate(self):
        rows=[dict(runtime_adapters=dict(state_export=dict(live_display=dict(send_errors=0,deadline_includes_publication=True))))]*2
        viewer=dict(status='LIVE_VIEWER_COMPLETE',physics_steps=0,policy_inferences=0,cases={
            str(i):dict(fresh_frames=1500,first_step=2,last_step=3000,max_age_ms=10.,max_step_gap=2) for i in (1,2)})
        self.assertTrue(support.visual_result(viewer,dict(cases=rows))['complete'])
        viewer['cases']['2']['max_step_gap']=100
        self.assertFalse(support.visual_result(viewer,dict(cases=rows))['complete'])
        viewer['cases']['2']['max_step_gap']=2
        viewer['status']='LIVE_VIEWER_CLOSED_BY_USER'
        self.assertFalse(support.visual_result(viewer,dict(cases=rows))['complete'])

    def test_viewer_core_never_shares_physical_core_or_smt_sibling(self):
        rows=[dict(cpu=i,package=0,core=i%3,maximum_frequency_kHz=1000,busy_fraction=.1) for i in range(6)]
        placement=dict(enabled=True,control_cpu=0,bridge_cpu=1,candidates=rows)
        self.assertIn(support.viewer_cpu(placement),(2,5))
        placement['candidates']=[r for r in rows if r['core']<2]
        with self.assertRaises(RuntimeError): support.viewer_cpu(placement)

    def test_accepted_summary_still_verifies_without_heavy_imports(self):
        source=Path(os.environ['MICRODUCK_ACCEPTED_REPORT'])
        code='''import sys,json
sys.path.insert(0,sys.argv[1])
from loop_evidence import verify_reports
r=json.load(open(sys.argv[2]))
a=verify_reports(r['simulator'],r['bridge'],r['orin_server'])
assert a==r['acceptance'] and a['full_loop_realtime_acceptance']
assert not {'torch','numpy','mujoco','rclpy','onnxruntime'} & set(sys.modules)
'''
        result=subprocess.run([sys.executable,'-I','-S','-c',code,str(HERE/'package'),str(source)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__=='__main__': unittest.main(verbosity=2)
