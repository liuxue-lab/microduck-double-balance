"""Real GLFW window + renderer on a test X display, using synthetic states."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import socket
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

from test_live import HERE, model_fixture, sample_row, np, mujoco, stream


@unittest.skipUnless(os.environ.get('MICRODUCK_TEST_DISPLAY'), 'A dedicated test X display is required')
class Window(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.display_process=None
        if os.environ.get('MICRODUCK_XVFB_ROOT'):
            root=Path(os.environ['MICRODUCK_XVFB_ROOT'])
            cls.display_log=tempfile.TemporaryFile()
            cls.display_process=subprocess.Popen([str(root/'usr/bin/Xvfb'),':95','-screen','0','1280x800x24',
                '-ac','-listen','tcp','-nolisten','local','-nolisten','unix','-fp',str(root/'usr/share/fonts/X11/misc')],
                stdout=cls.display_log,stderr=cls.display_log)
            for _ in range(100):
                try:
                    sock=socket.create_connection(('127.0.0.1',6095),timeout=.1);sock.close();break
                except OSError:
                    if cls.display_process.poll() is not None:
                        cls.display_log.seek(0);raise RuntimeError(cls.display_log.read().decode())
                    time.sleep(.05)
            else:
                cls.tearDownClass();raise RuntimeError('Dedicated test display did not start')

    @classmethod
    def tearDownClass(cls):
        if cls.display_process is not None:
            cls.display_process.terminate();cls.display_process.wait(timeout=8)
            cls.display_log.close()

    def exercise(self, close_early=False):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);model=model_fixture();row=sample_row(model)
            binary=root/'model.mjb';mujoco.mj_saveModel(model,str(binary),None)
            state=root/'state.npz';np.savez(state,qpos=[row[:35]],qvel=[row[35:67]],ctrl=[row[67:]])
            run=root/'run';run.mkdir()
            worker=root/'window.py'
            worker.write_text('''import sys,os
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,%r)
import live_viewer as view
import glfw
import mujoco as mj
import numpy as np
view.MODEL_SHA=%r
callbacks={};frames=[0];window=[None];context=[None]
real_key=glfw.set_key_callback
def key_set(w,cb):callbacks['key']=cb;window[0]=w;return real_key(w,cb)
glfw.set_key_callback=key_set
real_scroll=glfw.set_scroll_callback
def scroll_set(w,cb):callbacks['scroll']=cb;return real_scroll(w,cb)
glfw.set_scroll_callback=scroll_set
real_render=mj.mjr_render
def render(rect,scene,ctx):context[0]=ctx;return real_render(rect,scene,ctx)
mj.mjr_render=render
real_swap=glfw.swap_buffers
def swap(w):
 frames[0]+=1
 if frames[0]==40 and os.environ.get('MICRODUCK_TEST_MEDIA'):
  from PIL import Image
  width,height=glfw.get_framebuffer_size(w);pixels=np.empty((height,width,3),dtype=np.uint8)
  mj.mjr_readPixels(pixels,None,mj.MjrRect(0,0,width,height),context[0])
  Image.fromarray(pixels[::-1]).save(Path(os.environ['MICRODUCK_TEST_MEDIA'])/'synthetic-live-window.png')
 return real_swap(w)
glfw.swap_buffers=swap
real_poll=glfw.poll_events
def poll():
 real_poll()
 if frames[0]==18:callbacks['key'](window[0],glfw.KEY_2,0,glfw.PRESS,0)
 if frames[0]==24:callbacks['scroll'](window[0],0.,1.)
 if frames[0]==30:callbacks['key'](window[0],glfw.KEY_1,0,glfw.PRESS,0)
 if %r and frames[0]>=28:glfw.set_window_should_close(window[0],True)
glfw.poll_events=poll
raise SystemExit(view.main(SimpleNamespace(run=Path(%r),model=Path(%r),state=Path(%r),fd=int(sys.argv[1]))))
'''%(str(HERE/'package'),hashlib.sha256(binary.read_bytes()).hexdigest(),close_early,str(run),str(binary),str(state)))
            writer,reader=stream.channel()
            env=dict(os.environ,DISPLAY=os.environ['MICRODUCK_TEST_DISPLAY'],MUJOCO_GL='glfw',LP_NUM_THREADS='1')
            env.pop('PYOPENGL_PLATFORM',None)
            log=(root/'window.log').open('wb')
            process=subprocess.Popen([sys.executable,str(worker),str(reader.fileno())],env=env,
                                     pass_fds=(reader.fileno(),),stdout=log,stderr=log)
            reader.close()
            try:
                limit=time.monotonic()+20
                while not (run/'viewer-ready.json').is_file():
                    if process.poll() is not None:self.fail((root/'window.log').read_text())
                    if time.monotonic()>limit:self.fail('Window readiness timed out')
                    time.sleep(.02)
                for case in (1,2):
                    publisher=stream.Publisher(writer.fileno(),case,SimpleNamespace(rows=[]),np)
                    for step in range(61):
                        row[0]=np.sin(step*.1)*.02
                        publisher.publish(row,step)
                        time.sleep(.02)
                if not close_early:(run/'VIEWER_FINISH.json').write_text('{}')
                process.wait(timeout=8)
                result=json.loads((run/'viewer-report.json').read_text())
                self.assertEqual(result['physics_steps'],0)
                self.assertEqual(result['policy_inferences'],0)
                if close_early:
                    self.assertEqual(result['status'],'LIVE_VIEWER_CLOSED_BY_USER')
                    self.assertEqual(process.returncode,2)
                    self.assertGreater(publisher.errors,0)
                else:
                    self.assertEqual(result['status'],'LIVE_VIEWER_COMPLETE',result)
                    self.assertEqual(process.returncode,0)
                    self.assertEqual(set(result['cases']),{'1','2'})
                    self.assertTrue(all(c['fresh_frames']>=15 and c['last_step']>=59 for c in result['cases'].values()),result)
            finally:
                writer.close()
                if process.poll() is None:process.terminate();process.wait(timeout=8)
                log.close()

    def test_real_window_two_cases_camera_switch_and_exit(self):self.exercise()
    def test_real_window_close_does_not_block_publisher(self):self.exercise(close_early=True)


if __name__=='__main__':unittest.main(verbosity=2)
