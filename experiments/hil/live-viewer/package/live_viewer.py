"""Read-only native 3D window in its own process and physical CPU core."""
import argparse
import os
from pathlib import Path
import signal
import socket
import time
import traceback

from live_stream import latest
from live_support import write, require, sha, MODEL_SHA


def scene_ids(model):
    def one(rows, label):
        require(len(rows) == 1, 'Expected one visual ' + label)
        return rows[0]
    head = one([i for i in range(model.nsite)
                if model.site(i).name.split('/')[-1] == 'stage11_head_material_frame'], 'head site')
    def free(prefix):
        return one([i for i in range(1, model.nbody)
                    if model.body(i).name.startswith(prefix + '/') and model.body_jntnum[i]
                    and model.jnt_type[model.body_jntadr[i]] == 0], prefix)
    return dict(head=head, lower=free('ball'), upper=free('top_ball'))


class LiveScene:
    def __init__(self, mj, np, model):
        require((model.nq, model.nv, model.nu) == (35, 32, 14), 'Live model dimensions differ')
        self.mj, self.np, self.model = mj, np, model
        self.data = mj.MjData(model)
        self.ids = scene_ids(model)
        self.camera = mj.MjvCamera()
        mj.mjv_defaultCamera(self.camera)
        self.option = mj.MjvOption()
        self.scene = mj.MjvScene(model, maxgeom=10000)
        for name in ('mjRND_SHADOW', 'mjRND_REFLECTION'):
            self.scene.flags[getattr(mj.mjtRndFlag, name)] = False
        self.follow = True
        self.mode = 1
        self.reset_camera(1)

    def reset_camera(self, mode):
        self.mode, self.follow = mode, True
        self.camera.type = self.mj.mjtCamera.mjCAMERA_FREE
        self.camera.azimuth = 135. if mode == 1 else 45.
        self.camera.elevation = -20. if mode == 1 else -12.
        self.camera.distance = 1.20 if mode == 1 else .30

    def apply(self, packet):
        row = packet['row']
        self.np.copyto(self.data.qpos, row[:35])
        self.np.copyto(self.data.qvel, row[35:67])
        self.np.copyto(self.data.ctrl, row[67:])
        self.data.time = packet['sim_time']
        self.mj.mj_kinematics(self.model, self.data)
        self.mj.mj_comPos(self.model, self.data)
        self.mj.mj_camlight(self.model, self.data)

    def update(self):
        if self.follow:
            a = self.data.xpos[self.ids['lower']] if self.mode == 1 else self.data.site_xpos[self.ids['head']]
            self.camera.lookat[:] = (a + self.data.xpos[self.ids['upper']]) * .5
        self.mj.mjv_updateScene(self.model, self.data, self.option, None, self.camera,
                               self.mj.mjtCatBit.mjCAT_ALL, self.scene)


class Seen:
    def __init__(self):
        self.cases = {}
        self.last = None
        self.frames = 0
        self.invalid_order = 0

    def observe(self, packet, age_ms):
        key = (packet['case'], packet['step'])
        if self.last is not None and key < self.last:
            self.invalid_order += 1
            raise RuntimeError('Live state sequence went backwards')
        if key == self.last or packet['step'] == 0:
            return
        self.last = key
        self.frames += 1
        case = self.cases.setdefault(str(key[0]), dict(fresh_frames=0, first_step=key[1],
            last_step=key[1], max_age_ms=0., max_step_gap=0, display_coalesced_samples=0))
        gap = key[1] - case['last_step']
        case['max_step_gap'] = max(case['max_step_gap'], gap)
        case['display_coalesced_samples'] += max(0, gap - 1)
        case['last_step'] = key[1]
        case['fresh_frames'] += 1
        case['max_age_ms'] = max(case['max_age_ms'], age_ms)


def main(args):
    run = Path(args.run)
    result = dict(status='LIVE_VIEWER_FAILED_PRESERVE', target_fps=25, fresh_frames=0, cases={},
                  physics_steps=0, policy_inferences=0, motor_io=False, receives_only=True,
                  whole_window_close_stops_display_only=True, display_frames_may_coalesce=True)
    stopped = [False]
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    window = context = sock = scene = None
    glfw = None
    seen = Seen()
    try:
        os.nice(5)
        # All model loading, GL allocation and first render happen before READY.
        import numpy as np
        import mujoco as mj
        import glfw
        require(mj.__version__ == '3.10.0', 'Live viewer needs the existing MuJoCo 3.10.0')
        def no_integration(*_args, **_kwargs):
            raise RuntimeError('The read-only viewer cannot advance physics')
        for function in ('mj_step', 'mj_step1', 'mj_step2'):
            setattr(mj, function, no_integration)
        require(sha(args.model) == MODEL_SHA, 'Accepted visual model checksum differs')
        model = mj.MjModel.from_binary_path(str(args.model))
        model.vis.quality.shadowsize = 512
        model.vis.quality.offsamples = 0
        scene = LiveScene(mj, np, model)
        with np.load(args.state, allow_pickle=False) as states:
            initial = dict(row=np.concatenate([states[k][0] for k in ('qpos', 'qvel', 'ctrl')]), sim_time=0.)
        scene.apply(initial)
        require(glfw.init(), 'Cannot open desktop graphics; run from the laptop graphical terminal')
        glfw.window_hint(glfw.SAMPLES, 0)
        window = glfw.create_window(1100, 700, 'Microduck LIVE - Orin HIL', None, None)
        require(window is not None, 'GLFW window creation failed')
        glfw.set_window_size_limits(window, 640, 400, 1280, 800)
        glfw.make_context_current(window)
        glfw.swap_interval(0)
        context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150)
        sock = socket.socket(fileno=args.fd)
        sock.setblocking(False)
        cursor = [0., 0.]
        def move(_window, x, y):
            dx, dy = x-cursor[0], y-cursor[1]
            cursor[:] = [x, y]
            height = max(1, glfw.get_window_size(window)[1])
            shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
                     glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
            if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
                action = mj.mjtMouse.mjMOUSE_MOVE_H if shift else mj.mjtMouse.mjMOUSE_MOVE_V
                scene.follow = False
            elif glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
                action = mj.mjtMouse.mjMOUSE_ROTATE_H if shift else mj.mjtMouse.mjMOUSE_ROTATE_V
            elif glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS:
                action = mj.mjtMouse.mjMOUSE_ZOOM
            else:
                return
            mj.mjv_moveCamera(model, action, dx/height, dy/height, scene.scene, scene.camera)
        def scroll(_window, _x, y):
            mj.mjv_moveCamera(model, mj.mjtMouse.mjMOUSE_ZOOM, 0., -.05*y, scene.scene, scene.camera)
        def key(_window, code, _scan, action, _mods):
            if action != glfw.PRESS:
                return
            if code in (glfw.KEY_1, glfw.KEY_2):
                scene.reset_camera(1 if code == glfw.KEY_1 else 2)
            elif code == glfw.KEY_F:
                scene.follow = not scene.follow
            elif code == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
        glfw.set_cursor_pos_callback(window, move)
        glfw.set_scroll_callback(window, scroll)
        glfw.set_key_callback(window, key)
        def draw(left, right):
            width, height = glfw.get_framebuffer_size(window)
            if width <= 0 or height <= 0:
                return False
            viewport = mj.MjrRect(0, 0, width, height)
            scene.update()
            mj.mjr_render(viewport, scene.scene, context)
            mj.mjr_overlay(mj.mjtFontScale.mjFONTSCALE_150, mj.mjtGridPos.mjGRID_TOPLEFT,
                           viewport, left, right, context)
            glfw.swap_buffers(window)
            return True
        for _ in range(8):
            draw('LIVE HIL\nPreparing before control starts', '1 Full body | 2 Head | F Follow\nMouse: rotate / pan / zoom')
            glfw.poll_events()
        require(not glfw.window_should_close(window), 'Window closed before HIL was armed')
        write(run/'viewer-ready.json', dict(status='READY', pid=os.getpid(), model_sha256=MODEL_SHA,
              affinity=sorted(os.sched_getaffinity(0)), target_fps=25, warmed_frames=8,
              desktop=os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')))
        current = None
        last_progress = 0.
        begun = time.monotonic()
        while not stopped[0] and not glfw.window_should_close(window):
            frame_start = time.monotonic()
            glfw.poll_events()
            packet = latest(sock, np)
            if packet is not None:
                scene.apply(packet)
                current = packet
            age = (time.monotonic_ns()-current['stamp_ns'])/1e6 if current else 0.
            require(age >= 0, 'Live timestamp is in the future')
            if current:
                label = ('quiet-2407' if current['case'] == 1 else 'regression-2500')
                left = '%s | step %d/3000 | t %.2fs\nLatest completed cycle #%d: %.3f ms | misses: %d\nState age %.0f ms | %s' % (
                    label, current['step'], current['sim_time'], current['last_step'],
                    current['last_cycle_ms'], current['misses'], age,
                    'LIVE' if age <= 500 else 'WAITING - no fresh state')
            else:
                left = 'LIVE HIL ready\nWaiting for simulator / Orin startup'
            shown = draw(left, '1 Full body | 2 Head | F Follow\nMouse: rotate / pan / zoom\nEsc: close DISPLAY only | Ctrl+C in terminal: stop HIL')
            if shown and current and packet is not None:
                seen.observe(current, age)
            if frame_start-last_progress >= 1:
                write(run/'viewer-progress.json', dict(status='LIVE_VIEWER_RUNNING', fresh_frames=seen.frames,
                      cases=seen.cases, elapsed_seconds=frame_start-begun,
                      state_age_ms=age, minimized=not shown))
                last_progress = frame_start
            if (run/'VIEWER_FINISH.json').is_file():
                result['status'] = 'LIVE_VIEWER_COMPLETE'
                break
            if (run/'STOP.json').is_file():
                result['status'] = 'LIVE_VIEWER_STOPPED_PRESERVE'
                break
            require(frame_start-begun < 2400, 'Bounded viewer wall-clock limit reached')
            time.sleep(max(0., .04-(time.monotonic()-frame_start)))
        else:
            result['status'] = ('LIVE_VIEWER_STOPPED_PRESERVE' if stopped[0]
                                else 'LIVE_VIEWER_CLOSED_BY_USER')
        result['elapsed_seconds'] = time.monotonic()-begun
    except BaseException as exc:
        result.update(error=repr(exc), traceback=traceback.format_exc()[-7000:])
    finally:
        result.update(fresh_frames=seen.frames, cases=seen.cases, invalid_order=seen.invalid_order)
        if sock is not None:
            sock.close()
        if context is not None:
            context.free()
        if glfw is not None:
            if window is not None:
                glfw.destroy_window(window)
            glfw.terminate()
        write(run/'viewer-report.json', result)
    return 0 if result['status'] == 'LIVE_VIEWER_COMPLETE' else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--fd', type=int, required=True)
    raise SystemExit(main(parser.parse_args()))
