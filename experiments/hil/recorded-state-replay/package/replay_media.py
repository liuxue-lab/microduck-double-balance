"""Native MuJoCo views and bounded existing H.264 encoder."""
import importlib.util
from pathlib import Path
import shutil
import struct
import subprocess
import zlib
from replay_common import need, sha


def encoder_path():
    path = shutil.which('ffmpeg')
    if path is None and importlib.util.find_spec('imageio_ffmpeg') is not None:
        import imageio_ffmpeg
        path = imageio_ffmpeg.get_ffmpeg_exe()
    need(path and Path(path).is_file(), 'Existing ffmpeg not found; no dependencies were installed')
    probe = subprocess.run([path, '-hide_banner', '-encoders'], capture_output=True, timeout=15)
    need(probe.returncode == 0 and b'libx264' in probe.stdout, 'Existing ffmpeg has no libx264 encoder')
    return path


def png(path, frame):
    def chunk(kind, raw):
        return struct.pack('!I', len(raw))+kind+raw+struct.pack('!I', zlib.crc32(kind+raw)&0xffffffff)
    height, width, channels = frame.shape
    need(channels == 3 and str(frame.dtype) == 'uint8', 'Unexpected RGB frame')
    pixels = b''.join(b'\0'+frame[y].tobytes() for y in range(height))
    raw = b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR', struct.pack('!IIBBBBB', width, height, 8, 2, 0, 0, 0))
    raw += chunk(b'IDAT', zlib.compress(pixels))+chunk(b'IEND', b'')
    with Path(path).open('xb') as stream:
        stream.write(raw)


class Views:
    def __init__(self, mujoco, model, ids, width=640, height=480):
        self.mj, self.model, self.ids = mujoco, model, ids
        model.vis.global_.offwidth = max(width, model.vis.global_.offwidth)
        model.vis.global_.offheight = max(height, model.vis.global_.offheight)
        self.renderer = mujoco.Renderer(model, width=width, height=height)
        self.cameras = [mujoco.MjvCamera(), mujoco.MjvCamera()]
        for camera, azimuth, elevation in zip(self.cameras, (135., 45.), (-20., -12.)):
            mujoco.mjv_defaultCamera(camera)
            camera.azimuth, camera.elevation = azimuth, elevation
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        for name in ('mjRND_SHADOW', 'mjRND_REFLECTION'):
            self.renderer.scene.flags[getattr(mujoco.mjtRndFlag, name)] = False

    def frame(self, data):
        import numpy as np
        lower = data.xpos[self.ids['lower']]
        upper = data.xpos[self.ids['upper']]
        head = data.site_xpos[self.ids['head_site']]
        whole, detail = self.cameras
        whole.lookat[:] = (lower+upper)/2
        whole.distance = max(1.15, float(np.linalg.norm(upper-lower))*2.5)
        detail.lookat[:] = (head+upper)/2
        detail.distance = max(.30, float(np.linalg.norm(upper-head))*3.)
        images = []
        for camera in self.cameras:
            self.renderer.update_scene(data, camera=camera)
            images.append(self.renderer.render().copy())
        frame = np.ascontiguousarray(np.concatenate(images, axis=1))
        need(frame.dtype == np.uint8 and frame.ndim == 3, 'Invalid replay image')
        return frame

    def close(self):
        self.renderer.close()


class Video:
    def __init__(self, encoder, path, width, height, fps=50):
        self.encoder, self.path, self.fps = encoder, Path(path), fps
        self.width, self.height, self.count = width, height, 0
        self.log = self.path.with_suffix('.ffmpeg.log').open('xb')
        self.process = None
        try:
            self.process = subprocess.Popen([encoder, '-hide_banner', '-loglevel', 'error', '-nostdin',
                '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(fps),
                '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
                '-threads', '2', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-n', str(path)],
                stdin=subprocess.PIPE, stdout=self.log, stderr=self.log)
        except BaseException:
            self.log.close()
            raise

    def add(self, frame):
        need(frame.shape == (self.height, self.width, 3) and str(frame.dtype) == 'uint8', 'Video frame shape differs')
        self.process.stdin.write(frame.tobytes())
        self.count += 1

    def finish(self, expected):
        need(self.count == expected, 'Missing replay frames')
        self.process.stdin.close()
        self.process.wait(timeout=40)
        need(self.process.returncode == 0, 'Video encoder failed; inspect '+str(self.path.with_suffix('.ffmpeg.log')))
        self.log.close()
        decode = subprocess.run([self.encoder, '-nostdin', '-v', 'error', '-i', str(self.path),
            '-map', '0:v:0', '-f', 'null', '-', '-progress', 'pipe:1', '-nostats'],
            stdin=subprocess.DEVNULL, capture_output=True, timeout=120)
        frames = [int(line.split('=', 1)[1]) for line in decode.stdout.decode().splitlines() if line.startswith('frame=')]
        need(decode.returncode == 0 and frames and frames[-1] == expected,
             'Video full decode/frame count failed: '+decode.stderr.decode(errors='replace')[-1000:])
        return dict(path=str(self.path), bytes=self.path.stat().st_size, sha256=sha(self.path),
                    frames=expected, fps=self.fps, duration_seconds=expected/self.fps,
                    codec='H.264', pixel_format='yuv420p', width=self.width, height=self.height,
                    full_decode_verified=True, decoded_frames=frames[-1])

    def close(self):
        if self.process is not None and self.process.poll() is None:
            # Do not flush a blocked encoder pipe while handling a cancellation.
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        if self.process is not None and self.process.stdin and not self.process.stdin.closed:
            try:
                self.process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        if not self.log.closed:
            self.log.close()
