"""One-way, bounded nonblocking visualization; never receives control commands."""
import errno
import os
import socket
import struct
import time

MAGIC = b'MD12LV01'
HEADER = struct.Struct('<8sIIQIIdd')
WIDTH = 35 + 32 + 14
SIZE = HEADER.size + WIDTH * 8


def channel():
    writer, reader = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    for sock in (writer, reader):
        sock.setblocking(False)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    return writer, reader


class Publisher:
    def __init__(self, fd, case, timing, np):
        if os.get_blocking(fd):
            raise RuntimeError('Live display descriptor must be nonblocking')
        self.fd, self.case, self.timing = fd, case, timing
        self.buffer = bytearray(SIZE)
        self.payload = np.ndarray((WIDTH,), dtype='<f8', buffer=self.buffer, offset=HEADER.size)
        self.np = np
        self.sent = self.skipped = self.errors = self.misses = 0
        self.disabled = False
        self.last_counted = 0

    def publish(self, row, step):
        # This call is inside state_capture_ms and the formal cycle deadline.
        if self.disabled:
            self.skipped += 1
            return
        last = self.timing.rows[-1] if self.timing.rows else None
        last_step, duration = (last['step'], last['release_to_finish_ms']) if last else (0, 0.)
        if last_step > self.last_counted:
            self.misses += int(last['finished_ns'] > last['deadline_ns'])
            self.last_counted = last_step
        HEADER.pack_into(self.buffer, 0, MAGIC, self.case, step, time.monotonic_ns(),
                         last_step, self.misses, duration, step * .02)
        self.np.copyto(self.payload, row)
        try:
            count = os.write(self.fd, self.buffer)
            if count != SIZE:
                self.errors += 1
                self.disabled = True
            else:
                self.sent += 1
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                self.skipped += 1
            else:
                # A closed/crashed viewer cannot stop, stall, or alter HIL.
                self.errors += 1
                self.skipped += 1
                self.disabled = True

    def snapshot(self):
        return dict(published_frames=self.sent, skipped_display_frames=self.skipped,
                    send_errors=self.errors, publisher_disabled=self.disabled,
                    nonblocking=True, packet_bytes=SIZE, cadence_hz=50,
                    dropped_control_ticks=0, deadline_includes_publication=True,
                    channel='private inherited Unix datagram socket; one way')


def decode(raw, np):
    if len(raw) != SIZE:
        raise RuntimeError('Invalid live packet size')
    magic, case, step, stamp, last_step, misses, duration, sim_time = HEADER.unpack_from(raw)
    if (magic != MAGIC or case not in (1, 2) or not 0 <= step <= 3000 or
            not 0 <= last_step <= step or not 0 <= misses <= 3000 or
            not np.isfinite(duration) or duration < 0 or sim_time != step * .02):
        raise RuntimeError('Invalid live state identity or timing')
    row = np.frombuffer(raw, dtype='<f8', offset=HEADER.size)
    if not np.isfinite(row).all():
        raise RuntimeError('Nonfinite live state')
    return dict(case=case, step=step, stamp_ns=stamp, last_step=last_step,
                misses=misses, last_cycle_ms=duration, sim_time=sim_time, row=row)


def latest(sock, np):
    packet = None
    # Bounded even under a faulty producer; the real producer writes at 50Hz.
    for _ in range(64):
        try:
            raw = sock.recv(SIZE + 1)
        except BlockingIOError:
            break
        packet = decode(raw, np)
    return packet
