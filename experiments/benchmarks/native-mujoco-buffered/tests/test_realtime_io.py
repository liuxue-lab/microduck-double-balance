import gc
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from realtime_io import DeferredTrace,ControlGC
import live_progress

class RealtimeIO(unittest.TestCase):
    def test_no_disk_io_during_trace_writes_and_partial_export_is_exact(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'trace.jsonl';trace=DeferredTrace(path,8192,3)
            data=['{"step":1}\n','{"step":2}\n']
            with patch('os.fsync',side_effect=AssertionError('No control-time fsync')):
                for row in data:trace.write(row);trace.flush()
                self.assertEqual(path.stat().st_size,0)
            trace.close();trace.close()
            self.assertEqual(path.read_bytes(),''.join(data).encode())
            self.assertEqual(trace.snapshot()['rows'],2)
            self.assertTrue(trace.snapshot()['durable'])

    def test_overflow_preserves_prefix_without_dropping_records(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'trace';trace=DeferredTrace(path,8,1)
            trace.write('first\n')
            with self.assertRaises(RuntimeError):trace.write('second\n')
            trace.close();self.assertEqual(path.read_text(),'first\n')

    def test_gc_restored_after_control_failure(self):
        original=gc.isenabled()
        for enabled in (False,True):
            gc.enable() if enabled else gc.disable()
            guard=ControlGC()
            try:
                self.assertFalse(gc.isenabled())
                raise RuntimeError('simulated failure')
            except RuntimeError:pass
            finally:guard.close()
            self.assertEqual(gc.isenabled(),enabled)
            self.assertEqual(guard.snapshot()['control_interval_collections'],[0,0,0])
            self.assertTrue(guard.snapshot()['restored'])
        gc.enable() if original else gc.disable()

    def test_full_progress_pipe_never_blocks_or_drops_control_data(self):
        read_fd,write_fd=os.pipe2(os.O_NONBLOCK)
        row=dict(total_control_steps=6000,completed_control_steps=25,detail='中文')
        try:
            with patch.dict(os.environ,NATIVE_PROGRESS_FD=str(write_fd)),patch.object(live_progress,'_fd',None):
                live_progress.configure()
                while True:
                    try:os.write(write_fd,b'x'*4096)
                    except BlockingIOError:break
                live_progress.emit(row)
                self.assertEqual(live_progress.snapshot()['coalesced_live_display_updates'],1)
                self.assertEqual(live_progress.snapshot()['control_ticks_dropped'],0)
                while True:
                    try:os.read(read_fd,65536)
                    except BlockingIOError:break
                live_progress.emit(row)
                self.assertEqual(live_progress.parse(os.read(read_fd,65536).decode()),row)
        finally:os.close(read_fd);os.close(write_fd)
