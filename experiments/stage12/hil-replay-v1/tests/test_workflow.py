"""Workflow integration, baseline review, and supervisor interruption/reentry."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE/'package'))
import replay_common as common
import replay_ops as ops


class BaselineReview(unittest.TestCase):
    def test_real_accepted_report_revalidates_without_heavy_imports(self):
        old = Path(os.environ.get('MICRODUCK_V3_PACKAGE', HERE.parent/'native-mujoco-v3/package'))
        code = '''import sys,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from loop_evidence import verify_reports, verify_own_package
r=json.loads(Path(sys.argv[2]).read_bytes())
a=verify_reports(r['simulator'],r['bridge'],r['orin_server'])
assert a==r['acceptance'] and a['full_loop_realtime_acceptance']
assert verify_own_package()=='a057b01ab6a3bdf2b33a825331fdc631ab39b094010d7df296858a54bb0b8ce4'
assert not {'torch','mujoco','numpy','rclpy','onnxruntime'} & set(sys.modules)
print('accepted v3: 6000 ticks, 0 deadline misses')
'''
        result = subprocess.run([sys.executable, '-I', '-S', '-c', code, str(old),
            str(HERE/'tests/fixtures/accepted-v3-summary.json')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)


class Supervisor(unittest.TestCase):
    def test_duplicate_run_stop_and_preserved_reentry(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); package = root/'package'; package.mkdir()
            (package/'replay_render.py').write_text('''import sys,time
from pathlib import Path
if sys.argv[1]=='probe': print('ready')
else:
 Path(sys.argv[2],'partial.txt').write_text('preserve me')
 while True: time.sleep(.1)
''')
            launcher = root/'launch.py'
            launcher.write_text('''import sys
from pathlib import Path
sys.path.insert(0, %r)
import replay_ops as ops
ops.ROOT=Path(%r);ops.HERE=Path(%r);ops.PYTHON=Path(sys.executable)
ops.verify_source=lambda:{}
ops.input_path=lambda *args:Path('unused-model')
ops.package_identity=lambda *args:'test-package'
raise SystemExit(ops.run())
'''%(str(HERE/'package'), str(root/'outputs'), str(package)))
            log = (root/'parent.log').open('wb')
            worker = subprocess.Popen([sys.executable, str(launcher)], stdout=log, stderr=log)
            try:
                deadline = time.monotonic()+8
                report_path = root/'outputs/report.json'
                while time.monotonic() < deadline:
                    if report_path.exists():
                        report = common.read(report_path)
                        if report.get('phase') == 'render_saved_states': break
                    if worker.poll() is not None: self.fail((root/'parent.log').read_text())
                    time.sleep(.03)
                else: self.fail('Supervisor never launched its render worker')
                run = Path(report['directory'])
                duplicate = subprocess.run([sys.executable, str(launcher)], capture_output=True, timeout=5)
                self.assertEqual(duplicate.returncode, 0)
                self.assertEqual(common.read(report_path)['directory'], str(run))
                with patch.object(ops, 'ROOT', root/'outputs'):
                    ops.stop()
                worker.wait(timeout=8)
                latest = common.read(report_path)
                self.assertEqual(latest['status'], 'HIL_REPLAY_FAILED_PRESERVE')
                self.assertTrue(latest['owned_worker_exited'])
                self.assertFalse(common.alive(latest.get('worker')))
                self.assertEqual((run/'partial.txt').read_text(), 'preserve me')
                again = subprocess.run([sys.executable, str(launcher)], capture_output=True, timeout=5)
                self.assertEqual(again.returncode, 2)
                self.assertEqual(common.read(report_path)['directory'], str(run))
                self.assertEqual(len(list((root/'outputs/runs').iterdir())), 1)
            finally:
                if worker.poll() is None: worker.terminate(); worker.wait(timeout=8)
                log.close()


if __name__ == '__main__': unittest.main(verbosity=2)
