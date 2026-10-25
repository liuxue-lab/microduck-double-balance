"""CPU regressions for durable runtime installation and pass/fail boundaries."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / 'package'))
import fixed_runtime as fixed
import finalize
import timeline_service as service
import probe_helper as helper

REPORT = Path(os.environ.get('MICRODUCK_REVIEWED_REPORT', str(fixed.EVIDENCE if fixed.EVIDENCE.exists() else fixed.SOURCE_REPORT)))


class ReviewedEvidence(unittest.TestCase):
    def test_actual_gpu_report_is_complete(self):
        report = fixed.reviewed(REPORT)
        self.assertTrue(service.project_ok(report))
        self.assertEqual(report['arms']['inference-only']['full_environment_step_ms']['over_20ms'], 150)
        self.assertFalse(report['full_loop_realtime_acceptance'])

    def test_changed_report_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'report.json'
            path.write_bytes(REPORT.read_bytes() + b' ')
            with self.assertRaisesRegex(RuntimeError, 'reviewed successful'):
                fixed.reviewed(path)

    def test_cpu_only_or_incomplete_project_never_passes(self):
        report = json.loads(REPORT.read_bytes())
        for change in ('gpu', 'parity', 'restored', 'audit', 'mixed', 'window'):
            row = copy.deepcopy(report)
            if change == 'gpu': row['arms']['timeline']['cuda_timeline']['summary']['cuda_kernel_events'] = 0
            if change == 'parity': row['comparison']['all_numeric_state_differences_zero'] = False
            if change == 'restored': row['arms']['timeline']['runtime_bindings_restored'] = False
            if change == 'audit': row['comparison']['full_audit_equal'] = False
            if change == 'mixed': row['application_libraries_verified'] = False
            if change == 'window': row['arms']['timeline']['cuda_timeline']['completed_steps'] = list(range(101, 108))
            with self.subTest(change=change): self.assertFalse(service.project_ok(row))

    def test_minimal_gpu_proof_has_exact_library_and_warp_coverage(self):
        report = json.loads(REPORT.read_bytes())
        path = Path(report['candidate_library'])
        for mode, p in report['probes'].items():
            self.assertTrue(service.minimal_ok(p, mode, path))
            bad = copy.deepcopy(p)
            bad['result']['loaded_cupti'].append(copy.deepcopy(bad['result']['loaded_cupti'][0]))
            self.assertFalse(service.minimal_ok(bad, mode, path))
            bad = copy.deepcopy(p)
            bad['timed_out'] = True
            self.assertFalse(service.minimal_ok(bad, mode, path))


class PersistentInstall(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / 'fixed'
        self.src = self.base / 'source'
        self.src.mkdir()
        self.original = self.base / 'original.so'
        self.original.write_bytes(b'original untouched')
        self.source_report = self.base / 'source-report.json'
        self.source_report.write_bytes(b'{"reviewed":true}\n')
        self.lib_files = {}
        for n in fixed.LIB_FILES:
            data = ('verified library ' + n).encode()
            (self.src / n).write_bytes(data)
            self.lib_files[n] = (len(data), hashlib.sha256(data).hexdigest())
        self.changes = dict(ROOT=self.root, LIB_DIR=self.root / 'runtime/cupti-12.9.79',
            LIBRARY=self.root / 'runtime/cupti-12.9.79/libcupti.so.12', SOURCE_LIB_DIR=self.src,
            EVIDENCE=self.root / 'evidence/reviewed-repair-report.json', SOURCE_REPORT=self.source_report,
            OLD_LIBRARY=self.original, OLD_SHA=fixed.sha(self.original), LIB_FILES=self.lib_files,
            REVIEW_SHA=fixed.sha(self.source_report))
        self.patcher = patch.multiple(fixed, **self.changes)
        self.patcher.start()
        self.review = patch.object(fixed, 'reviewed', return_value={'status': 'reviewed'})
        self.review.start()

    def tearDown(self):
        self.review.stop()
        self.patcher.stop()
        self.temp.cleanup()

    def test_install_idempotence_source_retention_and_license(self):
        original = self.original.read_bytes()
        first = fixed.install_runtime()
        mtimes = {p.name: p.stat().st_mtime_ns for p in fixed.LIB_DIR.iterdir()}
        self.assertEqual(first, fixed.install_runtime())
        self.assertEqual(mtimes, {p.name: p.stat().st_mtime_ns for p in fixed.LIB_DIR.iterdir()})
        self.assertEqual(original, self.original.read_bytes())
        self.assertEqual(self.source_report.read_bytes(), fixed.EVIDENCE.read_bytes())
        self.assertTrue((self.src / 'NVIDIA-License.txt').exists())
        self.assertTrue((fixed.LIB_DIR / 'NVIDIA-License.txt').exists())

    def test_conflicting_runtime_never_overwritten(self):
        fixed.install_runtime()
        fixed.LIBRARY.write_bytes(b'conflict to preserve')
        with self.assertRaisesRegex(RuntimeError, 'file changed'):
            fixed.install_runtime()
        self.assertEqual(fixed.LIBRARY.read_bytes(), b'conflict to preserve')

    def test_source_corruption_never_committed(self):
        (self.src / 'libcupti.so.12').write_bytes(b'corrupted source')
        with self.assertRaises(RuntimeError): fixed.install_runtime()
        self.assertFalse(fixed.LIB_DIR.exists())
        self.assertFalse(fixed.EVIDENCE.exists())

    def test_symlink_directory_is_rejected(self):
        fixed.LIB_DIR.parent.mkdir(parents=True)
        fixed.LIB_DIR.symlink_to(self.src, target_is_directory=True)
        with self.assertRaises(RuntimeError): fixed.install_runtime()

    def test_environment_is_child_only_and_no_implicit_fallback(self):
        fixed.install_runtime()
        clean = {'PATH': '/usr/bin', 'LD_LIBRARY_PATH': '/approved'}
        before = dict(os.environ)
        env = fixed.fixed_environment(clean)
        self.assertEqual(env['LD_PRELOAD'], str(fixed.LIBRARY))
        self.assertEqual(env['LD_LIBRARY_PATH'], str(fixed.LIB_DIR) + ':/approved')
        self.assertNotIn('LD_PRELOAD', clean)
        self.assertEqual(dict(os.environ), before)
        with self.assertRaises(RuntimeError): fixed.fixed_environment({'LD_PRELOAD': '/foreign.so'})
        fixed.LIBRARY.unlink()
        with self.assertRaises(RuntimeError): fixed.fixed_environment(clean)

    def fake_child(self, ok=True, fail_mode=None):
        reviewed = json.loads(REPORT.read_bytes())
        def run(command, folder, seconds, env):
            folder.mkdir()
            if '--worker' not in command:
                result = {'status': 'FROZEN_INPUTS_VERIFIED', 'inputs': ['unchanged']}
            else:
                mode = command[command.index('--worker') + 1]
                result = copy.deepcopy(reviewed['probes'][mode]['result'])
                result['loaded_cupti'][0]['path'] = str(fixed.LIBRARY)
                if not ok and mode == fail_mode:
                    result['summaries'][0]['cuda_kernel_events'] = 0
            return {'exit_code': 0, 'timed_out': False, 'result': result}
        return run

    def test_finalize_success_uses_stable_path_and_no_simulation(self):
        reviewed = json.loads(REPORT.read_bytes())
        with patch.object(fixed, 'reviewed', return_value=reviewed), patch.object(helper, 'run_child', side_effect=self.fake_child()) as runner:
            self.assertEqual(finalize.apply(), 0)
            calls = runner.call_args_list
            self.assertEqual(len(calls), 4)
            self.assertTrue(all('timeline_worker.py' not in str(c) for c in calls))
            for call in calls:
                if '--worker' in call.args[0]: self.assertEqual(call.args[3]['LD_PRELOAD'], str(fixed.LIBRARY))
            report = json.loads((fixed.ROOT / 'fix-report.json').read_bytes())
            self.assertEqual(report['status'], 'TIMELINE_FIX_INSTALLED_VERIFIED')
            self.assertFalse(report['full_loop_realtime_acceptance'])
            self.assertEqual(report['new_project_physics_steps'], 0)
            self.assertEqual(finalize.apply(), 0)
            self.assertEqual(len(runner.call_args_list), 4)

    def test_failed_probe_stops_and_retains_after_check(self):
        reviewed = json.loads(REPORT.read_bytes())
        with patch.object(fixed, 'reviewed', return_value=reviewed), patch.object(helper, 'run_child', side_effect=self.fake_child(False, 'torch-only')) as runner:
            self.assertEqual(finalize.apply(), 2)
            self.assertEqual(len(runner.call_args_list), 3)
            report = json.loads((fixed.ROOT / 'fix-report.json').read_bytes())
            self.assertEqual(report['status'], 'TIMELINE_FIX_FAILED_PRESERVE')
            self.assertTrue(report['original_inputs_unchanged'])
            with self.assertRaisesRegex(RuntimeError, '不自动重复'): finalize.apply()


class ProcessLifecycle(unittest.TestCase):
    def test_future_timeline_uses_fixed_library_only_for_profiled_arm(self):
        evidence = json.loads(REPORT.read_bytes())
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'fixed'
            root.mkdir()
            probe = Path(td) / 'prior.json'
            probe.write_text(json.dumps({'status': 'PC_CUPTI_PROBE_COLLECTED'}))
            old_timeline = Path(td) / 'old-timeline.json'
            old_timeline.write_text('{}')
            old_library = Path(td) / 'original.so'
            old_library.write_bytes(b'original')
            library = Path(td) / 'fixed.so'
            old_sha = fixed.sha(old_library)
            calls = []
            def child(command, folder, seconds, env):
                folder.mkdir()
                mode = command[command.index('--mode') + 1]
                calls.append((mode, dict(env)))
                if mode == 'inspect':
                    result = copy.deepcopy(evidence['processes']['before']['result'])
                elif mode == 'compare':
                    result = copy.deepcopy(evidence['comparison'])
                else:
                    (folder / 'worker-report.json').write_text(json.dumps(evidence['arms'][mode]))
                    profiled = mode == 'timeline'
                    result = {'loaded_cupti': [{'path': str(library if profiled else old_library),
                        'sha256': service.NEW_LIB_SHA if profiled else old_sha,
                        'api_version': 28 if profiled else 26, 'return_code': 0}]}
                return {'exit_code': 0, 'timed_out': False, 'result': result}
            with patch.multiple(service, ROOT=root, PYTHON=Path(sys.executable), PRIOR=probe,
                                PRIOR_SHA=fixed.sha(probe), TIMELINE_PRIOR=old_timeline,
                                TIMELINE_SHA=fixed.sha(old_timeline), OLD_LIB=old_library, OLD_LIB_SHA=old_sha), \
                 patch.object(fixed, 'LIBRARY', library), \
                 patch.object(fixed, 'verify_installed', return_value={'minimal_passes': evidence['minimal_passes']}), \
                 patch.object(helper, 'run_child', side_effect=child), \
                 patch.object(helper, 'capture_command', return_value={'exit_code': 0}):
                self.assertEqual(service.execute(), 0)
            row = json.loads((root / 'report.json').read_bytes())
            self.assertEqual(row['status'], 'PC_TIMELINE_FIXED_MEASURED')
            self.assertFalse(row['full_loop_realtime_acceptance'])
            self.assertEqual(len(calls), 5)
            for mode, env in calls:
                if mode == 'timeline': self.assertEqual(env['LD_PRELOAD'], str(library))
                else: self.assertNotIn('LD_PRELOAD', env)

    def test_owned_timeout_reaps_child_and_keeps_report(self):
        with tempfile.TemporaryDirectory() as td:
            result = helper.run_child([sys.executable, '-c', 'import time; time.sleep(30)'], Path(td) / 'child', .08, dict(os.environ))
            self.assertTrue(result['timed_out'])
            with self.assertRaises(ProcessLookupError): os.kill(result['owned_worker_pid'], 0)
            self.assertTrue((Path(td) / 'child/process.json').is_file())

    def test_stop_rejects_reused_pid_without_signalling(self):
        with patch.object(helper, 'process_identity', return_value={'pid': 31337, 'state': 'S', 'start_ticks': 'new'}):
            self.assertFalse(service.controller_alive({'pid': 31337, 'start_ticks': 'old'}))
        self.assertFalse(service.controller_alive({'pid': 1, 'start_ticks': 'x'}))


if __name__ == '__main__':
    unittest.main(verbosity=2)
