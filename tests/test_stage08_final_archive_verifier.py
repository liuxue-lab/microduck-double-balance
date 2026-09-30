"""Fail-closed checks for the local tag prerequisite; no simulator or GPU."""
import importlib.util
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

SPEC = importlib.util.spec_from_file_location('stage08_final_verifier',
    Path(__file__).resolve().parents[1] / 'scripts/verify_stage08_local_archive.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FinalArchiveVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / 'repo'
        self.audits = self.repo / 'docs/audits'
        self.root = self.base / 'artifacts/stage08/returned/hash/evidence'
        self.local = self.base / 'local-video'
        for directory in (self.audits, self.root, self.local):
            directory.mkdir(parents=True)
        self.final = {'selection_sha256': self.put('final-selection.json', b'selection'),
                      'batch_reference': {'sha256': self.put('final-evaluation/batch-evaluation.json', b'batch')},
                      'selection': {'development_reports': []}, 'models': []}
        models = []
        for i in range(5):
            checksum = self.put(f'models/{i}.pt', f'model-{i}'.encode())
            model = {'path': str(MODULE.CLOUD_ROOT / f'models/{i}.pt'), 'sha256': checksum}
            models.append(model)
            protocols = {}
            for protocol in ('nominal', 'dev', 'test-8201', 'test-8202', 'test-8203'):
                initial_sha = self.put(f'initial-states/{protocol}.pt', protocol.encode())
                trace_sha = self.put(f'evals/{i}/{protocol}/trace.pt', f'trace-{i}-{protocol}'.encode())
                report = {'status': 'PASS', 'protocol': protocol, 'checkpoint_sha256': checksum,
                          'initial_states': {'path': str(MODULE.CLOUD_ROOT / f'initial-states/{protocol}.pt'),
                                             'sha256': initial_sha}, 'trace_sha256': trace_sha,
                          'selection_sha256': self.final['selection_sha256']}
                relative = f'evals/{i}/{protocol}/evaluation.json'
                report_sha = self.put(relative, report)
                protocols[protocol] = {'report_reference': {'path': str(MODULE.CLOUD_ROOT / relative), 'sha256': report_sha}}
            self.final['models'].append({'identity': model, 'protocols': protocols})
        self.receipt = {'extracted_directory': str(self.root), 'ready_to_shutdown': True,
                        'no_remaining_cloud_work': True, 'necessary_files_returned_and_sha256_verified': True,
                        'verification': {'models': models}}
        self.write_audit('stage-08-return-verified.json', self.receipt)
        self.write_audit('stage-08-final-evaluation-review.json', self.final)
        self.write_audit('stage-08-local-render-report.json',
                         {'output_directory': str(self.local), 'videos': [{} for _ in range(9)], 'new_ppo_updates': 0})
        package = self.local / 'microduck-stage08-local-review.zip'
        manifest = []
        with zipfile.ZipFile(package, 'w') as z:
            for i in range(52):
                name = f'file-{i}'
                data = name.encode()
                z.writestr(name, data)
                manifest.append({'path': name, 'size_bytes': len(data),
                                  'sha256': MODULE.hashlib.sha256(data).hexdigest()})
            z.writestr('review-files.json', json.dumps(manifest))
        self.acceptance = {'technical_work_complete': True, 'policy_objective_status': 'PARTIAL',
                           'local_video_review': {'archive_sha256': MODULE.sha(package)}}
        self.refresh_audit_hashes()

    def put(self, relative, data):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data if isinstance(data, bytes) else json.dumps(data).encode())
        return MODULE.sha(path)

    def write_audit(self, name, data):
        (self.audits / name).write_text(json.dumps(data))

    def refresh_audit_hashes(self):
        self.acceptance['evidence_files'] = [
            {'path': str(p.relative_to(self.repo)), 'sha256': MODULE.sha(p)}
            for p in self.audits.glob('*.json') if p.name != 'stage-08-final-acceptance.json']
        self.write_audit('stage-08-final-acceptance.json', self.acceptance)

    def run_verifier(self):
        with patch.object(MODULE, 'REPO', self.repo), \
             patch.object(MODULE.subprocess, 'check_output', return_value='a'*40 + '\n'), \
             redirect_stdout(io.StringIO()):
            MODULE.main()

    def test_complete_evidence_passes_without_gpu_or_network(self):
        self.run_verifier()
        record = json.loads(next((self.root.parents[2] / 'finalization').glob('*.json')).read_text())
        self.assertEqual(record['final_protocols'], 25)
        self.assertFalse(record['gpu_compute_executed'])
        self.assertFalse(record['cloud_contacted'])

    def test_corrupt_cloud_trace_stops_tag_prerequisite(self):
        (self.root / 'evals/4/test-8203/trace.pt').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'Cloud metric trace differs'):
            self.run_verifier()
        self.assertFalse((self.root.parents[2] / 'finalization').exists())

    def test_mismatched_report_model_rejected_even_with_updated_report_hash(self):
        relative = 'evals/0/nominal/evaluation.json'
        report = MODULE.read(self.root / relative)
        report['checkpoint_sha256'] = 'other-model'
        self.final['models'][0]['protocols']['nominal']['report_reference']['sha256'] = self.put(relative, report)
        self.write_audit('stage-08-final-evaluation-review.json', self.final)
        self.refresh_audit_hashes()
        with self.assertRaisesRegex(ValueError, 'Final report/model mismatch'):
            self.run_verifier()

    def test_changed_video_package_rejected(self):
        with (self.local / 'microduck-stage08-local-review.zip').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'Local video package differs'):
            self.run_verifier()


if __name__ == '__main__':
    unittest.main()
