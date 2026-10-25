"""Standard-library checks for checkpoint selection and archive integrity."""
import hashlib
import io
import json
from pathlib import Path
import runpy
import tarfile
import tempfile
import unittest

C = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/collect_stage07_results.py'))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


class Stage07CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'cloud'
        self.root.mkdir()
        self.first = self.root / 'train-first'
        self.last = self.root / 'train-last'
        self.add_evaluation(self.first, 1000, .90)
        self.add_evaluation(self.last, 3000, .69)
        self.add_evaluation(self.last, 6000, .35)
        write_json(self.first / 'training.json', dict(initialization_sha256=C['INITIAL_SHA'], status='RUNNING'))
        self.training = dict(initialization_sha256=C['INITIAL_SHA'], status='TRAINING_COMPLETE',
                             finite_checks='PASS', completed_updates=6000, target_updates=6000,
                             latest_checkpoint=str(self.last / 'checkpoints/update_006000.pt'))
        write_json(self.last / 'training.json', self.training)
        (self.root / 'latest-training').symlink_to(self.last)

    def add_evaluation(self, run, update, stable):
        checkpoint = run / f'checkpoints/update_{update:06d}.pt'
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f'fixture checkpoint {update}'.encode())
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        write_json(checkpoint.with_suffix('.json'), dict(sha256=digest, size_bytes=checkpoint.stat().st_size,
                                                       completed_updates=update))
        write_json(run / f'evaluations/update_{update:06d}/evaluation.json', dict(
            status='PASS', protocol=C['PROTOCOL'], num_envs=64, horizon_seconds=10., assistance=0.,
            success_fraction=0., mean_survival_seconds=10., mean_stable_fraction=stable,
            mean_top_center_error_m=.004, completed_updates=update,
            checkpoint=str(checkpoint), checkpoint_sha256=digest))

    def test_selects_across_segments_and_verifies_download(self):
        archive = Path(self.temp.name) / 'result.tar.gz'
        with archive.open('wb') as out:
            C['write_archive'](self.root, out)
        destination = Path(self.temp.name) / 'extracted'
        destination.mkdir()
        manifest = C['verify_extract'](archive, destination)
        self.assertEqual(manifest['selected']['best_nominal']['completed_updates'], 1000)
        self.assertEqual(manifest['selected']['final_state']['completed_updates'], 6000)
        self.assertEqual(len(list(destination.rglob('*.pt'))), 2)
        self.assertTrue((destination / 'train-last/checkpoints/update_003000.json').is_file())
        self.assertFalse((destination / 'train-last/checkpoints/update_003000.pt').exists())

    def test_refuses_unfinished_run(self):
        self.training['status'] = 'RUNNING'
        write_json(self.last / 'training.json', self.training)
        with self.assertRaisesRegex(ValueError, 'has not finished'):
            C['collect_plan'](self.root)

    def test_rejects_corrupt_checkpoint_on_cloud(self):
        (self.first / 'checkpoints/update_001000.pt').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            C['collect_plan'](self.root)

    def test_refuses_missing_final_evaluation(self):
        (self.last / 'evaluations/update_006000/evaluation.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Final evaluation'):
            C['collect_plan'](self.root)

    def test_rejects_traversal_before_extraction(self):
        archive = Path(self.temp.name) / 'unsafe.tar.gz'
        with tarfile.open(archive, 'w:gz') as out:
            info = tarfile.TarInfo('../escaped')
            info.size = 1
            out.addfile(info, io.BytesIO(b'x'))
        destination = Path(self.temp.name) / 'safe'
        destination.mkdir()
        with self.assertRaisesRegex(ValueError, 'Unsafe archive'):
            C['verify_extract'](archive, destination)
        self.assertFalse((destination.parent / 'escaped').exists())

    def test_rejects_transfer_content_mismatch(self):
        archive = Path(self.temp.name) / 'corrupt.tar.gz'
        manifest = dict(schema_version=1, stage=7, files={
            'checkpoint.pt': dict(size_bytes=1, sha256=hashlib.sha256(b'a').hexdigest())})
        with tarfile.open(archive, 'w:gz') as out:
            for name, content in [('checkpoint.pt', b'b'), ('collection-manifest.json', json.dumps(manifest).encode())]:
                info = tarfile.TarInfo(name)
                info.size = len(content)
                out.addfile(info, io.BytesIO(content))
        destination = Path(self.temp.name) / 'checked'
        destination.mkdir()
        with self.assertRaisesRegex(ValueError, 'Transfer checksum mismatch'):
            C['verify_extract'](archive, destination)


if __name__ == '__main__':
    unittest.main()
