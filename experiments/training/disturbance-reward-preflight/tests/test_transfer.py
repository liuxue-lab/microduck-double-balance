"""Offline artifact inventories and ZIP output; no remote git or training."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'package'))
import transfer
import common


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.slot=Path(self.temp.name)
        for n in ('run/checkpoints','run/candidates','evaluations','logs','package','cache'):(self.slot/n).mkdir(parents=True)
        self.model=self.slot/'run/checkpoints/push10_000020.pt';self.model.write_bytes(b'new-model')
        self.receipt=dict(path=str(self.model),sha256=common.sha(self.model),bytes=self.model.stat().st_size,
                          completed_updates=20,policy_updates=6030)
        common.write(self.slot/'run/latest.json',self.receipt)
        common.write(self.slot/'run/candidates/candidate-final.json',self.receipt)
        common.write(self.slot/'started.json',{'started_utc':1000.})
        common.write(self.slot/'source-commit.json',{'commit':'a'*40})
        common.write(self.slot/'exit.json',{'status':'CAMPAIGN_COMPLETE'})
        self.state=dict(status='COMPUTE_COMPLETE_RETURNING',compute_active=False,
            processes_alive={'supervisor':True,'child':False},
            exit={'status':'CAMPAIGN_COMPLETE','checked_baseline_goal_passed':False,'disturbance_goal_passed':False})
        self.stack=contextlib.ExitStack()
        self.stack.enter_context(patch.object(transfer,'SLOT',self.slot))
        self.stack.enter_context(patch.object(transfer,'snapshot',return_value=self.state))
        self.stack.enter_context(patch.object(transfer,'input_checkpoint',return_value=self.receipt))
        self.stack.enter_context(patch.object(transfer,'verify_package',return_value='b'*64))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
    def tearDown(self):self.stack.close();self.temp.cleanup()

    def test_return_allowed_while_supervisor_only_handles_cpu_transfer(self):
        inventory=transfer.make_inventory()
        self.assertIn('run/checkpoints/push10_000020.pt',inventory['included'])
        self.assertFalse(inventory['disturbance_goal_passed'])

    def test_active_compute_blocks_packaging(self):
        self.state['compute_active']=True
        with self.assertRaisesRegex(RuntimeError,'not yet quiescent'):transfer.make_inventory()

    def test_large_log_is_preserved_and_disclosed_not_a_transfer_block(self):
        path=self.slot/'logs/huge.log';path.write_bytes(b'x'*100)
        with patch.object(transfer,'MAX_GROUP_BYTES',90):value=transfer.make_inventory()
        self.assertIn('logs/huge.log',value['omitted'])
        self.assertEqual(value['omitted']['logs/huge.log']['sha256'],common.sha(path))
        self.assertTrue(path.exists())

    def test_active_transfer_log_and_receipt_are_not_archived(self):
        (self.slot/'logs/automatic-transfer.log').write_text('still writing')
        common.write(self.slot/'transfer-worker.json',{'pid':1})
        value=transfer.make_inventory()
        self.assertNotIn('logs/automatic-transfer.log',value['included'])
        self.assertNotIn('transfer-worker.json',value['included'])

    def test_intermediate_checkpoint_is_kept_on_cloud_and_hashed(self):
        p=self.slot/'run/checkpoints/intermediate.pt';p.write_bytes(b'unselected')
        value=transfer.make_inventory();self.assertIn('run/checkpoints/intermediate.pt',value['omitted'])
        self.assertTrue(p.exists())

    def test_pack_only_is_repeatable_and_archive_hashes_are_checked(self):
        with patch.object(sys,'argv',['transfer','--pack-only']):transfer.main()
        index_path=self.slot/'transfer/final/return-index.json';index=common.read(index_path)
        self.assertEqual(index['schema'],'stage12-push10-final6h-return-v1')
        for name,pin in index['archives'].items():
            archive=index_path.parent/name;self.assertEqual(common.sha(archive),pin['sha256'])
            with zipfile.ZipFile(archive) as z:self.assertEqual(z.namelist(),pin['members'])
        before=index_path.read_bytes()
        with patch.object(sys,'argv',['transfer','--pack-only']):transfer.main()
        self.assertEqual(index_path.read_bytes(),before)

    def test_repair_uses_new_transfer_directory_and_preserves_old_archive(self):
        previous=self.slot/'transfer/final';previous.mkdir(parents=True)
        old=previous/'return-index.json';old.write_bytes(b'old immutable archive index')
        common.write(self.slot/'repair-applied.json',{'repair_id':'reward-scaling-v2'})
        with patch.object(sys,'argv',['transfer','--pack-only']):transfer.main()
        new=self.slot/'transfer/final-reward-scaling-v2/return-index.json'
        self.assertEqual(common.read(new)['session'],2)
        self.assertEqual(old.read_bytes(),b'old immutable archive index')


if __name__=='__main__':unittest.main(verbosity=2)
