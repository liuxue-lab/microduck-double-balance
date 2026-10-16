"""Synthetic WAL tests; no native operations are performed."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from integration.local_campaign import reconcile_native
from integration.planning import proposal


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.plan = proposal()
        self.ledgers = []
        for point in self.plan['points']:
            parent = self.out/point['name']
            parent.mkdir()
            (parent/'started-v1.json').write_text('{}')
            for p in point['owners']:
                owner = parent/('owner-%d'%p['logical_world'])
                owner.mkdir()
                ledger = owner/'native-ledger.jsonl'
                self.ledgers.append(ledger)
        for count,path in zip((2,3,4),self.ledgers,strict=True):
            rows = []
            for n in range(1,count+1):
                for kind in ('attempt','complete'):
                    rows.append(dict(seq=len(rows)+1,kind=kind,label='integrations',number=n))
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))

    def test_three_owners_add_baseline_once(self):
        r = reconcile_native(self.out,self.plan)
        self.assertTrue(r['all_operation_ledgers_clean'])
        self.assertEqual(r['cumulative_completed'],[674465,674465])
        self.assertEqual(r['cumulative_attempted'],[674466,674466])

    def test_pending_attempt_never_inferred_completed(self):
        with self.ledgers[-1].open('a') as f:
            f.write(json.dumps(dict(seq=9,kind='attempt',label='integrations',number=5))+'\n')
        r = reconcile_native(self.out,self.plan)
        self.assertFalse(r['all_operation_ledgers_clean'])
        self.assertFalse(r['integration_counts_exact'])
        self.assertEqual(r['cumulative_completed'],[674465,674466])
        self.assertEqual(r['cumulative_attempted'],[674467,674467])

    def test_corrupt_ledger_is_bounded_unknown_not_zero(self):
        self.ledgers[-1].write_text('corrupt\n')
        r = reconcile_native(self.out,self.plan)
        self.assertFalse(r['integration_counts_exact'])
        self.assertEqual(r['native_added']['completed_lower'],5)
        self.assertEqual(r['native_added']['completed_upper'],5+self.plan['points'][1]['owners'][1]['caps']['integrations'])

    def test_started_without_ledger_is_not_clean_acceptance(self):
        self.ledgers[-1].unlink()
        r = reconcile_native(self.out,self.plan)
        self.assertFalse(r['all_operation_ledgers_clean'])
        self.assertEqual(r['native_added']['attempted_lower'],5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
