"""Immutable new authorization, accounting and time boundaries; no GPU work."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'package'))
import budget


class BudgetTests(unittest.TestCase):
    def test_six_hour_total_bounded_training_and_empty_new_ledger(self):
        b=budget.new(100.)
        self.assertEqual(b['deadline_utc'],21700.)
        self.assertEqual(b['train_stop_utc'],17200.)
        self.assertEqual(b['completed_updates'],0)
        self.assertFalse(b['automatic_extension'])
        self.assertTrue(budget.validate(b))

    def test_smoke_must_commit_five_before_formal(self):
        b=budget.new(0.)
        with self.assertRaises(RuntimeError):budget.reserve_update(b,'formal',1.)
        for i in range(5):
            budget.reserve_update(b,'smoke',1.+i)
            self.assertEqual(b['attempted_updates'],i+1)
            budget.commit_update(b,'smoke')
        with self.assertRaises(RuntimeError):budget.reserve_update(b,'smoke',10.)
        budget.reserve_update(b,'formal',10.);budget.commit_update(b,'formal')
        self.assertEqual(b['completed_updates'],6)

    def test_interrupted_attempt_is_spent_and_not_automatically_replayed(self):
        b=budget.new(0.);budget.reserve_update(b,'smoke',1.)
        with self.assertRaises(RuntimeError):budget.reserve_update(b,'smoke',2.)
        self.assertEqual(b['attempted_updates'],1);self.assertEqual(b['completed_updates'],0)
        # Safe retained checkpoint may be inspected without refunding an attempt.
        budget.reserve_evaluation(b,'selection','case-0','model-sha',100.)
        self.assertEqual(b['pending']['attempt'],1)

    def test_training_cannot_start_at_cutoff_and_eval_cannot_start_at_total_cutoff(self):
        b=budget.new(0.)
        with self.assertRaises(RuntimeError):budget.reserve_update(b,'smoke',17100.)
        budget.reserve_evaluation(b,'selection','last','model',21599.)
        with self.assertRaises(RuntimeError):budget.reserve_evaluation(b,'heldout','too-late','model',21600.)

    def test_deadline_and_ceiling_mutations_rejected(self):
        for key in ('deadline_utc','train_stop_utc','max_updates'):
            b=budget.new(0.);b[key]+=1
            with self.assertRaises(RuntimeError):budget.reserve_update(b,'smoke',1.)

    def test_evaluation_once_only_and_global_cap(self):
        b=budget.new(0.)
        budget.reserve_evaluation(b,'selection','0','sha',1.)
        with self.assertRaises(RuntimeError):budget.reserve_evaluation(b,'selection','0','sha',2.)
        budget.finish_evaluation(b,'0','result-sha')
        with self.assertRaises(RuntimeError):budget.finish_evaluation(b,'0','another')
        for i in range(1,27):budget.reserve_evaluation(b,'heldout',str(i),'sha',2.)
        with self.assertRaises(RuntimeError):budget.reserve_evaluation(b,'heldout','28','sha',3.)


if __name__=='__main__':unittest.main(verbosity=2)
