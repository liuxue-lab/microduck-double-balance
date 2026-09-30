"""Zero-PPO checks for the authorized 25-attempt F2 power-loss recovery."""
from datetime import datetime,timezone,timedelta
import hashlib,json,tempfile,unittest
from pathlib import Path
from mjlab_microduck.double_balance_stage09_budget import (
    BudgetLedger,create_ledger,activate_powerloss_recovery,attempt_caps,
    RECOVERY_COUNTS,ORIGINAL_RUNTIME_SHA,validate_local_runtime,
)
from mjlab_microduck.double_balance_stage09_plan import progress

class RecoveryTests(unittest.TestCase):
    def original(self,path):
        create_ledger(path,'5090',(datetime.now(timezone.utc)-timedelta(minutes=5)).isoformat())
        state=json.loads(path.read_text());state['charged_updates']=dict(RECOVERY_COUNTS)
        path.write_text(json.dumps(state));return state

    def test_activation_preserves_counts_and_start_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'ledger.json';before=self.original(path)
            self.assertTrue(activate_powerloss_recovery(path,'a'*64))
            self.assertFalse(activate_powerloss_recovery(path,'a'*64))
            after=json.loads(path.read_text())
            for key in before:
                if key!='maximum_total_updates':self.assertEqual(before[key],after[key],key)
            self.assertEqual(after['maximum_total_updates'],2025)
            ledger=BudgetLedger(path)
            self.assertEqual(ledger.profile_limit('F2'),525)
            self.assertEqual(ledger.profile_limit('F3'),500)
            self.assertGreaterEqual(ledger.snapshot()['elapsed_seconds'],before['elapsed_seconds'])

    def test_caps_still_apply_after_reload_and_all_lost_attempts_remain(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'ledger.json';self.original(path);activate_powerloss_recovery(path,'a'*64)
            state=json.loads(path.read_text());state['charged_updates']['F2']=524;state['charged_updates']['F3']=500
            path.write_text(json.dumps(state));ledger=BudgetLedger(path)
            self.assertEqual(ledger.charge_update('F2'),525)
            self.assertEqual(sum(ledger.snapshot()['charged_updates'].values()),2025)
            for name in ('F0','F1','F2','F3'):
                with self.assertRaises(ValueError):BudgetLedger(path).charge_update(name)

    def test_no_general_or_repeat_allowance_or_counter_rollback(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'ledger.json';before=self.original(path)
            before['charged_updates']['F2']=176;path.write_text(json.dumps(before))
            with self.assertRaises(ValueError):activate_powerloss_recovery(path,'a'*64)
            before['charged_updates']['F2']=175;path.write_text(json.dumps(before));activate_powerloss_recovery(path,'a'*64)
            state=json.loads(path.read_text());state['approved_recovery']['lost_updates']=26
            with self.assertRaises(ValueError):attempt_caps(state)
            state['approved_recovery']['lost_updates']=25;state['charged_updates']['F2']=150
            with self.assertRaises(ValueError):attempt_caps(state)

    def test_model_progress_and_adam_are_not_extended(self):
        self.assertEqual(progress(150)['expected_adam_steps'],103000)
        self.assertEqual(progress(500)['expected_adam_steps'],110000)
        with self.assertRaises(ValueError):progress(525)

    def test_original_evidence_admits_only_the_recorded_recovery_source_diff(self):
        repo=Path(__file__).resolve().parents[1]
        self.assertTrue(validate_local_runtime(repo,ORIGINAL_RUNTIME_SHA))
        with tempfile.TemporaryDirectory() as folder:
            other=Path(folder);(other/'docs/audits').mkdir(parents=True)
            for name in ('stage-09-runtime-hashes.json','stage-09-runtime-before-powerloss.json','stage-09-recovery-20261001.json'):
                (other/'docs/audits'/name).write_bytes((repo/'docs/audits'/name).read_bytes())
            path=other/'docs/audits/stage-09-runtime-hashes.json';data=json.loads(path.read_text())
            data['src/mjlab_microduck/tasks/mdp.py']='0'*64;path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):validate_local_runtime(other,ORIGINAL_RUNTIME_SHA)

if __name__=='__main__':unittest.main()
