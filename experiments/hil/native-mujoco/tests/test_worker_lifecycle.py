"""The supervisor preserves first failures and never silently repeats a case."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from support import c, loop_sim, case_row, adapter_stats, native_preflight

class WorkerLifecycle(unittest.TestCase):
    def scenario(self,failing=False,realtime=False,changed_inputs=False):
        with tempfile.TemporaryDirectory() as d,ExitStack() as stack:
            run=Path(d);export=run/'export';export.mkdir()
            (export/'contract.json').write_text('{}')
            checkpoint=dict(path=str(loop_sim.old.INPUTS/'run/checkpoints/final6h_005304.pt'))
            calls,finished=[],[]
            client=NS(seq=0,close=lambda:None,snapshot=lambda:{},finish=lambda:finished.append(True))
            def evaluate_one(context,folder,case,cp,identity,boundary,progress):
                boundary();calls.append(case)
                context.stats.update(compared_actions=3000,max_action_absolute_error=0.,sessions=[True],adapters=adapter_stats())
                client.seq+=3000
                row=case_row(case,realtime);row['acceptance']['passed']=not failing
                return row
            stack.enter_context(patch.object(native_preflight,'preflight',return_value={'package_identity':'fixture'}))
            stack.enter_context(patch.object(loop_sim.old,'verify_inputs',side_effect=[{'same':True},{'same':not changed_inputs}]))
            for name,val in (('frozen',{}),('reviewed_evidence',{'verified':True}),('verify_own_package','fixture'),('validate_report',checkpoint)):
                stack.enter_context(patch.object(loop_sim,name,return_value=val))
            stack.enter_context(patch.object(loop_sim.evaluate,'load_runtime',return_value=NS()))
            stack.enter_context(patch.object(loop_sim.evaluate,'evaluate_one',side_effect=evaluate_one))
            stack.enter_context(patch.object(loop_sim,'linked_context',side_effect=lambda b,c,i,s,k:NS(stats=s)))
            stack.enter_context(patch.object(loop_sim,'LoopClient',return_value=client))
            stack.enter_context(patch('review_buffered_loop.compare_states',return_value={'all_bit_identical':True}))
            stack.enter_context(patch.object(loop_sim.Timing,'export',return_value=case_row(c.CASES[0],realtime)['timing']))
            stack.enter_context(patch.object(loop_sim.signal,'signal'))
            stack.enter_context(patch.object(loop_sim.signal,'alarm'))
            stack.enter_context(patch('builtins.print'))
            rc=loop_sim.run(NS(run=d,export=str(export),socket='/fixture',session='a'*32))
            return rc,json.loads((run/'simulator-report.json').read_bytes()),calls,finished

    def test_first_failed_case_stops_without_retry(self):
        rc,report,calls,finished=self.scenario(failing=True)
        self.assertEqual(rc,2);self.assertEqual(len(calls),1);self.assertFalse(finished)
        self.assertEqual(report['status'],'LOOP_SIM_FAILED_PRESERVE')

    def test_two_cases_and_distinct_realtime_results(self):
        for realtime in (False,True):
            rc,report,calls,finished=self.scenario(realtime=realtime)
            self.assertEqual(rc,0,report.get('error'));self.assertEqual(calls,list(c.CASES))
            self.assertEqual(finished,[True]);self.assertEqual(report['completed_control_steps'],6000)
            self.assertIs(report['full_loop_realtime_acceptance'],realtime)
            self.assertFalse(report['motor_io'])

    def test_cleanup_failure_cannot_keep_success_flag(self):
        rc,report,calls,finished=self.scenario(realtime=True,changed_inputs=True)
        self.assertEqual(rc,2);self.assertEqual(report['status'],'LOOP_SIM_FAILED_PRESERVE')
        self.assertFalse(report['full_loop_realtime_acceptance'])
        self.assertFalse(report['realtime_acceptance'])
