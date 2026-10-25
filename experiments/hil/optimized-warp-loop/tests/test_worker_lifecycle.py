"""The real two-case supervisor stops after the first bad case, without a retry."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from support import c, loop_sim, case_row, adapter_stats
from telemetry import FIELDS


class WorkerLifecycle(unittest.TestCase):
    def scenario(self, failing):
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            run = Path(d)
            export = run/'export'
            export.mkdir()
            (export/'contract.json').write_text('{}')
            checkpoint = dict(path=str(loop_sim.old.INPUTS/'run/checkpoints/final6h_005304.pt'))
            calls, finished = [], []
            client = NS(seq=0, close=lambda:None, snapshot=lambda:{}, finish=lambda:finished.append(True))
            def evaluate_one(context, folder, case, cp, identity, boundary, progress):
                boundary()
                calls.append(case)
                stats=context.stats
                stats.update(compared_actions=3000, max_action_absolute_error=0., sessions=[NS()], adapters=adapter_stats())
                stats['timing'].rows = [dict(step=i+1, simulation_time_s=(i+1)*.02,
                    **{key:(None if key=='start_to_start_ms' and i==2999 else .5) for key in FIELDS}) for i in range(3000)]
                client.seq += 3000
                row=case_row(case)
                row['acceptance']['passed']=not failing
                return row
            for name,value in (('preflight',dict(package_identity='fixture')),('verify_inputs',{'unchanged':True})):
                stack.enter_context(patch.object(loop_sim.old,name,return_value=value))
            stack.enter_context(patch.object(loop_sim,'frozen',return_value={}))
            stack.enter_context(patch.object(loop_sim,'reviewed_evidence',return_value={'verified':True}))
            stack.enter_context(patch.object(loop_sim,'verify_own_package',return_value='fixture'))
            stack.enter_context(patch.object(loop_sim,'validate_report',return_value=checkpoint))
            stack.enter_context(patch.object(loop_sim.evaluate,'load_runtime',return_value=NS()))
            stack.enter_context(patch.object(loop_sim.evaluate,'evaluate_one',side_effect=evaluate_one))
            stack.enter_context(patch.object(loop_sim,'linked_context',side_effect=lambda b,c,i,s,k:NS(stats=s)))
            stack.enter_context(patch.object(loop_sim,'LoopClient',return_value=client))
            stack.enter_context(patch.object(loop_sim.signal,'signal'))
            stack.enter_context(patch.object(loop_sim.signal,'alarm'))
            stack.enter_context(patch('builtins.print'))
            rc=loop_sim.run(NS(run=d,export=str(export),socket='/fixture',session='a'*32))
            report=json.loads((run/'simulator-report.json').read_bytes())
            return rc,report,calls,finished

    def test_failed_first_case_is_preserved_without_second_case(self):
        rc,report,calls,finished=self.scenario(True)
        self.assertEqual(rc,2)
        self.assertEqual(len(calls),1)
        self.assertFalse(finished)
        self.assertEqual(report['status'],'LOOP_SIM_FAILED_PRESERVE')
        self.assertTrue(report['original_records_unchanged'])

    def test_two_cases_finish_once_without_realtime_or_hardware_claim(self):
        rc,report,calls,finished=self.scenario(False)
        self.assertEqual(rc,0)
        self.assertEqual(calls,list(c.CASES))
        self.assertEqual(finished,[True])
        self.assertEqual(report['completed_control_steps'],6000)
        self.assertFalse(report['full_loop_realtime_acceptance'])
        self.assertFalse(report['motor_io'])
