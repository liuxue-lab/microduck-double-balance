import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

from support import loop_sim
import review_v2
import test_native_timing


class V3Evidence(unittest.TestCase):
    def rows(self):
        result=[]
        for i in range(3000):
            start=1_000_000_000+i*20_000_000
            duration=26_225_318 if i==18 else 11_000_000
            result.append(dict(step=str(i+1),release_ns=str(start),deadline_ns=str(start+20_000_000),
                finished_ns=str(start+duration),release_to_finish_ms=str(duration/1e6),
                reference_inference_ms='16.741282' if i==18 else '0.6'))
        return result

    def test_previous_miss_is_retained_with_exact_step_and_domain(self):
        rows=self.rows()
        expected=dict(deadline_misses=1,all_steps={'release_to_finish_ms':{'max':26.225318}})
        receipt=review_v2.timing_details(rows,expected)
        self.assertEqual(receipt['deadline_events'][0]['step'],19)
        self.assertEqual(receipt['deadline_events'][0]['reference_inference_ms'],16.741282)
        self.assertEqual(receipt['first_tick']['step'],1)
        self.assertEqual(receipt['peak_reference_tick']['step'],19)
        rows[19]['release_ns']=str(int(rows[19]['release_ns'])+1)
        with self.assertRaisesRegex(RuntimeError,'schedule'):
            review_v2.timing_details(rows,expected)

    def test_new_report_includes_first_tick_and_all_missed_tick_evidence(self):
        timing,result,_=test_native_timing.TimingTests().run_ticks([28,15,5])
        self.assertEqual(result['deadline_misses'],2)
        self.assertEqual([r['step'] for r in result['deadline_events']],[1,2])
        self.assertEqual(result['first_tick']['release_to_finish_ms'],28.)
        self.assertFalse(result['deadline_events_truncated'])
        self.assertEqual(result['worst_ticks'][0]['step'],1)
        self.assertIn('major_page_faults',result['first_tick'])
        self.assertEqual(result['resource_counters']['major_page_faults'],
                         sum(r['major_page_faults'] for r in timing.rows))

    def test_full_state_comparison_detects_a_single_changed_value(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=root/'old';b=root/'new';a.mkdir();b.mkdir()
            rows=[dict(case_id='quiet-2407'),dict(case_id='regression-2500')]
            for base in (a,b):
                for i in (1,2):
                    folder=base/('case-%02d'%i);folder.mkdir()
                    np.savez_compressed(folder/'state-data.npz',qpos=np.zeros((3001,3)),
                        qvel=np.zeros((3001,3)),ctrl=np.zeros((3001,14)))
            with patch.object(review_v2,'report_v2',return_value={}), \
                 patch.object(review_v2,'checked_export',side_effect=lambda report,name:a/name):
                good=review_v2.compare_states(b,rows)
                self.assertTrue(good['all_bit_identical'])
                changed=np.zeros((3001,3));changed[2000,1]=np.nextafter(0.,1.)
                np.savez_compressed(b/'case-02/state-data.npz',qpos=changed,
                    qvel=np.zeros((3001,3)),ctrl=np.zeros((3001,14)))
                bad=review_v2.compare_states(b,rows)
                self.assertFalse(bad['all_bit_identical'])
                self.assertTrue(bad['cases'][0]['bit_identical'])
                self.assertFalse(bad['cases'][1]['fields']['qpos']['bit_identical'])
