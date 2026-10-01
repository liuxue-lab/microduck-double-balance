"""Offline tests for diagnostic correctness and frozen evaluation command scope."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

spec=importlib.util.spec_from_file_location('stage10',Path(__file__).with_name('microduck-stage10-diagnostics.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class DiagnosticTests(unittest.TestCase):
    def test_six_frozen_development_replays(self):
        jobs=m.job_specs()
        self.assertEqual(len(jobs),6)
        self.assertEqual([x['video_env_id'] for x in jobs],[None,99,None,99,None,99])
        self.assertEqual(len({x['name'] for x in jobs}),6)
        for j in jobs:
            cmd=m.evaluation_command('python','checkpoint.pt','out','inputs',j['video_env_id'])
            self.assertIn('--local',cmd);self.assertEqual(cmd[4],'evaluate')
            self.assertEqual(cmd[cmd.index('--protocol')+1],'dev')
            self.assertNotIn('train',cmd);self.assertNotIn('--reward-probe',cmd)
            self.assertNotIn('--ledger',cmd)

    def test_first_difference_ignores_ended_episodes_and_reports_actual_time(self):
        a=np.zeros((6,3,4),np.float32);b=a.copy();active=np.ones((6,3),bool)
        active[0,2]=False;b[0,2,0]=123.;b[2,1,2]=1e-8;b[4,0,0]=1e-4
        times=np.arange(1,7)*.02
        result=m.first_difference(a,b,active,times)
        self.assertEqual(result['sample_index'],2);self.assertEqual(result['env_ids'],[1])
        self.assertAlmostEqual(result['time_s'],.06)
        result=m.first_difference(a,b,active,times,1e-7)
        self.assertEqual(result['sample_index'],4)
        self.assertIsNone(m.first_difference(a,a,active,times))

    def test_nonfinite_active_samples_fail(self):
        a=np.zeros((3,2,4),np.float32);b=a.copy();b[1,1,0]=np.nan
        with self.assertRaisesRegex(RuntimeError,'Nonfinite'):
            m.first_difference(a,b,np.ones((3,2),bool),np.arange(1,4)*.02)

    def test_labels_cannot_be_pooled_or_compared_across_different_initial_states(self):
        left={'episodes':[dict(env_id=i,initial_state_id=str(i),success=s,official_timer_seconds=6.)
                          for i,s in enumerate([True,False])]}
        right=json.loads(json.dumps(left))
        right['episodes'][0]['success']=False;right['episodes'][1]['success']=True
        result=m.label_comparison(left,right)
        self.assertEqual(result['left_successes'],result['right_successes'])
        self.assertEqual(result['label_disagreements'],2)
        right['episodes'][1]['initial_state_id']='different'
        with self.assertRaisesRegex(RuntimeError,'initial IDs'):
            m.label_comparison(left,right)

    def test_timer_shadow_counts_terminal_run_not_best_run(self):
        self.assertEqual(m.trailing_count([True]*300+[False]+[True]*249),249)
        self.assertEqual(m.trailing_count([False]+[True]*250),250)
        self.assertEqual(m.trailing_count([True]*500+[False]),0)

    def test_hash_mismatch_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x';p.write_text('original');rows=m.inventory(Path(d))
            p.write_text('changed')
            with self.assertRaisesRegex(RuntimeError,'SHA256'):m.verify_inventory(d,rows)
            with self.assertRaisesRegex(RuntimeError,'Unsafe'):m.verify_inventory(d,[{'path':'../x','sha256':'x','size_bytes':1}])

    def test_repack_manifest_excludes_itself(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);m.write(root/'run.json',{'status':'STOPPED'})
            for _ in range(2):
                with patch('builtins.print'):m.package_review(root)
                rows=m.read(root/'review/review-files.json')
                self.assertNotIn('review-files.json',[r['path'] for r in rows])
                m.verify_inventory(root/'review',rows)

    def test_six_run_analysis_preserves_official_boundary_labels(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);jobs=m.job_specs();m.write(root/'plan.json',{'jobs':jobs})
            for number,job in enumerate(jobs):
                run=root/job['name']/'attempt-fixture';run.mkdir(parents=True)
                metric=np.zeros((500,128,3),np.float32)
                metric[250:,0,0]=1.;metric[-1,0,2]=4.999996185302734
                success=bool(number%2)
                if success:metric[:,1,0:2]=1.;metric[-1,1,2]=10.
                fields={k:np.zeros((500,128,1),np.float32) for k in m.FIELDS}
                fields['policy_action'][2,3,0]=number*1e-6
                fields['metric_values']=metric
                np.savez_compressed(run/'head-trace.npz',**fields,
                    active_first_episode=np.ones((500,128),bool),time_s=np.arange(1,501)*.02,
                    metric_columns=np.array(['cached_stable','cached_success','official_timer_s']))
                np.savez_compressed(run/'head-initial.npz',joint_pos_rad=np.zeros((128,4),np.float32))
                episodes=[dict(env_id=i,initial_state_id=str(i),steps=500,terminated=False,
                    success=success and i==1,official_timer_seconds=4.999996185302734 if i==0 else (10. if success and i==1 else 0.))
                    for i in range(128)]
                report=dict(status='PASS',protocol='dev',checkpoint_sha256=m.PRIMARY,git_head=m.BASE,
                    initial_states={'sha256':m.INITIAL},num_envs=128,horizon_seconds=10.,assistance=0.,
                    video_env_id=job['video_env_id'],reward_probe_only=False,episodes_count=128,
                    episodes=episodes,successes=int(success))
                m.write(run/'evaluation.json',report)
                m.write(run.parent/'completed.json',{'attempt':run.name,'files':m.inventory(run)})
            result=m.analyze(root)
            self.assertEqual(len(result['pairs']),15)
            self.assertEqual(result['pairs'][0]['label_disagreements'],1)
            self.assertEqual(result['pairs'][0]['first_recorded_differences']['policy_action']['sample_index'],2)
            self.assertEqual([j['timer_shadow_disagreements'] for j in result['jobs']],[1]*6)
            self.assertFalse(result['stage10_complete']);self.assertFalse(result['old_results_replaced'])
            for job in jobs:
                original=m.read(root/job['name']/'attempt-fixture/evaluation.json')
                self.assertFalse(original['episodes'][0]['success'])


if __name__=='__main__':unittest.main(verbosity=2)
