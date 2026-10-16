"""Offline regression for newly added orchestration. Never invokes cloud or physics."""
from pathlib import Path
import contextlib
import hashlib
import importlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import types
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'candidate')]
import common as c
import cloud_run as cr
import local as lc
import native_stage as ns


class Core(unittest.TestCase):
    def test_fixed_deadline(self):
        import datetime
        self.assertEqual(datetime.datetime.fromtimestamp(c.CAMPAIGN_DEADLINE,datetime.timezone.utc).isoformat(),'2026-10-17T03:53:14+00:00')

    def test_host_key_fingerprint(self):
        import base64
        raw=base64.b64decode(c.HOST_KEY.split()[2]);fp='SHA256:'+base64.b64encode(hashlib.sha256(raw).digest()).decode().rstrip('=')
        self.assertEqual(fp,c.HOST_FINGERPRINT)

    def test_caps_and_zero_training(self):
        points=[ns.point_plan(n) for n,_,_ in ns.POINTS]
        self.assertEqual(sum(x['max_integrations'] for x in points),8738240)
        self.assertEqual(sum(len(x['owners']) for x in points),7)
        for point in points:
            for p in point['owners']:
                self.assertEqual(p['caps']['PPO'],0);self.assertEqual(p['caps']['cloud'],0)
                self.assertEqual(p['tiers'],[32,64,128,256,512])
                from integration.native_factory import admission
                admission(p,ns.auth(p))
                bad=ns.auth(p);bad['canonical_before']={'completed':693528,'attempted':693529}
                with self.assertRaises(RuntimeError):admission(p,bad)

    def test_write_exclusive_and_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a';c.publish(p,b'old')
            with self.assertRaises(FileExistsError):c.publish(p,b'new')
            self.assertEqual(p.read_bytes(),b'old')
            with self.assertRaises(RuntimeError):c.same_or_new(p,b'new')

    def test_path_traversal(self):
        for x in ('../a','/a','a/../b','a\\b','./b',''):
            with self.assertRaises(RuntimeError):c.safe_name(x)

    def test_source_archive_410(self):
        with tempfile.TemporaryDirectory() as td:
            m=c.extract_source(ROOT/'cloud-inputs.zip',Path(td)/'inputs')
            self.assertEqual(len(m['files']),410)
            self.assertEqual(m,c.extract_source(ROOT/'cloud-inputs.zip',Path(td)/'inputs'))

    def test_completed_not_run_again(self):
        with tempfile.TemporaryDirectory() as td,patch('common.time.time',return_value=c.CAMPAIGN_DEADLINE-7200):
            seen=[]
            first=c.run_once(td,'one',lambda:seen.append(1) or {'done':True})
            self.assertEqual(first,c.run_once(td,'one',lambda:seen.append(2)))
            self.assertEqual(seen,[1])

    def test_failed_attempt_not_retried(self):
        with tempfile.TemporaryDirectory() as td,patch('common.time.time',return_value=c.CAMPAIGN_DEADLINE-7200):
            def fail():raise RuntimeError('interrupted')
            with self.assertRaises(RuntimeError):c.run_once(td,'one',fail)
            with self.assertRaises(RuntimeError):c.run_once(td,'one',lambda:None)
            self.assertTrue((Path(td)/'one.started.json').exists())
            self.assertTrue((Path(td)/'one.failed.json').exists())

    def test_budget_not_refreshed(self):
        with tempfile.TemporaryDirectory() as td,patch('common.time.time',return_value=c.CAMPAIGN_DEADLINE):
            with self.assertRaises(RuntimeError):c.run_once(td,'one',lambda:None)
            self.assertFalse((Path(td)/'one.started.json').exists())

    def test_setup_explicit_resume_keeps_failed_marker(self):
        with tempfile.TemporaryDirectory() as td,patch('common.time.time',return_value=c.CAMPAIGN_DEADLINE-7200):
            def fail(label):raise RuntimeError('download interrupted')
            with self.assertRaises(RuntimeError):c.preparation_once(td,'install',fail)
            with self.assertRaises(RuntimeError):c.preparation_once(td,'install',lambda label:{'label':label})
            r=c.preparation_once(td,'install',lambda label:{'label':label},retry=True)
            self.assertEqual(r['label'],'install-attempt-02')
            self.assertTrue((Path(td)/'install.started.json').exists())
            self.assertTrue((Path(td)/'install.failed.json').exists())
            self.assertEqual(c.preparation_once(td,'install',lambda label:self.fail('rerun'),retry=True),r)

    def test_feasibility_time_and_disk(self):
        prior={'n1-rollout':dict(status='NATIVE_POINT_PASS',maximum_step_seconds=200,evidence_bytes=10**8,public_steps_per_owner=24)}
        with tempfile.TemporaryDirectory() as td:
            r=cr.feasibility(ns.point_plan('n1-full-episode'),prior,Path(td),now=c.CAMPAIGN_DEADLINE-5000)
            self.assertFalse(r['admitted']);self.assertIn('REMAINING_TOTAL_TIME_INSUFFICIENT',r['reasons'])
            with patch('cloud_run.shutil.disk_usage',return_value=shutil._ntuple_diskusage(100,90,10)):
                r=cr.feasibility(ns.point_plan('n1-pilot'),{},Path(td),now=c.CAMPAIGN_DEADLINE-8000)
                self.assertIn('EVIDENCE_AND_RETURN_DISK_INSUFFICIENT',r['reasons'])

    def test_native_guard_and_cleanup(self):
        from integration.cloud_limits import check_boundary
        with patch.dict(os.environ,MD12_NATIVE_DEADLINE='1'):
            with self.assertRaises(RuntimeError):check_boundary(ROOT)
            check_boundary(ROOT,force=True)

    def test_environment_clears_git(self):
        with patch.dict(os.environ,{'GIT_SSH_COMMAND':'bad','PYTHONPATH':'bad','UV_PROJECT_ENVIRONMENT':'old'}):
            env=c.native_env()
            self.assertFalse(any(k.startswith(('GIT_','UV_')) for k in env))
            self.assertNotIn('PYTHONPATH',env)


class Transfer(unittest.TestCase):
    def test_local_prepare_repeat_reuses_commit(self):
        with tempfile.TemporaryDirectory() as td:
            t=Path(td);repo=t/'repo';repo.mkdir();root=t/'package';root.mkdir();base=t/'artifacts'
            c.command(['git','init',repo]);c.command(['git','-C',repo,'config','user.name','Offline test'])
            c.command(['git','-C',repo,'config','user.email','offline@example.invalid'])
            c.write(root/'payload.json',{'PPO':0});c.write(root/'PACKAGE.json',c.manifest(root,('PACKAGE.json',)))
            identity={'head':'73ee01930a031d76a9d57419afdfd4464a547ab3','status':'retained-user-change'}
            with patch.object(lc,'ROOT',root),patch.object(lc,'BASE',base),patch.object(lc,'REPO',repo),patch.object(lc,'local_identity',return_value=identity),contextlib.redirect_stdout(io.StringIO()):
                a=lc.prepare();b=lc.prepare();self.assertEqual(a,b)
            self.assertEqual(c.command(['git','-C',base/'transfer','rev-list','--count','HEAD']),'1')

    def test_complete_large_wal_git_return_and_second_receive(self):
        # Real local Git objects and >49 MiB WAL; SSH/GPU/network are not used.
        with tempfile.TemporaryDirectory() as td:
            t=Path(td);work=t/'work';work.mkdir();transfer=t/'transfer';transfer.mkdir()
            for d in ('reports','logs','phases','native/point/owner-0'):(work/d).mkdir(parents=True,exist_ok=True)
            c.write(work/'campaign-report.json',dict(status='fixture',PPO=0,cloud_training=0))
            c.write(work/'started.json',dict(no_retry=True))
            big=work/'native/point/owner-0/native-ledger.jsonl'
            # Large enough to exercise segmentation without a large random ZIP.
            with big.open('wb') as f:
                for _ in range(55):f.write(b'0123456789abcdef'*(1024**2//16))
            idx=cr.pack_evidence(work,transfer)
            self.assertEqual(idx['file_count'],3)
            self.assertGreater(len(idx['members']),3)
            self.assertTrue(all(a['bytes']<49*1024**2 for a in idx['archives']))
            self.assertTrue(all(p['bytes']<=512*1024 for a in idx['archives'] for p in a['parts']))
            self.assertEqual(idx,cr.pack_evidence(work,transfer))
            row=dict(run_id='FIXTURE',commit='a'*40,package_sha256='b'*64)
            c.write(transfer/'input-identity.json',{'commit':row['commit'],'package_sha256':row['package_sha256']})
            c.write(transfer/'MANIFEST.json',c.manifest(transfer,('MANIFEST.json',)))
            bare=t/'origin.git';c.command(['git','init','--bare',bare])
            branch='stage12-cloud-execution-v1-return-fixture'
            with patch('common.ORIGIN',str(bare)):
                c.prepare_git(transfer,branch,dict(name='Offline test',email='offline@example.invalid'))
            commit=c.commit_git(transfer,'offline fixture')
            c.command(['git','-C',bare,'fetch',transfer,'HEAD:refs/heads/'+branch])
            base=t/'local';base.mkdir()
            with patch('local.BASE',base),patch('local.ORIGIN',str(bare)),contextlib.redirect_stdout(io.StringIO()):
                lc.receive(row);lc.receive(row)
            returned=base/'returned/evidence/native/point/owner-0/native-ledger.jsonl'
            self.assertEqual(c.info(big),c.info(returned))
            self.assertEqual(c.read(base/'return-verified.json')['commit'],commit)
            # A completed received byte stream is preserved, never overwritten.
            with self.assertRaises(RuntimeError):c.same_or_new(returned,b'tamper')

    def test_empty_and_partial_wal_reconcile(self):
        with tempfile.TemporaryDirectory() as td:
            point=ns.point_plan('n1-pilot');r=ns.reconcile(td,point)
            self.assertEqual(r['native_added']['attempted_upper'],0)
            owner=Path(td)/'owner-0';owner.mkdir()
            (owner/'native-ledger.jsonl').write_bytes(b'{"truncated')
            r=ns.reconcile(td,point)
            self.assertFalse(r['all_ledgers_exact'])
            self.assertGreaterEqual(r['native_added']['attempted_upper'],0)


class VectorFlow(unittest.TestCase):
    def exercise(self,name,early=False):
        # Explicit transport fixture, not real Torch/native acceptance evidence.
        class Values(list):
            def tolist(self):return list(self)
            def long(self):return Values(int(x) for x in self)
            def __or__(self,other):return Values(a or b for a,b in zip(self,other))
            def __and__(self,other):return Values(a and b for a,b in zip(self,other))
            def __invert__(self):return Values(not x for x in self)
        torch=types.ModuleType('torch');torch.bool=bool
        torch.full=lambda shape,value,**kw:Values([value]*shape[0])
        torch.zeros=lambda n,*shape:Values([0.0]*n)
        tdmod=types.ModuleType('tensordict');tdmod.TensorDict=dict
        import integration.process_owners as owners
        point=ns.point_plan(name)
        class Fake:
            def __init__(self,repo,output,plans,authorizations,*args,**kwargs):
                self.folder=Path(output);self.folder.mkdir();self.n=len(plans);self.steps=0;self.episodes=[0]*self.n
                self.resets=[1]*self.n;self.num_envs=self.n;self.transport=type('T',(),{'timeout':1000})()
            def inspect(self):
                return [dict(public_steps=self.steps,macros=self.steps*10,resets=self.resets[i],
                             counts={'integrations_completed':self.steps*1600,'integrations_attempted':self.steps*1600},
                             rng_sha256=str(self.resets[i]),observation_sha256=str(self.steps),episode_length=[self.episodes[i]]) for i in range(self.n)]
            def get_observations(self):return None
            def reset(self,ids):
                for i in ids:self.resets[i]+=1;self.episodes[i]=0
            def step(self,action):
                self.steps+=1;self.episodes=[n+1 for n in self.episodes]
                terminated=torch.full((self.n,),early and self.steps==2,dtype=torch.bool)
                truncated=torch.full((self.n,),self.steps==500,dtype=torch.bool)
                done=terminated|truncated
                for i,x in enumerate(done):
                    if x:self.resets[i]+=1;self.episodes[i]=0
                return None,torch.zeros(self.n),done.long(),{'stage12_transition':{'terminated':terminated,'truncated':truncated,'bootstrap_mask':truncated&~terminated}}
            def __enter__(self):return self
            def __exit__(self,*args):c.write(self.folder/'owner-closeout.json',{'all_clean':True})
        with tempfile.TemporaryDirectory() as td,patch.dict(sys.modules,{'torch':torch,'tensordict':tdmod}),patch.object(owners,'OwnerProcesses',Fake),patch.object(ns,'independent_review',return_value={'test_fixture':True}),contextlib.redirect_stdout(io.StringIO()):
            return ns.measure(Path(td),Path(td)/'point',point,10**12)
    def test_n1_flow(self):self.assertEqual(self.exercise('n1-pilot')['public_steps_per_owner'],3)
    def test_n2_flow(self):self.assertEqual(self.exercise('n2-pilot')['owners'],2)
    def test_rollout_flow(self):self.assertEqual(self.exercise('n2-rollout')['timed_steps_per_owner'],23)
    def test_full_episode_unforced_timeout(self):self.assertTrue(self.exercise('n1-full-episode')['full_10s_episode_proven'])
    def test_early_termination_never_claims_full_pass(self):
        r=self.exercise('n1-full-episode',early=True)
        self.assertEqual(r['status'],'FULL_EPISODE_NOT_COMPLETED');self.assertFalse(r['full_10s_episode_proven'])


if __name__=='__main__':unittest.main(verbosity=2)
