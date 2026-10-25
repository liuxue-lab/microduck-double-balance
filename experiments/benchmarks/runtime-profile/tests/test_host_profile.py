"""Profiler scope, original failure semantics, and evidence-selection checks."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'package'))
from host_profile import HostProfile
import profile_contract as contract
import profile_ops as ops


class HostProfileTests(unittest.TestCase):
    def test_nested_elapsed_and_exclusive_time_are_not_double_counted(self):
        clock=iter((0,5_000_000,9_000_000,20_000_000))
        p=HostProfile(clock=lambda:next(clock));obj=NS(inner=lambda x:x+1)
        obj.outer=lambda x:obj.inner(x)*2
        original=dict(vars(obj))
        p.attach(obj,'outer','parent');p.attach(obj,'inner','child')
        self.assertEqual(obj.outer(2),6)  # warmup does not consume timer samples
        p.start()
        try:self.assertEqual(obj.outer(2),6)
        finally:p.stop();p.close()
        r=p.report();self.assertEqual(vars(obj),original)
        self.assertEqual(r['scopes']['parent']['inclusive_sum_ms'],20.)
        self.assertEqual(r['scopes']['parent']['exclusive_sum_ms'],16.)
        self.assertEqual(r['scopes']['child']['inclusive_sum_ms'],4.)
        self.assertTrue(r['timing_semantics']['nested_values_are_not_additive'])

    def test_exception_unwinds_nested_stack_and_restores_class_method(self):
        class Thing:
            def inner(self):raise ValueError('original failure')
            def outer(self):return self.inner()
        o=Thing();p=HostProfile();p.attach(o,'outer','parent');p.attach(o,'inner','child')
        p.start()
        try:
            with self.assertRaisesRegex(ValueError,'original failure'):o.outer()
        finally:p.stop();p.close();p.close()
        self.assertIsNone(sys.getprofile());self.assertEqual(vars(o),{})
        r=p.report();self.assertTrue(r['stack_empty'])
        self.assertTrue(all(v['inclusive_ms']['count']==1 for v in r['scopes'].values()))

    def test_existing_profile_hook_is_preserved_and_no_window_started(self):
        hook=lambda *_:None;p=HostProfile()
        sys.setprofile(hook)
        try:
            with self.assertRaisesRegex(RuntimeError,'Existing Python profiling'):p.start()
            p.close();self.assertIs(sys.getprofile(),hook)
            self.assertEqual(p.windows,0)
        finally:sys.setprofile(None)

    def test_profile_json_retains_recursion_counts_and_callers(self):
        def recurse(n):return 1 if not n else recurse(n-1)+1
        p=HostProfile();p.start()
        try:self.assertEqual(recurse(4),5)
        finally:p.stop();p.close()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'profile.pstats';r=p.report(path)
            self.assertGreater(path.stat().st_size,0)
            rows=json.loads(json.dumps(r,allow_nan=False))['cprofile']['functions']
        row=next(v for v in rows if v['function']=='recurse')
        self.assertEqual((row['primitive_calls'],row['total_calls']),(1,5))
        edge=next(v for v in row['callers'] if v['function']=='recurse')
        self.assertEqual((edge['primitive_calls'],edge['total_calls']),(1,4))

    def test_partial_attachment_is_rolled_back(self):
        s=NS(raw=NS(sim=NS(step=lambda:7)),physics_step_original=lambda:8,guard=NS())
        a=s.raw.sim.step;b=s.physics_step_original;p=HostProfile()
        with self.assertRaises(AttributeError):p.attach_session(s)
        self.assertIs(s.raw.sim.step,a);self.assertIs(s.physics_step_original,b)
        self.assertFalse(p.saved);self.assertTrue(p.closed)

    def test_original_safety_failures_keep_counts_and_only_clear_owned_force(self):
        import torch
        from model_limits import LimitStop
        from test_packed_checks import make_session
        torch.set_num_threads(1)
        for fault in ('motor_nan','post_force','post_joint_nan','integrator_error','stale','foreign_force'):
            outcomes=[]
            for profiled in (False,True):
                s,_,events=make_session();s.physics_calls=1000
                if fault=='motor_nan':s.guard.groups[0].last_diagnostics['motor_torque'][0,4]=float('nan')
                if fault=='stale':s.guard.last_seen[0]=s.guard.groups[0].last_diagnostics
                if fault=='foreign_force':s.raw.sim.data.xfrc_applied[0,0,4]=.125
                original=s.physics_step_original
                def integrate(s=s,original=original):
                    if fault=='integrator_error':raise RuntimeError('original integrator failure')
                    value=original()
                    if fault=='post_force':s.raw.sim.data.xfrc_applied[0,0,4]=.125
                    if fault=='post_joint_nan':s.raw.sim.data.qpos[0,5]=float('nan')
                    return value
                s.physics_step_original=integrate
                p=HostProfile()
                if profiled:
                    for obj,name,label in ((s.raw.sim,'step','physics.total'),
                        (s,'physics_step_original','physics.original'),
                        (s.guard,'before_integration','before.read'),
                        (s.guard,'after_integration','after.read'),
                        (s.guard.audit,'before_integration','before.python'),
                        (s.guard.audit,'after_integration','after.python'),
                        (s,'_clear_owned','force.clear'),(s,'assert_zero_wrench','force.zero')):
                        p.attach(obj,name,label)
                    p.start()
                try:
                    with self.assertRaises((RuntimeError,LimitStop)) as caught:s.raw.sim.step()
                finally:p.stop();p.close()
                self.assertIsNone(sys.getprofile());self.assertFalse(p.stack);self.assertFalse(p.saved)
                self.assertFalse(s.owned_force_present)
                self.assertTrue(torch.equal(s.raw.sim.data.xfrc_applied[0,2],torch.zeros(6)))
                outcomes.append((type(caught.exception),str(caught.exception),events.integrated,
                    s.physics_calls,s.checked_physics_steps,s.guard.audit.snapshot(),s.pulses,
                    s.schedule_verified,s.raw.sim.data.xfrc_applied.tolist()))
            self.assertEqual(outcomes[0],outcomes[1],fault)


class ProfileContractTests(unittest.TestCase):
    def test_exact_latest_measurement_and_accepted_runtime_are_required(self):
        path=ROOT/'reference/command-views-mixed-index-measured.json';prior=ROOT/'reference/inference-loop-measured.json'
        r=contract.reviewed(path,prior)
        self.assertEqual(r['status'],'PC_COMMANDVIEW_MEASURED')
        self.assertEqual(r['recommendation']['choice'],'KEEP_INFERENCE_ONLY')
        with patch.object(contract,'sha',return_value='0'*64),self.assertRaises(RuntimeError):
            contract.reviewed(path,prior)
        changed=copy.deepcopy(r);changed['comparison']['full_audit_equal']=False
        original=contract.old.read
        with patch.object(contract.old,'read',side_effect=lambda p:changed if p==path else original(p)),self.assertRaises(RuntimeError):
            contract.reviewed(path,prior)

    def test_profiler_timing_never_selects_a_new_optimization(self):
        baseline=dict(full_environment_step_ms=dict(mean=25.,p95=27.))
        for mean,p95 in ((10.,11.),(25.,27.),(40.,45.)):
            for parity in (True,False):
                r=ops.recommendation(baseline,dict(full_environment_step_ms=dict(mean=mean,p95=p95)),dict(passed=parity))
                self.assertEqual(r['choice'],'KEEP_INFERENCE_ONLY');self.assertTrue(r['diagnostic_only'])
                self.assertFalse(r['production_activated']);self.assertFalse(r['full_loop_realtime_acceptance'])
                self.assertEqual(r['parity_passed'],parity)
                self.assertAlmostEqual(r['profile_overhead_mean_fraction'],mean/25.-1.)


if __name__=='__main__':unittest.main()
