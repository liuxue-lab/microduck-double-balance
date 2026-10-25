"""Exact helper arithmetic, complete Audit accounting and failure equivalence."""
import copy
import math
from pathlib import Path
import random
import struct
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'package'))
import model_limits as original
import scalar_audit as fast
import auditfast_contract as contract
import auditfast_ops as ops


def fixture(handling='record_and_continue_simulation'):
    joints=[dict(name=str(i),lower_rad=-1.,upper_rad=1.,soft_lower_rad=-.75,soft_upper_rad=.75)
            for i in range(14)]
    motors=[dict(name=str(i),local_joint_id=i,kt=1.,resistance_ohm=4.,max_pwm=1.,max_current_A=1.75,
                 force_lower_Nm=-1.,force_upper_Nm=1.) for i in range(14)]
    audit=original.Audit(joints,motors,joint_violation_handling=handling)
    audit.observe_joints([0.]*14,[0.]*14,initial=True)
    points=[dict(voltage=5.,velocity=.12,kd_scale=1.,duty=.1,torque=.095,ctrl=.095,
                 position=0.,position_error=.02) for _ in joints]
    return audit,points


def invoke(obj,name,*args,**kwargs):
    try:return getattr(obj,name)(*args,**kwargs),None
    except (Exception,original.LimitStop) as exc:return None,(type(exc),str(exc),getattr(exc,'event',None))


class ScalarHelperTests(unittest.TestCase):
    def check_tolerance(self,args):
        try:expected=original.tolerance(*args)
        except Exception as exc:
            with self.assertRaises(type(exc)) as caught:fast.tolerance(*args)
            self.assertEqual(str(caught.exception),str(exc))
        else:self.assertTrue(fast.exact(expected,fast.tolerance(*args)))

    def test_random_float64_bits_extremes_and_unbounded_ranges_match_exactly(self):
        rng=random.Random(20261024)
        values=[0.,-0.,math.inf,-math.inf,math.nan,1.,-1.,2.**-1074,-2.**-1074,
                float.fromhex('0x1.fffffffffffffp+1023')]
        values += [struct.unpack('!d',rng.getrandbits(64).to_bytes(8,'big'))[0] for _ in range(12000)]
        for i,x in enumerate(values):
            args=values[i:i+1+i%6]
            self.check_tolerance(args)
            lo=None if i%3==0 else values[(i+1)%len(values)]
            hi=None if i%4==0 else values[(i+2)%len(values)]
            self.assertTrue(fast.exact(original.excursion(x,lo,hi),fast.excursion(x,lo,hi)),(x,lo,hi))
        self.check_tolerance(())

    def test_original_method_code_and_global_helpers_remain_intact(self):
        audit,points=fixture();other,_=fixture();original_tolerance=original.tolerance
        before=dict(vars(audit));adapter=fast.ScalarAudit(audit)
        try:
            self.assertIs(audit.observe_joints.__func__.__code__,original.Audit.observe_joints.__code__)
            self.assertIsNot(audit.observe_joints.__func__.__globals__,vars(original))
            self.assertIs(original.tolerance,original_tolerance)
            self.assertIs(other.before_integration.__func__,original.Audit.before_integration)
            audit.before_integration(points);other.before_integration(points)
            self.assertTrue(fast.exact(fast.state(audit),fast.state(other)))
        finally:adapter.close();adapter.close()
        self.assertFalse(adapter.saved)
        self.assertEqual(set(vars(audit)),set(before))
        self.assertIs(audit.before_integration.__func__,original.Audit.before_integration)

    def test_source_or_preexisting_instance_override_rejected_without_mutation(self):
        a,_=fixture();before=dict(vars(a))
        with patch.object(fast,'sha',return_value='0'*64),self.assertRaises(RuntimeError):fast.ScalarAudit(a)
        self.assertEqual(vars(a),before)
        a.before_integration=lambda *_:None;before=dict(vars(a))
        with self.assertRaisesRegex(RuntimeError,'already replaced'):fast.ScalarAudit(a)
        self.assertEqual(vars(a),before)

    def test_shadow_detects_full_state_corruption_and_restores_bindings(self):
        a,points=fixture();adapter=fast.ScalarAudit(a)
        adapter.shadow.motor_stats['0']['peak_speed_rad_s']=999.
        try:
            with self.assertRaisesRegex(RuntimeError,'shadow comparison failed'):a.before_integration(points)
            self.assertIsNotNone(adapter.failure)
        finally:adapter.close()
        self.assertFalse(adapter.saved);self.assertNotIn('before_integration',vars(a))


class AuditAccountingTests(unittest.TestCase):
    def compare_call(self,a,b,name,*args,**kwargs):
        left,error_a=invoke(a,name,*copy.deepcopy(args),**copy.deepcopy(kwargs))
        right,error_b=invoke(b,name,*copy.deepcopy(args),**copy.deepcopy(kwargs))
        self.assertTrue(fast.exact(left,right))
        if error_a is None:self.assertIsNone(error_b)
        else:
            self.assertIsNotNone(error_b);self.assertEqual(error_a[:2],error_b[:2])
            self.assertTrue(fast.exact(error_a[2],error_b[2]))
        self.assertTrue(fast.exact(fast.state(a),fast.state(b)),name)
        self.assertTrue(fast.exact(a.snapshot(),b.snapshot()),name)
        return error_a

    def test_500_steps_keep_saturation_excursion_reentry_and_full_counters(self):
        a,points=fixture();b=copy.deepcopy(a);adapter=fast.ScalarAudit(b)
        try:
            for i in range(500):
                pos=[1.01 if i%40<8 else (.8 if i%40<20 else .2)]*14
                duty=1. if i%20<5 else .1
                for p in points:
                    p['voltage']=3.;p['duty']=duty;p['torque']=(3.*duty-.12)/4.;p['ctrl']=p['torque']
                    p['position']=pos[0];p['position_error']=.1
                self.assertIsNone(self.compare_call(a,b,'before_integration',points))
                forces=[1. if i%25<4 else -.2]*14
                self.assertIsNone(self.compare_call(a,b,'after_integration',pos,[.12]*14,forces))
            self.assertGreater(a.joint_stats['0']['violation_episodes'],1)
            self.assertGreater(a.motor_stats['0']['pwm_saturated_samples'],0)
            self.assertGreater(a.motor_stats['0']['force_saturated_samples'],0)
            self.assertEqual(adapter.checked,dict(before_integration=40,after_integration=40))
        finally:adapter.close()

    def test_numerical_tolerance_neighbors_and_stop_vs_record_modes(self):
        boundary=1.+original.tolerance(1.)
        vals=[math.nextafter(boundary,-math.inf),boundary,math.nextafter(boundary,math.inf),
              -math.nextafter(boundary,math.inf),.75,math.nextafter(.75+original.tolerance(.75),math.inf)]
        for handling in ('stop','record_and_continue_simulation'):
            for q in vals:
                a,_=fixture(handling);b=copy.deepcopy(a);adapter=fast.ScalarAudit(b)
                try:self.compare_call(a,b,'after_integration',[q]*14,[.1]*14,[0.]*14)
                finally:adapter.close()

    def test_nonfinite_mapping_formula_pwm_current_and_torque_failures_match(self):
        for fault in ('motor_nan','motor_inf','extra_nan','zero_voltage','gain','formula','ctrl','pwm',
                      'current','force_nan','force_inf','force_limit','position_nan','speed_inf',
                      'motor_count','force_count','joint_count','target_overflow'):
            with self.subTest(fault=fault):
                a,points=fixture();positions=[0.]*14;velocities=[0.]*14;forces=[0.]*14
                method='before_integration';args=(points,)
                p=points[0]
                if fault=='motor_nan':p['torque']=math.nan
                if fault=='motor_inf':p['velocity']=math.inf
                if fault=='extra_nan':p['extra']=math.nan
                if fault=='zero_voltage':p['voltage']=0.
                if fault=='gain':p['kd_scale']=0.
                if fault=='formula':p['torque']+=.001
                if fault=='ctrl':p['ctrl']+=.001
                if fault=='pwm':p['duty']=1.01;p['torque']=(5.*1.01-.12)/4.;p['ctrl']=p['torque']
                if fault=='current':p['voltage']=12.;p['duty']=.9;p['torque']=(12.*.9-.12)/4.;p['ctrl']=p['torque']
                if fault=='target_overflow':p['position']=p['position_error']=float.fromhex('0x1.fffffffffffffp+1023')
                if fault=='motor_count':points.pop()
                if fault.startswith('force') or fault in ('position_nan','speed_inf','joint_count'):
                    method='after_integration';args=(positions,velocities,forces)
                    if fault=='force_nan':forces[0]=math.nan
                    if fault=='force_inf':forces[0]=math.inf
                    if fault=='force_limit':forces[0]=1.01
                    if fault=='force_count':forces.pop()
                    if fault=='position_nan':positions[0]=math.nan
                    if fault=='speed_inf':velocities[0]=math.inf
                    if fault=='joint_count':positions.pop()
                b=copy.deepcopy(a);adapter=fast.ScalarAudit(b)
                try:self.assertIsNotNone(self.compare_call(a,b,method,*args))
                finally:adapter.close()


class CombinedPlantTests(unittest.TestCase):
    def test_1600_steps_full_original_plant_and_audit_remain_equal(self):
        import torch
        from test_packed_checks import make_session
        from bam_graph import BatchedAuditReads
        from packed_checks import PackedChecks
        from bridge_fixture import bridge_data
        torch.set_num_threads(1)
        a,set_a,ea=make_session();b,set_b,eb=make_session()
        a.raw.sim.data=bridge_data(a.raw.sim.data);b.raw.sim.data=bridge_data(b.raw.sim.data)
        original_reads=BatchedAuditReads(a.guard,torch);scalar=fast.ScalarAudit(b.guard.audit);packed=PackedChecks(b,torch)
        try:
            for i in range(1600):
                set_a(i);set_b(i)
                self.assertEqual(a.physics_step(),packed.physics_step())
                for name in ('qpos','qvel','ctrl','actuator_force','xfrc_applied'):
                    self.assertTrue(torch.equal(getattr(a.raw.sim.data,name),getattr(b.raw.sim.data,name)),(i,name))
            self.assertTrue(fast.exact(a.guard.audit.snapshot(),b.guard.audit.snapshot()))
            self.assertEqual(a.pulses,b.pulses);self.assertEqual(b.applied_physics_steps,50)
            self.assertEqual((b.guard.audit.motor_checks,b.guard.audit.joint_checks),(1600,1601))
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(ea.wrench_samples,eb.wrench_samples)))
            self.assertEqual(scalar.checked,dict(before_integration=40,after_integration=40))
        finally:scalar.close();original_reads.close()

    def test_combination_preserves_original_failure_phase_and_owned_force_cleanup(self):
        import torch
        from test_packed_checks import make_session
        from bam_graph import BatchedAuditReads
        from packed_checks import PackedChecks
        for fault in ('motor_nan','post_force','post_joint_nan','integrator_error','stale','foreign_force','cancel'):
            outcomes=[]
            for use_fast in (False,True):
                s,_,events=make_session();s.physics_calls=1000
                if fault=='motor_nan':s.guard.groups[0].last_diagnostics['motor_torque'][0,4]=float('nan')
                if fault=='stale':s.guard.last_seen[0]=s.guard.groups[0].last_diagnostics
                if fault=='foreign_force':s.raw.sim.data.xfrc_applied[0,0,4]=.125
                if fault=='cancel':
                    def cancel():raise RuntimeError('original cancellation')
                    s.boundary=cancel
                integrate=s.physics_step_original
                def faulty(s=s,integrate=integrate):
                    if fault=='integrator_error':raise RuntimeError('original integrator failure')
                    result=integrate()
                    if fault=='post_force':s.raw.sim.data.xfrc_applied[0,0,4]=.125
                    if fault=='post_joint_nan':s.raw.sim.data.qpos[0,5]=math.nan
                    return result
                s.physics_step_original=faulty
                scalar=fast.ScalarAudit(s.guard.audit) if use_fast else None
                reads=None if use_fast else BatchedAuditReads(s.guard,torch)
                method=PackedChecks(s,torch).physics_step if use_fast else s.physics_step
                try:
                    with self.assertRaises((RuntimeError,original.LimitStop)) as caught:method()
                    self.assertFalse(s.owned_force_present)
                    self.assertTrue(torch.equal(s.raw.sim.data.xfrc_applied[0,2],torch.zeros(6)))
                    outcomes.append((type(caught.exception),str(caught.exception),events.integrated,s.physics_calls,
                        s.checked_physics_steps,s.guard.audit.snapshot(),s.pulses,s.schedule_verified,
                        s.raw.sim.data.xfrc_applied.tolist()))
                finally:
                    if scalar:scalar.close()
                    if reads:reads.close()
            self.assertEqual(outcomes[0],outcomes[1],fault)


class AuditFastContractTests(unittest.TestCase):
    def test_exact_profile_report_and_all_windows_are_required(self):
        path=ROOT/'reference/runtime-profile-measured.json';prior=ROOT/'reference/inference-loop-measured.json'
        r=contract.reviewed(path,prior);self.assertEqual(r['status'],'PC_PROFILE_MEASURED')
        with patch.object(contract,'sha',return_value='0'*64),self.assertRaises(RuntimeError):contract.reviewed(path,prior)
        changed=copy.deepcopy(r);changed['arms']['profile']['detailed_profile']['measured_control_windows']=149
        read=contract.old.read
        with patch.object(contract.old,'read',side_effect=lambda p:changed if p==path else read(p)),self.assertRaises(RuntimeError):
            contract.reviewed(path,prior)

    def test_adoption_needs_both_speed_thresholds_and_parity(self):
        base=dict(full_environment_step_ms=dict(mean=25.,p95=26.))
        for mean,p95,passed,expected in ((23.75,24.7,True,'AUDIT_FAST_CANDIDATE'),
            (24.,24.,True,'KEEP_INFERENCE_ONLY'),(23.,25.,True,'KEEP_INFERENCE_ONLY'),
            (20.,20.,False,'KEEP_INFERENCE_ONLY')):
            r=ops.recommendation(base,dict(full_environment_step_ms=dict(mean=mean,p95=p95)),dict(passed=passed))
            self.assertEqual(r['choice'],expected);self.assertFalse(r['production_activated'])
            self.assertFalse(r['full_loop_realtime_acceptance'])


if __name__=='__main__':unittest.main()
