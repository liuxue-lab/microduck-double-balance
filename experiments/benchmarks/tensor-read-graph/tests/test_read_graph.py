"""Original Audit/force lifecycle with fixed-address read graphs emulated on CPU."""
import ast
from contextlib import ExitStack,contextmanager
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'package'))
import read_graph as rg
import bam_graph as bg
from model_limits import LimitStop
from test_packed_checks import make_session
import test_bam_graph as bam_fixture

tree=ast.parse((ROOT/'tests/fixtures/entity-data.py').read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EntityData')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='joint_pos')
node=ast.ClassDef(name='EntityDataFixture',bases=[],keywords=[],decorator_list=[],body=[method])
namespace={'torch':torch}
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<original-joint-pos>','exec'),namespace)
EntityDataFixture=namespace['EntityDataFixture']


def prepare_entity(guard):
    entity=EntityDataFixture();entity.data=guard.raw.sim.data
    entity.indexing=NS(joint_q_adr=torch.tensor(guard.qids,dtype=torch.long))
    guard.robot.data=entity


REAL_READ_GRAPH=rg.ReadGraph
def emulate_read(torch,inputs,gather):
    r=REAL_READ_GRAPH.__new__(REAL_READ_GRAPH)
    r.torch=torch;r.inputs=inputs;r.gather=gather;r.fields=inputs();r.signatures=r.signature(r.fields)
    r.calls=r.checked_calls=r.scalar_comparisons=0;r.output=gather().clone()
    r.output_signature=bg.storage_signature(r.output)
    r.graph=NS(replay=lambda:r.output.copy_(gather()))
    return r


@contextmanager
def graphs(session):
    prepare_entity(session.guard)
    with ExitStack() as stack:
        stack.enter_context(patch.object(rg,'ReadGraph',side_effect=emulate_read))
        module=NS(EntityData=EntityDataFixture,__file__=str(ROOT/'tests/fixtures/entity-data.py'))
        with patch.object(rg.importlib,'import_module',return_value=module):
            candidate=rg.GraphPackedChecks(session,torch)
        stack.enter_context(patch.object(bg,'GraphedBam',side_effect=lambda g,t,c:bam_fixture.OriginalArithmeticTests().emulate_capture(g,c)))
        bam=bg.BamGraphs(session.guard.groups,torch)
        try:yield candidate,bam
        finally:candidate.close();bam.close()


class ReadGraphTests(unittest.TestCase):
    def test_multiple_groups_and_nonidentity_joint_addresses_keep_order(self):
        import copy
        s,_,_=make_session();s.guard.qids=list(range(13,-1,-1));prepare_entity(s.guard)
        s.raw.sim.data.qpos.copy_(torch.linspace(-.03,.03,14).reshape(1,14))
        group=s.guard.groups[0];groups=[]
        for start in (0,7):
            g=copy.copy(group);g.global_ctrl_ids=group.global_ctrl_ids[start:start+7]
            g.target_ids=group.target_ids[start:start+7]
            g.last_diagnostics={k:v[:,start:start+7] if v.ndim==2 and v.shape[1]==14 else v
                                for k,v in group.last_diagnostics.items()}
            groups.append(g)
        s.guard.groups=groups;s.guard.last_seen=[None,None]
        with patch.object(rg,'verify_entity',return_value={}),patch.object(rg,'ReadGraph',side_effect=emulate_read):
            candidate=rg.GraphPackedChecks(s,torch)
            s.guard.before_integration();expected=copy.deepcopy(s.guard.audit.latest_motor_samples)
            s.guard.last_seen=[None,None]
            candidate.before([0.]*24)
            self.assertEqual(s.guard.audit.latest_motor_samples,expected)
            candidate.close()

    def test_1600_substeps_full_audit_force_and_plant_equal(self):
        a,set_a,ea=make_session();b,set_b,eb=make_session();reads=bg.BatchedAuditReads(a.guard,torch)
        try:
            with graphs(b) as (candidate,bam):
                for i in range(1600):
                    set_a(i);set_b(i)
                    self.assertEqual(a.physics_step(),candidate.physics_step())
                    for name in ('qpos','qvel','ctrl','actuator_force','xfrc_applied'):
                        self.assertTrue(torch.equal(getattr(a.raw.sim.data,name),getattr(b.raw.sim.data,name)))
                self.assertEqual(a.guard.audit.snapshot(),b.guard.audit.snapshot())
                self.assertEqual(a.pulses,b.pulses);self.assertEqual(b.applied_physics_steps,50)
                self.assertEqual(b.guard.audit.motor_checks,1600);self.assertEqual(b.guard.audit.joint_checks,1601)
                self.assertEqual((candidate.pre_reads,candidate.post_reads,candidate.zero_checks),(1600,1600,1600))
                self.assertEqual(ea.boundaries,eb.boundaries)
                for key in ('pre_read_graph','post_read_graph'):
                    stats=candidate.stats()[key]
                    self.assertEqual(stats['replays'],1600);self.assertEqual(stats['same_input_checked_calls'],40)
                    self.assertFalse(stats['live_writes'])
        finally:reads.close()

    def test_original_failures_keep_phase_counts_event_and_force_cleanup(self):
        for fault in ('motor_nan','post_force','post_joint_nan','integrator_error','stale'):
            results=[]
            for captured in (False,True):
                with ExitStack() as stack:
                    s,command,events=make_session();s.physics_calls=1000
                    candidate=stack.enter_context(graphs(s))[0] if captured else None
                    command();integrate=s.physics_step_original
                    if fault=='motor_nan':s.guard.groups[0].last_diagnostics['motor_torque'][0,4]=float('nan')
                    if fault=='stale':s.guard.last_seen[0]=s.guard.groups[0].last_diagnostics
                    def faulty(fault=fault,s=s,integrate=integrate):
                        if fault=='integrator_error':raise RuntimeError('injected integrator failure')
                        value=integrate()
                        if fault=='post_force':s.raw.sim.data.xfrc_applied[0,0,4]=.125
                        if fault=='post_joint_nan':s.raw.sim.data.qpos[0,5]=float('nan')
                        return value
                    s.physics_step_original=faulty
                    with self.assertRaises((RuntimeError,LimitStop)) as caught:
                        (candidate.physics_step if captured else s.physics_step)()
                    self.assertFalse(s.owned_force_present);self.assertFalse(s.schedule_verified)
                    self.assertTrue(torch.equal(s.raw.sim.data.xfrc_applied[0,2],torch.zeros(6)))
                    results.append((type(caught.exception),str(caught.exception),events.integrated,
                        s.physics_calls,s.checked_physics_steps,s.guard.audit.snapshot(),s.pulses))
            self.assertEqual(results[0],results[1],fault)

    def test_unknown_wrench_and_cancellation_precede_write(self):
        for fault in ('wrench','cancel'):
            s,command,events=make_session();s.physics_calls=1000
            with graphs(s) as (candidate,_):
                command()
                if fault=='wrench':s.raw.sim.data.xfrc_applied[0,0,5]=1e-30
                else:
                    def stop():raise RuntimeError('cancelled')
                    s.boundary=stop
                before=s.raw.sim.data.xfrc_applied.clone()
                with self.assertRaises(RuntimeError):candidate.physics_step()
                self.assertTrue(torch.equal(before,s.raw.sim.data.xfrc_applied))
                self.assertEqual(events.integrated,0);self.assertEqual(s.pulses[0]['force_write_attempts'],0)

    def test_read_capture_never_writes_live_values_and_returns_owned_snapshots(self):
        s,command,_=make_session()
        with graphs(s) as (candidate,_):
            command();before={k:v.clone() for k,v in candidate.base_inputs().items()}
            runner=emulate_read(torch,candidate.pre_inputs,candidate.pre_values)
            first=runner.read();retained=list(first)
            for k,v in before.items():self.assertTrue(torch.equal(v,candidate.base_inputs()[k]))
            s.raw.sim.data.qpos[0,0]=.2;command(3);second=runner.read()
            self.assertEqual(first,retained);self.assertNotEqual(first,second)

    def test_changed_input_or_output_storage_is_rejected(self):
        for change in ('input','index','output'):
            s,command,_=make_session()
            with graphs(s) as (candidate,_):
                command();runner=emulate_read(torch,candidate.pre_inputs,candidate.pre_values)
                if change=='input':s.raw.sim.data.qpos=s.raw.sim.data.qpos.clone()
                elif change=='index':s.guard.robot.data.indexing.joint_q_adr=s.guard.robot.data.indexing.joint_q_adr.clone()
                else:runner.output=runner.output.clone()
                with self.assertRaisesRegex(RuntimeError,'storage'):runner.read()
                self.assertEqual(runner.calls,0)

    def test_corrupted_read_rejected_before_original_audit(self):
        s,command,events=make_session()
        with graphs(s) as (candidate,_):
            command();candidate.pre_graph=emulate_read(torch,candidate.pre_inputs,candidate.pre_values)
            replay=candidate.pre_graph.graph.replay
            def corrupt():replay();candidate.pre_graph.output[-1]+=.1
            candidate.pre_graph.graph.replay=corrupt
            with self.assertRaisesRegex(RuntimeError,'same-input'):candidate.physics_step()
            self.assertEqual(events.integrated,0);self.assertEqual(s.guard.audit.motor_checks,0)

    def test_capture_failure_preserves_owned_force_cleanup(self):
        s,command,events=make_session();s.physics_calls=1000
        with graphs(s) as (candidate,_):
            command()
            with patch.object(rg,'ReadGraph',side_effect=RuntimeError('capture failed')):
                with self.assertRaisesRegex(RuntimeError,'capture failed'):candidate.physics_step()
            self.assertEqual(events.integrated,0);self.assertFalse(s.owned_force_present)
            self.assertTrue(torch.equal(s.raw.sim.data.xfrc_applied,torch.zeros_like(s.raw.sim.data.xfrc_applied)))

    def test_entity_source_and_live_qpos_identity_required(self):
        s,_,_=make_session();prepare_entity(s.guard)
        path=ROOT/'tests/fixtures/entity-data.py'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),rg.ENTITY_SHA)
        module=NS(EntityData=EntityDataFixture,__file__=str(path))
        with patch.object(rg.importlib,'import_module',return_value=module):
            self.assertEqual(rg.verify_entity(s.guard)['sha256'],rg.ENTITY_SHA)
            with patch.object(rg,'sha',return_value='0'*64),self.assertRaises(RuntimeError):rg.verify_entity(s.guard)
            s.guard.robot.data.data=NS(qpos=s.raw.sim.data.qpos.clone())
            with self.assertRaisesRegex(RuntimeError,'current simulator qpos'):rg.verify_entity(s.guard)


if __name__=='__main__':unittest.main()
