from contextlib import ExitStack
from copy import deepcopy
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'package'))
import command_views as cv
import bam_graph as bg
import commandview_contract as contract
import commandview_ops as ops
from command_fixture import MODULES,ACT,ENTITY,prepare_commands
from bridge_fixture import bridge_data
from test_packed_checks import make_session
import test_bam_graph as bam_fixture


class CommandViewTests(unittest.TestCase):
    def setUp(self):
        original=cv.importlib.import_module
        p=patch.object(cv.importlib,'import_module',side_effect=lambda n:MODULES[n] if n in MODULES else original(n))
        p.start();self.addCleanup(p.stop)
        p=patch.object(bg,'GraphedBam',side_effect=lambda g,t,c:bam_fixture.OriginalArithmeticTests().emulate_capture(g,c))
        p.start();self.addCleanup(p.stop)

    def setup(self,delay=False):
        s,_,_=make_session();s.raw.sim.data=bridge_data(s.raw.sim.data)
        d=prepare_commands(s.guard,delay);graphs=bg.BamGraphs(s.guard.groups,torch)
        self.addCleanup(graphs.close)
        views=cv.CommandViews(s.guard,torch,graphs);self.addCleanup(views.close)
        return s,d,graphs,views

    def test_original_sources_and_shared_current_views_all_five_fields(self):
        s,d,_,views=self.setup();g=s.guard.groups[0];runner=views.runners[0]
        for i in range(50):
            d.data.qpos[...].fill_(i*.001);d.data.qvel[...].fill_(i*.002)
            d.joint_pos_target.fill_(i*.003);d.joint_vel_target.fill_(i*.004);d.joint_effort_target.fill_(i*.005)
            actual=g.get_command(d);expected=runner.original(d)
            for name,x in runner.sources().items():
                a,b=getattr(actual,name),getattr(expected,name)
                self.assertTrue(torch.equal(a,b))
                self.assertEqual(a.untyped_storage().data_ptr(),x.untyped_storage().data_ptr())
        self.assertEqual(runner.checked_calls,40);self.assertEqual(runner.scalar_checks,2800)
        # Replacing live state storage is safe: each call takes a fresh view.
        from command_fixture import BRIDGE
        from bridge_fixture import CpuWarpArray
        d.data.struct.qpos=CpuWarpArray(torch.full((1,14),.7));d.data.clear_cache()
        self.assertTrue(torch.equal(g.get_command(d).pos,torch.full((1,14),.7)))
        with patch.object(cv,'sha',return_value='0'*64),self.assertRaises(RuntimeError):cv.verify_sources()

    def test_nonidentity_contiguous_addresses_and_noncontiguous_fallback(self):
        s,_,_=make_session();s.raw.sim.data=bridge_data(s.raw.sim.data);d=prepare_commands(s.guard)
        g=s.guard.groups[0]
        # Start at a nonzero local joint offset; no hardcoded servo ordering.
        g.target_ids=torch.arange(3,10)
        graphs=bg.BamGraphs(s.guard.groups,torch);views=cv.CommandViews(s.guard,torch,graphs)
        try:
            d.data.qpos[...].copy_(torch.arange(14).reshape(1,14))
            self.assertTrue(torch.equal(g.get_command(d).pos,torch.arange(3,10).reshape(1,7).float()))
        finally:views.close();graphs.close()
        g.target_ids=torch.tensor([8,1,11,3]);graphs=bg.BamGraphs(s.guard.groups,torch)
        views=cv.CommandViews(s.guard,torch,graphs)
        try:
            actual=g.get_command(d);expected=views.runners[0].original(d)
            self.assertTrue(torch.equal(actual.pos,expected.pos));self.assertEqual(views.runners[0].fallback_calls,1)
            self.assertEqual(actual.pos.tolist(),[[8.,1.,11.,3.]])
        finally:views.close();graphs.close()

    def test_original_delay_history_rng_and_BAM_staging_keep_owned_snapshots(self):
        a,da,ga,va=self.setup(delay=True);b,db,gb,vb=self.setup(delay=True)
        va.close();g=a.guard.groups[0];h=b.guard.groups[0]
        for i in range(100):
            for d in (da,db):d.joint_pos_target.fill_(.02*math.sin(i*.021))
            left=g.apply_delay(g.get_command(da));right=h.apply_delay(h.get_command(db))
            self.assertTrue(torch.equal(left.position_target,right.position_target))
            self.assertTrue(torch.equal(g.compute(left),h.compute(right)))
            self.assertTrue(torch.equal(g._delay_buffer.generator.get_state(),h._delay_buffer.generator.get_state()))
            self.assertTrue(torch.equal(g._delay_buffer._buffer._buffer,h._delay_buffer._buffer._buffer))
        captured=gb.runners[0][0];stored=captured.command.position_target.clone()
        db.joint_pos_target.fill_(99.)
        self.assertTrue(torch.equal(captured.command.position_target,stored))
        self.assertFalse(torch.any(h._delay_buffer._buffer._buffer==99.))

    def test_consumer_index_and_source_changes_stop_without_actuating(self):
        s,d,graphs,views=self.setup();g=s.guard.groups[0];compute=g.compute
        g.compute=lambda x:x
        with self.assertRaisesRegex(RuntimeError,'consumer changed'):g.get_command(d)
        g.compute=compute
        old=g.target_ids;g.target_ids=old.clone()
        with self.assertRaisesRegex(RuntimeError,'address storage'):g.get_command(d)
        g.target_ids=old
        old_target=d.joint_pos_target;d.joint_pos_target=torch.zeros(1,15)
        with self.assertRaisesRegex(RuntimeError,'schema changed'):g.get_command(d)
        d.joint_pos_target=old_target
        self.assertEqual(views.runners[0].calls,0);self.assertIsNone(graphs.runners[0][0])

    def test_comparison_and_final_mapping_failures_restore_hooks(self):
        s,d,_,views=self.setup();g=s.guard.groups[0];r=views.runners[0]
        original=r.original
        def corrupted(data):
            c=original(data);c.pos.add_(1);return c
        r.original=corrupted
        with self.assertRaisesRegex(RuntimeError,'differs from original'):g.get_command(d)
        self.assertTrue(torch.equal(d.data.qpos[...],torch.zeros(1,14)))
        r.original=original;g.target_ids[0]=1
        with self.assertRaisesRegex(RuntimeError,'address values changed'):views.close()
        self.assertTrue(views.closed);self.assertFalse(views.saved)
        self.assertIs(g.get_command.__func__,ACT.Actuator.get_command)

    def test_1600_original_entity_BAM_audit_and_disturbance_steps_equal(self):
        a,da,ga,va=self.setup(delay=True);b,db,gb,vb=self.setup(delay=True);va.close()
        reads_a=bg.BatchedAuditReads(a.guard,torch);reads_b=bg.BatchedAuditReads(b.guard,torch)
        self.addCleanup(reads_a.close);self.addCleanup(reads_b.close)
        entities=[]
        from types import SimpleNamespace as NS
        for s,d in ((a,da),(b,db)):
            g=s.guard.groups[0];e=ENTITY.Entity();e._data=d;e._custom_actuators=[g]
            e._builtin_group=NS(apply_controls=lambda data:None);g.ctrl_ids=g.global_ctrl_ids
            d.write_ctrl=lambda torque,ids,d=d:d.data.ctrl.__setitem__((slice(None),ids),torque)
            entities.append(e)
        for i in range(1600):
            for s,d,e in zip((a,b),(da,db),entities):
                d.joint_pos_target.fill_(.02*math.sin(i*.021))
                e._apply_actuator_controls();s.physics_step()
            for key in ('qpos','qvel','ctrl','actuator_force','xfrc_applied'):
                self.assertTrue(torch.equal(getattr(da.data,key)[...],getattr(db.data,key)[...]))
        self.assertEqual(a.guard.audit.snapshot(),b.guard.audit.snapshot());self.assertEqual(a.pulses,b.pulses)
        self.assertEqual(b.physics_calls,1600);self.assertEqual(b.guard.audit.motor_checks,1600)
        self.assertEqual(b.guard.audit.joint_checks,1601);self.assertEqual(b.applied_physics_steps,50)
        self.assertEqual(vb.runners[0].fast_calls,1600)


class CommandContractTests(unittest.TestCase):
    def test_exact_latest_measurement_and_retained_baseline(self):
        r=contract.reviewed(ROOT/'reference/bridged-read-graph-measured.json',ROOT/'reference/inference-loop-measured.json')
        self.assertEqual(r['recommendation']['choice'],'KEEP_INFERENCE_ONLY')
        with patch.object(contract,'sha',return_value='0'*64),self.assertRaises(RuntimeError):
            contract.reviewed(ROOT/'reference/bridged-read-graph-measured.json',ROOT/'reference/inference-loop-measured.json')

    def test_bounded_stop_and_no_automatic_activation(self):
        from test_cache import valid_row
        row=valid_row();self.assertFalse(contract.valid_partial(row))
        row['stop_reason']='COMMANDVIEW_SAMPLE_COMPLETE';self.assertTrue(contract.valid_partial(row))
        base=dict(full_environment_step_ms=dict(mean=25.,p95=26.))
        fast=dict(full_environment_step_ms=dict(mean=20.,p95=21.))
        result=ops.recommendation(base,fast,dict(passed=True))
        self.assertEqual(result['choice'],'COMMAND_VIEW_CANDIDATE');self.assertFalse(result['production_activated'])
        self.assertEqual(ops.recommendation(base,fast,dict(passed=False))['choice'],'KEEP_INFERENCE_ONLY')


if __name__=='__main__':unittest.main()
