"""Reproduce the reported TorchArray failure using original mjlab bridge code."""
from contextlib import ExitStack
import gc
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import weakref
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'package'))
import bridge_views as bv
import read_graph as rg
import bam_graph as bg
from bridge_fixture import BRIDGE,bridge_data,wrap
from test_packed_checks import make_session
from test_read_graph import graphs,emulate_read,prepare_entity


class BridgeTests(unittest.TestCase):
    def setUp(self):
        original=bv.importlib.import_module
        p=patch.object(bv.importlib,'import_module',side_effect=lambda name:BRIDGE if name=='mjlab.sim.sim_data' else original(name))
        p.start();self.addCleanup(p.stop)

    def test_v1_reported_failure_reproduced_before_any_capture(self):
        path=ROOT/'tests/fixtures/tensor-read-graph-baseline.py'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
            '226445a61be337e5cf3cc98a2043012e703899c646f2012546b83660041535f5')
        spec=importlib.util.spec_from_file_location('legacy_readgraph_v1',path)
        legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
        s,_,_=make_session();s.raw.sim.data=bridge_data(s.raw.sim.data);prepare_entity(s.guard)
        with patch.object(legacy,'verify_entity',return_value={}):old=legacy.GraphPackedChecks(s,torch)
        self.assertFalse(isinstance(old.pre_inputs()['qpos'],torch.Tensor))
        with self.assertRaisesRegex(RuntimeError,'Unexpected captured input type: qpos'):
            legacy.ReadGraph(torch,old.pre_inputs,old.pre_values)
        self.assertEqual(s.physics_calls,0)

    def test_real_bridge_fields_are_shared_views_with_original_owners(self):
        tensor=torch.arange(24,dtype=torch.float32).reshape(1,4,6);proxy=wrap(tensor)
        views=bv.BridgeViews(torch);result=views.view(proxy,'xfrc_applied')
        self.assertIsInstance(result,torch.Tensor)
        self.assertEqual(bg.storage_signature(result),bg.storage_signature(tensor))
        self.assertEqual(result.untyped_storage().data_ptr(),tensor.untyped_storage().data_ptr())
        tensor[0,2,3]=123.;self.assertEqual(result[0,2,3].item(),123.)
        owner=weakref.ref(proxy);del proxy;gc.collect();self.assertIsNotNone(owner())
        self.assertIs(views.view(tensor,'plain'),tensor)
        self.assertEqual(views.stats()['unwrapped_fields'],['xfrc_applied'])

    def test_all_wrapped_fields_pass_real_read_constructor_and_alias_checks(self):
        s,command,_=make_session();s.raw.sim.data=bridge_data(s.raw.sim.data)
        with graphs(s) as (candidate,_):
            command()
            self.assertTrue(all(isinstance(v,torch.Tensor) for v in candidate.base_inputs().values()))
            pre=emulate_read(torch,candidate.pre_inputs,candidate.pre_values)
            post=emulate_read(torch,candidate.post_inputs,candidate.post_values)
            self.assertEqual(pre.read(),candidate.pre_values().tolist())
            self.assertEqual(post.read(),candidate.post_values().tolist())
            self.assertEqual(set(candidate.views.owners),{'qpos','qvel','ctrl','actuator_force','xfrc_applied','entity_qpos'})

    def test_wrapped_1600_substeps_match_plain_original_audit_and_force(self):
        a,set_a,_=make_session();b,set_b,_=make_session();b.raw.sim.data=bridge_data(b.raw.sim.data)
        reads=bg.BatchedAuditReads(a.guard,torch)
        try:
            with graphs(b) as (candidate,_):
                for i in range(1600):
                    set_a(i);set_b(i)
                    self.assertEqual(a.physics_step(),candidate.physics_step())
                    for key in ('qpos','qvel','ctrl','actuator_force','xfrc_applied'):
                        self.assertTrue(torch.equal(getattr(a.raw.sim.data,key),getattr(b.raw.sim.data,key)[...]))
                self.assertEqual(a.guard.audit.snapshot(),b.guard.audit.snapshot())
                self.assertEqual(a.pulses,b.pulses)
                self.assertEqual(b.guard.audit.motor_checks,1600);self.assertEqual(b.guard.audit.joint_checks,1601)
                self.assertEqual(b.applied_physics_steps,50)
        finally:reads.close()

    def test_wrapper_cache_refresh_keeps_current_values_and_storage_guard(self):
        s,command,_=make_session();s.raw.sim.data=bridge_data(s.raw.sim.data)
        with graphs(s) as (candidate,_):
            command();runner=emulate_read(torch,candidate.pre_inputs,candidate.pre_values)
            old_owner=candidate.views.owners['qpos'];before=runner.read()
            s.raw.sim.data.clear_cache();s.raw.sim.data.qpos[0,0]=.2
            self.assertIsNot(s.raw.sim.data.qpos,old_owner)
            self.assertNotEqual(before,runner.read())
            self.assertIs(candidate.views.owners['qpos'],old_owner)
            # Simulate replacement of the underlying Warp array, then rebuild
            # the official wrapper cache. The fixed captured storage must fail.
            from bridge_fixture import CpuWarpArray
            s.raw.sim.data.struct.qpos=CpuWarpArray(s.raw.sim.data.qpos[...].clone())
            s.raw.sim.data.clear_cache()
            with self.assertRaisesRegex(RuntimeError,'backing storage'):runner.read()

    def test_source_mismatch_and_arbitrary_tensor_like_objects_are_rejected(self):
        with patch.object(bv,'sha',return_value='0'*64),self.assertRaises(RuntimeError):bv.BridgeViews(torch)
        views=bv.BridgeViews(torch)
        with self.assertRaisesRegex(RuntimeError,'Unsupported live array type'):
            views.view(NS(_tensor=torch.zeros(1)),'qpos')
        tensor=torch.zeros(1);proxy=wrap(tensor)
        with patch.object(BRIDGE.TorchArray,'__getitem__',lambda self,index:self._tensor.clone()):
            with self.assertRaisesRegex(RuntimeError,'copied or changed'):views.view(proxy,'qpos')


if __name__=='__main__':unittest.main()
