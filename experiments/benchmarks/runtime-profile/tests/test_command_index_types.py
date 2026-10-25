"""Reproduce v1's rejection with original mjlab indexing construction methods."""
from contextlib import ExitStack
import hashlib
import importlib.util
import itertools
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'package'))
import command_views as cv
import commandview_contract as contract
import bam_graph as bg
from command_fixture import MODULES,original_indexing,original_actuator_indices,prepare_commands
from bridge_fixture import bridge_data
from test_packed_checks import make_session


class IndexCompatibilityTests(unittest.TestCase):
    def setUp(self):
        original=cv.importlib.import_module
        p=patch.object(cv.importlib,'import_module',side_effect=lambda n:MODULES[n] if n in MODULES else original(n))
        p.start();self.addCleanup(p.stop)

    def fixture(self):
        s,_,_=make_session();s.raw.sim.data=bridge_data(s.raw.sim.data)
        d=prepare_commands(s.guard);g=s.guard.groups[0]
        graphs=bg.BamGraphs(s.guard.groups,torch);self.addCleanup(graphs.close)
        return s,d,g,graphs

    def test_original_index_builders_reproduce_v1_error_before_physics(self):
        indexing=original_indexing(list(range(7,21)),list(range(6,20)),with_free=True)
        targets,controls,global_controls=original_actuator_indices(indexing,range(14),range(14))
        self.assertEqual(indexing.joint_q_adr.dtype,torch.int32)
        self.assertEqual(indexing.joint_v_adr.dtype,torch.int32)
        self.assertEqual(indexing.ctrl_ids.dtype,torch.int32)
        self.assertEqual(indexing.free_joint_q_adr.tolist(),list(range(7)))
        self.assertEqual(indexing.free_joint_v_adr.tolist(),list(range(6)))
        self.assertTrue(all(x.dtype==torch.int64 for x in (targets,controls,global_controls)))
        s,d,g,graphs=self.fixture()
        path=ROOT/'tests/fixtures/command-views-int64-baseline.py'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         '4c0d5f54eb7f3f90f8b9c12058826d490a3c35a1a5ff333cf34e781019c206d9')
        spec=importlib.util.spec_from_file_location('legacy_command_views_v1',path)
        legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
        with self.assertRaisesRegex(RuntimeError,'Unexpected joint address type'):
            legacy.CommandView(g,d,torch,MODULES)
        repaired=cv.CommandViews(s.guard,torch,graphs)
        try:
            cmd=g.get_command(d)
            self.assertEqual(cmd.pos.shape,(1,14))
            schema=repaired.runners[0].stats()['index_schema']
            self.assertEqual(schema['target']['dtype'],'torch.int64')
            self.assertEqual(schema['qadr']['dtype'],'torch.int32')
            self.assertEqual(schema['vadr']['dtype'],'torch.int32')
            self.assertEqual(s.physics_calls,0)
        finally:repaired.close()

    def test_all_int32_int64_combinations_preserve_dtype_storage_and_mapping(self):
        for types in itertools.product((torch.int32,torch.int64),repeat=3):
            with self.subTest(types=types):
                s,d,g,graphs=self.fixture()
                g.target_ids=g.target_ids.to(types[0])
                d.indexing.joint_q_adr=d.indexing.joint_q_adr.to(types[1])
                d.indexing.joint_v_adr=d.indexing.joint_v_adr.to(types[2])
                fields=(g.target_ids,d.indexing.joint_q_adr,d.indexing.joint_v_adr)
                identities=[(id(x),x.data_ptr(),x.dtype,x.tolist()) for x in fields]
                d.data.qpos[...].copy_(torch.arange(14,dtype=torch.float32).reshape(1,14))
                d.data.qvel[...].copy_(torch.arange(14,dtype=torch.float32).reshape(1,14)*.1)
                views=cv.CommandViews(s.guard,torch,graphs)
                try:
                    actual=g.get_command(d);expected=views.runners[0].original(d)
                    for name in cv.FIELDS:self.assertTrue(torch.equal(getattr(actual,name),getattr(expected,name)))
                    self.assertEqual(identities,[(id(x),x.data_ptr(),x.dtype,x.tolist()) for x in fields])
                finally:views.close();graphs.close()

    def test_nonindex_types_shapes_and_non_tensors_fail_with_field_details(self):
        for name in ('target','qadr','vadr'):
            for invalid in (torch.zeros(14,dtype=torch.float32),torch.zeros(14,dtype=torch.bool),
                            torch.zeros(14,dtype=torch.int16),torch.zeros(1,14,dtype=torch.int32),[0,1]):
                with self.subTest(field=name,invalid_type=type(invalid).__name__):
                    s,d,g,graphs=self.fixture()
                    owner,field=(g,'target_ids') if name=='target' else (d.indexing,'joint_q_adr' if name=='qadr' else 'joint_v_adr')
                    setattr(owner,field,invalid)
                    with self.assertRaisesRegex(RuntimeError,'Unexpected joint address type: '+name):
                        cv.CommandViews(s.guard,torch,graphs)
                    self.assertNotIn('get_command',vars(g));self.assertEqual(s.physics_calls,0)
                    graphs.close()

    def test_exact_failed_report_and_zero_candidate_physics_are_required(self):
        r=contract.reviewed_failure(ROOT/'reference/command-views-int64-failure.json')
        self.assertEqual(r['arms']['command-views']['physics_steps'],0)
        self.assertEqual(r['arms']['inference-only']['physics_steps'],1600)
        with patch.object(contract,'sha',return_value='0'*64),self.assertRaises(RuntimeError):
            contract.reviewed_failure(ROOT/'reference/command-views-int64-failure.json')


if __name__=='__main__':unittest.main()
