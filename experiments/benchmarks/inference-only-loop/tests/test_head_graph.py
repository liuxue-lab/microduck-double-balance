"""Original source math with fixed-address CPU replay; not a CUDA benchmark."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'package'))
import head_graph as hg
from phase_cache import PhaseCache

REAL_GRAPH_CLASS = hg.GraphedHead
FIXTURES = Path(__file__).parent / 'fixtures'
torch.set_num_threads(1)


def original_math():
    math_path = FIXTURES / 'mjlab-math-v1.3.0.py'
    mdp_path = FIXTURES / 'mdp-reviewed.py'
    assert hashlib.sha256(math_path.read_bytes()).hexdigest() == hg.MATH_SHA
    assert hashlib.sha256(mdp_path.read_bytes()).hexdigest() == hg.MDP_SHA
    spec = importlib.util.spec_from_file_location('reviewed_mjlab_math', math_path)
    math_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = math_module; spec.loader.exec_module(math_module)
    names = ('_stage11_head_kinematics', 'stage11_head_material_values', 'stage11_head_features_from_values')
    nodes = [n for n in ast.parse(mdp_path.read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = dict(torch=torch, matrix_from_quat=math_module.matrix_from_quat)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(mdp_path), 'exec'), namespace)
    return NS(**{n:namespace[n] for n in names}, matrix_from_quat=math_module.matrix_from_quat,
              __file__=str(mdp_path)), math_module


MDP, MATH = original_math()


def fixture():
    robot = NS(data=NS(site_pose_w=torch.zeros(1, 3, 7),
                      site_lin_vel_w=torch.zeros(1, 3, 3), site_ang_vel_w=torch.zeros(1, 3, 3)))
    robot.data.site_pose_w[..., 3] = 1.
    robot.find_sites = lambda pattern: ([2], [hg.REFERENCE_SITE]) if pattern == '^' + hg.REFERENCE_SITE + '$' else ([], [])
    ball = NS(data=NS(root_link_pos_w=torch.tensor([[.005, -.002, .02]]),
                     root_link_lin_vel_w=torch.zeros(1, 3), root_link_ang_vel_w=torch.zeros(1, 3)))
    return NS(scene={'robot':robot, 'top_ball':ball})


def emulate_capture(raw, torch_module, original, reference_site):
    runner = REAL_GRAPH_CLASS.__new__(REAL_GRAPH_CLASS)
    runner.raw, runner.torch, runner.original, runner.reference_site = raw, torch_module, original, reference_site
    runner.staged = torch.empty(1, sum(hg.WIDTHS))
    runner.stage(hg.read_inputs(raw, reference_site))
    runner.shadow = hg.shadow_scene(runner.staged, reference_site)
    runner.calls = runner.checked_calls = runner.scalar_comparisons = runner.snapshot_clones = 0
    runner.maximum_error = 0.; runner.last_failure = None
    runner.output = runner.calculate().clone()
    def replay(): runner.output.copy_(runner.calculate())
    runner.graph = NS(replay=replay)
    runner.signatures = (hg.storage_signature(runner.staged), hg.storage_signature(runner.output))
    return runner


class HeadGraphTests(unittest.TestCase):
    def test_source_hashes_and_original_rotation_binding_are_required(self):
        with patch.dict(sys.modules, {'mjlab.utils.lab_api.math':MATH}):
            sources = hg.verify_sources(MDP)
            self.assertEqual(sources['mdp']['sha256'], hg.MDP_SHA)
            with patch.object(MDP, 'matrix_from_quat', lambda q:q):
                with self.assertRaisesRegex(RuntimeError, 'binding changed'): hg.verify_sources(MDP)

    def test_original_math_matches_for_dynamic_inputs_and_preserves_live_data(self):
        raw = fixture(); runner = emulate_capture(raw, torch, MDP._stage11_head_kinematics, hg.REFERENCE_SITE)
        generator = torch.Generator().manual_seed(20261024)
        for _ in range(160):
            data = raw.scene['robot'].data
            data.site_pose_w.copy_(torch.randn(1, 3, 7, generator=generator))
            quat = data.site_pose_w[..., 3:]; quat.div_(torch.linalg.vector_norm(quat, dim=-1, keepdim=True))
            for value in (data.site_lin_vel_w, data.site_ang_vel_w,
                          *vars(raw.scene['top_ball'].data).values()):
                value.copy_(torch.randn(value.shape, generator=generator))
            before = [t.clone() for t in hg.read_inputs(raw, hg.REFERENCE_SITE)]
            expected = MDP._stage11_head_kinematics(raw)
            actual = runner()
            for a, b in zip(expected, actual): self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.equal(MDP.stage11_head_features_from_values(*expected, privileged=True),
                                        MDP.stage11_head_features_from_values(*actual, privileged=True)))
            for a, b in zip(before, hg.read_inputs(raw, hg.REFERENCE_SITE)): self.assertTrue(torch.equal(a, b))
        self.assertEqual(runner.checked_calls, 40); self.assertEqual(runner.scalar_comparisons, 360)

    def test_replay_cannot_overwrite_a_retained_earlier_result(self):
        raw = fixture(); runner = emulate_capture(raw, torch, MDP._stage11_head_kinematics, hg.REFERENCE_SITE)
        previous = runner(); saved = [t.clone() for t in previous]
        raw.scene['top_ball'].data.root_link_pos_w.add_(.2)
        current = runner()
        for a, b in zip(previous, saved): self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(previous[0], current[0]))
        self.assertNotEqual(previous[0].untyped_storage().data_ptr(), runner.output.untyped_storage().data_ptr())
        self.assertNotEqual(previous[0].untyped_storage().data_ptr(), current[0].untyped_storage().data_ptr())

    def test_reallocated_live_property_storage_is_copied_fresh(self):
        raw = fixture(); runner = emulate_capture(raw, torch, MDP._stage11_head_kinematics, hg.REFERENCE_SITE)
        runner()
        raw.scene['top_ball'].data.root_link_pos_w = torch.tensor([[.08, .03, .05]])
        for a, b in zip(runner(), MDP._stage11_head_kinematics(raw)): self.assertTrue(torch.equal(a, b))

    def test_bad_shape_precision_and_graph_storage_are_rejected(self):
        for fault in ('shape', 'precision', 'storage'):
            raw = fixture(); runner = emulate_capture(raw, torch, MDP._stage11_head_kinematics, hg.REFERENCE_SITE)
            if fault == 'shape': raw.scene['top_ball'].data.root_link_pos_w = torch.zeros(2, 3)
            if fault == 'precision': raw.scene['top_ball'].data.root_link_pos_w = torch.zeros(1, 3, dtype=torch.float64)
            if fault == 'storage': runner.output = runner.output.clone()
            with self.assertRaises(RuntimeError): runner()
            self.assertEqual(runner.calls, 0)

    def test_corrupted_or_nonfinite_replay_stops_before_returning_outputs(self):
        raw = fixture(); runner = emulate_capture(raw, torch, MDP._stage11_head_kinematics, hg.REFERENCE_SITE)
        replay = runner.graph.replay
        def corrupt(): replay(); runner.output[0, 4] += .1
        runner.graph.replay = corrupt
        with self.assertRaises(hg.HeadComparisonFailure): runner()
        self.assertEqual(runner.last_failure['field'], 'rate')
        self.assertEqual(runner.last_failure['attempted_replay'], 1)
        self.assertEqual(runner.calls, 0)
        raw.scene['robot'].data.site_pose_w[0, 2, 3:] = 0.
        with self.assertRaisesRegex(RuntimeError, 'Nonfinite'): runner()

    def test_independent_probe_recomputes_original_outside_manager_graph(self):
        raw = fixture(); calls = []
        def original(env, site=hg.REFERENCE_SITE):
            calls.append(env); return MDP._stage11_head_kinematics(env, site)
        mdp = NS(_stage11_head_kinematics=original)
        def compute():
            return mdp._stage11_head_kinematics(raw), mdp._stage11_head_kinematics(raw)
        raw.reward_manager = NS(compute=compute); raw.termination_manager = NS(compute=compute)
        raw.observation_manager = NS(compute=compute)
        raw.sim = NS(step=lambda:None, forward=lambda:None, reset=lambda:None)
        class Scene(dict): pass
        raw.scene = Scene(raw.scene); raw.scene.write_data_to_sim = lambda:None
        probe = NS(original=compute); cache = PhaseCache(raw, mdp, probe)
        with patch.object(hg, 'verify_sources', return_value={}), patch.object(hg, 'GraphedHead', side_effect=emulate_capture):
            head = hg.HeadGraph(cache, torch, mdp)
            try:
                raw.reward_manager.compute()
                self.assertEqual(head.runner.calls, 1)
                saved_calls = len(calls)
                raw.scene['top_ball'].data.root_link_pos_w.add_(.1)
                direct = mdp._stage11_head_kinematics(raw)
                self.assertEqual(len(calls), saved_calls + 1)
                self.assertEqual(head.runner.calls, 1); self.assertEqual(head.independent_calls, 1)
                for a, b in zip(direct, MDP._stage11_head_kinematics(raw)): self.assertTrue(torch.equal(a, b))
            finally: head.close(); cache.close()
        self.assertIs(mdp._stage11_head_kinematics, original)

    def test_capture_failure_restores_original_binding(self):
        raw = fixture()
        original = MDP._stage11_head_kinematics
        cache = NS(raw=raw, original=original, depth=1)
        with patch.object(hg, 'verify_sources', return_value={}), patch.object(hg, 'GraphedHead', side_effect=RuntimeError('capture failed')):
            head = hg.HeadGraph(cache, torch, MDP)
            with self.assertRaisesRegex(RuntimeError, 'capture failed'): cache.original(raw)
            head.close(); head.close()
        self.assertIs(cache.original, original)


if __name__ == '__main__': unittest.main()
