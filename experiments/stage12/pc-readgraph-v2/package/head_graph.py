"""Capture the original head-frame calculation on isolated, staged inputs.

Only manager-scoped calls use the graph. The independent geometric probe still
executes the original function on the live scene outside the phase cache.
All six live input blocks are copied each call; graph outputs are cloned before
returning, so a later replay cannot overwrite a retained earlier result.
"""
import importlib
from pathlib import Path
from types import SimpleNamespace as NS

from deploy_contract import require, sha, ATOL, RTOL
from phase_cache import Bindings
from bam_graph import storage_signature

REFERENCE_SITE = 'stage11_head_material_frame'
HEAD_CHECK_CALLS = 40
WIDTHS = (7, 3, 3, 3, 3, 3)
FIELDS = ('position', 'rate', 'spin')
MDP_SHA = '9f970ae07c6912873fc44ee1850b20629fabc82cb20d3a95cf08f5cbd3f81e76'
MATH_SHA = '47757748386173cc4ded5fb5dc5619336393ee6296769fa8b985738ebe50d436'


def verify_sources(mdp):
    math_module = importlib.import_module('mjlab.utils.lab_api.math')
    rows = {}
    for name, module, expected in (('mdp', mdp, MDP_SHA), ('mjlab_math', math_module, MATH_SHA)):
        path = Path(module.__file__)
        require(sha(path) == expected, 'Head calculation source differs: ' + name)
        rows[name] = dict(path=str(path), sha256=expected)
    require(mdp.matrix_from_quat is math_module.matrix_from_quat,
            'Head rotation function binding changed')
    return rows


def read_inputs(raw, reference_site):
    robot, ball = raw.scene['robot'], raw.scene['top_ball']
    ids, _ = robot.find_sites('^' + reference_site + '$')
    require(len(ids) == 1, 'Expected one head material reference: ' + reference_site)
    site = ids[0]
    # Preserve the original entity-property read order.
    pose = robot.data.site_pose_w[:, site]
    return (pose, ball.data.root_link_pos_w, ball.data.root_link_lin_vel_w,
            ball.data.root_link_ang_vel_w, robot.data.site_lin_vel_w[:, site],
            robot.data.site_ang_vel_w[:, site])


def shadow_scene(staged, reference_site):
    pose, pos, vel, omega, site_vel, site_omega = staged.split(WIDTHS, dim=-1)
    def find_sites(pattern):
        require(pattern == '^' + reference_site + '$', 'Shadow reference site changed')
        return [0], [reference_site]
    robot = NS(find_sites=find_sites, data=NS(site_pose_w=pose.unsqueeze(1),
        site_lin_vel_w=site_vel.unsqueeze(1), site_ang_vel_w=site_omega.unsqueeze(1)))
    ball = NS(data=NS(root_link_pos_w=pos, root_link_lin_vel_w=vel, root_link_ang_vel_w=omega))
    return NS(scene={'robot': robot, 'top_ball': ball})


class HeadComparisonFailure(RuntimeError):
    def __init__(self, details):
        self.details = details
        super().__init__('Head same-input comparison failed: ' + details['field'] +
            '; max_error=' + str(details['maximum_absolute_error']))


def compare_outputs(torch, expected, actual):
    require(len(expected) == len(actual) == 3, 'Head output schema changed')
    maximum, count = 0., 0
    for name, ref, value in zip(FIELDS, expected, actual):
        require(ref.shape == value.shape == (1, 3) and ref.dtype == value.dtype == torch.float32,
                'Head output shape/dtype changed')
        require(bool(torch.isfinite(ref).all()) and bool(torch.isfinite(value).all()),
                'Nonfinite head same-input output: ' + name)
        error = (value - ref).abs()
        worst = float(error.max().item())
        maximum = max(maximum, worst); count += ref.numel()
        threshold = ATOL + RTOL * ref.abs()
        if not bool((error <= threshold).all()):
            raise HeadComparisonFailure(dict(field=name, maximum_absolute_error=worst,
                values_outside_tolerance=int((error > threshold).sum().item()), atol=ATOL, rtol=RTOL,
                expected=ref.detach().reshape(-1).cpu().tolist(),
                actual=value.detach().reshape(-1).cpu().tolist()))
    return maximum, count


class GraphedHead:
    def __init__(self, raw, torch, original, reference_site):
        self.raw, self.torch, self.original, self.reference_site = raw, torch, original, reference_site
        blocks = read_inputs(raw, reference_site)
        require(all(t.is_cuda for t in blocks), 'Head graph requires CUDA inputs')
        self.staged = torch.empty((1, sum(WIDTHS)), device=blocks[0].device, dtype=torch.float32)
        self.stage(blocks)
        self.shadow = shadow_scene(self.staged, reference_site)
        self.calls = self.checked_calls = self.scalar_comparisons = self.snapshot_clones = 0
        self.maximum_error = 0.; self.last_failure = None
        current = torch.cuda.current_stream(self.staged.device)
        stream = torch.cuda.Stream(device=self.staged.device); stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(3): self.calculate()
        current.wait_stream(stream); torch.cuda.synchronize(self.staged.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.output = self.calculate()
        current.wait_stream(stream)
        self.signatures = (storage_signature(self.staged), storage_signature(self.output))
        require(self.staged.untyped_storage().data_ptr() != self.output.untyped_storage().data_ptr(),
                'Head graph input and output share storage')

    def stage(self, blocks):
        require(len(blocks) == len(WIDTHS) and all(t.shape == (1, width)
            and t.dtype == self.torch.float32 and t.device == self.staged.device and not t.requires_grad
            for t, width in zip(blocks, WIDTHS)), 'Head input shape/dtype/device changed')
        # A single gather copies current values even if live property storage changes.
        self.torch.cat(blocks, dim=-1, out=self.staged)

    def calculate(self):
        # Call the original function, including its original quaternion transform.
        return self.torch.cat(self.original(self.shadow, self.reference_site), dim=-1)

    def __call__(self):
        require((storage_signature(self.staged), storage_signature(self.output)) == self.signatures,
                'Head graph storage changed; no automatic recapture')
        self.stage(read_inputs(self.raw, self.reference_site))
        self.graph.replay()
        outputs = self.output.split(3, dim=-1)
        if self.checked_calls < HEAD_CHECK_CALLS:
            expected = self.original(self.shadow, self.reference_site)
            try:
                worst, count = compare_outputs(self.torch, expected, outputs)
            except HeadComparisonFailure as exc:
                self.last_failure = dict(**exc.details, attempted_replay=self.calls + 1)
                raise
            self.maximum_error = max(self.maximum_error, worst)
            self.scalar_comparisons += count; self.checked_calls += 1
        self.calls += 1
        snapshot = self.output.clone()
        self.snapshot_clones += 1
        return snapshot.split(3, dim=-1)

    def stats(self):
        return dict(replays=self.calls, same_input_checked_calls=self.checked_calls,
            scalar_comparisons=self.scalar_comparisons, maximum_absolute_error=self.maximum_error,
            last_comparison_failure=self.last_failure, cloned_output_snapshots=self.snapshot_clones,
            input_scalars_per_call=sum(WIDTHS), all_live_inputs_copied_each_call=True,
            warmup_physics_steps=0, precision='original FP32',
            math='original _stage11_head_kinematics and original matrix_from_quat',
            outputs_do_not_alias_reusable_graph_buffer=True)


class HeadGraph(Bindings):
    def __init__(self, cache, torch, mdp):
        super().__init__()
        self.cache, self.torch, self.original = cache, torch, cache.original
        self.source_checks = verify_sources(mdp)
        self.runner = None; self.independent_calls = 0
        self.set(cache, 'original', self.run)

    def run(self, env, reference_site=REFERENCE_SITE):
        if env is not self.cache.raw or self.cache.depth == 0:
            self.independent_calls += 1
            return self.original(env, reference_site)
        require(reference_site == REFERENCE_SITE, 'Unexpected scoped head reference')
        if self.runner is None:
            self.runner = GraphedHead(env, self.torch, self.original, reference_site)
        return self.runner()

    def stats(self):
        return dict(source_checks=self.source_checks,
            independent_original_calls=self.independent_calls,
            capture_scope='manager-phase only; independent probe always calls original live calculation',
            runner=None if self.runner is None else self.runner.stats())
