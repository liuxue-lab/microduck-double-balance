"""Capture original action writes, preserving every 500-Hz write and check.

Original process_action, including action history, noise and hold-force math,
runs once per control tick. Its five outputs are staged once per control tick.
Each substep replays the original apply_action methods with current encoder
bias and current destination contents. Warmup writes only to private copies.
No graph advances physics, samples RNG, or applies the evaluation disturbance.
"""
import copy
import importlib
from pathlib import Path
from types import SimpleNamespace

from bootstrap import require, sha
from phase_cache import Bindings
from bam_graph import storage_signature
from bridge_views import BridgeViews

CHECK_CALLS = 40
PINS = {
    'mjlab.managers.action_manager': 'fec79f933fd9437dc9fcb784e4f214b9bae4aad2252c2d2d1387b11b605a5332',
    'mjlab.envs.mdp.actions.actions': '9cae83a2b0006320a8e15e040a0edbf5b4ae76d023832563f1ac36630f3b28d1',
    'mjlab.entity.data': 'fd1116692f82278bda721d44c015d27e253d2e866608c8e12824df039dfb4657',
    'mjlab.entity.entity': '817896ec3ce69c8ef18cb3d8e2519b85d1db25f70126af5e643c34a0b054e572',
    'mjlab_microduck.tasks.mdp': '9f970ae07c6912873fc44ee1850b20629fabc82cb20d3a95cf08f5cbd3f81e76',
}
FORCES = ('_force', '_torque', '_duck_force', '_duck_torque')


def verify_sources():
    modules, records = {}, {}
    for name, pin in PINS.items():
        module = importlib.import_module(name)
        path = Path(module.__file__)
        require(sha(path) == pin, 'Unreviewed action-write source: ' + name)
        modules[name] = module
        records[name] = dict(path=str(path), sha256=pin)
    return modules, records


def original_method(obj, name, expected):
    require(getattr(getattr(obj, name), '__func__', None) is expected,
            'Action-write method already changed: ' + name)


class PrivateWrites:
    """Original methods on shallow entity copies with isolated destinations."""
    def __init__(self, manager, joint, hold, inputs, outputs, torch=None):
        self.manager = copy.copy(manager)
        self.joint, self.hold = copy.copy(joint), copy.copy(hold)
        self.manager._terms = dict(joint_pos=self.joint, ball_hold=self.hold)
        self.outputs = outputs
        # Original BallHoldAction passes the Python list [0]. During capture,
        # avoid an implicit CPU-to-CUDA index conversion by passing the same
        # one-element int64 index already on the original device. The
        # independent reference keeps the original Python-list path.
        self.body_zero = None if torch is None else torch.tensor(
            [0], dtype=torch.int64, device=inputs['processed'].device)
        entities = {}
        for original in (joint._entity, hold._entity, hold._duck):
            if id(original) in entities:
                continue
            entity = copy.copy(original)
            entity._data = copy.copy(original.data)
            # apply_action reads only encoder bias / indexing and writes these
            # two fields. No simulator owner or live model is replaced.
            entity._data.data = SimpleNamespace(xfrc_applied=outputs['wrench'])
            if original is joint._entity:
                entity._data.joint_pos_target = outputs['targets']
            if self.body_zero is not None:
                write = entity._data.write_external_wrench
                def write_indexed(force, torque, body_ids=None, env_ids=None, _write=write):
                    require(type(body_ids) is list and body_ids == [0],
                            'Unexpected original action body selection')
                    return _write(force, torque, self.body_zero, env_ids)
                entity._data.write_external_wrench = write_indexed
            entities[id(original)] = entity
        self.joint._entity = entities[id(joint._entity)]
        self.hold._entity = entities[id(hold._entity)]
        self.hold._duck = entities[id(hold._duck)]
        self.set_inputs(inputs)

    def set_inputs(self, inputs):
        self.joint._processed_actions = inputs['processed']
        for name in FORCES:
            setattr(self.hold, name, inputs[name])

    def compute(self, live):
        # Copy ALL destination bits before each replay, preserving untouched
        # bodies, torques and non-target joints. Never zero a whole wrench.
        for name, value in live.items():
            self.outputs[name].copy_(value)
        self.manager.apply_action()


class ActionGraph(Bindings):
    def __init__(self, raw, torch):
        super().__init__()
        self.raw, self.torch = raw, torch
        self.manager = raw.action_manager
        self.calls = self.checked_calls = self.scalar_checks = self.controls = 0
        self.staged_controls = self.current_substeps = 0
        self.closed = False
        self.comparison_failure = None
        self.graph = self.private = self.reference = None
        self.inputs = self.input_schema = self.live = self.output_signatures = None
        self.modules, self.sources = verify_sources()
        self.views = BridgeViews(torch)
        manager_type = self.modules['mjlab.managers.action_manager'].ActionManager
        joint_type = self.modules['mjlab.envs.mdp.actions.actions'].JointPositionAction
        hold_type = self.modules['mjlab_microduck.tasks.mdp'].BallHoldAction
        require(type(self.manager) is manager_type, 'Expected original ActionManager')
        require(list(self.manager._terms) == ['joint_pos', 'ball_hold'], 'Action terms/order differ')
        self.joint, self.hold = self.manager._terms.values()
        require(type(self.joint) is joint_type and type(self.hold) is hold_type,
                'Expected original joint-position and ball-hold actions')
        require(self.joint._entity is self.hold._duck is raw.scene['robot'] and
                self.hold._entity is raw.scene['ball'], 'Action entity mapping differs')
        original_method(self.manager, 'process_action', manager_type.process_action)
        original_method(self.manager, 'apply_action', manager_type.apply_action)
        original_method(self.joint, 'apply_actions', joint_type.apply_actions)
        original_method(self.hold, 'apply_actions', hold_type.apply_actions)
        original_method(self.joint, 'process_actions',
                        self.modules['mjlab.envs.mdp.actions.actions'].BaseAction.process_actions)
        original_method(self.hold, 'process_actions', hold_type.process_actions)
        for entity in (self.joint._entity, self.hold._entity):
            entity_type = self.modules['mjlab.entity.entity'].Entity
            data_type = self.modules['mjlab.entity.data'].EntityData
            require(type(entity) is entity_type and type(entity.data) is data_type,
                    'Expected original action entity and data')
            original_method(entity, 'write_external_wrench_to_sim', entity_type.write_external_wrench_to_sim)
            original_method(entity.data, 'write_external_wrench', data_type.write_external_wrench)
        original_method(self.joint._entity, 'set_joint_position_target',
                        self.modules['mjlab.entity.entity'].Entity.set_joint_position_target)
        self.original_process = self.manager.process_action
        self.field_refs = self.fixed_fields()
        self.field_signatures = {k: storage_signature(v) for k, v in self.field_refs.items()}
        self.index_values = {k: v.detach().cpu().tolist() for k, v in self.field_refs.items()
                             if k != 'encoder_bias'}
        try:
            self.set(self.manager, 'process_action', self.process)
            self.set(self.manager, 'apply_action', self.apply)
        except BaseException:
            self.close()
            raise

    def fixed_fields(self):
        return dict(target_ids=self.joint._target_ids,
                    encoder_bias=self.joint._entity.data.encoder_bias,
                    robot_bodies=self.hold._duck.data.indexing.body_ids,
                    ball_bodies=self.hold._entity.data.indexing.body_ids)

    def live_outputs(self):
        wrench = self.views.view(self.raw.sim.data.xfrc_applied, 'wrench')
        for entity in (self.joint._entity, self.hold._entity):
            require(storage_signature(entity.data.data.xfrc_applied) == storage_signature(wrench),
                    'Action entity has another wrench destination')
        return dict(targets=self.joint._entity.data.joint_pos_target, wrench=wrench)

    def current_inputs(self):
        values = dict(processed=self.joint._processed_actions)
        values.update({name: getattr(self.hold, name, None) for name in FORCES})
        require(all(isinstance(x, self.torch.Tensor) and x.dtype == self.torch.float32
                    and not x.requires_grad for x in values.values()),
                'Expected original FP32 processed action and four hold-wrench tensors')
        require(values['processed'].shape == (1, 14) and
                all(values[k].shape == (1, 1, 3) for k in FORCES), 'Action input layout differs')
        return values

    @staticmethod
    def schema(values):
        return {k: (tuple(x.shape), str(x.dtype), str(x.device)) for k, x in values.items()}

    def process(self, actions):
        require(not self.closed, 'Action graph closed')
        require(self.controls == 0 or self.current_substeps == 10,
                'Expected ten action-write substeps per processed control')
        result = self.original_process(actions)
        self.controls += 1
        self.current_substeps = 0
        current = self.current_inputs()
        if self.graph is not None:
            require(self.schema(current) == self.input_schema, 'Action input schema changed')
            for name, value in current.items():
                self.inputs[name].copy_(value)
            self.staged_controls += 1
        self.source_refs = current
        return result

    def prepare(self):
        t = self.torch
        self.live = self.live_outputs()
        require(all(x.dtype == t.float32 and not x.requires_grad for x in self.live.values()),
                'Action destinations must retain FP32 inference state')
        self.output_signatures = {k: storage_signature(x) for k, x in self.live.items()}
        self.inputs = {k: x.detach().clone() for k, x in self.source_refs.items()}
        self.input_schema = self.schema(self.inputs)
        outputs = {k: x.detach().clone() for k, x in self.live.items()}
        reference = {k: x.detach().clone() for k, x in self.live.items()}
        self.private = PrivateWrites(self.manager, self.joint, self.hold, self.inputs, outputs, t)
        self.reference = PrivateWrites(self.manager, self.joint, self.hold, self.source_refs, reference)
        # The manager was copied after installing its instance hooks. Remove
        # only those private copies so original class methods are invoked.
        for shadow in (self.private, self.reference):
            for name in ('process_action', 'apply_action'):
                vars(shadow.manager).pop(name, None)
        self.capture()
        self.staged_controls += 1

    def compute_commit(self):
        self.private.compute(self.live)
        for name, value in self.live.items():
            value.copy_(self.private.outputs[name])

    def capture(self):
        t = self.torch
        require(all(x.is_cuda for x in (*self.live.values(), *self.inputs.values())),
                'Action graph needs the existing CUDA tensors')
        device = self.inputs['processed'].device
        current = t.cuda.current_stream(device)
        stream = t.cuda.Stream(device=device)
        stream.wait_stream(current)
        # Warmup invokes only private writes: no additional live action write,
        # simulator step, delay sample, audit sample or disturbance application.
        with t.cuda.stream(stream):
            for _ in range(3):
                self.private.compute(self.live)
        current.wait_stream(stream)
        t.cuda.synchronize(device)
        graph = t.cuda.CUDAGraph()
        with t.cuda.graph(graph, stream=stream):
            self.compute_commit()
        current.wait_stream(stream)
        self.graph = graph

    def apply(self):
        require(not self.closed and self.controls > 0 and self.current_substeps < 10,
                'Action write outside its processed control')
        require({k: storage_signature(x) for k, x in self.fixed_fields().items()} == self.field_signatures,
                'Action indexing or encoder-bias storage changed')
        require(all(x is self.source_refs[k] for k, x in self.current_inputs().items()),
                'Processed action changed between original process_action calls')
        if self.graph is None:
            self.prepare()
        require({k: storage_signature(x) for k, x in self.live_outputs().items()} == self.output_signatures,
                'Action destination storage changed')
        checking = self.checked_calls < CHECK_CALLS
        if checking:
            # Reference reads exact live pre-replay destinations and original
            # inputs, with original class methods and independent output memory.
            self.reference.set_inputs(self.source_refs)
            self.reference.compute(self.live)
        self.graph.replay()
        if checking:
            for name, value in self.live.items():
                reference = self.reference.outputs[name]
                if not bool(self.torch.equal(reference.view(self.torch.int32), value.view(self.torch.int32))):
                    self.comparison_failure = dict(call=self.calls+1, field=name,
                        original_bits=reference.detach().view(self.torch.int32).reshape(-1).cpu().tolist(),
                        candidate_bits=value.detach().view(self.torch.int32).reshape(-1).cpu().tolist())
                    raise RuntimeError('Action graph differs from original FP32 bits: ' + name)
                self.scalar_checks += value.numel()
            self.checked_calls += 1
        self.calls += 1
        self.current_substeps += 1

    def close(self):
        if self.closed:
            return
        try:
            current = {k: v.detach().cpu().tolist() for k, v in self.fixed_fields().items()
                       if k != 'encoder_bias'}
            require(current == self.index_values, 'Action mapping values changed during run')
        finally:
            super().close()
            self.closed = True

    def stats(self):
        return dict(sources=self.sources, replays=self.calls, controls=self.controls,
                    capture_completed=self.graph is not None, comparison_failure=self.comparison_failure,
                    staged_controls=self.staged_controls, same_input_checked_calls=self.checked_calls,
                    same_input_scalar_comparisons=self.scalar_checks, closed=self.closed,
                    bindings_restored=not self.saved, original_apply_action_methods=True,
                    original_process_action=True, comparison='exact FP32 bits; full destinations',
                    untouched_body_wrenches_preserved=True, live_warmup_writes=0,
                    body_index='original [0] represented once as device int64 [0]; no remap',
                    extra_physics_steps=0, extra_RNG_draws=0, physics_hz=500,
                    force_schedule_in_graph=False, bridge_views=self.views.stats())
