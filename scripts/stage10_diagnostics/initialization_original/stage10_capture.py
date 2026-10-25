"""Read-only wrappers around existing calls; never replay or replace an operation."""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import random

import numpy as np

from stage10_core import require, write

PHYSICAL = ('time', 'qpos', 'qvel', 'qacc', 'qacc_warmstart', 'ctrl', 'act', 'act_dot',
            'qfrc_applied', 'qfrc_actuator', 'qfrc_passive', 'qfrc_bias', 'qfrc_constraint',
            'xfrc_applied', 'actuator_force', 'solver_niter', 'nefc', 'ncon', 'nacon', 'nisland')
CONTACT = ('worldid', 'geom', 'dim', 'dist', 'pos', 'frame', 'friction',
           'solref', 'solreffriction', 'solimp', 'efc_address')


class Hooks:
    def __init__(self):
        self.saved = []

    def wrap(self, obj, name, factory):
        original = getattr(obj, name)
        existed = name in vars(obj)
        old = vars(obj).get(name)
        replacement = factory(original)
        setattr(obj, name, replacement)
        self.saved.append((obj, name, existed, old))

    def close(self):
        for obj, name, existed, old in reversed(self.saved):
            if existed:
                setattr(obj, name, old)
            else:
                delattr(obj, name)
        self.saved.clear()


def code_identity(obj):
    cls = type(obj)
    out = {'type': cls.__module__ + '.' + cls.__qualname__, 'sources': {}}
    for base in cls.__mro__:
        if base is object:
            continue
        try:
            source = inspect.getsource(base)
            # Runtime dependency code, not unrelated files or environment secrets.
            out['sources'][base.__module__ + '.' + base.__qualname__] = source[:400000]
        except (TypeError, OSError):
            out['sources'][base.__module__ + '.' + base.__qualname__] = 'SOURCE_UNAVAILABLE'
    return out


class Capture:
    def __init__(self, raw, runner, policy, bam, writer, torch, output):
        self.raw, self.runner, self.policy, self.bam = raw, runner, policy, bam
        self.writer, self.torch, self.output = writer, torch, output
        self.fields, self.schema = {}, {'physical': {}, 'contact': {}, 'object_inventory': {}}
        self.control, self.steps, self.forwards, self.bam_calls, self.actions = 0, 0, 0, 0, 0
        self.applies = 0
        self.position_action = raw.action_manager.get_term('joint_pos')
        self.hold_action = raw.action_manager.get_term('ball_hold')
        self.hooks = Hooks()
        self.physical_readers = {}
        data = raw.sim.data
        wp_data = getattr(raw.sim, 'wp_data', None)
        for name in PHYSICAL:
            choices = ((data, 'sim.data'), (wp_data, 'sim.wp_data'))
            for obj, location in choices:
                if obj is None:
                    continue
                try:
                    value = getattr(obj, name)
                    self.numpy(value)
                except (AttributeError, TypeError, ValueError) as exc:
                    continue
                self.physical_readers[name] = (obj, name)
                self.schema['physical'][name] = {'location': location, 'shape': list(self.numpy(value).shape)}
                break
            else:
                self.schema['physical'][name] = {'status': 'UNAVAILABLE'}
        self.contact = getattr(wp_data, 'contact', None) if wp_data is not None else None
        self.nacon = getattr(wp_data, 'nacon', None) if wp_data is not None else None
        self.schema['contact']['sampling'] = 'Prefix [0:nacon[0]] when scalar aggregate count is available; storage semantics require source review'
        self.schema['contact']['validity_not_independently_established'] = True
        self.schema['gaps'] = [
            'No promise of complete MuJoCo/Warp integrator, CUDA graph or device RNG internal state.',
            'No additional forward, reset, policy call, reward call or success recomputation.',
            'Contact order differences are reported separately and do not prove physical divergence.',
            'Only direct tensor/scalar state and selected nested delay/buffer objects are read.',
        ]
        for label, obj in [('bam', bam), ('actor', runner.alg.actor), ('critic', runner.alg.critic),
                           ('position_action', self.position_action), ('hold_action', self.hold_action)]:
            self.schema['object_inventory'][label] = self.object_inventory(obj)
        sources = {label: code_identity(obj) for label, obj in
                   [('env', raw), ('simulation', raw.sim), ('bam', bam),
                    ('actor', runner.alg.actor), ('critic', runner.alg.critic), ('policy', policy),
                    ('position_action', self.position_action), ('hold_action', self.hold_action)]}
        if wp_data is not None:
            sources['warp_data'] = code_identity(wp_data)
        if self.contact is not None:
            sources['warp_contact'] = code_identity(self.contact)
        write(output / 'dependency-sources.json', sources)
        write(output / 'capture-schema.json', self.schema)
        require(all(name in self.physical_readers for name in ('qpos', 'qvel', 'ctrl', 'qacc')),
                'Critical qpos/qvel/ctrl/qacc fields unavailable; return schema and log')
        require(all(self.torch.is_tensor(getattr(bam, key, None)) for key in
                    ('vin_tensor', 'kp_scale', 'kd_scale', 'friction_scale')),
                'Required BAM voltage/gain/friction tensors unavailable; return schema and source')

    def numpy(self, value):
        if isinstance(value, np.ndarray):
            return value
        if self.torch.is_tensor(value):
            return value.detach().cpu().numpy()
        if type(value).__module__.startswith('warp') and hasattr(value, 'numpy'):
            return value.numpy()
        if isinstance(value, (bool, int, float, np.generic)):
            return np.asarray(value)
        raise TypeError('Unsupported numeric field: ' + type(value).__name__)

    def put(self, key, value, axis=None):
        a = self.numpy(value)
        if axis is not None and (a.ndim <= axis or a.shape[axis] != 128):
            axis = None
        self.fields[key] = self.writer.array(a, axis)

    def tree(self, prefix, value, axis=None, depth=0):
        if depth > 5:
            self.fields[prefix + '/capture_status'] = 'DEPTH_LIMIT'
        elif self.torch.is_tensor(value) or isinstance(value, np.ndarray):
            self.put(prefix, value, axis)
        elif value is None or isinstance(value, (bool, int, str)):
            self.fields[prefix] = value
        elif isinstance(value, (float, np.floating)):
            self.fields[prefix] = float(value) if math.isfinite(value) else str(value)
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                self.tree(prefix + '/' + str(i), item, axis, depth + 1)
        elif hasattr(value, 'items') and callable(value.items):
            for key, item in value.items():
                self.tree(prefix + '/' + str(key), item, axis, depth + 1)
        else:
            self.fields[prefix + '/capture_status'] = 'UNSUPPORTED:' + type(value).__name__

    def object_inventory(self, obj):
        return {k: {'type': type(v).__module__ + '.' + type(v).__name__,
                    'shape': list(v.shape) if self.torch.is_tensor(v) else None}
                for k, v in sorted(vars(obj).items()) if not callable(v)}

    def direct_state(self, prefix, obj, depth=0, seen=None, axis=0):
        """Read vars only: do not invoke stateful lazy properties."""
        if seen is None:
            seen = set()
        if id(obj) in seen or not hasattr(obj, '__dict__'):
            return
        seen.add(id(obj))
        for key, value in sorted(vars(obj).items()):
            if callable(value) or key in ('_parameters', '_modules', '_flat_weights', '_flat_weight_refs', '_all_weights'):
                continue
            path = prefix + '/' + key
            if self.torch.is_tensor(value) or isinstance(value, (np.ndarray, bool, int, float, str, type(None), list, tuple, dict)):
                self.tree(path, value, axis)
            elif depth < 3 and any(token in key.lower() for token in ('delay', 'buffer', 'history', 'cfg', 'bam_model', 'actuator')):
                self.direct_state(path, value, depth + 1, seen, axis)

    def model(self, label, model):
        if hasattr(model, 'named_modules'):
            for name, module in model.named_modules():
                # LSTM hidden state is normally (layers, envs, features).
                self.direct_state(label + '/' + name, module, axis=1)
        else:
            self.direct_state(label, model, axis=1)

    def policy_state(self, obs):
        self.tree('observations', obs, 0)
        self.model('actor', self.runner.alg.actor)
        self.model('critic', self.runner.alg.critic)
        if self.policy is not self.runner.alg.actor:
            self.model('inference_policy', self.policy)
        self.put('rng/torch_cpu', self.torch.get_rng_state())
        for i, state in enumerate(self.torch.cuda.get_rng_state_all()):
            self.put('rng/torch_cuda/' + str(i), state)
        state = np.random.get_state()
        self.tree('rng/numpy', state)
        self.tree('rng/python', random.getstate())

    def physical(self):
        for name, (obj, key) in self.physical_readers.items():
            self.put('physical/' + name, getattr(obj, key), 0)

    def contacts(self):
        if self.contact is None or self.nacon is None:
            self.fields['contact/status'] = 'NO_AGGREGATE_CONTACT_ACCESS'
            return
        count_array = self.numpy(self.nacon)
        if count_array.size != 1:
            self.fields['contact/status'] = 'COUNT_LAYOUT_UNRECOGNIZED'
            return
        count = int(count_array.reshape(-1)[0])
        require(count >= 0, 'Negative aggregate contact count')
        self.fields['contact/count'] = count
        for key in CONTACT:
            value = getattr(self.contact, key, None)
            if value is None:
                self.fields['contact/' + key + '/status'] = 'UNAVAILABLE'
                continue
            # Warp and torch slicing have different APIs. Transfer the observed
            # backing array, then retain only the aggregate-count prefix.
            shape = tuple(getattr(value, 'shape', ()))
            if not shape or shape[0] < count:
                self.fields['contact/' + key + '/status'] = 'LAYOUT_UNRECOGNIZED'
                continue
            if self.torch.is_tensor(value):
                sample = self.numpy(value[:count])
            elif type(value).__module__.startswith('warp'):
                import warp as wp
                sample = self.numpy(wp.to_torch(value)[:count])
            else:
                sample = self.numpy(value)[:count]
            self.put('contact/' + key, sample)

    def emit(self, phase, *, substep=None, physical=False, bam=False, contact=False, action=False, obs=None, extras=None):
        self.fields = {}
        if physical:
            self.physical()
        if bam:
            self.direct_state('bam', self.bam)
        if action:
            self.direct_state('position_action', self.position_action)
            self.direct_state('hold_action', self.hold_action)
            for key in ('joint_pos_target', 'joint_vel_target', 'joint_effort_target'):
                self.tree('actuation/' + key, getattr(self.raw.scene['robot'].data, key, None), 0)
        if contact:
            self.contacts()
        if obs is not None:
            self.policy_state(obs)
        if extras:
            for key, value in extras.items():
                self.tree(key, value, 0)
        self.writer.record(phase, self.control, self.steps % 10 if substep is None else substep, self.fields)

    def install(self):
        def process_factory(original):
            def process(*args, **kwargs):
                result = original(*args, **kwargs)
                self.actions += 1
                self.emit('action_processed', substep=0, action=True,
                          extras={'action_manager_input': args[0],
                                  'action_manager_action': self.raw.action_manager.action})
                return result
            return process

        def apply_factory(original):
            def apply(*args, **kwargs):
                require(self.applies < 100, 'Action-application budget exceeded')
                result = original(*args, **kwargs)
                self.applies += 1
                self.emit('action_applied', substep=self.steps % 10 + 1, action=True)
                return result
            return apply

        def bam_factory(original):
            def compute(cmd):
                require(self.bam_calls < 100, 'BAM-call budget exceeded')
                substep = self.steps % 10 + 1
                command = {k: getattr(cmd, k) for k in ('position_target', 'velocity_target', 'effort', 'pos', 'vel') if hasattr(cmd, k)}
                self.emit('bam_before', substep=substep, bam=True, extras={'command': command})
                result = original(cmd)
                self.bam_calls += 1
                self.emit('bam_after', substep=substep, bam=True, extras={'returned_torque': result})
                return result
            return compute

        def step_factory(original):
            def step(*args, **kwargs):
                require(self.steps < 100, 'Physics-step budget exceeded')
                substep = self.steps % 10 + 1
                self.emit('physics_before', substep=substep, physical=True, contact=True)
                result = original(*args, **kwargs)
                self.steps += 1
                self.emit('physics_after', substep=substep, physical=True, contact=True)
                return result
            return step

        def forward_factory(original):
            def forward(*args, **kwargs):
                require(self.forwards < 10, 'Forward-call budget exceeded')
                self.emit('forward_before', substep=10, physical=True, contact=True)
                result = original(*args, **kwargs)
                self.forwards += 1
                self.emit('forward_after', substep=10, physical=True, contact=True)
                return result
            return forward

        self.hooks.wrap(self.raw.action_manager, 'process_action', process_factory)
        self.hooks.wrap(self.raw.action_manager, 'apply_action', apply_factory)
        self.hooks.wrap(self.bam, 'compute', bam_factory)
        self.hooks.wrap(self.raw.sim, 'step', step_factory)
        self.hooks.wrap(self.raw.sim, 'forward', forward_factory)

    def verify_counts(self):
        require((self.steps, self.forwards, self.bam_calls, self.actions, self.applies) == (100, 10, 100, 10, 100),
                f'Unexpected operation counts: {self.steps}, {self.forwards}, {self.bam_calls}, {self.actions}, {self.applies}')


def module_digest(model, torch):
    """Fingerprint loaded parameters/buffers without saving another checkpoint."""
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        require(torch.is_tensor(value), 'Non-tensor model state needs explicit review')
        a = value.detach().cpu().numpy()
        h.update(json.dumps([name, a.dtype.str, list(a.shape)]).encode() + b'\0' + a.tobytes())
    return h.hexdigest()
