"""Observe original initialization calls exactly once; no added physics calls."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import dataclasses
import inspect
import math
import numbers

import numpy as np

from stage10_core import require, write
from stage10_capture import Capture, Hooks, code_identity, PHYSICAL

EXTRA_DATA = ('eq_active', 'mocap_pos', 'mocap_quat', 'ne', 'nf', 'nl', 'ncollision',
              'xpos', 'xquat', 'xmat', 'xipos', 'ximat', 'subtree_com', 'cdof', 'cinert',
              'crb', 'qM', 'qLD', 'qLDiagInv', 'cvel', 'cdof_dot', 'actuator_length',
              'actuator_velocity', 'actuator_moment', 'qfrc_spring', 'qfrc_damper',
              'qfrc_gravcomp', 'qfrc_fluid', 'qfrc_smooth', 'qacc_smooth', 'cacc',
              'cfrc_int', 'cfrc_ext', 'tree_island', 'sensordata')
INPUT_FIELDS = ('time', 'qpos', 'qvel', 'act', 'ctrl', 'qacc_warmstart', 'qfrc_applied',
                'xfrc_applied', 'eq_active', 'mocap_pos', 'mocap_quat')
LIMITS = {'raw.reset': 1, 'raw._reset_idx': 1, 'sim.reset': 1, 'scene.reset': 1,
          'event.apply': 1, 'scene.write_data_to_sim': 1, 'sim.forward': 2, 'sim.sense': 1,
          'bam.compute': 1, 'command.compute': 1, 'observation.compute': 2, 'policy.reset': 1}
MAX_ARRAY_BYTES = 8 * 1024**2


class BoundaryHooks:
    def __init__(self, snapshot, counts_path):
        self.snapshot, self.counts_path = snapshot, counts_path
        self.hooks, self.counts, self.stack = Hooks(), Counter(), []

    def install_one(self, obj, method, label):
        def factory(original):
            def observed(*args, **kwargs):
                self.counts[label] += 1
                write(self.counts_path, {'counts': dict(self.counts), 'stack': self.stack})
                require(self.counts[label] <= LIMITS[label], 'Unexpected extra original call: ' + label)
                name = label + '#' + str(self.counts[label])
                self.stack.append(name)
                observed_args = {k: v for k,v in kwargs.items() if k in
                                 ('seed', 'env_ids', 'mode', 'global_env_step_count', 'dt', 'update_history')}
                if args and label in ('raw._reset_idx', 'sim.reset', 'scene.reset'):
                    observed_args['env_ids_positional'] = args[0]
                if args and label == 'bam.compute':
                    observed_args['command'] = {k: getattr(args[0], k) for k in
                                                ('position_target', 'velocity_target', 'effort', 'pos', 'vel')
                                                if hasattr(args[0], k)}
                self.snapshot(name + '.before', None, list(self.stack), observed_args)
                result = original(*args, **kwargs)  # One original call, unchanged arguments.
                returned_obs = result if label in ('observation.compute', 'bam.compute') else None
                self.snapshot(name + '.after', returned_obs, list(self.stack), None)
                self.stack.pop()
                return result
            return observed
        self.hooks.wrap(obj, method, factory)

    def close(self):
        self.hooks.close()

    def verify(self):
        require(dict(self.counts) == LIMITS and not self.stack,
                'Original initialization call counts differ; preserve the trace for review')


class InitCapture(Capture):
    """Reuse reviewed read-only tensor access without its rollout wrappers."""
    def __init__(self, raw, runner, policy, writer, torch, output):
        self.raw, self.runner, self.policy = raw, runner, policy
        self.writer, self.torch, self.output = writer, torch, output
        self.bam = raw.scene['robot'].actuators[0]
        self.position_action = raw.action_manager.get_term('joint_pos')
        self.hold_action = raw.action_manager.get_term('ball_hold')
        self.fields, self.gaps = {}, {}
        self.schema = {'physical': {}, 'model': {}, 'workspace': {},
                       'input_fields': list(INPUT_FIELDS),
                       'limits': [
                           'Snapshots synchronize GPU work and may alter ordering.',
                           'Before reset/forward, derived arrays can be stale from earlier initialization.',
                           'EFC arrays retain padded storage; inactive entries are not physical constraints.',
                           'Model enumeration is bounded and does not prove complete hidden-state equivalence.',
                           'No extra policy, reset, forward, sense, observation or rollout call is introduced.']}
        self.physical_readers = {}
        wp_data = raw.sim.wp_data
        self.contact, self.nacon = wp_data.contact, wp_data.nacon
        for name in dict.fromkeys((*PHYSICAL, *EXTRA_DATA)):
            if hasattr(wp_data, name):
                value = getattr(wp_data, name)
                self.physical_readers[name] = (wp_data, name)
                self.schema['physical'][name] = {'shape': list(getattr(value, 'shape', ())),
                                                  'location': 'sim.wp_data'}
            else:
                self.schema['physical'][name] = {'status': 'UNAVAILABLE'}
        require(all(k in self.physical_readers for k in INPUT_FIELDS),
                'A required simulator input is unavailable; stop before observed reset')
        self.model_objects = [('model/wp', raw.sim.wp_model), ('model/cpu', raw.sim.mj_model)]
        self.boundaries = BoundaryHooks(self.snapshot, output/'boundary-counts.json')
        write(output/'capture-schema.json', self.schema)
        # Class source is captured before installing boundary wrappers.
        objects = [('environment', raw), ('simulation', raw.sim), ('scene', raw.scene),
                   ('observations', raw.observation_manager), ('commands', raw.command_manager),
                   ('warp_model', raw.sim.wp_model), ('warp_data', wp_data),
                   ('warp_efc', wp_data.efc), ('warp_contact', wp_data.contact),
                   ('actor', runner.alg.actor), ('critic', runner.alg.critic), ('bam', self.bam)]
        write(output/'dependency-sources.json', {k: code_identity(v) for k,v in objects})

    def bounded_array(self, key, value, *, axis=None, category='model'):
        try:
            if self.torch.is_tensor(value):
                count_bytes = value.numel() * value.element_size()
                if count_bytes > MAX_ARRAY_BYTES:
                    self.gap(key, 'ARRAY_EXCEEDS_8_MIB', category, shape=list(value.shape), bytes=count_bytes)
                    return
                a = value.detach().cpu().numpy()
            elif isinstance(value, np.ndarray):
                if value.nbytes > MAX_ARRAY_BYTES or value.dtype.hasobject:
                    self.gap(key, 'ARRAY_TOO_LARGE_OR_OBJECT', category,
                             shape=list(value.shape), bytes=value.nbytes)
                    return
                a = value
            elif type(value).__module__.startswith('warp') and hasattr(value, 'numpy'):
                # A zero-copy view gives a byte bound before any device-to-host copy.
                import warp as wp
                view = wp.to_torch(value)
                return self.bounded_array(key, view, axis=axis, category=category)
            else:
                raise TypeError('Unsupported array class ' + type(value).__name__)
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            self.gap(key, 'UNREADABLE_ARRAY', category, error=type(exc).__name__ + ': ' + str(exc)[:300])
            return
        if axis is not None and (a.ndim <= axis or a.shape[axis] != 128):
            axis = None
        self.fields[key] = self.writer.array(a, axis)
        self.schema[category][key] = {'dtype': a.dtype.str, 'shape': list(a.shape),
                                       'nonfinite_in_last_snapshot': int((~np.isfinite(a)).sum())
                                       if a.dtype.kind in 'buif' else None}

    def gap(self, key, status, category, **detail):
        value = {'status': status, **detail}
        self.gaps[key] = value
        self.schema[category][key] = value
        self.fields[key + '/capture_status'] = value

    def public_numeric(self, prefix, obj, category='model', depth=0):
        if dataclasses.is_dataclass(obj):
            names = [f.name for f in dataclasses.fields(obj)]
        elif hasattr(obj, '__dict__') and not type(obj).__module__.startswith('mujoco.'):
            names = sorted(vars(obj))
        else:
            names = [n for n in dir(obj) if not n.startswith('_')]
        require(len(names) <= 2500, 'Unexpected model/constraint field count')
        for name in names:
            if name.startswith('_'):
                continue
            key = prefix + '/' + name
            try:
                value = getattr(obj, name)
            except (AttributeError, TypeError, ValueError) as exc:
                self.gap(key, 'ATTRIBUTE_UNAVAILABLE', category, error=type(exc).__name__)
                continue
            if callable(value):
                continue
            if value is None or isinstance(value, (str, bool, int)):
                self.fields[key] = value
            elif isinstance(value, numbers.Real):
                self.fields[key] = float(value) if math.isfinite(value) else str(value)
            elif isinstance(value, bytes):
                import hashlib
                self.fields[key] = {'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()}
            elif isinstance(value, np.ndarray) or self.torch.is_tensor(value) or (
                    type(value).__module__.startswith('warp') and hasattr(value, 'numpy')):
                self.bounded_array(key, value, category=category)
            elif name in ('opt', 'stat') and depth < 2:
                self.public_numeric(key, value, category, depth+1)
            elif isinstance(value, (list, tuple)) and all(isinstance(x, (str, bool, int, float)) for x in value):
                self.fields[key] = [float(x) if isinstance(x, float) and math.isfinite(x) else
                                    str(x) if isinstance(x, float) else x for x in value]
            else:
                self.gap(key, 'UNENUMERATED_OBJECT', category,
                         type=type(value).__module__ + '.' + type(value).__name__)

    def snapshot(self, phase, returned_obs=None, stack=None, observed_args=None):
        require(self.writer.events < 40, 'Initialization snapshot limit reached')
        self.fields = {}
        self.physical()
        self.contacts()
        self.direct_state('bam', self.bam)
        self.direct_state('position_action', self.position_action)
        self.direct_state('hold_action', self.hold_action)
        self.policy_state({})  # Read model state and RNG; never compute observations.
        self.fields['context/common_step_counter'] = int(self.raw.common_step_counter)
        self.fields['context/sim_step_counter'] = int(self.raw._sim_step_counter)
        self.fields['context/call_stack'] = stack or []
        self.fields['context/use_cuda_graph'] = bool(self.raw.sim.use_cuda_graph)
        for name in ('step_graph', 'forward_graph', 'reset_graph', 'sense_graph'):
            self.fields['context/' + name + '_present'] = getattr(self.raw.sim, name, None) is not None
        self.put('context/episode_length_buf', self.raw.episode_length_buf, 0)
        self.put('context/assistance_hold', self.raw._basketball_state.hold, 0)
        if observed_args is not None:
            self.tree('call_args', observed_args, 0)
        if returned_obs is not None:
            self.tree('returned_torque' if phase.startswith('bam.compute') else 'observations_returned', returned_obs, 0)
        for prefix, obj in self.model_objects:
            self.public_numeric(prefix, obj)
        self.public_numeric('workspace/efc', self.raw.sim.wp_data.efc, 'workspace')
        for key in ('body_mass', 'body_inertia', 'dof_damping', 'dof_armature',
                    'dof_frictionloss', 'geom_contype', 'geom_conaffinity'):
            require('model/cpu/' + key in self.fields, 'Required compiled model array unavailable: ' + key)
        for key in ('timestep', 'gravity', 'solver', 'integrator', 'iterations', 'tolerance', 'disableflags'):
            require('model/cpu/opt/' + key in self.fields, 'Required compiled option unavailable: ' + key)
        self.writer.record(phase, 0, 0, self.fields)
        self.schema['gaps'] = self.gaps
        write(self.output/'capture-schema.json', self.schema)

    def install(self):
        mappings = [(self.raw, 'reset', 'raw.reset'), (self.raw, '_reset_idx', 'raw._reset_idx'),
                    (self.raw.sim, 'reset', 'sim.reset'), (self.raw.scene, 'reset', 'scene.reset'),
                    (self.raw.event_manager, 'apply', 'event.apply'),
                    (self.raw.scene, 'write_data_to_sim', 'scene.write_data_to_sim'),
                    (self.raw.sim, 'forward', 'sim.forward'), (self.raw.sim, 'sense', 'sim.sense'),
                    (self.raw.command_manager, 'compute', 'command.compute'),
                    (self.raw.observation_manager, 'compute', 'observation.compute'),
                    (self.bam, 'compute', 'bam.compute'),
                    (self.policy, 'reset', 'policy.reset')]
        for obj, name, label in mappings:
            self.boundaries.install_one(obj, name, label)

    def close(self):
        self.boundaries.close()


@contextmanager
def no_rollout_and_count_bootstrap(env_cls, simulation_cls, mjwarp, output):
    """Count Python initialization entries; captured graph launches are separate.

    Original graph-capture internals are allowed during construction. These
    counts must not be presented as counts of every GPU kernel/graph execution.
    """
    hooks, counts = Hooks(), Counter()
    state = {'phase': 'construction', 'python_entry_counts': {},
             'rollout_calls': 0, 'counts_are_not_gpu_kernel_or_graph_launch_counts': True}
    def record():
        state['python_entry_counts'] = dict(counts)
        write(output/'bootstrap-calls.json', state)
    def deny_factory(original):
        def deny(*args, **kwargs):
            state['rollout_calls'] += 1
            record()
            raise RuntimeError('env.step/sim.step is forbidden in initialization-only diagnostics')
        return deny
    def count_factory(label):
        def factory(original):
            def counted(*args, **kwargs):
                key = state['phase'] + ':' + label
                counts[key] += 1
                record()
                require(sum(counts.values()) <= 256, 'Unexpected initialization entry count')
                if label == 'mjwarp.step':
                    require(state['phase'] == 'construction', 'Unexpected low-level physics step after construction')
                return original(*args, **kwargs)
            return counted
        return factory
    try:
        hooks.wrap(env_cls, 'step', deny_factory)
        hooks.wrap(simulation_cls, 'step', deny_factory)
        for cls, label in ((env_cls, 'env'), (simulation_cls, 'sim')):
            for name in ('reset', 'forward', 'sense'):
                if hasattr(cls, name):
                    hooks.wrap(cls, name, count_factory(label + '.' + name))
        for name in ('step', 'forward', 'reset_data'):
            hooks.wrap(mjwarp, name, count_factory('mjwarp.' + name))
        record()
        yield state
    finally:
        record()
        hooks.close()


@contextmanager
def no_policy_inference(runner):
    hooks = Hooks()
    def factory(original):
        def denied(*args, **kwargs):
            raise RuntimeError('Policy/critic forward is forbidden in initialization-only diagnostics')
        return denied
    try:
        seen = set()
        for model in (runner.alg.actor, runner.alg.critic):
            for module in model.modules():
                if id(module) not in seen:
                    hooks.wrap(module, 'forward', factory)
                    seen.add(id(module))
        yield
    finally:
        hooks.close()
