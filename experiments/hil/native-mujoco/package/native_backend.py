"""One-world native MuJoCo integration, with CPU tensors for existing task code.

No mujoco_warp physics, CUDA graphs, rendering, or GPU copies run here. Float32
sensor views preserve the policy's input dtype; MjData keeps native float64
integration state. Unmodified qpos/qvel views are never written back (rounding
the state on every substep would change the native integrator).
"""
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import mujoco
import torch


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


class ModelView:
    def __init__(self, native):
        self.native, self.cache, self.published = native, {}, {}
        self.arrays={}

    def __getattr__(self, name):
        if name not in self.cache:
            value = getattr(self.native, name)
            if not isinstance(value, np.ndarray):
                return value
            # mjlab's real-valued model fields have a world dimension. Integer
            # topology/address arrays are shared, without that dimension.
            array = value.astype(np.float32) if value.dtype.kind == 'f' else value.copy()
            if value.dtype.kind == 'f':
                array = array[None]
            self.cache[name] = torch.from_numpy(array.copy())
            self.published[name] = array.copy()
            self.arrays[name]=self.cache[name].numpy()
        return self.cache[name]

    def push(self):
        for name,array in self.arrays.items():
            if np.array_equal(array, self.published[name]):
                continue
            target = getattr(self.native, name)
            need(np.isfinite(array).all(), 'Nonfinite native model field: ' + name)
            np.copyto(target, array[0] if target.dtype.kind == 'f' else array, casting='unsafe')
            np.copyto(self.published[name],array)

    def pull(self):
        for name, target in self.arrays.items():
            value = getattr(self.native, name)
            array = value[None] if value.dtype.kind == 'f' else value
            np.copyto(target, array, casting='unsafe')
            np.copyto(self.published[name],target)


class DataView:
    INPUTS = frozenset(('qpos', 'qvel', 'act', 'ctrl', 'qfrc_applied',
                       'xfrc_applied', 'mocap_pos', 'mocap_quat', 'eq_active'))

    def __init__(self, native):
        self.native, self.cache, self.published = native, {}, {}
        self.arrays={}
        self.inputs={}
        self.nworld = 1
        self.revision = 0

    def array(self, name, cast=True):
        value = getattr(self.native, name)
        array = np.asarray(value)
        if name.endswith('xmat'):
            array = array.reshape(-1, 3, 3)
        if cast and array.dtype.kind == 'f':
            array = array.astype(np.float32)
        return array[None]

    def __getattr__(self, name):
        if name == 'efc':
            # Constraint arrays can change size after every solve. BAM fetches
            # fresh views each time and masks by nefc before its friction scan.
            return SimpleNamespace(**{key: torch.from_numpy(self.array('efc_' + key).copy())
                                      for key in ('type', 'id', 'force')})
        if name not in self.cache:
            array = self.array(name)
            self.cache[name] = torch.from_numpy(array.copy())
            self.arrays[name]=self.cache[name].numpy()
            if name in self.INPUTS:
                self.published[name] = array.copy()
                self.inputs[name]=self.arrays[name]
        return self.cache[name]

    def push(self):
        for name,array in self.inputs.items():
            if not np.array_equal(array, self.published[name]):
                need(np.isfinite(array).all(), 'Nonfinite native input: ' + name)
                np.copyto(getattr(self.native, name), array[0], casting='unsafe')
                np.copyto(self.published[name],array)

    def pull(self):
        self.revision += 1
        for name,target in self.arrays.items():
            array = self.array(name,cast=False)
            need(target.shape == array.shape, 'Native state shape changed: ' + name)
            np.copyto(target, array, casting='unsafe')
            if name in self.INPUTS:
                np.copyto(self.published[name],target)


class NativeSimulation:
    def __init__(self, num_envs, cfg, model, device):
        need(num_envs == 1 and device == 'cpu', 'Native backend requires one CPU world')
        need(mujoco.__version__ == '3.10.0', 'Native MuJoCo version differs')
        self.cfg, self.device, self.num_envs = cfg, device, num_envs
        cfg.mujoco.apply(model)
        # Keep the compile-time model immutable for original DR defaults and
        # geometry/limit metadata. The solver owns a separate randomized model.
        self.mj_model = model
        self.native_model = deepcopy(model)
        self.mj_data = mujoco.MjData(self.native_model)
        mujoco.mj_forward(self.native_model, self.mj_data)
        self.model, self.data = ModelView(self.native_model), DataView(self.mj_data)
        self.default_model_fields, self.expanded_fields = {}, set()
        self.use_cuda_graph = False
        self.step_calls = self.forward_calls = self.reset_calls = self.sense_calls = 0
        self.after_step = lambda: None
        self._sensor_context = None
        self._warning_counts = np.array(self.mj_data.warning.number, copy=True)

    def expand_model_fields(self, fields):
        for name in fields:
            need(hasattr(self.mj_model, name), 'Unknown randomized model field: ' + name)
            getattr(self.model, name)
        self.expanded_fields.update(fields)

    def get_default_field(self, field):
        if field not in self.default_model_fields:
            original = getattr(self.mj_model, field)
            self.default_model_fields[field] = torch.as_tensor(
                original, dtype=getattr(self.model, field).dtype, device='cpu').clone()
        return self.default_model_fields[field]

    def recompute_constants(self, level):
        need(level.name in ('set_const', 'set_const_0', 'set_const_fixed'), 'Unknown DR recomputation')
        self.model.push()
        # mj_setConst uses qpos0 in an independent scratch data, preserving the
        # live reset state, time, warm-start and forces.
        mujoco.mj_setConst(self.native_model, mujoco.MjData(self.native_model))
        self.model.pull()

    def _check(self):
        need(all(np.isfinite(getattr(self.mj_data, name)).all()
                 for name in ('qpos', 'qvel', 'qacc', 'ctrl', 'sensordata')),
             'Native MuJoCo produced nonfinite state')
        warnings = np.array(self.mj_data.warning.number, copy=True)
        need(np.array_equal(warnings, self._warning_counts),
             'Native MuJoCo warning; preserve state: ' + str(warnings.tolist()))

    def forward(self):
        self.model.push()
        self.data.push()
        mujoco.mj_forward(self.native_model, self.mj_data)
        self._check()
        self.data.pull()
        self.forward_calls += 1

    def step(self):
        self.model.push()
        self.data.push()
        before = float(self.mj_data.time)
        mujoco.mj_step(self.native_model, self.mj_data)
        need(abs(float(self.mj_data.time) - before - self.native_model.opt.timestep) < 1e-10,
             'Native integrator reset or changed simulation time')
        self._check()
        self.data.pull()
        self.step_calls += 1
        self.after_step()

    def reset(self, env_ids=None):
        if env_ids is not None and len(env_ids) == 0:
            return
        need(env_ids is None or list(env_ids) == [0], 'Unexpected native reset selection')
        self.model.push()
        mujoco.mj_resetData(self.native_model, self.mj_data)
        self._warning_counts = np.array(self.mj_data.warning.number, copy=True)
        self.data.pull()
        self.reset_calls += 1

    def set_sensor_context(self, ctx):
        need(isinstance(ctx, NativeSensorContext), 'Unexpected native sensor context')
        self._sensor_context = ctx

    def sense(self):
        # IMU/contact sensors come from forward; foot rays use native mj_ray.
        if self._sensor_context is not None:
            self._sensor_context.sense()
        self.sense_calls += 1

    def snapshot(self):
        return dict(backend='mujoco.mj_step', mujoco_version=mujoco.__version__,
            device='cpu', physics_dt=self.native_model.opt.timestep,
            native_step_calls=self.step_calls, native_forward_calls=self.forward_calls,
            reset_calls=self.reset_calls, sense_calls=self.sense_calls,
            integration_dtype='float64', policy_observation_dtype='float32',
            warp_physics_calls=0, cuda_graphs=False, warning_counts=self._warning_counts.tolist())


class NativeSensorContext:
    """Retain the original terrain-ray preparation/reduction, using mj_ray."""
    def __init__(self, mj_model, model, data, camera_sensors, raycast_sensors, device):
        need(device == 'cpu' and not camera_sensors, 'Native validation has no camera input')
        self.model, self.data = model.native, data.native
        self.raycast_sensors = raycast_sensors
        self.camera_sensors = []
        self.has_cameras, self.has_raycasts = False, bool(raycast_sensors)
        for sensor in raycast_sensors:
            need(sensor.cfg.name == 'foot_height_scan', 'Unreviewed native ray sensor')
            sensor.set_context(self)

    def sense(self):
        import warp as wp
        for sensor in self.raycast_sensors:
            sensor.prepare_rays()
            origins = wp.to_torch(sensor._ray_pnt).numpy()[0]
            vectors = wp.to_torch(sensor._ray_vec).numpy()[0]
            excludes = wp.to_torch(sensor._ray_bodyexclude).numpy()
            distances = wp.to_torch(sensor._ray_dist).numpy()[0]
            geomids = wp.to_torch(sensor._ray_geomid).numpy()[0]
            normals = wp.to_torch(sensor._ray_normal).numpy()[0]
            groups = None
            if sensor.cfg.include_geom_groups is not None:
                groups = np.zeros(6,dtype=np.uint8)
                groups[list(sensor.cfg.include_geom_groups)] = 1
            for i,(origin,vector) in enumerate(zip(origins,vectors)):
                geom = np.array([-1],dtype=np.int32)
                normal = np.zeros(3,dtype=np.float64)
                distances[i] = mujoco.mj_ray(self.model,self.data,origin.astype(np.float64),
                    vector.astype(np.float64),groups,True,int(excludes[i]),geom,normal)
                geomids[i], normals[i] = geom[0], normal
            sensor.postprocess_rays()


@contextmanager
def native_environment_factory():
    """Scope the existing environment's backend selection to this process only."""
    import mjlab.envs.manager_based_rl_env as module
    import importlib
    scene_module = importlib.import_module('mjlab.scene.scene')
    original = module.Simulation
    original_context = scene_module.SensorContext
    module.Simulation = NativeSimulation
    scene_module.SensorContext = NativeSensorContext
    try:
        yield
    finally:
        module.Simulation = original
        scene_module.SensorContext = original_context
