"""Reuse input/source verification without initializing or requiring a GPU."""
import os
import sys
from bootstrap import load_base


def preflight():
    old = load_base()
    old.require(sys.platform=='linux' and old.PYTHON.is_file(), 'Existing project Python required')
    old.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'Native worker must hide CUDA devices')
    package = old.verify_package()
    versions = old.verify_runtime()
    inputs = old.verify_inputs()
    return dict(package_identity=package, versions=versions, inputs=inputs,
        backend='native MuJoCo 3.10.0 CPU', GPU_initialized=False,
        physics_dt=.005, control_dt=.02, decimation=4,
        physics_steps=0, PPO_updates=0, cloud_connections=0)
