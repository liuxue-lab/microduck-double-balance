"""Real Warp initialization and offline NVRTC compilation; no physics/GPU run."""
from pathlib import Path
import importlib
import importlib.util
import json
import os

ROOT = Path(__file__).resolve().parents[2]


def run(output, repo, legacy_root=None):
    for key in list(os.environ):
        if key.startswith('GIT_'): os.environ.pop(key)
    from integration.child_bootstrap import owner_bootstrap
    from integration.durable import write_json, file_digest
    from validation.integration.bootstrap_cpu import import_runtime
    reproduce_v3 = legacy_root is not None
    if reproduce_v3:
        spec=importlib.util.spec_from_file_location('legacy_bootstrap',legacy_root/'integration/child_bootstrap.py')
        legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
        owner_bootstrap=legacy.owner_bootstrap
    output.mkdir(parents=True)
    owner = output/'owner-0'
    result = dict(native_integrations=0,CPU_forward=0,GPU_forward=0,
                  environment_constructions=0,PPO=0,cloud_training=0,
                  GPU_execution=False,compiled_modules=[],long_evidence_path=str(owner))
    try:
        with owner_bootstrap(ROOT,owner) as bootstrap:
            owner.mkdir()
            import torch
            result['imports'] = import_runtime(repo,owner,torch,legacy=reproduce_v3)
            import warp as wp
            result['temporary_directory']=os.environ['TMPDIR']
            result['temporary_directory_bytes']=len(os.fsencode(os.environ['TMPDIR']))
            names=('mujoco_warp._src.render_util','mujoco_warp._src.ray',
                   'rigid_metadata_kernel','order_kernels','batch_kernels','solver_order_kernel')
            for name in names:
                module = importlib.import_module(name)
                # Actual NVRTC CUBIN compilation of the failed sensor dependency,
                # plus the four accepted native modules. No GPU launch/load.
                dest=Path(wp.config.kernel_cache_dir)/name
                wp.compile_aot_module(module,arch=120,module_dir=dest,use_ptx=False)
                cubins=list(dest.glob('*.cubin'));assert len(cubins)==1 and cubins[0].stat().st_size>0
                result['compiled_modules'].append(dict(name=name,source_sha256=file_digest(module.__file__),
                    target='sm_120 CUBIN, offline only',sha256=file_digest(cubins[0]),bytes=cubins[0].stat().st_size))
            assert not reproduce_v3, 'Legacy long-path compilation unexpectedly succeeded'
            assert not torch.cuda.is_initialized() and not wp.is_cuda_available()
            result['status']='PASS'
    except Exception as exc:
        if not reproduce_v3 or 'CUDA kernel build failed with error code 6' not in str(exc):
            raise
        result.update(status='V3_LONG_TMPDIR_DEFECT_REPRODUCED',error=repr(exc))
    write_json(owner/'runtime-compile-report.json',result)
    print(json.dumps(result),flush=True)
    return result
