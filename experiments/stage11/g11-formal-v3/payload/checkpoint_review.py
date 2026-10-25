"""Read the pinned retained policy on CPU; never overwrite or load a runner."""
from pathlib import Path
from dataclasses import fields, is_dataclass
from collections.abc import Mapping
import hashlib
import inspect
import json
from functools import partial
from runtime_accounting import require, save

CHECKPOINT = '/home/lx/microduck-double-balance/artifacts/double-balance-stage08/returned/5d998fc322345102/evidence-20260930T093645-8955/extension/E-20260929/segment-001/checkpoints/update_004000.pt'
CHECKPOINT_SHA256 = '86d55c3703c18fcf4817db49c6e4dcf839b3f5195d68ae2c559db7e222bded97'


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def describe(value, depth=0):
    require(depth < 18, 'Configuration recursion limit')
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, Mapping):
        return {str(k): describe(v, depth+1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [describe(v, depth+1) for v in value]
    if isinstance(value, partial):
        return dict(partial=describe(value.func, depth+1), args=describe(value.args, depth+1), keywords=describe(value.keywords, depth+1))
    if callable(value):
        p = inspect.getsourcefile(value)
        return dict(callable=value.__module__+'.'+value.__qualname__,
                    source_sha256=sha_file(p) if p else None)
    if is_dataclass(value):
        return dict(type=type(value).__module__+'.'+type(value).__qualname__,
                    **{f.name: describe(getattr(value, f.name), depth+1) for f in fields(value)})
    if hasattr(value, 'detach'):
        return dict(tensor=value.detach().cpu().tolist(), dtype=str(value.dtype))
    if isinstance(value, slice):
        return dict(slice=[value.start, value.stop, value.step])
    if hasattr(value, 'value'):
        return dict(enum=type(value).__name__, value=describe(value.value, depth+1))
    return dict(type=type(value).__module__+'.'+type(value).__qualname__, value=str(value))


def config_manifest(cfg, env=None):
    result = dict(actor_observations=describe(cfg.observations['actor']),
                  critic_observations=describe(cfg.observations['critic']),
                  actions=describe(cfg.actions), rewards=describe(cfg.rewards),
                  terminations=describe(cfg.terminations),
                  physics_dt_s=cfg.sim.mujoco.timestep, decimation=cfg.decimation,
                  step_dt_s=cfg.sim.mujoco.timestep*cfg.decimation,
                  observation_term_order={k: list(cfg.observations[k].terms) for k in ('actor','critic')})
    if env is not None:
        a = env.action_manager.get_term('joint_pos')
        result['resolved_action'] = dict(joint_order=list(a._target_names), scale=describe(a._scale),
            offset=describe(a._offset), clip=describe(getattr(a, '_clip', None)))
        result['resolved_observation_dimensions'] = env.observation_manager.group_obs_dim
        result['BAM_delay'] = describe({k: getattr(env.scene['robot'].actuators[0]._delay_buffer, k)
            for k in ('min_lag','max_lag','update_period','per_env','hold_prob')})
    return result


def inspect_checkpoint(out, torch):
    out = Path(out)
    report = dict(path=CHECKPOINT, expected_sha256=CHECKPOINT_SHA256,
                  source_task='Stage08 E retained tray double-balance policy',
                  provenance='Fixed 056eaf2 handoff, immutable copy included',
                  runner_loaded=False, optimizer_loaded=False, saved_or_converted=False,
                  direct_resume='BLOCKED_TASK_AND_TIMING_DIFFER',
                  inherited_performance_is_new_task_performance=False,
                  future_transfer_requires='Explicit feature/normalizer/recurrent/optimizer/counter policy; no automatic warm start')
    try:
        p = Path(CHECKPOINT)
        require(p.is_file() and not p.is_symlink(), 'Pinned retained checkpoint is missing or symlinked')
        require(p.stat().st_size <= 512*1024*1024, 'Checkpoint exceeds declared 512 MiB read cap')
        report['bytes'] = p.stat().st_size
        report['observed_sha256'] = sha_file(p)
        require(report['observed_sha256'] == CHECKPOINT_SHA256, 'Pinned checkpoint bytes differ')
        checkpoint = torch.load(p, map_location='cpu', weights_only=True)
        require(isinstance(checkpoint, Mapping), 'Checkpoint is not a mapping')
        schema = {}; tensors = 0
        def walk(v, path):
            nonlocal tensors
            require(len(schema) < 20000, 'Checkpoint schema exceeds declared bound')
            if isinstance(v, torch.Tensor):
                require(v.device.type == 'cpu' and v.layout == torch.strided, 'Unsupported tensor layout')
                b = v.detach().contiguous()
                schema[path] = dict(kind='tensor', shape=list(v.shape), dtype=str(v.dtype),
                    finite=bool(torch.isfinite(v).all()),
                    sha256=hashlib.sha256(b.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
                tensors += 1
            elif isinstance(v, Mapping):
                schema[path] = dict(kind='mapping', keys=[str(k) for k in v])
                for k, x in v.items():
                    walk(x, path+'.'+str(k))
            elif isinstance(v, (tuple, list)):
                schema[path] = dict(kind=type(v).__name__, length=len(v))
                for i, x in enumerate(v):
                    walk(x, path+'.'+str(i))
            elif v is None or type(v) in (str, int, float, bool):
                schema[path] = dict(kind=type(v).__name__, value=v)
            else:
                raise RuntimeError('Unknown checkpoint field '+path+': '+type(v).__name__)
        walk(checkpoint, 'checkpoint')
        save(out/'checkpoint-schema.json', schema)
        from checkpoint_candidate import migrate_checkpoint_keys, MigrationError
        try:
            mapped, key_report = migrate_checkpoint_keys(checkpoint)
            key_report['actual_checkpoint_read'] = True
            report['key_mapping'] = key_report
        except MigrationError as exc:
            report['key_mapping'] = dict(status='BLOCKED_UNMAPPED_OR_AMBIGUOUS_KEYS', error=str(exc))
        report.update(status='PINNED_CHECKPOINT_INSPECTED_DIRECT_RESUME_REJECTED', tensors=tensors,
                      all_tensors_finite=all(r.get('finite', True) for r in schema.values()))
        require(report['all_tensors_finite'], 'Checkpoint contains nonfinite tensors')
        require(sha_file(p) == CHECKPOINT_SHA256, 'Checkpoint changed while being inspected')
    except Exception as exc:
        report.update(status='CHECKPOINT_READ_BLOCKED_REVIEW_REQUIRED', error=repr(exc))
    save(out/'checkpoint-review.json', report, durable=True)
    return report


def reuse_checkpoint_review(out):
    """Reuse the fixed v1 tensor inspection; verify file identity without torch.load."""
    root = Path(__file__).parent/'reused-v1'
    audit = json.loads((root/'audit.json').read_bytes())
    require(audit['commit']=='5f01cdacd565c0be9c93ccbf592821b1224f47b3' and
            audit['physics_attempted']==audit['physics_completed']==0, 'V1 provenance differs')
    for name in ('checkpoint-review','checkpoint-schema'):
        key = name.replace('-', '_')+'_sha256'
        require(sha_file(root/(name+'.json'))==audit[key], 'V1 checkpoint evidence differs')
    old = json.loads((root/'checkpoint-review.json').read_bytes())
    p = Path(CHECKPOINT)
    require(p.is_file() and not p.is_symlink() and p.stat().st_size==old['bytes'],
            'Retained checkpoint identity differs')
    observed = sha_file(p)
    require(observed==old['observed_sha256']==CHECKPOINT_SHA256, 'Retained checkpoint bytes differ')
    require(old['tensors']==94 and old['all_tensors_finite'] and
            old['status']=='PINNED_CHECKPOINT_INSPECTED_DIRECT_RESUME_REJECTED', 'V1 inspection not reusable')
    report = dict(status='REUSED_VERIFIED_V1_CHECKPOINT_INSPECTION',
        source_commit=audit['commit'], source_schema_sha256=audit['checkpoint_schema_sha256'],
        path=CHECKPOINT, bytes=old['bytes'], observed_sha256=observed,
        tensors=94, all_tensors_finite=True, tensor_inspection_repeated=False,
        torch_load_calls_this_batch=0, runner_loaded=False, optimizer_loaded=False,
        saved_or_converted=False, direct_resume=old['direct_resume'],
        key_mapping=old['key_mapping'], inherited_performance_is_new_task_performance=False)
    save(Path(out)/'checkpoint-review.json', report, durable=True)
    return report
