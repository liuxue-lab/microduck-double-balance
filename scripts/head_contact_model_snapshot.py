"""Snapshot two single-world BAM Model fields without changing live layouts.

MuJoCo Warp retains a zero leading stride on singleton Model arrays. wp.clone
canonicalizes this unused stride. Preserve it only on the new owned snapshot,
after validating the exact supported layout; the strict Data copier is unchanged.
"""
from head_contact_data_commit_r3 import describe


def snapshot_model_fields(fields, api):
    if type(fields) is not dict or tuple(fields)!=('dof_frictionloss','dof_damping'):
        raise ValueError('Expected exactly the two ordered BAM Model fields')
    shape=None
    for name,a in fields.items():
        if not isinstance(a,api.array) or a.dtype!=api.float32:
            raise ValueError('Expected float32 Warp Model array: '+name)
        if len(a.shape)!=2 or a.shape[0]!=1 or a.shape[1]<1:
            raise ValueError('Only single-world nonempty Model rows supported: '+name)
        if shape is not None and tuple(a.shape)!=shape:
            raise ValueError('BAM Model row shapes differ')
        shape=tuple(a.shape)
        if tuple(a.strides) not in ((0,4),(4*a.shape[1],4)) or not a.is_contiguous:
            raise ValueError('Unsupported Model stride layout: '+name)
        if a.ptr is None or int(a.ptr)<=0 or int(a.capacity)<4*a.shape[1]:
            raise ValueError('Invalid Model storage: '+name)
    if len({id(a) for a in fields.values()})!=2:
        raise ValueError('Distinct BAM Model fields must not alias')
    original,_,owners,regions=describe(fields,api.array)
    saved={}
    for name,a in fields.items():
        clone=api.clone(a)
        if clone.ptr==a.ptr:raise ValueError('Model snapshot shares live storage')
        # The first index can only be zero (shape[0] == 1), so this metadata
        # assignment does not change addressed elements. Never modify live a.
        clone.strides=tuple(a.strides)
        saved[name]=clone
    api.synchronize()
    cloned,_,_,saved_regions=describe(saved,api.array)
    if cloned!=original or describe(fields,api.array)[2]!=owners:
        raise ValueError('Model snapshot descriptor/owner mismatch')
    if any(d==sd and not (hi<=slo or lo>=shi)
           for d,lo,hi in regions for sd,slo,shi in saved_regions):
        raise ValueError('Model snapshot overlaps live storage')
    for name,a in fields.items():
        if a.numpy().tobytes()!=saved[name].numpy().tobytes():
            raise ValueError('Model snapshot content mismatch: '+name)
    return saved


def best_effort_restore(actions):
    """Attempt independent cleanup components even when one fails; never resume."""
    records={}
    for name,action in actions:
        if name in records:raise ValueError('Duplicate cleanup component')
        try:records[name]={'status':'RESTORED','result':action()}
        except Exception as exc:records[name]={'status':'FAILED','error':repr(exc)}
    return records
