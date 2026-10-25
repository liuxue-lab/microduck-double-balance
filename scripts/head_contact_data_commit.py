"""Copy accepted scratch Data into existing owners, with preflight validation.

No simulator import. The caller supplies the array class and copy/sync methods.
This is a memory transaction helper, not an integrator or full macro rollback.
"""
from __future__ import annotations
import dataclasses


def describe(root, array_type):
    arrays, descriptors, owners, seen = {}, {}, {}, {}

    def walk(value, path):
        mutable = isinstance(value, array_type) or dataclasses.is_dataclass(value) or isinstance(value, (list, dict, tuple))
        if mutable:
            owners[path] = id(value)
            if id(value) in seen:
                descriptors[path] = ('alias', seen[id(value)])
                return
            seen[id(value)] = path
        if isinstance(value, array_type):
            if value.size and not value.is_contiguous:
                raise ValueError('Noncontiguous Data array: '+path)
            descriptors[path] = ('array', str(value.dtype), tuple(value.shape), tuple(value.strides), str(value.device))
            arrays[path] = value
            owners[path] = (id(value), int(value.ptr), int(value.capacity))
        elif dataclasses.is_dataclass(value):
            fields = [f.name for f in dataclasses.fields(value)]
            if set(vars(value)) != set(fields):
                raise ValueError('Dynamic Data attributes: '+path)
            descriptors[path] = ('dataclass', type(value).__module__, type(value).__qualname__, tuple(fields))
            for name in fields:
                walk(getattr(value, name), path+'.'+name)
        elif isinstance(value, (list, tuple)):
            descriptors[path] = (type(value).__name__, len(value))
            for i, item in enumerate(value):
                walk(item, path+'['+str(i)+']')
        elif isinstance(value, dict):
            if not all(isinstance(k, str) for k in value):
                raise ValueError('Only string Data dictionary keys supported')
            descriptors[path] = ('dict', tuple(value))
            for key, item in value.items():
                walk(item, path+'['+repr(key)+']')
        elif value is None or type(value) in (int, float, bool, str):
            descriptors[path] = ('scalar', type(value).__name__, value)
        else:
            raise ValueError('Unsupported Data value at '+path+': '+type(value).__name__)

    walk(root, 'Data')
    regions = []
    for path, value in arrays.items():
        if value.size:
            region = (str(value.device), int(value.ptr), int(value.ptr)+int(value.capacity))
            if any(region[0] == d and not (region[2] <= lo or region[1] >= hi) for d, lo, hi in regions):
                raise ValueError('Distinct overlapping Data views: '+path)
            regions.append(region)
    return descriptors, arrays, owners, regions


def commit_data(destination, source, api):
    """Validate every field before copying; keep all destination owner identities.

    It is not atomic against device faults: callers stop on a copy/sync failure.
    Scalar/topology changes, overlap, and stale/different layouts fail pre-copy.
    """
    dst_desc, dst, owners, dst_regions = describe(destination, api.array)
    src_desc, src, _, src_regions = describe(source, api.array)
    if dst_desc != src_desc:
        raise ValueError('Data topology/alias/scalar/layout mismatch before commit')
    if any(d == sd and not (hi <= slo or lo >= shi)
           for d, lo, hi in dst_regions for sd, slo, shi in src_regions):
        raise ValueError('Source and destination share mutable array storage')
    copied = 0
    for path in dst:
        if dst[path].size:
            api.copy(dst[path], src[path])
            copied += 1
    api.synchronize()
    after_desc, _, after_owners, _ = describe(destination, api.array)
    if after_desc != dst_desc or after_owners != owners:
        raise RuntimeError('Destination owner/array identity changed during commit')
    return copied
