"""Bounded entry-point accounting and read-only ownership inventory.

No simulator imports. Native BVH/mesh handles are reported as opaque; this
inventory is not a complete snapshot or a rollback implementation.
"""
import dataclasses
import hashlib
import time


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


class Gate:
    def __init__(self, counts, persist, deadline, sync):
        self.counts, self.persist, self.deadline, self.sync = counts, persist, deadline, sync
        self.originals = []

    def patch(self, obj, name, fn):
        own = hasattr(obj, '__dict__') and name in vars(obj)
        self.originals.append((obj, name, getattr(obj, name), own))
        setattr(obj, name, fn)

    def limited(self, label, limit, native, check=None):
        def call(*args, **kwargs):
            require(time.monotonic() < self.deadline, 'Wall budget exhausted')
            if check is not None:
                check(args, kwargs)
            attempted, completed = label + '_attempted', label + '_completed'
            require(self.counts.get(attempted, 0) < limit, label + ' budget exhausted')
            self.counts[attempted] = self.counts.get(attempted, 0) + 1
            self.persist()
            value = native(*args, **kwargs)
            self.sync()
            self.counts[completed] = self.counts.get(completed, 0) + 1
            self.persist()
            return value
        return call

    def blocked(self, label):
        def call(*args, **kwargs):
            self.counts['blocked_' + label] = self.counts.get('blocked_' + label, 0) + 1
            self.persist()
            raise RuntimeError('Prohibited outside bounded diagnostic entry points: ' + label)
        return call

    def close(self):
        for obj, name, original, own in reversed(self.originals):
            if own:
                setattr(obj, name, original)
            else:
                delattr(obj, name)
        self.originals.clear()


def inventory(roots, torch, wp, np, *, max_nodes=5000, max_bytes=64*1024*1024):
    """Record explicit roots only, preserving aliases and labelling opaque leaves.

    Configuration and simulator bridge owners remain references. Native handles
    are never traversed or treated as cloned. No property getters are evaluated.
    """
    records, seen = {}, {}
    read_bytes = 0
    allowed = ('mjlab.sensor.',)

    def walk(value, path):
        nonlocal read_bytes
        require(len(records) < max_nodes, 'Inventory node budget exceeded')
        if value is None or type(value) in (str, int, float, bool):
            records[path] = {'kind': 'scalar', 'value': value}
            return
        if id(value) in seen:
            records[path] = {'kind': 'alias', 'target': seen[id(value)], 'owner': id(value)}
            return
        seen[id(value)] = path
        typ = type(value).__module__ + '.' + type(value).__qualname__
        row = {'type': typ, 'owner': id(value)}
        records[path] = row
        if isinstance(value, torch.Tensor):
            row.update(kind='torch', shape=list(value.shape), strides=list(value.stride()),
                       dtype=str(value.dtype), device=str(value.device), pointer=value.data_ptr(),
                       storage_pointer=value.untyped_storage().data_ptr(), offset=value.storage_offset())
            size = value.numel()*value.element_size()
            require(read_bytes + size <= max_bytes, 'Inventory byte budget exceeded')
            raw = value.detach().cpu().numpy()
        elif isinstance(value, wp.array):
            row.update(kind='warp', shape=list(value.shape), strides=list(value.strides),
                       dtype=str(value.dtype), device=str(value.device), pointer=value.ptr,
                       capacity=int(value.capacity))
            require(read_bytes + int(value.capacity) <= max_bytes, 'Inventory byte budget exceeded')
            if 'Texture' in str(value.dtype):
                row['contents'] = 'opaque resource handles; not copied'
                return
            raw = value.numpy()
        elif isinstance(value, np.ndarray):
            row.update(kind='numpy', shape=list(value.shape), dtype=str(value.dtype))
            require(read_bytes + value.nbytes <= max_bytes, 'Inventory byte budget exceeded')
            raw = value
        else:
            if isinstance(value, dict):
                row['kind'] = 'dict'
                for key, item in value.items():
                    walk(item, path + '[' + repr(key) + ']')
            elif isinstance(value, (list, tuple)):
                row['kind'] = type(value).__name__
                for i, item in enumerate(value):
                    walk(item, path + '[' + str(i) + ']')
            elif dataclasses.is_dataclass(value) or typ.startswith(allowed):
                row['kind'] = 'object'
                # cfg is immutable for this probe; bridges/native model and
                # device objects are references, never recursively expanded.
                for name, item in vars(value).items():
                    if name in ('cfg', '_mj_model', '_data', '_model', '_wp_device', 'wp_device'):
                        records[path + '.' + name] = {'kind': 'reference', 'owner': id(item),
                                                     'type': type(item).__module__ + '.' + type(item).__qualname__}
                    else:
                        walk(item, path + '.' + name)
            else:
                row['kind'] = 'opaque'
                row['scope'] = 'Owner only; native mutable contents not validated'
            return
        raw = np.asarray(raw)
        read_bytes += int(raw.nbytes)
        row.update(bytes=int(raw.nbytes), sha256=hashlib.sha256(raw.tobytes()).hexdigest())
        if raw.dtype.kind in 'biuf':
            row['finite'] = bool(np.isfinite(raw).all())
            if raw.size <= 32:
                row['values'] = raw.tolist()

    for name, value in roots.items():
        walk(value, name)
    return {'records': records, 'host_read_bytes': read_bytes,
            'is_complete_snapshot': False,
            'opaque_paths': [p for p, v in records.items() if v['kind'] == 'opaque']}


def delta(before, after):
    a, b = before['records'], after['records']
    common = sorted(set(a) & set(b))
    return {
        'added': sorted(set(b)-set(a)), 'removed': sorted(set(a)-set(b)),
        'owner_changed': [p for p in common if a[p].get('owner') != b[p].get('owner')],
        'content_changed': [p for p in common if a[p].get('sha256') != b[p].get('sha256')],
        'value_changed': [p for p in common if a[p].get('value') != b[p].get('value')],
    }
