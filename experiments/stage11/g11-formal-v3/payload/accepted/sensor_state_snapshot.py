"""Selected sensor owners/aliases plus full Warp Data array restore.

Native contexts/meshes/BVH/textures are references, never cloned. A separate
real sense call must refit the native BVH and verify recovered ray outputs.
No simulator imports; APIs are supplied by the bounded worker.
"""
import hashlib

from head_contact_data_commit_r3 import describe
from lifecycle_support import require


def digest(raw):
    return hashlib.sha256(raw.tobytes()).hexdigest()


def tensor_meta(value):
    return (tuple(value.shape), tuple(value.stride()), str(value.dtype),
            str(value.device), str(value.layout), value.data_ptr(),
            value.untyped_storage().data_ptr(), value.storage_offset(),
            bool(value.requires_grad))


def warp_meta(value):
    return (tuple(value.shape), tuple(value.strides), str(value.dtype),
            str(value.device), value.ptr, int(value.capacity))


def disjoint_tensor(value):
    """Reject internally overlapping/expanded layouts before copying back."""
    if value.numel() == 0:
        return
    axes = sorted((stride, size) for size, stride in zip(value.shape, value.stride()) if size > 1)
    extent = 1
    for stride, size in axes:
        require(stride >= extent, 'Internally overlapping Torch view is outside this diagnostic')
        extent += (size - 1) * stride


class DataSnapshot:
    def __init__(self, data, wp, max_bytes=64*1024*1024):
        self.data, self.wp = data, wp
        self.desc, self.arrays, self.owners, _ = describe(data, wp.array)
        self.bytes = sum(int(a.capacity) for a in self.arrays.values())
        require(self.bytes <= max_bytes, 'Full Data snapshot capacity exceeded')
        self.saved = {n: wp.clone(a) for n, a in self.arrays.items()}
        wp.synchronize()
        self.hashes = self.fingerprint()
        require(all(digest(self.saved[n].numpy()) == h for n, h in self.hashes.items()),
                'Data clone differs from source')

    def preflight(self):
        desc, _, owners, _ = describe(self.data, self.wp.array)
        require(desc == self.desc and owners == self.owners, 'Data topology/layout/owner changed')

    def fingerprint(self):
        return {n: digest(a.numpy()) for n, a in self.arrays.items()}

    def restore(self):
        self.preflight()
        for n, a in self.arrays.items():
            if a.size:
                self.wp.copy(a, self.saved[n])
        self.wp.synchronize()
        self.assert_restored()

    def assert_restored(self):
        self.preflight()
        require(self.fingerprint() == self.hashes, 'Full Data bytes not restored')


class OwnerSnapshot:
    """Capture explicit root attributes and original views, including rebinding.

Different non-overlapping views of shared storage are allowed. Every view is
saved at the same synchronized point; all original view bindings are retained.
This helper does not change tensor shape/storage metadata and never traverses
an arbitrary simulator/model or opaque native handle.
"""
    REFERENCES = frozenset(('cfg', '_mj_model', '_data', '_model', '_ctx',
                            '_wp_device', 'wp_device', '_geomgroup'))

    def __init__(self, roots, torch, wp, max_bytes=64*1024*1024):
        self.torch, self.wp, self.limit = torch, wp, max_bytes
        self.nodes, self.memo, self.roots = [], {}, []
        self.bytes = 0
        slots = set()
        for label, owner, names in roots:
            for name in names:
                require((id(owner), name) not in slots, 'Duplicate root attribute')
                slots.add((id(owner), name))
                self.roots.append((label+'.'+name, owner, name, self.capture(getattr(owner, name), name)))
        wp.synchronize()

    def capture(self, value, field=''):
        if field in self.REFERENCES:
            return ('reference', value)
        if value is None or type(value) in (str, int, float, bool):
            return ('scalar', value)
        if id(value) in self.memo:
            return ('node', self.memo[id(value)])
        require(len(self.nodes) < 5000, 'Owner graph node cap exceeded')
        index = len(self.nodes)
        self.memo[id(value)] = index
        node = {'original': value}
        self.nodes.append(node)
        if isinstance(value, self.torch.Tensor):
            require(value.layout == self.torch.strided and not value.requires_grad,
                    'Only non-autograd strided Torch state is in scope')
            disjoint_tensor(value)
            self.bytes += value.numel()*value.element_size()
            require(self.bytes <= self.limit, 'Owner snapshot byte cap exceeded')
            node.update(kind='torch', meta=tensor_meta(value), saved=value.detach().clone(),
                        hash=digest(value.detach().cpu().numpy()))
        elif isinstance(value, self.wp.array):
            require(value.is_contiguous and 'Texture' not in str(value.dtype),
                    'Only numeric contiguous sensor Warp arrays are in scope')
            self.bytes += int(value.capacity)
            require(self.bytes <= self.limit, 'Owner snapshot byte cap exceeded')
            node.update(kind='warp', meta=warp_meta(value), saved=self.wp.clone(value), hash=digest(value.numpy()))
        elif isinstance(value, dict):
            require(all(type(k) in (str, int) for k in value), 'Unexpected state dictionary key')
            node.update(kind='dict', children={k: self.capture(v) for k, v in value.items()})
        elif isinstance(value, (list, tuple)):
            node.update(kind=type(value).__name__, children=[self.capture(v) for v in value])
        elif type(value).__module__.startswith(('mjlab.sensor.', 'mjlab.utils.buffers.')):
            node.update(kind='object', children={k: self.capture(v, k) for k, v in vars(value).items()})
        else:
            raise RuntimeError('Unsupported selected sensor state: '+type(value).__module__+'.'+type(value).__qualname__)
        return ('node', index)

    def value(self, ref):
        return self.nodes[ref[1]]['original'] if ref[0] == 'node' else ref[1]

    def preflight(self):
        for label, owner, name, _ in self.roots:
            require(hasattr(owner, name), 'Root attribute removed: '+label)
        for node in self.nodes:
            value, kind = node['original'], node['kind']
            if kind == 'torch':
                require(tensor_meta(value) == node['meta'], 'Saved Torch view layout/storage changed')
            elif kind == 'warp':
                require(warp_meta(value) == node['meta'], 'Saved Warp owner layout/storage changed')
            elif kind == 'object':
                require(set(vars(value)) == set(node['children']), 'Selected object schema changed')

    def restore(self):
        self.preflight()  # Validate every saved owner before the first copy.
        with self.torch.no_grad():
            for node in self.nodes:
                value, kind = node['original'], node['kind']
                if kind == 'torch':
                    value.copy_(node['saved'])
                elif kind == 'warp' and value.size:
                    self.wp.copy(value, node['saved'])
            self.wp.synchronize()
            for node in self.nodes:
                value, kind = node['original'], node['kind']
                if kind == 'dict':
                    value.clear()
                    value.update({k: self.value(v) for k, v in node['children'].items()})
                elif kind == 'list':
                    value[:] = [self.value(v) for v in node['children']]
                elif kind == 'object':
                    for name, ref in node['children'].items():
                        setattr(value, name, self.value(ref))
            for _, owner, name, ref in self.roots:
                setattr(owner, name, self.value(ref))
        self.assert_restored()

    def assert_restored(self, mutable_data_views=frozenset()):
        # Only the trial caller supplies IDs proven to be fully backed by its
        # numeric Data snapshot. Layouts and all bindings remain checked.
        self.preflight()
        def binding(actual, ref):
            expected = self.value(ref)
            if ref[0] == 'scalar':
                require(type(actual) is type(expected) and actual == expected, 'Scalar state differs')
            else:
                require(actual is expected, 'Original owner binding not restored')
        for _, owner, name, ref in self.roots:
            binding(getattr(owner, name), ref)
        for node in self.nodes:
            value, kind = node['original'], node['kind']
            if kind == 'torch':
                require(id(value) in mutable_data_views or digest(value.detach().cpu().numpy()) == node['hash'], 'Torch view content differs')
            elif kind == 'warp':
                require(id(value) in mutable_data_views or digest(value.numpy()) == node['hash'], 'Warp sensor content differs')
            elif kind in ('dict', 'object'):
                actual = value if kind == 'dict' else vars(value)
                require(set(actual) == set(node['children']), 'Container fields differ')
                for key, ref in node['children'].items():
                    binding(actual[key], ref)
            elif kind in ('list', 'tuple'):
                require(len(value) == len(node['children']), 'Sequence length differs')
                for actual, ref in zip(value, node['children']):
                    binding(actual, ref)

    def manifest(self):
        return {'bytes': self.bytes, 'nodes': len(self.nodes),
                'roots': {label: id(getattr(owner, name)) for label, owner, name, ref in self.roots
                          if ref[0] != 'scalar'},
                'arrays': [{'kind': n['kind'], 'owner': id(n['original']), 'metadata': n['meta'], 'sha256': n['hash']}
                           for n in self.nodes if n['kind'] in ('torch', 'warp')],
                'native_context_contents_cloned': False}


def sense_and_invalidate(sim, sensors):
    """Refresh once, then invalidate caches without another timing/history update."""
    sim.sense()
    for sensor in sensors.values():
        sensor._invalidate_cache()
