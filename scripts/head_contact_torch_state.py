"""Snapshot selected Torch/BAM state, preserving original owners on rollback.

No implicit traversal into a simulator or an arbitrary model. Callers specify
root attributes and explicitly permitted buffer classes. Data/Model device
copies are a separate responsibility. Device failures are not atomic.
"""


class TorchStateSnapshot:
    def __init__(self, torch, roots, buffer_types=(), cuda_device=None,
                 max_tensor_bytes=64*1024*1024):
        self.torch=torch;self.buffer_types=tuple(buffer_types);self.cuda_device=cuda_device
        self.nodes=[];self.memo={};self.roots=[];self.tensor_bytes=0
        self.max_tensor_bytes=max_tensor_bytes
        labels=set();slots=set()
        for label,obj,names in roots:
            if label in labels:raise ValueError('Duplicate snapshot root label')
            labels.add(label)
            for name in names:
                if (id(obj),name) in slots:raise ValueError('Duplicate snapshot root slot')
                slots.add((id(obj),name))
                self.roots.append((label+'.'+name,obj,name,self._capture(getattr(obj,name))))
        self.cpu_rng=torch.get_rng_state().clone()
        self.cuda_rng=None if cuda_device is None else torch.cuda.get_rng_state(cuda_device).clone()

    @staticmethod
    def _tensor_meta(t):
        return (tuple(t.shape),tuple(t.stride()),str(t.dtype),str(t.device),
                str(t.layout),int(t.data_ptr()),bool(t.requires_grad))

    def _capture(self,value):
        if value is None or type(value) in (int,float,bool,str):return ('scalar',value)
        if id(value) in self.memo:return ('node',self.memo[id(value)])
        index=len(self.nodes);self.memo[id(value)]=index
        node={'original':value};self.nodes.append(node)
        t=self.torch
        if isinstance(value,t.Tensor):
            if value.layout!=t.strided or not value.is_contiguous():
                raise ValueError('Only contiguous strided Torch snapshot tensors supported')
            self.tensor_bytes+=value.numel()*value.element_size()
            if self.tensor_bytes>self.max_tensor_bytes:raise ValueError('Torch snapshot capacity exceeded')
            node.update(kind='tensor',meta=self._tensor_meta(value),saved=value.detach().clone())
        elif isinstance(value,t.Generator):
            node.update(kind='generator',saved=value.get_state().clone())
        elif type(value) in (dict,list,tuple):
            if isinstance(value,dict):
                if not all(type(k) is str for k in value):raise ValueError('Only string snapshot keys supported')
                node.update(kind='dict',children={k:self._capture(v) for k,v in value.items()})
            else:node.update(kind=type(value).__name__,children=[self._capture(v) for v in value])
        elif type(value) in self.buffer_types:
            node.update(kind='buffer',children={k:self._capture(v) for k,v in vars(value).items()})
        else:raise ValueError('Unsupported selected snapshot value: '+type(value).__qualname__)
        return ('node',index)

    def _value(self,ref):
        return ref[1] if ref[0]=='scalar' else self.nodes[ref[1]]['original']

    def preflight(self):
        # Rebinding is expected for BAM torque, firmware scalars and delay lags.
        # Saved original owners must still be valid before the first copy.
        for label,obj,name,ref in self.roots:
            if not hasattr(obj,name):raise ValueError('Snapshot root attribute removed: '+label)
        for node in self.nodes:
            value=node['original']
            if node['kind']=='tensor' and self._tensor_meta(value)!=node['meta']:
                raise ValueError('Saved Torch owner storage/layout changed before rollback')
            if node['kind']=='buffer' and set(vars(value))!=set(node['children']):
                raise ValueError('Selected buffer attributes changed before rollback')

    def restore(self):
        self.preflight()
        copies=0;t=self.torch
        with t.no_grad():
            for node in self.nodes:
                value=node['original'];kind=node['kind']
                if kind=='tensor':value.copy_(node['saved']);copies+=1
                elif kind=='generator':value.set_state(node['saved'])
            for node in self.nodes:
                value=node['original'];kind=node['kind']
                if kind=='dict':
                    value.clear();value.update({k:self._value(v) for k,v in node['children'].items()})
                elif kind=='list':value[:]=[self._value(v) for v in node['children']]
                elif kind=='buffer':
                    for name,ref in node['children'].items():setattr(value,name,self._value(ref))
            for label,obj,name,ref in self.roots:setattr(obj,name,self._value(ref))
        t.set_rng_state(self.cpu_rng)
        if self.cuda_device is not None:t.cuda.set_rng_state(self.cuda_rng,self.cuda_device)
        self.check_restored_bindings()
        return copies

    def check_restored_bindings(self):
        def check(current,ref):
            expected=self._value(ref)
            if ref[0]=='node':ok=current is expected
            else:ok=type(current) is type(expected) and current==expected
            if not ok:raise RuntimeError('Snapshot owner/value binding not restored')
        for label,obj,name,ref in self.roots:check(getattr(obj,name),ref)
        for node in self.nodes:
            value=node['original'];kind=node['kind']
            if kind in ('dict','buffer'):
                current=value if kind=='dict' else vars(value)
                if set(current)!=set(node['children']):raise RuntimeError('Snapshot container keys not restored')
                for name,ref in node['children'].items():check(current[name],ref)
            elif kind in ('list','tuple'):
                if len(value)!=len(node['children']):raise RuntimeError('Snapshot sequence not restored')
                for item,ref in zip(value,node['children']):check(item,ref)

    def owner_manifest(self):
        """Original owner identities, for verifying rollback in the same process."""
        return {'root_owners':{label:id(getattr(obj,name)) for label,obj,name,ref in self.roots
                               if ref[0]=='node'},
                'node_owners':[{ 'kind':n['kind'],'identity':id(n['original']),
                                 'tensor_meta':self._tensor_meta(n['original']) if n['kind']=='tensor' else None}
                                for n in self.nodes]}
