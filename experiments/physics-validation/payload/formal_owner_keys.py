"""Typed immutable dictionary keys, preserving original key and owner bindings.

The accepted helpers remain byte-identical. This private worker extension adds
bounded int/string tuple keys and a lossless evidence path encoding. It never
excludes a cache or stringifies live keys. No GPU operation occurs at import.
"""
import json
import math
from native_state_bridge import StrictOwnerSnapshot as AcceptedStrictOwnerSnapshot
from sensor_state_snapshot import OwnerSnapshot, tensor_meta, warp_meta
from runtime_accounting import require


def key_record(key, depth=0):
    require(depth<=4, 'Owner dictionary key nesting exceeds scope')
    if type(key) is str:
        require(len(key)<=1024, 'Owner dictionary string key exceeds scope')
        return ['str',key]
    if type(key) is int:
        return ['int',str(key)]
    if type(key) is tuple:
        require(len(key)<=16, 'Owner dictionary tuple key exceeds scope')
        return ['tuple',[key_record(k,depth+1) for k in key]]
    raise RuntimeError('Unsupported owner dictionary key type: '+type(key).__name__)


def supported_key(key):
    try:
        key_record(key)
        return True
    except RuntimeError:
        return False


def key_segment(key):
    # Hex cannot contain separators; type tags distinguish 1, "1", (1,), etc.
    return '/d'+json.dumps(key_record(key),ensure_ascii=True,separators=(',',':')).encode().hex()


class FormalOwnerSnapshot(AcceptedStrictOwnerSnapshot):
    def capture(self, value, field=''):
        if field in self.REFERENCES or not isinstance(value,dict):
            return super().capture(value,field)
        if id(value) in self.memo:
            return ('node',self.memo[id(value)])
        require(len(self.nodes)<5000, 'Owner graph node cap exceeded')
        for key in value: key_record(key)
        index=len(self.nodes); self.memo[id(value)]=index
        node=dict(original=value,kind='dict'); self.nodes.append(node)
        node['children']={k:self.capture(v) for k,v in value.items()}
        if any(type(k) is tuple for k in value):
            node['key_records']=[key_record(k) for k in value]
        return ('node',index)

    def assert_restored(self, mutable_data_views=frozenset()):
        super().assert_restored(mutable_data_views)
        for node in self.nodes:
            if node['kind']=='dict' and 'key_records' in node:
                actual=list(node['original']); expected=list(node['children'])
                require([key_record(k) for k in actual]==node['key_records'],
                        'Dictionary key types/order changed')
                require(all(a is b for a,b in zip(actual,expected) if type(b) is tuple),
                        'Original tuple key identity not restored')


def owner_state(roots, torch, wp):
    raw,scalars,topology={},{},{}; used=set()
    def walk(value,path,field='',ancestors=()):
        require(path not in used, 'Owner evidence path collision: '+path)
        used.add(path);require(len(used)<12000,'Selected owner record cap')
        if field in OwnerSnapshot.REFERENCES:
            topology[path]=dict(kind='reference',owner=id(value),type=str(type(value)));return
        if value is None or type(value) in (str,int,float,bool):
            scalars[path]={'nonfinite_float':repr(value)} if type(value) is float and not math.isfinite(value) else value
            return
        if isinstance(value,torch.Tensor):
            raw[path]=value.detach().cpu().numpy().copy()
            topology[path]=dict(kind='torch',owner=id(value),metadata=tensor_meta(value));return
        if isinstance(value,wp.array):
            require('Texture' not in str(value.dtype),'Opaque owner outside selected state')
            raw[path]=value.numpy().copy();topology[path]=dict(kind='warp',owner=id(value),metadata=warp_meta(value));return
        require(id(value) not in ancestors,'Cycle in selected owners')
        ancestors=ancestors+(id(value),)
        topology[path]=dict(kind=type(value).__name__,owner=id(value))
        if isinstance(value,dict):
            topology[path]['typed_keys']=[key_record(k) for k in value]
            for k,v in value.items(): walk(v,path+key_segment(k),'',ancestors)
        elif isinstance(value,(tuple,list)):
            for i,v in enumerate(value): walk(v,path+'/s'+str(i),'',ancestors)
        else:
            require(type(value).__module__.startswith(('mjlab.sensor.','mjlab.utils.buffers.')),'Unknown selected owner '+path)
            for k,v in vars(value).items(): walk(v,path+'.'+k,k,ancestors)
    for label,obj,names in roots:
        for name in names: walk(getattr(obj,name),label+'.'+name,name)
    require(sum(a.nbytes for a in raw.values())<=64*1024*1024,'Owner raw byte cap')
    return raw,scalars,topology


def install(gate):
    # These are all the accepted import bindings used by checkpointing,
    # endpoint hashing and EvidenceJournal._snapshot. Restore them on close.
    import importlib
    for name in ('native_state_bridge','macro_state_bridge','sensor_macro_bridge',
                 'pilot_state','portable_state','scene_evidence','evidence_journal',
                 'sensor_lifecycle_contract'):
        module=importlib.import_module(name)
        if hasattr(module,'StrictOwnerSnapshot'):
            gate.patch(module,'StrictOwnerSnapshot',FormalOwnerSnapshot)
        if hasattr(module,'owner_state'):
            gate.patch(module,'owner_state',owner_state)


def preflight_cache(env,torch,wp,mdp):
    """Warm only the native site lookup; mutate/restore only that numeric cache.

    No reward/observation/event compute, RNG draw or native integration.
    Whole environment Data/owners and counters are checked by the caller.
    """
    name='stage11_head_material_frame'
    robot=env.scene['robot']; ids,_=robot.find_sites(name)
    require(len(ids)==1,'Head site lookup changed')
    site=mdp._double_balance_tray_site_id(env,tray_site_name=name)
    cache=env._double_balance_tray_site_ids
    key=(id(robot),name)
    require(type(site) is int and cache=={key:site} and site==ids[0], 'Native tuple cache schema differs')
    original_key=next(iter(cache))
    snap=FormalOwnerSnapshot([('env',env,('_double_balance_tray_site_ids',))],torch,wp)
    snap.assert_restored()
    cache[original_key]=-1
    cache[(id(robot),'g11-preflight-diagnostic')]=site
    env._double_balance_tray_site_ids={key:-2}
    snap.restore();snap.assert_restored()
    require(env._double_balance_tray_site_ids is cache and next(iter(cache)) is original_key,
            'Site cache original container/key binding lost')
    require(mdp._double_balance_tray_site_id(env,tray_site_name=name)==site, 'Restored lookup differs')
    raw,scalars,topology=owner_state([('env',env,('_double_balance_tray_site_ids',))],torch,wp)
    require(not raw and list(scalars.values())==[site], 'Typed cache export differs')
    return dict(status='PASS',site_id=site,key=key_record(original_key),
        original_dict_and_key_restored=True,changed_values_and_rebinding_tested=True,
        raw_arrays=0,scalars=scalars,topology=topology,
        reward_computes=0,observation_computes=0,native_integrations=0)
