"""Reuse pure derived entity properties only until the next native data publish.

Return independent clones, preserving the original properties' copy semantics.
The original property computes each new value; no arithmetic is replaced. State
publication (including forward/reset) always changes DataView.revision. No cache
is shared across entity instances, native publications, or evaluation sessions.
"""
from phase_cache import Bindings
from sync_adapters import verify_sources, exact
from deploy_contract import require

FIELDS = ('root_link_pose_w', 'root_link_vel_w', 'site_pose_w', 'site_vel_w',
          'projected_gravity_b')


class PublishedProperties(Bindings):
    def __init__(self, raw, torch):
        super().__init__()
        modules,self.sources=verify_sources(('mjlab.entity.data',))
        cls=modules['mjlab.entity.data'].EntityData
        self.data=raw.sim.data
        require(type(self.data).__name__=='DataView' and hasattr(self.data,'revision'),
                'Published properties require native data publications')
        self.owners={id(ent.data):ent.data for ent in raw.scene.entities.values()}
        self.cache={};self.hits=0;self.computations=0;self.checked=0
        self.check_counts={}
        try:
            for name in FIELDS:
                original=getattr(cls,name)
                require(isinstance(original,property),'Expected pure entity property: '+name)
                def get(owner,name=name,original=original):
                    if id(owner) not in self.owners:return original.fget(owner)
                    key=(id(owner),name)
                    entry=self.cache.get(key)
                    if entry is None or entry[0]!=self.data.revision:
                        value=original.fget(owner)
                        entry=(self.data.revision,value)
                        self.cache[key]=entry;self.computations+=1
                    else:
                        self.hits+=1
                        count=self.check_counts.get(key,0)
                        if count<4:
                            exact(torch,entry[1],original.fget(owner),'published '+name)
                            self.check_counts[key]=count+1;self.checked+=1
                    return entry[1].clone()
                self.set(cls,name,property(get,doc=original.__doc__))
        except BaseException:self.close();raise

    def close(self):
        super().close();self.cache.clear()

    def stats(self):
        return dict(original_property_computations=self.computations,cache_hits=self.hits,
                    same_publication_comparisons=self.checked,sources=self.sources,
                    fields=list(FIELDS),arithmetic_unchanged=True,returned_values_are_copies=True,
                    invalidation='every native data pull, including step, forward and reset',
                    closed=not self.saved)
