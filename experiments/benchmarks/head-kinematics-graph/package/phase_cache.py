"""Reuse head-frame kinematics only inside one unchanged manager computation.

No cache survives an observation/metric/reward/termination call. Physics,
forward, reset and scene-write entry also invalidate it. The independent
geometric probe remains outside the metrics computation's cache scope.
"""
from functools import wraps
from deploy_contract import require


class Bindings:
    def __init__(self):self.saved=[]
    def set(self,obj,name,value):
        local=name in vars(obj)
        self.saved.append((obj,name,local,vars(obj).get(name)))
        setattr(obj,name,value)
    def close(self):
        for obj,name,local,value in reversed(self.saved):
            if local:setattr(obj,name,value)
            else:delattr(obj,name)
        self.saved.clear()


class PhaseCache:
    def __init__(self,raw,mdp,probe):
        self.raw=raw;self.depth=0;self.values={};self.bindings=Bindings()
        self.stats=dict(original_evaluations=0,cache_hits=0,scopes=0,barrier_invalidations=0)
        self.original=mdp._stage11_head_kinematics
        @wraps(self.original)
        def kinematics(env,reference_site='stage11_head_material_frame'):
            if env is not self.raw or self.depth==0:
                return self.original(env,reference_site)
            require(isinstance(reference_site,str),'Unexpected head reference argument')
            if reference_site in self.values:
                self.stats['cache_hits']+=1
                return self.values[reference_site]
            result=self.original(env,reference_site)
            self.stats['original_evaluations']+=1
            self.values[reference_site]=result
            return result
        try:
            self.bindings.set(mdp,'_stage11_head_kinematics',kinematics)
            for obj,name in ((raw.termination_manager,'compute'),(raw.reward_manager,'compute'),
                             (raw.observation_manager,'compute'),(probe,'original')):
                self.bindings.set(obj,name,self.scoped(getattr(obj,name)))
            for obj,name in ((raw.sim,'step'),(raw.sim,'forward'),(raw.sim,'reset'),
                             (raw.scene,'write_data_to_sim')):
                self.bindings.set(obj,name,self.barrier(getattr(obj,name)))
        except BaseException:
            self.close();raise
    def scoped(self,fn):
        @wraps(fn)
        def run(*args,**kwargs):
            self.values.clear();self.depth+=1;self.stats['scopes']+=1
            try:return fn(*args,**kwargs)
            finally:
                self.depth-=1;self.values.clear()
        return run
    def barrier(self,fn):
        @wraps(fn)
        def run(*args,**kwargs):
            if self.depth:
                self.stats['barrier_invalidations']+=1;self.values.clear()
                previous=self.depth;self.depth=0
                try:return fn(*args,**kwargs)
                finally:self.depth=previous;self.values.clear()
            return fn(*args,**kwargs)
        return run
    def close(self):
        self.values.clear();self.bindings.close()

