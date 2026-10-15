"""Pinned ManagerBasedRlEnv.step inner-loop wiring, not a complete env.step.

Nine managers and complete selected owner/control/RNG state must already exist.
No seed-only reconstruction, action processing, reset, observation-history push,
reward/termination or 20 ms environment commit is synthesized here.
Preparation occurs once, trials hold it, and Scene/metrics update once after
the public macro commit. Any error stops the adapter; no implicit retry.
"""
import json
from pathlib import Path
import numpy as np
from native_state_bridge import require
from ground_pair_driver import macro_cost

NAMES=('action','observation','event','reward','termination','metrics','command','curriculum','recorder')
def token(fn):return (getattr(fn,'__self__',None),getattr(fn,'__func__',fn))

class ManagerSubstepAdapter:
    def __init__(self,env,driver):
        self.env,self.driver=env,driver;self.phase='ready';self.calls=[]
        require(env.num_envs==1 and env.scene is driver.bridge.scene and env.sim is driver.bridge.sim,'Single-world installed environment bindings required')
        require(env.physics_dt==.002 and env.cfg.decimation==10 and env.cfg.auto_reset is False,'Pinned diagnostic cadence required')
        require(all(hasattr(env,n+'_manager') for n in NAMES),'All nine installed managers required')
        require(type(env._sim_step_counter) is int and env._sim_step_counter>=0,'Initialized Manager counter required')
        slots={(id(obj),name) for _,obj,names in driver.bridge.roots for name in names}
        require((id(env),'_sim_step_counter') in slots,'Manager counter missing from rollback roots')
        required=json.loads(Path(__file__).with_name('manager-owner-roster.json').read_text())['required']
        actual={}
        for label,obj,names in driver.bridge.roots:actual.setdefault(label,set()).update(names)
        require(all(set(names)<=actual.get(label,set()) for label,names in required.items()),'Incomplete pinned Manager/BAM/firmware owner roots')
        direct={'env':env,'RNGService':driver.bridge.bank}
        direct.update({n.title()+'Manager':getattr(env,n+'_manager') for n in ('action','observation','event','reward','termination','metrics','command','curriculum')})
        require(all(all(obj is direct[label] for lab,obj,_ in driver.bridge.roots if lab==label) for label in direct),'Manager root owner binding differs')
        self.owners={n:id(getattr(env,n+'_manager')) for n in NAMES}
        self.bindings={key:(obj,name,token(getattr(obj,name))) for key,obj,name in (
            ('Action_apply',env.action_manager,'apply_action'),('Scene_write',env.scene,'write_data_to_sim'),
            ('Scene_update',env.scene,'update'),('Metrics_substep',env.metrics_manager,'compute_substep'))}
    def preflight(self):
        require(self.phase=='ready' and self.driver.phase=='idle','Manager boundary stopped or private trial active')
        require(self.owners=={n:id(getattr(self.env,n+'_manager')) for n in NAMES},'Manager owner changed')
        require(all(token(getattr(obj,n))==t for obj,n,t in self.bindings.values()),'Manager/Scene callable changed')
        pending=self.env._manual_reset_pending.detach().cpu().numpy()
        require(pending.shape==(1,) and pending.dtype==np.bool_ and not pending.any(),'Pending manual reset; no substep before explicit recovery')
        self.driver.transaction.permit_owner_consumer()
    def invoke(self,key,**kwargs):
        obj,name,_=self.bindings[key]
        result=self.driver.bridge.budget.run(key,lambda:getattr(obj,name)(**kwargs))
        self.calls.append(key);return result
    def advance(self):
        self.preflight()
        cost=macro_cost()
        for key in ('Env_substep_counter',*self.bindings):cost[key]+=1
        self.driver.bridge.budget.reserve(**cost)
        self.phase='active';start=self.env._sim_step_counter
        try:
            self.driver.bridge.budget.run('Env_substep_counter',lambda:setattr(self.env,'_sim_step_counter',start+1))
            self.invoke('Action_apply');self.invoke('Scene_write')
            result=self.driver.advance()
            self.driver.transaction.permit_owner_consumer()
            self.invoke('Scene_update',dt=self.env.physics_dt);self.invoke('Metrics_substep')
            require(self.env._sim_step_counter==start+1,'Manager counter changed inside trials or post hooks')
            self.phase='ready';return result
        except Exception:
            # Preparation belongs to the already-issued control substep. A failure
            # after public commit cannot silently rewind a published clock.
            self.phase='stopped';raise
