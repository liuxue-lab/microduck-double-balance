"""Same installed substep wiring with the exact fixed128 diagnostic reservation."""
from continuation_driver import FixedContinuation,PrivateFreshness,continuation_cost
from ground_manager_adapter import ManagerSubstepAdapter
from lifecycle_support import require

class RecoveryFreshness(PrivateFreshness):
    def permit_owner_consumer(self):
        require(self.phase in ('idle','finished'),'Manager consumer inside private fixed trial')

class RecoveryContinuation(FixedContinuation):
    def __init__(self,*a,**k):
        super().__init__(*a,**k);self.transaction=RecoveryFreshness()

class RecoveryAdapter(ManagerSubstepAdapter):
    def advance(self):
        self.preflight();cost=continuation_cost()
        for key in ('Env_substep_counter',*self.bindings):cost[key]+=1
        self.driver.bridge.budget.reserve(**cost)
        self.phase='active';start=self.env._sim_step_counter
        try:
            self.driver.bridge.budget.run('Env_substep_counter',lambda:setattr(self.env,'_sim_step_counter',start+1))
            self.invoke('Action_apply');self.invoke('Scene_write')
            result=self.driver.advance()
            self.driver.transaction.permit_owner_consumer()
            self.invoke('Scene_update',dt=self.env.physics_dt);self.invoke('Metrics_substep')
            require(self.env._sim_step_counter==start+1,'Manager substep counter changed unexpectedly')
            self.phase='ready';return result
        except Exception:self.phase='stopped';raise
