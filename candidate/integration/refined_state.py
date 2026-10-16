"""Only the explicit trial-resolution roster changes; native owner guards stay."""
import numpy as np
from pilot_state import GroundPairStateBridge
from lifecycle_support import require
from integration.refinement_contract import TIERS


class RefinedStateBridge(GroundPairStateBridge):
    def options(self, resolution=None):
        self.check_protected()
        require(resolution is None or (type(resolution) is int and resolution in TIERS),
                'Outside the finite v7 tier roster; never add a tier after failure')
        def assign():
            self.sim.wp_model.opt.timestep.assign(self.saved.dt if resolution is None
                else np.array([.002/resolution], np.float32))
            self.sim.wp_model.opt.integrator = self.saved.integrator if resolution is None else 1
        self.budget.run('model_option_write', assign)
        self.check_protected()
