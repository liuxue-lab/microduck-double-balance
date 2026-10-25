"""Fresh diagnostic initialization/whole-run recovery support, no imports of CUDA.

Macro trials freeze Model except timestep/integrator. Whole diagnostic recovery
additionally restores the three explicit material/friction fields changed by
normal setup/BAM. Default native budgets remain zero in the inherited bridge.
"""
import numpy as np
from macro_state_bridge import MacroBudget
from sensor_macro_bridge import SensorAwareMacroStateBridge as MacroStateBridge
from native_state_bridge import model_tree,native_handles,StrictOwnerSnapshot,require,digest
from sensor_state_snapshot import DataSnapshot

class SharedMacroBudget(MacroBudget):
    delegated=frozenset(('GPU_forward','sense','Scene_write','Scene_update','Action_apply','Metrics_substep'))
    def run(self,name,fn):
        if name not in self.delegated:return super().run(name,fn)
        self.reserve(**{name:1});before={k:self.counts.get(name+'_'+k,0) for k in ('attempted','completed')}
        value=fn()
        require(all(self.counts.get(name+'_'+k,0)==v+1 for k,v in before.items()),'Delegated call not counted exactly once: '+name)
        return value

class GroundPairStateBridge(MacroStateBridge):
    def options(self,resolution=None):
        self.check_protected()
        require(resolution is None or resolution in (32,64,128),'Only ground pilot tiers32/64/128; no escalation')
        def assign():
            self.sim.wp_model.opt.timestep.assign(self.saved.dt if resolution is None else np.array([.002/resolution],np.float32))
            self.sim.wp_model.opt.integrator=self.saved.integrator if resolution is None else 1
        self.budget.run('model_option_write',assign);self.check_protected()

class CompleteCheckpoint:
    allowed_model=frozenset(('Model.dof_damping','Model.dof_frictionloss','Model.geom_solref'))
    def __init__(self,sim,scene,roots,bank,torch,wp,gate,caps):
        self.sim,self.scene,self.bank,self.torch,self.wp=sim,scene,bank,torch,wp
        self.roots=list(roots)
        slots={(id(obj),name) for _,obj,names in self.roots for name in names}
        for name in ('states','seeds','draw_counts'):
            if (id(bank),name) not in slots:self.roots.append(('private_rng',bank,(name,)))
        require(len(bank.states)==9 and set(bank.states)==set(bank.seeds)==set(bank.draw_counts),'Complete9-stream checkpoint required')
        self.gate,self.caps=gate,caps;self.poisoned=False;self.sync()
        self.data=gate.limited('Data_snapshots',caps['Data_snapshots'],lambda:DataSnapshot(sim.wp_data,wp))()
        self.owners=gate.limited('Owner_snapshots',caps['Owner_snapshots'],lambda:StrictOwnerSnapshot(self.roots,torch,wp))()
        self.arrays,self.meta=model_tree(sim.wp_model,wp)
        require(self.allowed_model<=set(self.arrays),'Explicit material/friction Model fields missing')
        self.hashes={n:digest(a.numpy()) for n,a in self.arrays.items()}
        self.raw={n:self.arrays[n].numpy().copy() for n in self.allowed_model}
        self.handles=native_handles(sim,scene,wp);self.rng=self.global_rng()
    def sync(self):self.wp.synchronize();self.torch.cuda.synchronize()
    def global_rng(self):
        return dict(cpu=digest(self.torch.get_rng_state().numpy()),cuda=digest(self.torch.cuda.get_rng_state('cuda:0').cpu().numpy()))
    def restore(self):
        require(not self.poisoned,'Failed whole checkpoint cannot be retried')
        try:
            self.sync();arrays,meta=model_tree(self.sim.wp_model,self.wp)
            require(meta==self.meta and native_handles(self.sim,self.scene,self.wp)==self.handles,'Whole checkpoint binding/options changed')
            require(self.global_rng()==self.rng,'Default RNG changed; never rewind to hide it')
            require(all(n in self.allowed_model or digest(a.numpy())==self.hashes[n] for n,a in arrays.items()),'Unapproved Model field changed')
            self.data.preflight();self.owners.preflight()
            self.gate.limited('External_Model_restore',self.caps['External_Model_restore'],self._model_restore)()
            self.gate.limited('Data_restore',self.caps['Data_restore'],self.data.restore)()
            self.gate.limited('Owner_restore',self.caps['Owner_restore'],self.owners.restore)()
            self.sync();self.data.assert_restored();self.owners.assert_restored()
            require({n:digest(a.numpy()) for n,a in arrays.items()}==self.hashes and self.global_rng()==self.rng,'Whole checkpoint bytes differ')
            return dict(Data_arrays=len(self.data.arrays),all_numeric_Data_exact=True,selected_owners_exact=True,
                numeric_Model_exact=True,default_RNG_unchanged=True,private_streams=9,opaque_handles_cloned=False)
        except Exception:self.poisoned=True;raise
    def _model_restore(self):
        for name,value in self.raw.items():self.arrays[name].assign(value)
