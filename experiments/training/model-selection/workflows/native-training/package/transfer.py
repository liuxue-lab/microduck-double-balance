"""Explicit old-tray -> material-head adaptation; no simulator dependency."""
from copy import deepcopy
from types import MethodType
from common import require

def migrate_stage08(source, torch):
    migrated=deepcopy(source)
    actor,critic=migrated['actor_state_dict'],migrated['critic_state_dict']
    if 'std' in actor:actor['distribution.std_param']=actor.pop('std')
    if 'log_std' in actor:actor['distribution.log_std_param']=actor.pop('log_std')
    require(tuple(actor['rnn.rnn.weight_ih_l0'].shape)==(1024,61) and tuple(critic['mlp.0.weight'].shape)==(512,85),'Stage08 transfer dimensions differ')
    # Retain all other columns, recurrent dynamics, output layers and action std.
    # These columns now represent a different material frame / rate convention.
    actor['rnn.rnn.weight_ih_l0'][:,55:61]=0.
    critic['mlp.0.weight'][:,76:85]=0.
    for sd,start in ((actor,55),(critic,76)):
        for key,value in (('_mean',0.),('_var',1.),('_std',1.)):
            sd['obs_normalizer.'+key][...,start:]=value
        sd['obs_normalizer.count'].fill_(10000)
        require(all(bool(torch.isfinite(v).all()) for v in sd.values()),'Nonfinite source network')
    return migrated

def configure_normalizer_updates(normalizer,start,torch):
    # The original normalizer has a single count. Give only changed features
    # the new prior; keep old proprioception/lower-ball moments exactly intact.
    def update(self,x):
        if not self.training:return
        with torch.no_grad():
            self.count+=x.shape[0]
            rate=x.shape[0]/self.count
            y=x[...,start:];mean=y.mean(dim=0,keepdim=True);var=y.var(dim=0,unbiased=False,keepdim=True)
            old=self._mean[...,start:];delta=mean-old
            old.add_(rate*delta)
            self._var[...,start:].add_(rate*(var-self._var[...,start:]+delta*(mean-old)))
            self._std[...,start:]=torch.sqrt(self._var[...,start:])
    normalizer.update=MethodType(update,normalizer)

def equal_state(actual,expected,torch,top_level=False):
    if isinstance(actual,dict):
        if not top_level and set(actual)!=set(expected):return False
        return all(k in expected and equal_state(v,expected[k],torch) for k,v in actual.items())
    if isinstance(actual,(tuple,list)):
        return len(actual)==len(expected) and all(equal_state(a,b,torch) for a,b in zip(actual,expected))
    if torch.is_tensor(actual):return torch.equal(actual.detach().cpu(),expected.detach().cpu())
    return actual==expected
