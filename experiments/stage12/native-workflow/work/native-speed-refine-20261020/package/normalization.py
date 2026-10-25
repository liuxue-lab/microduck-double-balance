from types import MethodType

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
