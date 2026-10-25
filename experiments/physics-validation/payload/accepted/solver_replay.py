"""Host-side fixed numeric fixture lease; not a full simulator checkpoint."""
import hashlib,inspect,functools
import numpy as np
from lifecycle_support import require

ALLOWED_SOLVER_WRITES=frozenset(('Data.qacc','Data.solver_niter','Data.qfrc_constraint','Data.efc.force','Data.efc.state','Data.efc.Ma'))
FIELDS_BY_PHASE={
    '_update_constraint':('Jaref','gauss','cost','prev_cost','done'),
    '_update_gradient':('grad','grad_dot','Mgrad','h','hfactor'),
    '_update_gradient_incremental':('grad','grad_dot','Mgrad','h','hfactor','changed_efc_count'),
    '_linesearch':('alpha','Jaref','mv','jv','quad','quad_gauss','done'),
    'init_context':('Jaref','gauss','cost','prev_cost','done','grad','grad_dot','Mgrad'),
    '_solver_iteration':('Jaref','search_dot','gauss','cost','prev_cost','done','grad','grad_dot','Mgrad','search','alpha','h','hfactor','changed_efc_count'),
}
def trace_adapter(original,before,after):
    """Bind the native signature without renaming or rewriting call arguments."""
    signature=inspect.signature(original)
    require(list(signature.parameters)[:3]==['m','d','ctx'],'Unexpected native trace signature')
    @functools.wraps(original)
    def invoke(*args,**kwargs):
        bound=signature.bind(*args,**kwargs);bound.apply_defaults()
        before(bound.arguments)
        result=original(*args,**kwargs)
        after(bound.arguments)
        return result
    return invoke

