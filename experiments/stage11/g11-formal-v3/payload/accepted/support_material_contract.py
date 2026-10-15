"""Bounded support material candidates with an explicit geometry-error budget."""
from pathlib import Path
import json,hashlib
import numpy as np
from lifecycle_support import require
from convergence_contract import PAIRS

HERE=Path(__file__).resolve().parent
PROFILES=(('tau005',.005),('tau0025',.0025))
SUPPORT_GEOMS=(29,79,82)

def candidate_solref(original,tau):
    require(tau in (.02,.005,.0025), 'Unbudgeted material candidate')
    a=np.asarray(original)
    require(a.shape==(1,85,2) and a.dtype==np.float32 and np.isfinite(a).all(), 'Material field shape/type differs')
    require(all(a[0,i,0]==np.float32(.02) and a[0,i,1]==1 for i in SUPPORT_GEOMS), 'Baseline support solref differs')
    result=a.copy();result[0,list(SUPPORT_GEOMS),0]=np.float32(tau)
    return result

def verify_effective(pair,dim,solref,solimp,friction,tau):
    pair=tuple(sorted(map(int,pair)))
    if pair not in PAIRS.values():return False
    foot_floor=pair in ((0,29),(0,79))
    upper=84 in pair
    expected_ref=np.array([.003 if upper else tau,1.],dtype=np.float32)
    expected_imp=np.array([.95,.99,.001,.5,2.] if upper else [.9,.95,.001,.5,2.],dtype=np.float32)
    expected_mu=np.array([.6,.6,.005,.0001,.0001] if upper else ([1.,1.,.005,.0001,.0001] if foot_floor else [1.2,1.2,.01,.001,.001]),dtype=np.float32)
    require(int(dim)==(3 if foot_floor else 4) and np.array_equal(solref,expected_ref) and np.array_equal(solimp,expected_imp) and np.array_equal(friction,expected_mu), 'Effective contact parameters differ: '+str(pair))
    return True

