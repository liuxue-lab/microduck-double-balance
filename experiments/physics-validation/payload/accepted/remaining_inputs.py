"""Exact input invariants across remaining branches in one corrected live owner."""
from pathlib import Path
import numpy as np
from lifecycle_support import require
from report_io import save
from scene_evidence import save_arrays,array_meta
from startup_mapping import fixture,PRIMARY,PARAMETERS
from owner_schedule import CASES

class InputInvariant:
    def __init__(self,world,plan,out):
        require(world==plan['target_logical_world']==1,'Only corrected world1 remaining regression')
        self.world,self.plan,self.out=world,plan,Path(out);self.initial=None;self.labels=[]
        self.expected_labels=['continuation-'+case+'-'+mode for case,mask,mode in CASES]
        self.parameters=fixture(plan)['parameters']
    def check(self,data,act,label):
        require(len(self.labels)<7 and label==self.expected_labels[len(self.labels)],'Foreign or repeated remaining input')
        raw={n:getattr(data,n.removeprefix('Data.')).numpy().copy() for n in PRIMARY}
        for n,a in raw.items():require(a.dtype==np.float32 and np.isfinite(a).all(),'Invalid primary input '+n)
        parameter_checks={}
        for n in PARAMETERS:
            a=getattr(act,n).detach().cpu().numpy();b=np.asarray(self.parameters[n],np.float32)[1:2]
            parameter_checks[n]=bool(a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes())
        if self.initial is None:self.initial={n:a.copy() for n,a in raw.items()}
        checks={n:bool(a.shape==self.initial[n].shape and a.dtype==self.initial[n].dtype and a.tobytes()==self.initial[n].tobytes()) for n,a in raw.items()}
        passed=all(checks.values()) and all(parameter_checks.values())
        proof=dict(label=label,logical_world=1,passed=passed,comparison='exact primary inputs against first cancel-control preparation in this live corrected owner',
            reference_label=self.expected_labels[0],checks=checks,startup_parameters=parameter_checks,arrays=array_meta(raw),
            historical_opaque_state_loaded=False,all_derived_solver_state_equivalence_claimed=False)
        save_arrays(self.out,label+'-input-preflight-data',raw);save(self.out/(label+'-input-preflight.json'),proof)
        require(passed,'Remaining input differs before physical integration; inspect saved input proof')
        self.labels.append(label);return proof
    def summary(self):return dict(checks=len(self.labels),labels=self.labels,primary_fields_each=len(PRIMARY),startup_fields_each=len(PARAMETERS),native_calls=0,physical_integrations_added=0)
