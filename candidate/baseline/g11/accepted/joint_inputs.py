"""Two logical startup mappings and exact branch-local primary input checks."""
from pathlib import Path
import numpy as np
from lifecycle_support import require
from report_io import save
from scene_evidence import save_arrays,array_meta
from startup_mapping import fixture,PRIMARY,PARAMETERS,verify_macro_input
from joint_cases import CASES

class InputInvariant:
    def __init__(self,world,plan,out):
        require(world in plan['execution_owners']==[0,1],'Foreign joint input owner')
        self.world,self.plan,self.out=world,plan,Path(out);self.initial=None;self.labels=[]
        self.parameters=fixture(plan)['parameters']
    def check(self,data,act,label,immediate,reset_passed):
        require(len(self.labels)<len(CASES),'Extra joint input')
        case,mask,mode=CASES[len(self.labels)]
        require(label=='continuation-'+case+'-'+mode,'Repeated or foreign joint input')
        selected=mask is not None and self.world in mask
        raw={n:getattr(data,n.removeprefix('Data.')).numpy().copy() for n in PRIMARY}
        for n,a in raw.items():require(a.dtype==np.float32 and np.isfinite(a).all(),'Invalid primary input '+n)
        parameters={}
        for n in PARAMETERS:
            a=getattr(act,n).detach().cpu().numpy();b=np.asarray(self.parameters[n],np.float32)[self.world:self.world+1]
            parameters[n]=bool(a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes())
        if self.initial is None:
            require(case=='cancel-control' and not selected,'First input must be cancellation control')
            self.initial={n:a.copy() for n,a in raw.items()}
        if not selected:
            basis='Exact primary inputs against this live owner cancel-control; no historical state import'
            checks={n:bool(a.shape==self.initial[n].shape and a.tobytes()==self.initial[n].tobytes()) for n,a in raw.items()}
        else:
            require(reset_passed is True,'Selected input lacks original reset-contract acceptance')
            basis='Exact primary state after installed selected reset; only control may change during Action/BAM preparation'
            checks={n:bool(a.shape==immediate['data'][n].shape and a.tobytes()==immediate['data'][n].tobytes()) for n,a in raw.items() if n!='Data.ctrl'}
            checks['Data.ctrl_finite']=bool(np.isfinite(raw['Data.ctrl']).all())
        proof=dict(label=label,logical_world=self.world,selected=selected,comparison=basis,checks=checks,
                   startup_parameters=parameters,arrays=array_meta(raw),historical_opaque_state_loaded=False)
        if selected and self.world==1:
            proof['verified_reset1_fixture']=verify_macro_input(data,act,self.world,self.plan,self.out,label)
            checks['reset1_fixture']=proof['verified_reset1_fixture']['passed']
        proof['passed']=all(checks.values()) and all(parameters.values())
        # The fixture helper uses input-preflight filenames; keep the generalized
        # joint proof in separate files, never overwrite its exact fixture proof.
        save_arrays(self.out,label+'-joint-input-data',raw);save(self.out/(label+'-joint-input.json'),proof)
        require(proof['passed'],'Joint primary input mismatch before integration')
        self.labels.append(label);return proof
    def summary(self):return dict(checks=len(self.labels),labels=self.labels,primary_fields_each=len(PRIMARY),startup_fields_each=len(PARAMETERS),native_calls=0)
