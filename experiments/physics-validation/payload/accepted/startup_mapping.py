"""Explicit logical-world initialization, never a native checkpoint loader."""
from pathlib import Path
import hashlib,json
import numpy as np
from lifecycle_support import require
from report_io import save
from scene_evidence import save_arrays,array_meta

PARAMETERS=('vin_tensor','vin_drop_gain')
PRIMARY=('Data.qpos','Data.qvel','Data.ctrl','Data.act','Data.qacc_warmstart','Data.qfrc_applied','Data.xfrc_applied','Data.time')

def fixture(plan):
    raw=Path(__file__).with_name('startup-fixture.json').read_bytes()
    require(hashlib.sha256(raw).hexdigest()==plan['startup_fixture_sha256'],'Startup fixture bytes changed')
    value=json.loads(raw)
    require(value['protocol']=='stage11-logical-startup-fixture-v1' and value['logical_world']==1,'Startup fixture scope differs')
    require(set(value['parameters'])==set(PARAMETERS) and set(value['target_macro_input'])==set(PRIMARY),'Startup/input fixture roster differs')
    return value

def initialize_logical_world(act,world,plan,torch,gate,out):
    value=fixture(plan);require(world in plan['execution_owners']==[0,1],'Foreign logical startup world')
    require(gate.counts.get('BAM_compute_attempted',0)==gate.counts.get('integrations_attempted',0)==0,'Startup mapping must precede control and physics')
    require(act._delay_buffer._buffer._buffer is None,'Cannot map startup parameters into live history')
    cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state_all()
    untouched={n:getattr(act,n).detach().cpu().numpy().copy() for n in
        ('_prev_motor_torque','kp_scale','kd_scale','default_kp_scale','default_kd_scale','friction_scale','default_friction_scale')}
    before={};after={};owners={}
    for name in PARAMETERS:
        target=getattr(act,name)
        require(isinstance(target,torch.Tensor) and tuple(target.shape)==(1,1) and target.dtype==torch.float32,'Unexpected startup tensor '+name)
        before[name]=target.detach().cpu().numpy().copy();owners[name]=(id(target),target.data_ptr())
        expected=np.asarray(value['parameters'][name],np.float32)
        require(expected.shape==(2,1) and np.isfinite(expected).all(),'Invalid two-world startup fixture')
        # The pre-repair singleton draw is recorded, not reused as logical world1.
        gate.limited('Startup_BAM_parameter_write',plan['owner_limits']['Startup_BAM_parameter_write'],
            lambda:target.copy_(torch.as_tensor(expected[world:world+1],dtype=target.dtype,device=target.device)))()
        after[name]=target.detach().cpu().numpy().copy()
        require(owners[name]==(id(target),target.data_ptr()) and after[name].tobytes()==expected[world:world+1].tobytes(),'Startup mapping changed storage or numeric bytes')
    require(torch.equal(cpu,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all())),'Startup mapping advanced default RNG')
    require(all(getattr(act,n).detach().cpu().numpy().tobytes()==a.tobytes() for n,a in untouched.items()),'Mapping changed undeclared BAM state')
    save_arrays(out,'startup-before',before);save_arrays(out,'startup-mapped',after)
    proof=dict(logical_world=world,fields=list(PARAMETERS),source=value['parameter_source'],source_archive_sha256=value['source_archive_sha256'],
        values_before={n:a.tolist() for n,a in before.items()},values_after={n:a.tolist() for n,a in after.items()},
        setup_writes=2,storage_preserved=True,default_rng_unchanged=True,other_selected_BAM_parameters_and_torque_unchanged=True,
        fresh_history=True,archived_dynamic_state_loaded=False,native_checkpoint_recovery=False)
    save(out/'startup-mapping.json',proof);return proof

def verify_macro_input(data,act,world,plan,out,label):
    value=fixture(plan);require(world==1 and label in ('continuation-reset-1-adaptive','continuation-reset-1-reference'),'Undeclared target input check')
    raw={n:getattr(data,n.removeprefix('Data.')).numpy().copy() for n in PRIMARY};checks={}
    for name,row in value['target_macro_input'].items():
        expected=np.asarray(row['values'],dtype=row['dtype']).reshape(row['shape']);actual=raw[name]
        same=actual.shape==expected.shape and actual.dtype==expected.dtype
        checks[name]=dict(passed=bool(same and actual.tobytes()==expected.tobytes()),
            max_abs=float(np.max(np.abs(actual.astype(np.float64)-expected.astype(np.float64)))) if same and actual.size else (0. if same else None),
            current=array_meta({name:actual})[name],expected=array_meta({name:expected})[name])
    for name in PARAMETERS:
        actual=getattr(act,name).detach().cpu().numpy();expected=np.asarray(value['parameters'][name],np.float32)[1:2]
        checks['BAM.'+name]=dict(passed=bool(actual.shape==expected.shape and actual.tobytes()==expected.tobytes()))
    proof=dict(label=label,logical_world=world,passed=all(v['passed'] for v in checks.values()),checks=checks,
        source=value['control_source'],comparison='exact dtype, shape and bytes; original physical thresholds unchanged',
        complete_derived_solver_state_equivalence_claimed=False,physical_integrations_in_this_target_before_check=0)
    save_arrays(out,label+'-input-preflight-data',raw);save(out/(label+'-input-preflight.json'),proof)
    require(proof['passed'],'Target primary input mismatch before integration; inspect saved input-preflight evidence')
    return proof
