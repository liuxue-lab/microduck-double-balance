"""Stage11 device tensor prototype audit using pinned fixed states; no integration."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,time,traceback,zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location('frozen_interface',ROOT/'scripts/audit_head_contact_interface.py')
a=importlib.util.module_from_spec(sp);sp.loader.exec_module(a)
save,require=a.save,a.require
PRIOR=Path('/home/lx/microduck-double-balance/artifacts/double-balance-stage11/head-interface/20261004T064257Z')
ZIP_SHA='a5ba697378bec4e846f92fc5145206b5516f99d9c1d2f942244cff6826e71eca'
CONTRACT=ROOT/'docs/audits/stage-11-head-tensor-contract.json'


def inputs(out):
    raw=PRIOR.with_suffix('.zip').read_bytes()
    require(hashlib.sha256(raw).hexdigest()==ZIP_SHA,'Pinned interface ZIP changed')
    paths=('run-summary.json','interface/configured-states.json','interface/cpu-forward-snapshots.json','interface/compiled-identities.json')
    result={};hashes={}
    with zipfile.ZipFile(PRIOR.with_suffix('.zip')) as z:
        inventory=json.loads(z.read('evidence-inventory.json'))
        for n in paths:
            data=z.read(n);sha=hashlib.sha256(data).hexdigest()
            require(sha==inventory[n]['sha256'] and len(data)==inventory[n]['bytes'],'Input inventory mismatch: '+n)
            require((PRIOR/n).read_bytes()==data,'Local returned input changed: '+n)
            result[n]=json.loads(data);hashes[n]=sha
    run=result['run-summary.json']
    require(run['execution_invariants_unchanged'] and run['physics_integration_steps']==run['physics_attempted_steps']==0 and run['canonical_total_known_completed_steps']==7895,'Unexpected prior counters')
    save(out/'input-provenance.json',{'zip_sha256':ZIP_SHA,'files':hashes,'old_configurations_reused_without_CPU_replay':True})
    return result


def additional_sources(out):
    contract=json.loads(CONTRACT.read_text());result={}
    for module,paths in contract['additional_reviewed_sources'].items():
        root=Path(importlib.util.find_spec(module).origin).parent
        for n,expected in paths.items():
            data=(root/n).read_bytes();actual=hashlib.sha256(data).hexdigest()
            result[module+'/'+n]={'sha256':actual,'reviewed_sha256':expected,'matches':actual==expected}
            p=out/'installed-source'/module/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    save(out/'additional-dependency-sources.json',result)
    require(all(v['matches'] for v in result.values()),'Stream implementation source differs; stop before model')
    return result


def tensor_cpu(value):
    return value.detach().cpu().numpy().copy()


def numpy_oracle(model,wd,wm,ids,revision):
    import mujoco_warp as mw
    import warp as wp
    ncon=int(wd.nacon.numpy()[0]);require(ncon < wd.naconmax,'Contact capacity reached')
    forces=wp.zeros(ncon,dtype=wp.spatial_vector,device='cuda:0')
    mw.contact_force(wm,wd,wp.array(np.arange(ncon,dtype=np.int32),dtype=wp.int32,device='cuda:0'),True,forces)
    wp.synchronize();ft=forces.numpy()
    raw={k:getattr(wd.contact,k).numpy()[:ncon].copy() for k in ('worldid','geom','frame','pos','dist','efc_address')}
    contacts=[dict(contact_id=i,world=int(raw['worldid'][i]),geom1=int(raw['geom'][i,0]),geom2=int(raw['geom'][i,1]),
                   frame=raw['frame'][i],pos=raw['pos'][i],dist=float(raw['dist'][i]),efc_address=int(raw['efc_address'][i,0]),
                   force_world=ft[i,:3],torque_world=ft[i,3:]) for i in range(ncon)]
    state={k:getattr(wd,k).numpy().copy() for k in a.DERIVED}
    return contacts,[a.snapshot({k:v[w] for k,v in state.items()},contacts,ids,phase='static_forward',revision=revision,world=w,data_time=0) for w in range(2)]


def compare_capture(capture,contacts,oracle):
    tensors={k:tensor_cpu(v) for k,v in capture['tensors'].items()}
    require(bool(tensors['valid_environment'].all()) and bool(tensors['contact_capacity_ok']) and int(tensors['invalid_world_contacts'])==0,'Tensor validity/capacity flags failed')
    errors={}
    for world,snap in enumerate(oracle):
        for name in ('ball_position','ball_velocity','ball_omega','head_omega','origin','rotation','position_material','rate_material'):
            expected=np.asarray(snap[name]);actual=tensors[name][world]
            error=float(np.max(np.abs(actual-expected)));errors[f'{world}/{name}']=error
            np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-6)
        for pair in ('head','floor'):
            expected=snap[pair+'_contacts'];indices=[int(c['contact_id']) for c in expected]
            mask=tensors[pair+'_mask'] & (tensors['contact_world']==world)
            require(np.array_equal(np.flatnonzero(mask),indices),'Global contact index/world mismatch')
            require(int(tensors[pair+'_count'][world])==len(expected) and int(tensors[pair+'_active_count'][world])==snap['active_'+pair],'Pair count mismatch')
            total=np.zeros(3)
            for c in expected:
                i=c['contact_id'];total+=c['force_on_ball']
                for name in ('force_on_ball','normal_to_ball','torque_on_ball'):
                    np.testing.assert_allclose(tensors[name][i],c[name],atol=2e-6,rtol=2e-6)
                if pair=='head':np.testing.assert_allclose(tensors['head_contact_slip'][i],c['slip'],atol=2e-6,rtol=2e-6)
            np.testing.assert_allclose(tensors[pair+'_force_on_ball'][world],total,atol=2e-6,rtol=2e-6)
    require(np.all(tensors['data_time']==0),'Tensor snapshot time advanced')
    return tensors,errors


def run(out,prior):
    import mujoco,mujoco_warp as mw,warp as wp,torch
    from mjlab.scene import Scene
    from mjlab_microduck.tasks import mdp
    from mjlab_microduck.tasks.microduck_head_contact_env_cfg import make_microduck_head_contact_env_cfg
    require(torch.cuda.is_available() and '5060' in torch.cuda.get_device_name(0),'Authorized local RTX5060 required')
    cfg=make_microduck_head_contact_env_cfg(static_audit_only=True)
    model=Scene(cfg.scene,device='cpu').compile();cfg.sim.mujoco.apply(model)
    ids=a.resolve_ids(model);require(ids==prior['interface/compiled-identities.json'],'Compiled identities changed')
    states=prior['interface/configured-states.json'];reference=prior['interface/cpu-forward-snapshots.json']
    require(len(states)==len(reference)==4,'Unexpected saved configuration set')
    fields=a.load_module('prototype_arrays',ROOT/'scripts/audit_head_task_prototype.py').PHYSICS_FIELDS
    hashes={k:hashlib.sha256(np.asarray(getattr(model,k)).tobytes()).hexdigest() for k in fields}
    reports=[];retained=[]
    with wp.ScopedDevice('cuda:0'):
        wm=mw.put_model(model);wd=mw.put_data(model,mujoco.MjData(model),nworld=2,nconmax=512,njmax=4096)
        sim=SimpleNamespace(mj_model=model,wp_model=wm,wp_data=wd,num_envs=2)
        alternate=torch.cuda.Stream(device='cuda:0')
        for revision,batch in enumerate(((0,2),(2,0),(1,3),(3,1))):
            for key in ('qpos','qvel','ctrl'):
                dest=getattr(wd,key);arr=np.stack([states[k][key] for k in batch]).astype(np.float32)
                wp.copy(dest,wp.array(arr,dtype=dest.dtype,device='cuda:0'))
            wd.qacc_warmstart.zero_()
            before={k:getattr(wd,k).numpy().copy() for k in ('qpos','qvel','time')}
            mw.forward(wm,wd)
            # Deliberately read on a non-default Torch stream before a host sync.
            with torch.cuda.stream(alternate):
                capture=mdp.stage11_head_contact_tensor_snapshot(sim,phase='static_forward',revision=revision,static_audit_only=True)
            alternate.synchronize()
            for k,v in before.items():require(np.array_equal(getattr(wd,k).numpy(),v),'Forward/snapshot advanced '+k)
            require(np.max(wd.nefc.numpy()) < wd.njmax,'Constraint capacity reached')
            contacts,oracle=numpy_oracle(model,wd,wm,ids,revision)
            save(out/f'raw-batch-{revision}.json',{'batch':batch,'contacts':contacts,'oracle':oracle})
            tensors,errors=compare_capture(capture,contacts,oracle)
            # Reconcile geometric state against pinned CPU evidence without replay.
            for world,case in enumerate(batch):
                for name in ('ball_position','ball_velocity','ball_omega','origin','rotation','position_material','rate_material'):
                    np.testing.assert_allclose(tensors[name][world],reference[case][name],atol=2e-5,rtol=0)
            retained.append((capture,tensors))
            reports.append({'revision':revision,'batch':batch,'max_tensor_numpy_kinematic_error':max(errors.values()),
                            'errors':errors,'head_active_count':tensors['head_active_count'],'floor_active_count':tensors['floor_active_count']})
            np.savez_compressed(out/f'tensor-batch-{revision}.npz',**tensors)
            save(out/'batch-review.json',reports)
        # Every original snapshot must survive all later simulator writes.
        for capture,expected in retained:
            for name,value in expected.items():np.testing.assert_array_equal(tensor_cpu(capture['tensors'][name]),value)
        # Adapter only: after last forward, qacc/ctrl/time/qpos/qvel must not change.
        state_before={k:getattr(wd,k).numpy().copy() for k in ('qpos','qvel','time','ctrl','qacc_warmstart')}
        with torch.cuda.stream(alternate):
            mdp.stage11_head_contact_tensor_snapshot(sim,phase='static_forward',revision=4,static_audit_only=True)
        alternate.synchronize()
        require(all(np.array_equal(getattr(wd,k).numpy(),v) for k,v in state_before.items()),'Adapter mutated simulator state')
    require(all(hashlib.sha256(np.asarray(getattr(model,k)).tobytes()).hexdigest()==v for k,v in hashes.items()),'Physical arrays changed')
    save(out/'physical-array-signatures.json',hashes)
    return {'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'worlds':2,'forward_calls':4,
            'tensor_captures':5,'configured_samples':8,'retained_snapshots_unchanged':True,'nondefault_torch_stream_tested':True,
            'no_adapter_state_mutation':True,'max_tensor_numpy_kinematic_error':max(r['max_tensor_numpy_kinematic_error'] for r in reports),
            'full_manager_execution':False,'native_sensor_cache_used':False,'new_integration_steps':0}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tensor-forward-only',action='store_true',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False);start=time.monotonic();attempts=[]
    report=dict(status='TENSOR_INTERFACE_EXECUTION_BLOCKED',stage11_status='MATERIAL_BLOCKED_REVIEW_REQUIRED',stage11_complete=False,
                new_integration_steps=0,new_ppo_updates=0,cloud_hours=0,canonical_total_completed_steps=7895,formal_task_migration=False,policy_success=None)
    try:
        report['prior_source_count']=len(a.source_audit(out));report['stream_source_count']=len(additional_sources(out))
        prior=inputs(out)
        with a.zero_integration_guard() as attempts:
            report['tensor']=run(out,prior)
            require(not attempts and time.monotonic()-start < 600,'Zero-step guard or wall budget failed')
        report['status']='STATIC_TENSOR_INTERFACE_REVIEW_REQUIRED'
    except Exception as exc:
        report['error']=str(exc);(out/'error.txt').write_text(traceback.format_exc())
    report['forbidden_integration_call_attempts']=attempts;report['wall_seconds']=time.monotonic()-start
    save(out/'tensor-review.json',report);print('Stage11TensorResult='+report['status'],flush=True)
    return 0 if report['status']=='STATIC_TENSOR_INTERFACE_REVIEW_REQUIRED' else 2


if __name__=='__main__':raise SystemExit(main())
