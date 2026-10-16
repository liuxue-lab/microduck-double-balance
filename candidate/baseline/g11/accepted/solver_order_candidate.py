"""Measured in-place change-list ordering and bounded solver evidence.

No host upload, no extra solve, no field exemption, no tolerance change.
This fixes a source-visible nondeterministic reduction input. GPU acceptance
and causal attribution require the new raw return; they are not asserted here.
"""
import hashlib,io,json
import numpy as np
from lifecycle_support import require
from report_io import save

sha=lambda b:hashlib.sha256(b).hexdigest()
def canonical(x):return (json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()

def expected_ids(before,after,ids,count,nefc,done,quadratic):
    require(before.shape==after.shape==ids.shape==(1,1024) and before.dtype==after.dtype==ids.dtype==np.int32,'Changed-row layout differs')
    require(type(count) is int and type(nefc) is int and 0<=count<=nefc<=1024,'Changed-row count invalid')
    actual=ids[0,:count].tolist()
    require(len(set(actual))==count and all(0<=i<nefc for i in actual),'Changed-row IDs duplicated or outside active constraints')
    expected=[] if done else np.flatnonzero((before[0,:nefc]==quadratic)!=(after[0,:nefc]==quadratic)).tolist()
    require(sorted(actual)==expected,'Native changed-row list does not match actual quadratic-state changes')
    require(before[0,nefc:].tobytes()==after[0,nefc:].tobytes(),'Inactive constraint state changed')
    return expected

def verify_sorted(before,after,count,expected):
    require(before.dtype==after.dtype==np.int32 and before.shape==after.shape==(1,1024),'Sorted list layout differs')
    require(after[0,:count].tolist()==expected and before[0,count:].tobytes()==after[0,count:].tobytes(),'Canonical list/tail differs')
    return True

class SolverOrder:
    def __init__(self,out,world,d,wp,solver,gate,caps,limits):
        import solver_order_kernel as k
        self.root=out/'solver-order';self.root.mkdir();(self.root/'blobs').mkdir();(self.root/'target-traces').mkdir()
        self.world,self.d,self.wp,self.gate,self.caps,self.limits,self.k=world,d,wp,gate,caps,limits,k
        self.quadratic=int(solver.types.ConstraintState.QUADRATIC.value)
        self.pending=None;self.rows=0;self.reordered=0;self.index_bytes=0;self.blob_bytes=0;self.blobs=set();self.digest=hashlib.sha256()
        self.target=None;self.targets=[];self.trace_rows=0;self.trace_bytes=0
        self.call('SolverOrder_module_load',wp.load_module,k,device=wp.get_device('cuda:0'))
    def call(self,name,fn,*args,**kwargs):return self.gate.limited(name,self.caps[name],fn)(*args,**kwargs)
    def begin_trial(self,ticks,resolution,role,trial):
        # Exactly the previously observed first-stage boundary, in fresh work.
        self.target=None
        if self.world==1 and ticks==(0,) and resolution==128 and role in ('retry','replay','fixed_continuation'):
            require(len(self.targets)<3 and role not in [r['role'] for r in self.targets],'Unexpected diagnostic target repetition')
            self.target=dict(world=self.world,resolution=resolution,role=role,trial=trial,forward=self.gate.counts.get('GPU_forward_attempted',0)+1)
            self.targets.append(dict(self.target))
    def trace(self,phase,ctx):
        if self.target is None or self.gate.counts.get('GPU_forward_attempted',0)!=self.target['forward']:return
        require(self.trace_rows<self.limits['target_trace_records'],'Solver trace record cap')
        self.gate.sync();arrays={}
        for n in ('qpos','qvel','ctrl','qacc','qacc_warmstart','qacc_smooth','qfrc_smooth','qfrc_constraint','qM','nefc','ne','nf','solver_niter'):
            arrays['Data.'+n]=getattr(self.d,n).numpy().copy()
        for n in ('J','D','aref','force','state','Ma'):arrays['Data.efc.'+n]=getattr(self.d.efc,n).numpy().copy()
        for n,a in vars(ctx).items():
            if isinstance(a,self.wp.array):arrays['ctx.'+n]=a.numpy().copy()
        buf=io.BytesIO();np.savez_compressed(buf,**arrays);raw=buf.getvalue()
        require(self.trace_bytes+len(raw)<=self.limits['target_trace_bytes'],'Solver trace byte cap')
        self.trace_rows+=1;name='record-%04d'%self.trace_rows
        with (self.root/'target-traces'/(name+'.npz')).open('xb') as f:f.write(raw)
        save(self.root/'target-traces'/(name+'.json'),dict(target=self.target,phase=phase,bytes=len(raw),sha256=sha(raw),fields={n:dict(shape=list(a.shape),dtype=str(a.dtype),sha256=sha(a.tobytes())) for n,a in arrays.items()}))
        self.trace_bytes+=len(raw)
    def before_constraint(self,ctx):
        require(self.pending is None,'Nested changed-row transaction')
        self.gate.sync();self.pending=(ctx,self.d.efc.state.numpy().copy(),int(self.d.nefc.numpy()[0]),bool(ctx.done.numpy()[0]))
    def blob(self,raw):
        h=sha(raw)
        if h not in self.blobs:
            require(self.blob_bytes+len(raw)<=self.limits['blob_bytes'] and len(self.blobs)<self.limits['blob_files'],'Changed-row blob cap')
            with (self.root/'blobs'/(h+'.bin')).open('xb') as f:f.write(raw)
            self.blobs.add(h);self.blob_bytes+=len(raw)
        return h
    def after_constraint(self,ctx):
        require(self.pending is not None and self.pending[0] is ctx,'Missing changed-row input')
        self.gate.sync();_,old,nefc,done=self.pending
        new=self.d.efc.state.numpy().copy();before=ctx.changed_efc_ids.numpy().copy();count_raw=ctx.changed_efc_count.numpy().copy();count=int(count_raw[0])
        expected=expected_ids(old,new,before,count,nefc,done,self.quadratic)
        self.trace('change_list_before_canonicalization',ctx)
        self.call('SolverOrder_kernel',self.wp.launch,self.k.canonical_changes,dim=1,inputs=[ctx.changed_efc_count,ctx.changed_efc_ids])
        after=ctx.changed_efc_ids.numpy().copy();verify_sorted(before,after,count,expected)
        require(ctx.changed_efc_count.numpy().tobytes()==count_raw.tobytes() and self.d.efc.state.numpy().tobytes()==new.tobytes(),'Canonicalization changed count/state')
        row=dict(serial=self.rows+1,forward=self.gate.counts['GPU_forward_attempted'],count=count,nefc=nefc,done=done,
            active_before=self.blob(before[0,:count].tobytes()),active_after=self.blob(after[0,:count].tobytes()),
            inactive_sha256=sha(before[0,count:].tobytes()),quadratic_state_changes_exact=True,host_uploads=0)
        raw=canonical(row);require(self.rows<self.limits['records'] and self.index_bytes+len(raw)<=self.limits['index_bytes'],'Changed-row index cap')
        with (self.root/'index.jsonl').open('ab' if self.rows else 'xb') as f:f.write(raw)
        self.digest.update(raw);self.rows+=1;self.index_bytes+=len(raw);self.reordered+=int(before.tobytes()!=after.tobytes());self.pending=None
        self.trace('change_list_after_canonicalization',ctx)
    def summary(self):
        return dict(protocol='dense-newton-canonical-changes-v1',records=self.rows,reordered_lists=self.reordered,index_bytes=self.index_bytes,index_sha256=self.digest.hexdigest(),unique_blobs=len(self.blobs),blob_bytes=self.blob_bytes,
            target_trace_records=self.trace_rows,target_trace_bytes=self.trace_bytes,targets=self.targets,host_uploads=0,pending=self.pending is not None,private_device_bytes=0,
            tolerance_changes=0,prior_failure_causal_attribution_proven=False)
