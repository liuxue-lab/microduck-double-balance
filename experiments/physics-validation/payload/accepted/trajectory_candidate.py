"""Scoped in-process candidate: fixed scalar reductions and explicit rigid metadata.

All original strict endpoint/stage comparators remain active. Every graph call
archives full before-metadata, before-sort and after-sort Contact bytes.
"""
import hashlib,json
import numpy as np
from lifecycle_support import require
from solver_replay import trace_adapter
from live_contact_graph import LiveContactGraph
from contact_order_contract import FIELDS,permutation,evidence,verify_evidence,canon

class FullContactArchive:
    def __init__(self,out,layout,limits):
        self.path=out/'full-contact';self.path.mkdir(exist_ok=False);(self.path/'blobs').mkdir()
        self.layout,self.limits=layout,limits;self.blobs={};self.bytes=0;self.index_bytes=0;self.rows=0;self.started=False;self.poisoned=False
    def blob(self,name,a):
        require(list(a.shape)==self.layout[name]['shape'] and str(a.dtype)==self.layout[name]['dtype'],'Full Contact archive layout differs')
        raw=a.tobytes();digest=hashlib.sha256(raw).hexdigest()
        if digest not in self.blobs:
            require(self.bytes+len(raw)<=self.limits['blob_bytes'] and len(self.blobs)<self.limits['blob_files'],'Full Contact raw evidence budget exhausted')
            with (self.path/'blobs'/(digest+'.bin')).open('xb') as f:f.write(raw)
            self.blobs[digest]=len(raw);self.bytes+=len(raw)
        return digest
    def write(self,before_metadata,before_sort,after_sort,proof):
        require(not self.poisoned and self.rows<self.limits['records'] and proof['serial']==self.rows+1,'Full Contact evidence schedule differs')
        try:
            row=dict(serial=proof['serial'],proof=proof)
            for label,raw in [('before_metadata',before_metadata),('before_sort',before_sort),('after_sort',after_sort)]:
                require(set(raw)==set(FIELDS),'Full Contact roster differs');row[label]={n:self.blob(n,raw[n]) for n in FIELDS}
            encoded=canon(row)+b'\n';require(self.index_bytes+len(encoded)<=self.limits['index_bytes'],'Full Contact index budget exhausted')
            with (self.path/'index.jsonl').open('ab' if self.started else 'xb') as f:f.write(encoded)
            self.started=True;self.index_bytes+=len(encoded);self.rows+=1
        except BaseException:self.poisoned=True;raise
    def summary(self):return dict(records=self.rows,unique_blobs=len(self.blobs),raw_blob_bytes=self.bytes,index_bytes=self.index_bytes,poisoned=self.poisoned,fields_per_record=16,capacity_rows=256,old_evidence_reconstructed=False)

class TrajectoryContactGraph(LiveContactGraph):
    CALL_CAP=8480
    def configure(self,archive,kernel):self.archive,self.metadata_kernel=archive,kernel
    def apply(self):
        require(self.prepared and not self.closed,'Live graph unavailable')
        self.gate.sync();self.ensure_binding();n=int(self.count_buffer.numpy()[0])
        require(0<=n<=256 and self.model.nflex==0,'Rigid metadata scope differs')
        before={name:a.numpy().copy() for name,a in self.buffers.items()}
        require(((before['geom'][:n]>=0)&(before['geom'][:n]<85)).all(),'Flex or foreign geometry cannot be initialized as rigid')
        self.gate.limited('Rigid_metadata_kernel',self.CALL_CAP,lambda:self.wp.launch(self.metadata_kernel,dim=256,inputs=[self.count_buffer,self.buffers['flex'],self.buffers['vert']],device=self.device))()
        raw={name:a.numpy().copy() for name,a in self.buffers.items()}
        for name,a in before.items():
            expected=a.copy()
            if name in ('flex','vert'):expected[:n]=0
            require(raw[name].tobytes()==expected.tobytes(),'Rigid metadata kernel changed undeclared bytes: '+name)
        expected,order=permutation(raw,n,self.layout)
        self.call('launch',lambda:self.apis['capture_launch'](self.graph));self.gate.sync()
        actual_order=self.order.numpy();require(int(self.status.numpy()[0])==0 and actual_order[:n].astype(np.int64).tobytes()==order.tobytes(),'Sorting graph disagrees with independent oracle')
        require(actual_order[n:].tobytes()==np.arange(n,256,dtype=np.int32).tobytes(),'Inactive order changed')
        after={name:a.numpy().copy() for name,a in self.buffers.items()}
        require(all(after[name].tobytes()==expected[name].tobytes() for name in FIELDS),'Full Contact permutation differs')
        self.ensure_binding();self.serial+=1;proof=evidence(raw,after,order,n,self.serial);verify_evidence(proof)
        self.last=proof;self.digest.update(canon(proof));self.archive.write(before,raw,after,proof)
    def summary(self):
        result=super().summary();result.update(implementation='rigid-metadata-complete-sort-candidate-v1',rigid_active_metadata_value=0,
            metadata_is_actual_kernel_write=True,full_contact_archive=self.archive.summary() if hasattr(self,'archive') else None)
        return result

class TrajectoryReductions:
    def __init__(self,m,d,wp,solver,gate,caps,*,out,world,order_limits):
        require(m.nv==m.nv_pad==32 and not m.is_sparse and m.opt.cone==0 and m.nflex==0 and d.nworld==1 and d.njmax==1024,'Unreviewed reduction scope')
        require(m.opt.iterations==10 and m.opt.ls_iterations==20 and m.opt.solver==2 and m.opt.ls_parallel is True,'Unreviewed solver options')
        import batch_kernels as k
        self.m,self.d,self.wp,self.solver,self.gate,self.caps,self.k=m,d,wp,solver,gate,caps,k
        self.originals={};self.installed=False
        self.call('Reduction_module_load',wp.load_module,k,device=wp.get_device('cuda:0'))
        self.rows=self.call('Reduction_allocation',wp.zeros,(1,1024),dtype=wp.float32,device=wp.get_device('cuda:0'))
        self.reference=self.call('Reduction_allocation',wp.zeros,1,dtype=wp.float32,device=wp.get_device('cuda:0'))
        require(self.rows.capacity+self.reference.capacity==4100,'Reduction scratch differs')
        from solver_order_candidate import SolverOrder
        self.order=SolverOrder(out,world,d,wp,solver,gate,caps,order_limits)
    def call(self,name,fn,*args,**kwargs):return self.gate.limited(name,self.caps[name],fn)(*args,**kwargs)
    def norm(self,values,target):self.call('Norm_kernel',self.wp.launch,self.k.norm32,dim=1,inputs=[values],outputs=[target])
    def cost(self,ctx):
        d=self.d;w=self.wp
        self.call('Cost_rows_kernel',w.launch,self.k.cost_rows,dim=(1,1024),inputs=[d.ne,d.nf,d.nefc,d.efc.D,d.efc.frictionloss,ctx.Jaref,ctx.done],outputs=[self.rows])
        self.call('Cost_reduce_kernel',w.launch,self.k.reduce_cost,dim=1,inputs=[d.nefc,ctx.done,self.rows,ctx.gauss,ctx.cost],outputs=[self.reference])
        self.call('Cost_override_copy',w.copy,ctx.cost,self.reference)
    def check(self,m,d):
        require(m is self.m and d is self.d and not self.wp.get_device('cuda:0').is_capturing,'Foreign/captured solver call')
        c=self.gate.counts;require(c.get('GPU_forward_attempted',0)==c.get('GPU_forward_completed',0)+1,'Solver must belong to one counted forward')
    def install(self):
        require(not self.installed,'Reduction installation is single-use');self.installed=True;solver=self.solver
        labels={'init_context':8480,'_update_constraint':93280,'_update_gradient':8480,'_update_gradient_incremental':84800,'_linesearch':84800,'_solver_iteration':84800}
        for name,cap in labels.items():
            original=getattr(solver,name);self.originals[name]=original
            def before(arguments,_name=name):
                self.check(arguments['m'],arguments['d'])
                self.order.trace('before:'+_name,arguments['ctx'])
                if _name=='_update_constraint' and arguments['track_changes']:
                    self.order.before_constraint(arguments['ctx'])
                if _name=='_linesearch':self.norm(arguments['ctx'].search,arguments['ctx'].search_dot)
            def after(arguments,_name=name):
                if _name=='_update_constraint':self.cost(arguments['ctx'])
                if _name=='_update_constraint' and arguments['track_changes']:
                    self.order.after_constraint(arguments['ctx'])
                if _name in ('_update_gradient','_update_gradient_incremental'):self.norm(arguments['ctx'].grad,arguments['ctx'].grad_dot)
                self.order.trace('after:'+_name,arguments['ctx'])
            self.gate.patch(solver,name,self.gate.limited('Solver'+name,cap,trace_adapter(original,before,after)))
        for name,label in [('create_solver_context','SolverContext_create'),('solve','GPU_solver')]:
            original=getattr(solver,name);self.originals[name]=original
            def counted(m,d,_original=original):self.check(m,d);return _original(m,d)
            self.gate.patch(solver,name,self.gate.limited(label,8480,counted))
    def summary(self):
        names=('GPU_solver','Solver','Reduction','Cost_','Norm_kernel')
        return dict(candidate_only=True,installed_sources_modified=False,solver_tolerances_unchanged=True,private_device_bytes=4100,
            counts={k:v for k,v in self.gate.counts.items() if k.startswith(names)},prior_batch_causal_improvement_established=False,
            changed_constraint_order=self.order.summary())

def collect_check(checks,key,fn):
    """Collect only read-only comparison failures; never wraps native mutation."""
    require(key not in checks,'Duplicate comparison')
    try:value=fn()
    except Exception as exc:
        checks[key]=dict(passed=False,error=repr(exc));return False
    checks[key]=dict(passed=True,result=value);return True

