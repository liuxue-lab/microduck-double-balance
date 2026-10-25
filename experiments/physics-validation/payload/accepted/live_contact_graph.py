"""Owned sorting subgraph over live Contact zero-copy uint32 views.

Physical forward remains eager. Only the previously measured sorting kernels
and 16 D2D copies are captured; host oracle/diagnostic reads remain outside.
"""
import gc,hashlib,json,time,weakref
import numpy as np
from contact_order_contract import ContactOrderHook,FIELDS,permutation,evidence,verify_evidence,canon
from lifecycle_support import require
from order_contract import Lease

class LiveContactGraph(ContactOrderHook):
    CALL_CAP=5013
    def __init__(self,model,data,wp,gate,counts,layout,cap,limits,apis,out):
        require(type(limits['launch']) is int and cap==limits['launch']==self.CALL_CAP,'Contact graph/forward cap disagreement')
        super().__init__(model,data,wp,gate,counts,layout,cap)
        self.limits,self.apis,self.out=limits,apis,out
        self.graph=None;self.lease=Lease();self.native_releases=[];self.events=[];self.original_destroy={}
        self.ctx=None;self.device=wp.get_device('cuda:0');self.prepared=False;self.closed=False;self.close_attempted=False;self.capture_active=False
        self.views={};self.scratch={};self.count_buffer=data.nacon;self.count_binding=self.token(data.nacon)
        self.contact_owner=id(data.contact);self.word_counts={};self.initial_binding=None
    def call(self,name,fn,cleanup=False):
        key='ContactGraph_'+name;attempt=key+'_attempted'
        if not cleanup:require(time.monotonic()<self.gate.deadline,'Graph inner wall budget exhausted')
        require(self.counts.get(attempt,0)<self.limits[name],'Live graph cap exhausted: '+name)
        self.counts[attempt]=self.counts.get(attempt,0)+1;self.gate.persist()
        result=fn();self.counts[key+'_completed']=self.counts.get(key+'_completed',0)+1;self.gate.persist();return result
    def binding(self):
        return dict(contact_owner=id(self.data.contact),nacon=list(self.token(self.data.nacon)),
            fields={name:list(self.token(getattr(self.data.contact,name))) for name in FIELDS},
            views={name:list(self.token(a)) for name,a in self.views.items()})
    def ensure_binding(self):
        require(id(self.data.contact)==self.contact_owner and self.data.nacon is self.count_buffer and self.token(self.count_buffer)==self.count_binding,'Live Contact/count owner changed')
        for name,a in self.buffers.items():
            require(getattr(self.data.contact,name) is a and self.token(a)==self.bindings[name],'Live Contact storage changed: '+name)
        require(self.binding()==self.initial_binding,'Live zero-copy view binding changed')
        self.lease.check(self.initial_binding)
    @staticmethod
    def pointer(p):return int(p.value or 0) if hasattr(p,'value') else int(p or 0)
    def prepare(self):
        require(not self.prepared and not self.closed,'Graph prepare is single use')
        wp=self.wp
        from warp._src import context
        import order_kernels
        self.ctx=context;self.kernels=order_kernels
        require(self.count_buffer.dtype==wp.int32 and self.count_buffer.shape==(1,) and self.count_buffer.strides==(4,),'nacon representation differs')
        self.call('module_load',lambda:wp.load_module(order_kernels,device=self.device))
        # Native view() expands vector/matrix scalar lanes; flatten() keeps the
        # owner reference. No pointer-built array and no host replacement.
        for name,a in self.buffers.items():
            raw=a.numpy();shape=tuple(self.layout[name]['shape']);dtype=self.layout[name]['dtype']
            require(raw.shape==shape and str(raw.dtype)==dtype and a.is_contiguous,'Live typed Contact layout differs: '+name)
            words=int(np.prod(shape));view=a.view(wp.uint32).flatten()
            require(view.ptr==a.ptr and view.capacity==a.capacity==words*4 and view.shape==(words,) and view.strides==(4,) and view.is_contiguous,'Non-zero-copy Contact view: '+name)
            require(view.dtype==wp.uint32 and view.device==a.device==self.device,'Contact view dtype/device differs')
            self.views[name]=view;self.word_counts[name]=words
            self.scratch[name]=self.call('allocation',lambda words=words:wp.zeros(words,dtype=wp.uint32,device=self.device))
        self.order=self.call('allocation',lambda:wp.zeros(256,dtype=wp.int32,device=self.device))
        self.status=self.call('allocation',lambda:wp.zeros(1,dtype=wp.int32,device=self.device))
        self.allocated_bytes=sum(x.capacity for x in [*self.scratch.values(),self.order,self.status])
        require(self.allocated_bytes<=self.limits['device_bytes'],'Scratch byte cap exceeded')
        arrays=[*self.buffers.values(),self.count_buffer,*self.scratch.values(),self.order,self.status]
        intervals=sorted((a.ptr,a.ptr+a.capacity) for a in arrays)
        require(all(end<=start for (_,end),(start,_) in zip(intervals,intervals[1:])),'Unexpected live/scratch alias')
        self.initial_binding=self.binding()
        for name in ('wp_cuda_graph_destroy','wp_cuda_graph_exec_destroy'):
            original=getattr(context.runtime.core,name);self.original_destroy[name]=original
            def observed(ctx,handle,name=name,original=original):
                key='ContactGraph_'+('exec_destroy' if name.endswith('exec_destroy') else 'graph_destroy')
                row=dict(api=name,context=self.pointer(ctx),handle=self.pointer(handle),capturing=bool(self.device.is_capturing));self.native_releases.append(row)
                self.counts[key+'_attempted']=self.counts.get(key+'_attempted',0)+1;self.gate.persist()
                value=original(ctx,handle);row['returned']=bool(value)
                if value:self.counts[key+'_completed']=self.counts.get(key+'_completed',0)+1
                self.gate.persist();return value
            setattr(context.runtime.core,name,observed)
        self.gate.sync();enabled=gc.isenabled();gc.disable()
        try:
            self.call('capture_begin',lambda:self.apis['capture_begin'](device=self.device,force_module_load=False));self.capture_active=True
            try:self.pipeline()
            finally:
                self.graph=self.call('capture_end',lambda:self.apis['capture_end'](device=self.device),cleanup=True);self.capture_active=False
        finally:
            if enabled:gc.enable()
        require(self.graph.graph is not None and self.graph.graph_exec is None,'Lazy graph initialization differs')
        self.lease.capture(self.initial_binding);self.prepared=True
        self.events.append(dict(event='captured',graph=self.pointer(self.graph.graph),graph_exec=None,bindings=self.initial_binding))
        self.save_state()
    def pipeline(self):
        w=self.wp;k=self.kernels
        self.call('kernel_api',lambda:w.launch(k.rank_rows,dim=1,inputs=[self.count_buffer,self.views['worldid'],self.views['geom'],self.views['geomcollisionid'],self.order,self.status],device=self.device))
        for name in FIELDS:
            rowwords=self.word_counts[name]//256;kind=1 if self.layout[name]['dtype']=='float32' else {'efc_address':2,'dim':3,'type':4}.get(name,0)
            self.call('kernel_api',lambda name=name,r=rowwords,t=kind:w.launch(k.validate_words,dim=1,inputs=[self.views[name],self.count_buffer,self.status,r,t],device=self.device))
        for name in FIELDS:
            self.call('kernel_api',lambda name=name:w.launch(k.gather_words,dim=self.word_counts[name],inputs=[self.views[name],self.scratch[name],self.count_buffer,self.order,self.status,self.word_counts[name]//256],device=self.device))
        for name in FIELDS:self.call('copy_api',lambda name=name:w.copy(self.views[name],self.scratch[name]))
    def apply(self):
        require(self.prepared and not self.closed,'Live graph unavailable')
        self.gate.sync();self.ensure_binding()
        raw={name:a.numpy().copy() for name,a in self.buffers.items()};n=int(self.count_buffer.numpy()[0])
        expected,expected_order=permutation(raw,n,self.layout)
        self.call('launch',lambda:self.apis['capture_launch'](self.graph));self.gate.sync()
        status=int(self.status.numpy()[0]);order=self.order.numpy().copy()
        require(status==0 and order[:n].astype(np.int64).tobytes()==expected_order.tobytes(),'Live graph status/order differs from independent host oracle')
        require(order[n:].tobytes()==np.arange(n,256,dtype=np.int32).tobytes(),'Inactive order entries changed')
        after={name:a.numpy().copy() for name,a in self.buffers.items()}
        require(all(after[name].tobytes()==expected[name].tobytes() for name in FIELDS),'Live GPU permutation differs from oracle')
        self.ensure_binding();self.serial+=1
        proof=evidence(raw,after,expected_order,n,self.serial);verify_evidence(proof)
        self.last=proof;self.digest.update(canon(proof))
        # Existing full-Data/RK evidence consumes this unmodified proof schema.
    def close(self):
        if self.closed:return
        require(not self.close_attempted,'Graph cleanup already attempted; no retry')
        self.close_attempted=True
        try:
            self.call('cleanup_sync',self.gate.sync,cleanup=True)
            require(not self.capture_active and not self.device.is_capturing,'Cannot release an active capture')
            binding_error=None
            if self.prepared:
                try:self.ensure_binding()
                except BaseException as exc:binding_error=repr(exc)
            self.lease.invalidate()
            if self.graph is not None:
                graph=self.graph;p=self.pointer(graph.graph);e=self.pointer(graph.graph_exec);ref=weakref.ref(graph)
                self.graph=None;graph=None;gc.collect();require(ref() is None,'Live graph still owned at close')
                expected=[('wp_cuda_graph_destroy',p)]+([('wp_cuda_graph_exec_destroy',e)] if e else [])
                require([(r['api'],r['handle']) for r in self.native_releases]==expected,'Live graph native release roster differs')
                require(all(r['returned'] and not r['capturing'] for r in self.native_releases),'Live graph destruction failed/deferred')
                self.events.append(dict(event='released',graph=p,graph_exec=e,weakref_dead=True))
            self.lease.released();self.closed=True
            require(binding_error is None,'Binding changed before cleanup: '+str(binding_error))
        finally:
            if self.ctx is not None:
                for name,original in self.original_destroy.items():setattr(self.ctx.runtime.core,name,original)
                self.original_destroy.clear()
            self.save_state()
    def save_state(self):
        (self.out/'live-contact-graph.json').write_text(json.dumps(dict(summary=self.summary(),events=self.events,native_releases=self.native_releases,
            initial_binding=self.initial_binding,current_binding=self.binding(),allocated_bytes=getattr(self,'allocated_bytes',None)),indent=2)+'\n')
    def summary(self):
        return dict(**super().summary(),implementation='live-zero-copy-contact-sort-subgraph-v1',graph_prepared=self.prepared,
            graph_closed=self.closed,graph_close_attempted=self.close_attempted,host_Contact_array_assignments=0,live_forward_calls=self.serial,
            graph_counts={k:v for k,v in self.counts.items() if k.startswith('ContactGraph_')},
            graph_contains_only_sorting=True,complete_simulation_graph_validated=False)
