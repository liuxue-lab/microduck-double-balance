"""Capture six declared pure device operations; preserve every host boundary.

Native RK4/forward, CPU compensation, host observations and adaptive decisions
remain unchanged. This is a segmented graph candidate, not a monolithic graph.
"""
import gc,json,weakref
from contextlib import contextmanager
from lifecycle_support import Gate,require
from report_io import save

LABELS=frozenset(('Norm_kernel','Cost_rows_kernel','Cost_reduce_kernel','Cost_override_copy','SolverOrder_kernel','Rigid_metadata_kernel'))
def ptr(v):return int(v.value or 0) if hasattr(v,'value') else int(v or 0)

class SegmentGate(Gate):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.segments=None;self.caps={}
    def limited(self,label,limit,native,check=None):
        limit=min(limit,self.caps.get(label,limit))
        def execute(*a,**k):
            s=self.segments
            if s is not None and s.enabled and label in LABELS:
                with s.scope(label):return native(*a,**k)
            return native(*a,**k)
        counted=super().limited(label,limit,execute,check)
        def call(*a,**k):
            try:return counted(*a,**k)
            finally:
                if label=='GPU_forward' and self.segments is not None:self.segments.retire()
        return call

class Segments:
    def __init__(self,wp,out,gate,apis,contact_order):
        self.wp,self.out,self.gate,self.apis=wp,out,gate,apis
        self.device=wp.get_device('cuda:0');self.raw_launch=wp.launch;self.raw_copy=wp.copy
        self.label=None;self.enabled=False;self.entries={};self.events=0;self.releases=[]
        self.targets=[];self.checked_stale=False;self.captures=0;self.launches=0;self.per_label={};self.closed=False;self.native_owned=set()
        from warp._src import context
        self.core=context.runtime.core;self.originals={}
        for name in ('wp_cuda_graph_destroy','wp_cuda_graph_exec_destroy'):
            original=getattr(self.core,name);self.originals[name]=original
            raw=contact_order.original_destroy[name]
            def observed(*args,_name=name,_original=original,_raw=raw):
                if (_name,ptr(args[-1])) not in self.native_owned:return _original(*args)
                row=dict(api=_name,handle=ptr(args[-1]),capturing=bool(self.device.is_capturing),returned_normally=False)
                self.releases.append(row);value=_raw(*args)
                row.update(returned_normally=True,result=bool(value));return value
            setattr(self.core,name,observed)
        gate.patch(wp,'launch',self.launch);gate.patch(wp,'copy',self.copy)
    def log(self,kind,**row):
        self.events+=1
        require(self.events<=1000000,'Segment evidence cap exhausted')
        with (self.out/'segments.jsonl').open('a') as f:f.write(json.dumps(dict(event=kind,sequence=self.events,**row),sort_keys=True)+'\n')
    @contextmanager
    def scope(self,label):
        require(self.label is None and label in LABELS,'Nested or foreign segment')
        self.label=label
        try:yield
        finally:self.label=None
    def token(self,value,refs):
        if isinstance(value,self.wp.array):
            refs.append(value)
            require(value.device==self.device,'Non-device array in graph segment')
            return ('array',id(value),None if value.ptr is None else int(value.ptr),int(value.capacity),tuple(value.shape),tuple(value.strides),str(value.dtype),str(value.device))
        if value is None or type(value) in (str,int,float,bool):return (type(value).__name__,value)
        if isinstance(value,(list,tuple)):return tuple(self.token(v,refs) for v in value)
        if isinstance(value,dict):return tuple((k,self.token(v,refs)) for k,v in sorted(value.items()))
        if value is self.device:return ('device',str(value))
        # Only pinned kernel objects are accepted as non-array operation inputs.
        require(hasattr(value,'module') and hasattr(value,'key'),'Unknown graph argument '+str(type(value)))
        return ('kernel',id(value),value.key)
    def launch(self,*args,**kwargs):
        if not (self.enabled and self.label):return self.raw_launch(*args,**kwargs)
        require(not kwargs.get('record_cmd',False) and not kwargs.get('adjoint',False),'Deferred/adjoint launch is outside segment scope')
        return self.execute('launch',self.raw_launch,args,kwargs)
    def copy(self,*args,**kwargs):
        if not (self.enabled and self.label):return self.raw_copy(*args,**kwargs)
        require(self.label=='Cost_override_copy','Only the declared cost D2D copy may be captured')
        return self.execute('copy',self.raw_copy,args,kwargs)
    def check(self,entry,token):require(entry['token']==token and not self.closed,'Stale graph binding rejected before CUDA dispatch')
    def native(self,label,fn):
        a,c='Segment_'+label+'_attempted','Segment_'+label+'_completed'
        require(self.gate.counts.get(a,0)<200000,'Segment capture API ceiling exceeded')
        self.gate.counts[a]=self.gate.counts.get(a,0)+1;self.gate.persist()
        result=fn();self.gate.counts[c]=self.gate.counts.get(c,0)+1;self.gate.persist();return result
    def execute(self,kind,fn,args,kwargs):
        require(not self.closed and not self.device.is_capturing,'Nested capture refused')
        refs=[];token=(self.label,kind,self.token(args,refs),self.token(kwargs,refs))
        entry=self.entries.get(token)
        if entry is None:
            require(len(self.entries)<256 and self.captures<200000,'Segment graph bound exceeded')
            if kind=='launch':self.wp.load_module(args[0].module,device=self.device)
            self.wp.synchronize();was=gc.isenabled();gc.disable()
            graph=None
            try:
                self.native('capture_begin',lambda:self.apis['capture_begin'](device=self.device,force_module_load=False))
                try:result=fn(*args,**kwargs)
                finally:graph=self.native('capture_end',lambda:self.apis['capture_end'](device=self.device))
            finally:
                if was:gc.enable()
            require(result is None and graph is not None,'Unexpected captured primitive result')
            self.captures+=1
            self.native_owned.add(('wp_cuda_graph_destroy',ptr(graph.graph)))
            entry=dict(graph=graph,token=token,refs=refs,serial=self.captures,label=self.label)
            self.entries[token]=entry
            self.log('capture',serial=self.captures,label=self.label,graph=ptr(graph.graph),binding=token,physics_integrations=0)
            if not self.checked_stale:
                if hasattr(self,'challenge'):
                    self.challenge(entry,args,kwargs,refs)
                else:
                    require(refs,'No array binding available for stale guard')
                    # wp.clone calls the public wp.copy internally. Keep this
                    # ownership challenge outside the captured numerical segment.
                    saved_label=self.label;self.label=None
                    try:clone=self.wp.clone(refs[0])
                    finally:self.label=saved_label
                    changed=[]
                    replacement=self.token(clone,changed)
                    def substitute(value):
                        if value is refs[0]:return clone
                        if isinstance(value,list):return [substitute(v) for v in value]
                        if isinstance(value,tuple):return tuple(substitute(v) for v in value)
                        if isinstance(value,dict):return {k:substitute(v) for k,v in value.items()}
                        return value
                    fake=(token[0],token[1],self.token(substitute(args),changed),self.token(substitute(kwargs),changed))
                    before=self.launches
                    try:self.check(entry,fake)
                    except RuntimeError:pass
                    else:raise RuntimeError('Stale binding dispatched')
                    require(before==self.launches,'Stale check launched a graph')
                    self.log('stale_rejected',serial=self.captures,new_array=list(replacement),before_dispatch=True)
                    clone=None;changed.clear();self.checked_stale=True
        self.check(entry,token)
        require(self.launches<1000000,'Segment launch ceiling exceeded')
        self.gate.counts['Segment_launch_attempted']=self.gate.counts.get('Segment_launch_attempted',0)+1;self.gate.persist()
        self.apis['capture_launch'](entry['graph']);self.wp.synchronize()
        self.native_owned.add(('wp_cuda_graph_exec_destroy',ptr(entry['graph'].graph_exec)))
        self.gate.counts['Segment_launch_completed']=self.gate.counts.get('Segment_launch_completed',0)+1;self.gate.persist()
        self.launches+=1;self.per_label[self.label]=self.per_label.get(self.label,0)+1
        self.log('launch',serial=entry['serial'],graph=ptr(entry['graph'].graph),exec=ptr(entry['graph'].graph_exec),label=self.label)
        return None
    def retire(self):
        if not self.entries:return
        require(not self.device.is_capturing,'Cannot retire during capture');self.wp.synchronize()
        items=[];offset=len(self.releases)
        for entry in self.entries.values():
            g=entry['graph'];items.append((weakref.ref(g),dict(serial=entry['serial'],graph=ptr(g.graph),exec=ptr(g.graph_exec))))
        g=None;entry=None;self.entries.clear()
        window=self.releases[offset:]
        for ref,row in items:
            require(ref() is None,'Segment graph remains owned')
            for api,key in (('wp_cuda_graph_destroy','graph'),('wp_cuda_graph_exec_destroy','exec')):
                matches=[x for x in window if x['api']==api and x['handle']==row[key]]
                require(row[key] and len(matches)==1 and matches[0]['returned_normally'] and matches[0]['result'] and not matches[0]['capturing'],'Segment native release missing')
            proof=[x for x in window if (x['api'],x['handle']) in {('wp_cuda_graph_destroy',row['graph']),('wp_cuda_graph_exec_destroy',row['exec'])}]
            self.log('release',**row,weakref_dead=True,native_callbacks=proof)
            self.native_owned.discard(('wp_cuda_graph_destroy',row['graph']));self.native_owned.discard(('wp_cuda_graph_exec_destroy',row['exec']))
        # Per-forward windows bound native callback memory; raw log retains proof.
        self.releases.clear()
    def close(self):
        self.retire();self.closed=True
        for name,original in self.originals.items():setattr(self.core,name,original)
        self.originals.clear()
        result=dict(captures=self.captures,launches=self.launches,per_label=self.per_label,stale_rejected=self.checked_stale,closed=True,host_decisions_preserved=True,full_native_RK4_graph=False)
        save(self.out/'segment-summary.json',result);return result
