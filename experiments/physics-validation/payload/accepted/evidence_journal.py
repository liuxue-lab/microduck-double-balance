"""Bounded raw evidence writer for a future already-authorized worker.

No physics is launched. Positive file limits authorize only fixture/archive I/O,
never native calls. Files are exclusive-create; partial failures are preserved.
Snapshots include all numeric Data and selected owners, including private RNG.
Opaque native contents remain outside this snapshot.
"""
from pathlib import Path
import hashlib,io,json
import numpy as np
from scene_evidence import owner_state,array_meta

def require(ok,msg):
    if not ok:raise RuntimeError(msg)
def normalize(x):
    if isinstance(x,np.ndarray):
        require(np.isfinite(x).all(),'Nonfinite recorded numeric evidence');return x.tolist()
    if isinstance(x,np.generic):return normalize(x.item())
    if isinstance(x,dict):return {k:normalize(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [normalize(v) for v in x]
    if x is None or type(x) in (str,int,float,bool):return x
    raise RuntimeError('Unsupported evidence type '+type(x).__name__)
def encoded(x):return (json.dumps(normalize(x),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
sha=lambda b:hashlib.sha256(b).hexdigest()

class EvidenceJournal:
    def __init__(self,path,bridge,*,byte_limit=0,file_limit=0,record_limit=0):
        require(all(type(v) is int and v>=0 for v in (byte_limit,file_limit,record_limit)),'Explicit nonnegative evidence limits required')
        self.path=Path(path);self.bridge=bridge
        self.byte_limit,self.file_limit,self.record_limit=byte_limit,file_limit,record_limit
        self.bytes=0;self.files={};self.events=[];self.started=False;self.stopped=False;self.closed=False
        self.attempted_files=0;self.attempted_records=0
    def _reserve(self,parts):
        require(not self.stopped and not self.closed,'Evidence journal stopped or finalized')
        require(self.bytes+sum(map(len,parts.values()))<=self.byte_limit and len(self.files)+len(parts)<=self.file_limit,'Evidence archive cap exhausted; no silent truncation')
        require(not any(name in self.files for name in parts),'Evidence file reuse forbidden')
    def _write(self,parts):
        self._reserve(parts)
        if not self.started:
            self.path.mkdir(parents=True,exist_ok=False);self.started=True
        for name,raw in parts.items():
            self.attempted_files+=1
            with (self.path/name).open('xb') as f:f.write(raw);f.flush()
            self.files[name]=dict(bytes=len(raw),sha256=sha(raw));self.bytes+=len(raw)
    def _snapshot(self,prefix):
        b=self.bridge;b.sync()
        require(b.saved is not None,'Raw evidence needs an active measured checkpoint')
        b.saved.data.preflight()
        data={name:a.numpy().copy() for name,a in b.saved.data.arrays.items()}
        owners,scalars,topology=owner_state(b.roots,b.torch,b.wp)
        parts={}
        for label,arrays in (('data',data),('owners',owners)):
            buf=io.BytesIO();np.savez_compressed(buf,**arrays);parts[prefix+'-'+label+'.npz']=buf.getvalue()
        parts[prefix+'-snapshot.json']=encoded(dict(Data=array_meta(data),owners=array_meta(owners),owner_scalars=scalars,owner_topology=topology,
            global_rng=b.global_rng(),model_hashes={n:sha(a.numpy().tobytes()) for n,a in b.saved.model_arrays.items()},
            scope='Full numeric Data and selected owner bytes; not an opaque native-handle snapshot'))
        return parts
    def __call__(self,kind,row):
        require(not self.closed and not self.stopped,'Evidence journal stopped or finalized')
        require(self.attempted_records<self.record_limit and self.file_limit>0 and self.byte_limit>0,'Evidence cap exhausted before native read or write')
        self.attempted_records+=1
        try:
            prefix='record-%05d'%self.attempted_records
            parts={}
            if kind in ('macro_start','trial_physics_complete','trial_complete','rollback','commit_prepared'):parts=self._snapshot(prefix)
            event=dict(sequence=self.attempted_records,kind=kind,row=row,snapshots=sorted(parts))
            name=prefix+'.json';parts[name]=encoded(event)
            self._write(parts);self.events.append(name)
        except Exception:self.stopped=True;raise
    def finalize(self):
        require(self.started and not self.stopped and not self.closed,'Cannot finalize incomplete/failed evidence')
        record=dict(events=self.events,files=self.files.copy(),payload_bytes=self.bytes,attempted_files=self.attempted_files,attempted_records=self.attempted_records,
            actual_GPU_execution_claimed=False,physics_authorization_granted=False)
        self._write({'evidence-inventory.json':encoded(record)});self.closed=True
        return record
