"""Additional immutable Model arrays nested inside tuple/list containers.

The old portable checkpoint remains byte-compatible. These fixed topology
tables are freshly built, separately archived and compared across processes.
"""
import dataclasses
from native_state_bridge import model_tree
from scene_evidence import save_arrays,array_meta
from portable_state import exact_arrays
from lifecycle_support import require
from report_io import save

def collect(model,wp):
    arrays={};seen=set()
    def walk(value,path):
        if isinstance(value,wp.array):arrays[path]=value;return
        if id(value) in seen:return
        if dataclasses.is_dataclass(value):
            seen.add(id(value))
            for field in dataclasses.fields(value):walk(getattr(value,field.name),path+'.'+field.name)
        elif isinstance(value,(tuple,list)):
            seen.add(id(value))
            for i,v in enumerate(value):walk(v,path+'['+str(i)+']')
        elif isinstance(value,dict):
            seen.add(id(value))
            for k,v in value.items():walk(v,path+'['+repr(k)+']')
    walk(model,'Model')
    known=model_tree(model,wp)[0]
    return {name:a for name,a in arrays.items() if name not in known}

class NestedModelTables:
    def __init__(self,model,wp,out):
        self.model,self.wp,self.out=model,wp,out
        arrays=collect(model,wp);require(arrays and all(n.startswith(('Model.body_tree[','Model.qM_tiles[','Model.qLD_updates[')) for n in arrays),'Unreviewed nested Model tables')
        self.owners=self.binding(arrays);self.raw={n:a.numpy().copy() for n,a in arrays.items()};self.checks=[]
        save_arrays(out,'model-nested-initial',self.raw,owners=self.owners,scope='static nested tables freshly constructed; not historical native pointers')
    def binding(self,arrays):
        return {n:(id(a),None if a.ptr is None else int(a.ptr),int(a.capacity),list(a.shape),list(a.strides),str(a.dtype),str(a.device)) for n,a in arrays.items()}
    def check(self,label):
        arrays=collect(self.model,self.wp);require(self.binding(arrays)==self.owners,'Nested Model allocation/schema changed')
        raw={n:a.numpy().copy() for n,a in arrays.items()};proof=exact_arrays(self.raw,raw)
        save_arrays(self.out,'model-nested-'+label,raw)
        self.checks.append(dict(label=label,**proof));save(self.out/'model-nested-review.json',dict(fields=len(raw),checks=self.checks,native_handle_imports=0))
        require(proof['passed'],'Static nested Model table values changed')
        return proof
