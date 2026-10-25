"""Independent host oracle and graph ownership guard; no native imports."""
import numpy as np
FIELDS=('dim','dist','efc_address','flex','frame','friction','geom','geomcollisionid','includemargin','pos','solimp','solref','solreffriction','type','vert','worldid')
def oracle(raw,n):
    assert set(raw)==set(FIELDS)
    out={k:v.copy() for k,v in raw.items()};status=0;order=np.arange(256,dtype=np.int32)
    if n<0 or n>256:return out,order,1
    keys=[(int(raw['worldid'][i]),*map(int,raw['geom'][i]),int(raw['geomcollisionid'][i])) for i in range(n)]
    if len(set(keys))!=len(keys):status|=2
    if any(w<0 or w>1 or a<0 or a>=85 or b<0 or b>=85 or a==b for w,a,b,c in keys):status|=16
    for name,a in raw.items():
        if a.dtype==np.dtype('float32') and not np.isfinite(a[:n]).all():status|=4
    if not (raw['efc_address'][:n]==-1).all():status|=8
    if not np.isin(raw['dim'][:n],(1,3,4,6)).all():status|=32
    if not np.isin(raw['type'][:n],(1,2,3)).all():status|=64
    if status==0:
        order[:n]=sorted(range(n),key=lambda i:keys[i])
        for k,v in raw.items():out[k][:n]=v[order[:n]]
    return out,order,status

def make_case(seed,n,name):
    raw={k:v.copy() for k,v in seed.items()}
    if name=='reverse':
        for k in raw:raw[k][:n]=raw[k][:n][::-1]
    elif name=='empty':n=0
    elif name=='full256':
        for k in raw:raw[k][:]=seed[k][np.arange(256)%n]
        raw['worldid'][:]=np.arange(256)%1
        raw['geom'][:,0]=np.arange(256)%80;raw['geom'][:,1]=82
        raw['geomcollisionid'][:]=np.arange(256)
        raw['efc_address'][:]=-1;n=256
    elif name=='duplicate':
        for k in ('worldid','geom','geomcollisionid'):raw[k][1]=raw[k][0]
    elif name=='nonfinite':raw['dist'][0]=np.nan
    elif name=='allocated_efc':raw['efc_address'][0,0]=0
    elif name=='overflow':n=513
    elif name!='original':raise ValueError(name)
    return raw,n

class Lease:
    def __init__(self):self.bound=None;self.state='empty'
    def capture(self,token):
        if self.state not in ('empty','released'):raise RuntimeError('Graph owner must be empty before capture')
        self.bound=token;self.state='ready'
    def check(self,token):
        if self.state!='ready' or token!=self.bound:raise RuntimeError('Stale or invalid graph lease; native launch refused')
    def invalidate(self):self.state='invalid'
    def released(self):self.bound=None;self.state='released'
