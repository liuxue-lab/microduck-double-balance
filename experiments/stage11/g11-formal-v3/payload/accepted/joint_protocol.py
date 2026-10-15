"""Two independent live owners, one explicit diagnostic publication epoch.

No GPU state is concatenated. A published descriptor references two sealed,
separately owned full states. Native pointers are never transferred. This is a
diagnostic consumer boundary, not a replacement for ManagerBasedRlEnv.step.
"""
from pathlib import Path
import hashlib,json,os,time,uuid

CASES=tuple((case,mask,mode) for case,mask in
    [('control',None),('reset-0',[0]),('reset-1',[1]),('empty',[])]
    for mode in ['adaptive','reference'])
CANCEL=('cancel-control',None,'adaptive')

def require(ok,message):
    if not ok:raise RuntimeError(message)

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()

def atomic_create(path,value):
    """Publish complete JSON once; an existing command/epoch is never overwritten."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_name(path.name+'.'+uuid.uuid4().hex+'.writing')
    with pending.open('xb') as f:f.write(canonical(value)+b'\n');f.flush();os.fsync(f.fileno())
    os.link(pending,path)
    pending.unlink()

def read(path):
    path=Path(path);require(path.is_file() and not path.is_symlink() and path.stat().st_size<=8*1024*1024,'Invalid IPC record')
    return json.loads(path.read_text())

def verify_publication_receipt(published,receipt,epoch,nonce,logical_world):
    """Compare the actual transmitted receipt, including its transport sequence."""
    require(type(receipt.get('sequence')) is int and receipt['sequence']>0,'Prepared wire sequence missing')
    require(logical_world in (0,1) and receipt['logical_world']==logical_world,'Foreign local publication owner')
    require(published.get('protocol')=='stage11-independent-owners-publication-v1' and published.get('row_splicing') is False,'Publication protocol differs')
    require(published['epoch']==epoch==receipt['epoch'] and published['nonce']==nonce==receipt['nonce'],'Publication epoch differs')
    require(len(published['states'])==2 and published['states'][logical_world]==receipt,'Publication complete-owner receipt differs')
    return True

class Barrier:
    def __init__(self,epoch,nonce,owners):
        require(type(epoch) is int and epoch>=0 and isinstance(nonce,str) and len(nonce)==32,'Invalid publication epoch')
        require(set(owners)=={0,1} and all(o['native_worlds']==1 and o['logical_world']==w for w,o in owners.items()),'Two declared singleton owners required')
        require(owners[0]['pid']!=owners[1]['pid'] and owners[0]['owner_id']!=owners[1]['owner_id'],'Independent process ownership required')
        self.epoch,self.nonce,self.owners=epoch,nonce,owners
        self.prepared={};self.validated={};self.phase='collecting'
    def check_owner(self,row):
        w=row['logical_world'];require(w in self.owners,'Foreign logical world')
        require(all(row[k]==self.owners[w][k] for k in ('pid','owner_id','native_worlds')),'Foreign execution owner')
        require(row['epoch']==self.epoch and row['nonce']==self.nonce,'Stale/replayed barrier record')
        return w
    def prepare(self,row):
        require(self.phase=='collecting','Barrier no longer collecting')
        w=self.check_owner(row);require(w not in self.prepared and row['state']=='prepared','Duplicate or invalid prepare')
        require(isinstance(row['state_sha256'],str) and len(row['state_sha256'])==64 and all(c in '0123456789abcdef' for c in row['state_sha256']),'Missing complete-state seal')
        require(type(row['checks_passed']) is bool and row['logical_tick']>=0,'Invalid measured verdict/tick')
        self.prepared[w]=row
    def validate(self,row):
        require(self.phase=='collecting' and set(self.prepared)=={0,1},'Both full states must be ready before validation')
        w=self.check_owner(row);require(w not in self.validated and row['state']=='validated','Duplicate or invalid validation')
        require(row['state_sha256']==self.prepared[w]['state_sha256'] and row['unchanged'] is True,'Prepared owner changed while waiting')
        self.validated[w]=row
    def publish(self,path):
        require(self.phase=='collecting' and set(self.validated)=={0,1},'Incomplete joint validation')
        require(all(r['checks_passed'] for r in self.prepared.values()),'Failed comparisons prohibit joint publication')
        require(self.prepared[0]['global_timer_sha256']==self.prepared[1]['global_timer_sha256'],'Replicated diagnostic global timer diverged')
        value=dict(protocol='stage11-independent-owners-publication-v1',epoch=self.epoch,nonce=self.nonce,
            states=[self.prepared[w] for w in (0,1)],row_splicing=False,macro_duration_s=.002,
            full_environment_step=False,G09='OPEN')
        # No native call is made between final owner validation and this barrier.
        atomic_create(path,value);self.phase='published';return value
    def abort(self):
        require(self.phase=='collecting','A published epoch cannot be silently rolled back')
        self.phase='aborted'

class WorkerMailbox:
    def __init__(self,out,parent_pid,deadline):
        self.out=Path(out);self.parent_pid=parent_pid;self.deadline=deadline;self.sequence=0
    def respond(self,value):
        require('sequence' not in value,'Response sequence is owned by the mailbox')
        wire=dict(value,sequence=self.sequence)
        atomic_create(self.out/'responses'/('%04d.json'%self.sequence),wire)
        return wire
    def next(self):
        self.sequence+=1;path=self.out/'commands'/('%04d.json'%self.sequence)
        while not path.exists():
            require(time.monotonic()<self.deadline,'IPC deadline expired; no native retry')
            os.kill(self.parent_pid,0)
            time.sleep(.2)
        value=read(path);require(value['sequence']==self.sequence,'Command sequence differs')
        return value
