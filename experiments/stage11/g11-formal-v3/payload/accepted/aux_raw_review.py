"""Validate actual full-contact blobs and recorded solver active-index lists."""
import hashlib,json
import numpy as np
from contact_order_contract import FIELDS,permutation,evidence
from lifecycle_support import require
sha=lambda b:hashlib.sha256(b).hexdigest()
def audit(path):
    counts=json.loads((path/'run-review.json').read_bytes())['counts'];contact=path/'full-contact';solver=path/'solver-order'
    cache={}
    def blob(folder,h):
        key=(str(folder),h)
        if key not in cache:
            b=(folder/'blobs'/(h+'.bin')).read_bytes();require(sha(b)==h,'Raw auxiliary blob hash differs');cache[key]=b
        return cache[key]
    ncontact=0
    with (contact/'index.jsonl').open() as f:
        for line in f:
            r=json.loads(line);ncontact+=1;p=r['proof'];require(r['serial']==p['serial']==ncontact,'Contact serial differs')
            layout={n:{k:p['arrays'][n][k] for k in ('shape','dtype')} for n in FIELDS}
            def arrays(label):return {n:np.frombuffer(blob(contact,r[label][n]),dtype=layout[n]['dtype']).reshape(layout[n]['shape']) for n in FIELDS}
            before,unsorted,sorted_actual=arrays('before_metadata'),arrays('before_sort'),arrays('after_sort')
            for n in FIELDS:
                expect=before[n].copy()
                if n in ('flex','vert'):expect[:p['nacon']]=0
                require(expect.tobytes()==unsorted[n].tobytes(),'Rigid metadata changed undeclared bytes')
            expected,order=permutation(unsorted,p['nacon'],layout)
            require(all(expected[n].tobytes()==sorted_actual[n].tobytes() for n in FIELDS),'Full contact oracle differs')
            require(evidence(unsorted,sorted_actual,order,p['nacon'],ncontact)==p,'Contact raw proof differs')
    nsolver=0
    with (solver/'index.jsonl').open() as f:
        for line in f:
            r=json.loads(line);nsolver+=1
            a=np.frombuffer(blob(solver,r['active_before']),np.int32);b=np.frombuffer(blob(solver,r['active_after']),np.int32)
            require(r['serial']==nsolver and len(a)==len(b)==r['count'] and 0<=r['count']<=r['nefc']<=1024,'Solver record/count differs')
            require(len(set(a.tolist()))==len(a) and np.all((a>=0)&(a<r['nefc'])) and np.array_equal(np.sort(a),b),'Solver active ordering differs')
            require(r['host_uploads']==0 and r['quadratic_state_changes_exact'] and 1<=r['forward']<=counts['GPU_forward_completed'],'Solver scope/counter differs')
    require(ncontact==counts['GPU_forward_completed']==counts['Contact_order_apply_completed'] and nsolver==counts['SolverOrder_kernel_completed'],'Auxiliary record counts differ')
    return dict(full_contact_records=ncontact,solver_active_index_records=nsolver,blob_hashes=len(cache),passed=True,
                solver_boundary_note='Active index bytes verified; full constraint before/after arrays are not archived for every iteration')
