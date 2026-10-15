"""Read-only exact replay differences; never grants an acceptance ticket."""
import hashlib,json
def compare(candidate,replay):
    a,b=json.loads(candidate['endpoint']),json.loads(replay['endpoint']);paths=[]
    def walk(x,y,path):
        if x==y:return
        if isinstance(x,dict) and isinstance(y,dict) and set(x)==set(y):
            for k in sorted(x):walk(x[k],y[k],path+[k])
        else:paths.append('.'.join(path))
    walk(a,b,[])
    return dict(endpoint_exact=candidate['endpoint']==replay['endpoint'],physical_stages_exact=candidate['stage_sha256']==replay['stage_sha256'],
        changed_endpoint_paths=paths,candidate_endpoint_sha256=hashlib.sha256(candidate['endpoint']).hexdigest(),replay_endpoint_sha256=hashlib.sha256(replay['endpoint']).hexdigest(),
        candidate_stage_sha256=candidate['stage_sha256'],replay_stage_sha256=replay['stage_sha256'],acceptance_authorized=False)
