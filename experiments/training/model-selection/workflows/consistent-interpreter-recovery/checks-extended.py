"""Exercise real preflight with disk fixtures; never import Torch or run physics."""
import contextlib
import io
import math
import os
from types import SimpleNamespace
import checks as base
from checks import *

extra=[]
def verified(name,value):
    assert value,name
    extra.append(name)

sample=c.summarize([row(i) for i in range(c.NUM_ENVS)])
changed=deepcopy(sample);changed['values_mean_over_first_episode_steps']['top_center_m']=5e-13
audit=c.summary_audit(changed,sample)
verified('small derived-float differences accepted and explicitly recorded',audit['valid'] and not audit['exact_equal'] and len(audit['differences'])==1)
changed['values_mean_over_first_episode_steps']['top_center_m']=1e-8
verified('material derived-float differences rejected with field and magnitude',not c.summary_audit(changed,sample)['valid'])
changed=deepcopy(sample);changed['successes']-=1
verified('one incorrect success count is rejected exactly',not c.summary_audit(changed,sample)['valid'])
changed=deepcopy(sample);changed['episodes']=float(changed['episodes'])
verified('integer count type cannot silently become float',not c.summary_audit(changed,sample)['valid'])
changed=deepcopy(sample);changed['mean_survival_seconds']=float('nan')
verified('nonfinite aggregate rejected',not c.summary_audit(changed,sample)['valid'])
changed=deepcopy(sample);changed.pop('success_wilson95')
verified('missing aggregate fields rejected',not c.summary_audit(changed,sample)['valid'])
original=(HERE.parent/'candidate-evaluation/package/eval_common.py').read_text()
oldns={'__name__':'frozen_common'};exec(compile(original,'<original>','exec'),oldns)
verified('success thresholds equal original task definition',c.THRESHOLDS==oldns['THRESHOLDS'])

with tempfile.TemporaryDirectory() as temp:
    root=Path(temp)
    paths={name:root/name for name in ('LEGACY','TRAIN','PRIOR','RECOVERY_V1','SLOT','SOURCE','HERE')}
    for p in paths.values():p.mkdir()
    train=paths['TRAIN'];prior=paths['PRIOR'];slot=paths['SLOT'];oldrec=paths['RECOVERY_V1']
    package=train/'package';package.mkdir();(train/'run/checkpoints').mkdir(parents=True)
    runtime=root/'existing-python';runtime.write_text('fixture runtime, not executable')
    source=paths['SOURCE']/'source.py';source.write_text('# fixture frozen project\n')
    c.write(package/'source-pins.json',{'source.py':{'bytes':source.stat().st_size,'sha256':c.sha(source)}})
    c.write(package/'manifest.json',{'identity':c.ORIGIN_ID,'files':{}})
    c.write(paths['HERE']/'origin-pins.json',{p.name:c.sha(p) for p in package.iterdir()})
    (prior/'package').mkdir();(prior/'package/fixture.py').write_text('# old evaluator\n')
    c.write(paths['HERE']/'prior-package-pins.json',{'fixture.py':c.sha(prior/'package/fixture.py')})
    c.write(paths['HERE']/'manifest.json',{'identity':'test-package'})
    (oldrec/'package').mkdir();(oldrec/'package/fixture.py').write_text('# failed recovery preserved\n')
    models={}
    for key in c.MODELS:
        p=train/'run/checkpoints'/('update_%06d.pt'%int(key))
        with p.open('wb') as f:f.truncate(10036060)
        models[key]=c.sha(p)
    c.write(train/'budget.json',dict(completed_updates=6000,attempted_updates=6000,max_updates=6000,pending=False))
    c.write(train/'run/progress.json',dict(status='TRAINING_BUDGET_COMPLETE'))
    c.write(train/'config.json',dict(package_identity=c.ORIGIN_ID))
    for rel in ('started.json','run/latest.json','run/best-training-metric.json'):c.write(train/rel,{})
    (train/'run/metrics.jsonl').write_text('{}\n')
    previous=dict(package_identity=c.PRIOR_ID,checkpoints=models,seeds=list(c.SEEDS),num_envs=128,
                  total_first_episodes=512,started_utc=agent.time.time()-60,deadline_utc=agent.time.time()+600,max_seconds=1800)
    c.write(prior/'started.json',previous)
    c.write(prior/'progress.json',dict(completed_groups=2,completed_first_episodes=256,group=3,model='6000',control_steps_in_group=0,error='do not share identical initial'))
    c.write(prior/'exit.json',dict(status='EVALUATION_ERROR_PRESERVE',original_training_files_unchanged=True))
    c.write(prior/'original-files.json',{})
    for seed in c.SEEDS:
        rows=[dict(row(i),seed=seed) for i in range(c.NUM_ENVS)]
        initial=dict(sha256='old-state',fields=['robot/root_link_pose_w'],scope='old')
        group=dict(status='EVALUATED',model='4500',seed=seed,checkpoint_sha256=models['4500'],
                   policy_and_normalizer_unchanged=True,episodes=rows,summary=c.summarize(rows),initial=initial)
        group['summary']['mean_survival_seconds']=math.nextafter(group['summary']['mean_survival_seconds'],math.inf)
        c.write(prior/('group-4500-%d.json'%seed),group)
        c.write(prior/('initial-seed-%d.json'%seed),initial)
    old_bytes={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.multiple(c,TRAIN=train,PRIOR=prior,MODELS=models))
        stack.enter_context(patch.multiple(agent,**paths,PYTHON=runtime,MODELS=models))
        baseline=agent.preflight()
        verified('real preflight accepts completed evidence with recorded last-bit differences',len(baseline)>10 and all(x['valid'] for x in agent.PREFLIGHT_DIAGNOSTICS['retained_4500_groups'].values()))
        snap=agent.snapshot()
        verified('read-only status returns both retained summaries and numeric diagnostics',snap['progress']['completed_first_episodes']==256 and all(x['reuse_audit']['differences'] for x in snap['retained_4500_groups'].values()))
        for seed in c.SEEDS:
            p=prior/('group-4500-%d.json'%seed);g=c.read(p);g['summary']['successes']+=1;c.write(p,g)
        try:agent.preflight();rejected=False
        except RuntimeError:rejected=True
        snap=agent.snapshot()
        verified('both invalid groups are collected in one failure without snapshot crashing',rejected and len(snap['retained_4500_groups'])==2 and all(not x['valid'] for x in snap['retained_4500_groups'].values()))
        for path,data in old_bytes.items():Path(path).write_bytes(data)
        c.write(oldrec/'started.json',{})
        try:agent.preflight();rejected=False
        except RuntimeError:rejected=True
        verified('an existing v1 start prevents a duplicate recovery',rejected)
        (oldrec/'started.json').unlink()
        expired=dict(previous,deadline_utc=agent.time.time()-1);c.write(prior/'started.json',expired)
        with patch.object(agent.subprocess,'Popen',side_effect=AssertionError('expired launch forbidden')):
            value=agent.launch()
        verified('real expired preflight collects 4500 without writing a new started marker',value['status']=='ORIGINAL_EVALUATION_DEADLINE_EXPIRED' and not (slot/'started.json').exists() and len(value['retained_4500_groups'])==2)
        c.write(prior/'started.json',previous)
        captured=[]
        def spawn(argv,**kw):
            captured.append((argv,kw))
            for fd in kw['pass_fds']:os.fstat(fd)
            return SimpleNamespace(pid=os.getpid())
        with patch.object(agent.subprocess,'Popen',side_effect=spawn),patch.object(agent,'identity',return_value=dict(pid=os.getpid(),start='test-process',state='R',group=os.getpid())):value=agent.launch()
        started=c.read(slot/'started.json')
        verified('real launch inherits deadline and passes all five locked descriptors',len(captured)==1 and len(captured[0][1]['pass_fds'])==5 and started['deadline_utc']==previous['deadline_utc'])
        verified('launch writes numeric audit before starting worker',(slot/'preflight-diagnostics.json').is_file() and started['new_first_episodes']==256)
        with patch.object(agent.subprocess,'Popen',side_effect=AssertionError('duplicate launch forbidden')):
            agent.launch()
        verified('repeated real launch attaches to prior start without spawning',True)
        verified('original train and prior evaluation evidence remain byte-identical',all(Path(p).read_bytes()==data for p,data in old_bytes.items()))

# Exercise the actual client's RPC parser on a well-formed remote failure.
remote=dict(error='fixture aggregate difference',started_marker_exists=False,state={'status':'NOT_STARTED','retained_4500_groups':{}},diagnostics={'field':'summary.values'})
stdout=('MD12_EVAL_ERROR='+json.dumps(remote)+'\n').encode()
with patch.object(ns['subprocess'],'run',return_value=SimpleNamespace(returncode=2,stdout=stdout,stderr=b'')):
    reply=ns['rpc']({'argv':lambda command:['fake-ssh',command]},['python3','-B','agent.py'])
verified('remote validation failure is not misreported as an SSH failure',reply['status']=='REMOTE_ERROR_PRESERVE' and reply['cloud_evaluation_state']=='NOT_STARTED' and reply['remote_error']==remote)
out=io.StringIO()
with contextlib.redirect_stdout(out):ns['show'](reply)
verified('terminal displays the specific remote failure',remote['error'] in out.getvalue())
with patch.dict(ns,request=lambda *_:None):pass
commands=[]
with patch.dict(ns,rpc=lambda ssh,args,data=None:commands.append(args)):
    ns['request']({},'status')
verified('status uses the same existing venv as evaluation','runtime=' in commands[0][3] and '.venv/bin/python' in commands[0][3] and "os.execv(runtime" in commands[0][3])
boot=(HERE/'bootstrap.py').read_text()
verified('bootstrap starts the existing venv, never installs dependencies',"os.execv(str(runtime)" in boot and '.venv/bin/python' in boot and 'pip install' not in boot)

receipt=json.loads((HERE/'offline-review.json').read_text())
receipt['checks']+=extra;receipt['count']=len(receipt['checks'])
receipt['scope']='Offline standard-library tests including real preflight, launch and status with isolated disk/process fixtures; no real cloud, Torch, CUDA, simulation or PPO.'
(HERE/'offline-review.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status='PASS',checks=receipt['count'],extra=len(extra),simulation_steps=0,PPO=0)))
