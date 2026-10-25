"""Pinned-source contract extraction only; no env or checkpoint deserialization."""
import ast,hashlib,json
from report_io import save

TARGETS={
 'mjlab/envs/manager_based_rl_env.py':('step','reset','_reset_idx'),
 'mjlab/managers/event_manager.py':('apply',),
 'mjlab/rl/runner.py':('save','load'),
 'src/mjlab_microduck/robot/head_contact.py':('physics_scene_pair',),
 'src/mjlab_microduck/tasks/microduck_head_contact_env_cfg.py':('make_microduck_head_contact_env_cfg',),
 'src/mjlab_microduck/tasks/mdp.py':('stage11_head_release_block','stage11_head_actor_state','stage11_head_critic_state','stage11_head_features_from_values'),
}

def collect(paths,out):
    records={};folder=out/'g11-source-contracts';folder.mkdir()
    for suffix,names in TARGETS.items():
        matches=[p for p in paths if str(p).endswith('/'+suffix)]
        if len(matches)!=1:raise RuntimeError('G11 pinned source path ambiguous: '+suffix)
        p=matches[0];raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        if digest!=paths[p]:raise RuntimeError('G11 source drift: '+suffix)
        source=raw.decode();tree=ast.parse(source);snippets={}
        for name in names:
            nodes=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name]
            if len(nodes)!=1:raise RuntimeError('G11 function ambiguous: '+name)
            n=nodes[0];calls=[dict(line=c.lineno,call=ast.unparse(c)) for c in sorted((x for x in ast.walk(n) if isinstance(x,ast.Call)),key=lambda x:(x.lineno,x.col_offset))]
            snippets[name]=dict(start_line=n.lineno,end_line=n.end_lineno,source=ast.get_source_segment(source,n),calls=calls)
        save(folder/(p.stem+'.json'),dict(path=suffix,sha256=digest,functions=snippets))
        records[suffix]=dict(sha256=digest,functions=list(snippets))
    result=dict(status='G11_PINNED_SOURCE_CONTRACTS_COLLECTED',files=records,
      physics_dt_s=.002,decimation=10,formal_step_dt_s=.020,
      integration_decisions=[
        'Preserve 10 physical macrosteps per formal step; apply interval/step events once with formal step_dt, not once per RK trial.',
        'Resolve native step reward/termination derived-state timing before adapter equivalence: stock forward is after reward and reset.',
        'Replace static head prototype release guards only through a separately reviewed formal task implementation.',
        'Checkpoint key migration exists in runner; observation meaning and normalizer compatibility remain unverified until a fixed checkpoint is inspected.'
      ],formal_environment_executed=False,checkpoint_loaded=False,release_guards_changed=False,
      native_integrations=0,G11='OPEN')
    save(out/'g11-source-contracts.json',result);return result
