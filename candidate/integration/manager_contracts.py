"""Independent interface checks before creating any model/CUDA environment."""
from types import SimpleNamespace
from integration.durable import require,write_json


def preflight(out):
    from mjlab.managers.recorder_manager import NullRecorderManager,RecorderManager
    from mjlab.managers.curriculum_manager import NullCurriculumManager
    from integration.owner_roster import recorder_terms
    from pathlib import Path
    import traceback
    checks=[]
    def null_recorder():
        m=NullRecorderManager()
        require(vars(m)==dict(active_terms=[],cfg=None),'Null recorder source schema differs')
        require(recorder_terms(m)==(),'Null recorder must have zero term owners')
        for name in ('record_pre_reset','record_post_reset'):getattr(m,name)([])
        m.record_post_step();m.close()
        require(vars(m)==dict(active_terms=[],cfg=None),'Null recorder changed state')
    def ordinary_recorder():
        m=RecorderManager({},SimpleNamespace(num_envs=1,device='cpu'))
        require(recorder_terms(m) is m._terms and m._terms==[],'Ordinary recorder roster lost')
        m.close()
    def null_curriculum():
        m=NullCurriculumManager();m.compute([])
        require(m.reset([])=={} and m.active_terms==[] and m._curriculum_state=={} and m.cfg is None,
                'Null curriculum schema or behavior differs')
    def unknown_rejected():
        try:recorder_terms(SimpleNamespace(active_terms=[],_terms=[]))
        except RuntimeError:return
        raise RuntimeError('Unknown recorder manager silently accepted')
    def changed_null_rejected():
        m=NullRecorderManager();m._terms=[]
        try:recorder_terms(m)
        except RuntimeError:return
        raise RuntimeError('Synthetic null-manager fields silently accepted')
    for name,fn in (('real-null-recorder',null_recorder),('real-empty-ordinary-recorder',ordinary_recorder),
                    ('real-null-curriculum',null_curriculum),('unknown-recorder-rejected',unknown_rejected),
                    ('changed-null-schema-rejected',changed_null_rejected)):
        try:fn();checks.append(dict(name=name,passed=True))
        except Exception as exc:checks.append(dict(name=name,passed=False,error=repr(exc),traceback=traceback.format_exc()))
    report=dict(passed=all(r['passed'] for r in checks),checks=checks,
        scope='Real manager classes on isolated CPU interface objects; not physical acceptance',
        environment_constructions=0,native_integrations=0,CPU_forward=0,GPU_forward=0,PPO=0,cloud_training=0)
    write_json(Path(out)/'manager-interface-preflight.json',report)
    require(report['passed'],'Manager interface preflight failed; all independent failures recorded before native construction')
    return report
