"""Import closure checks; never construct an environment or perform physics."""
import ast
import importlib
import json
from pathlib import Path
import sys

EXTERNAL={'torch','numpy','warp','tensordict','mjlab','mjlab_microduck','onnx'}


def check_local_closure(folder):
    folder=Path(folder);modules={p.stem for p in folder.glob('*.py')};references=[]
    allowed=set(sys.stdlib_module_names)|EXTERNAL|modules
    for path in sorted(folder.glob('*.py')):
        for node in ast.walk(ast.parse(path.read_bytes(),filename=str(path))):
            names=[x.name for x in node.names] if isinstance(node,ast.Import) else \
                  [node.module] if isinstance(node,ast.ImportFrom) and node.module else []
            for name in names:
                if name.split('.')[0] not in allowed:raise RuntimeError('Missing packaged dependency: '+path.name+' -> '+name)
                if name.split('.')[0] in modules:references.append([path.name,name])
    return dict(local_modules=len(modules),local_imports=references,physics_steps=0,PPO_updates=0)


def check_runtime_imports():
    from common import HERE,SOURCE
    result=check_local_closure(HERE)
    sys.path.insert(0,str(SOURCE/'src'))
    names=('reward_spec','native_task','speed_task','headroom_task','baseline_runtime','model_limits_runtime',
           'robust_task','force_runtime','train_motor_monitor','train','evaluate','onnx')
    loaded={name:importlib.import_module(name) for name in names}
    # Exercise the formerly deferred import, with the configuration factory uncalled.
    if not callable(loaded['speed_task'].make_cfg) or not callable(loaded['headroom_task'].make_cfg):
        raise RuntimeError('Task configuration entry point missing')
    result['runtime_imported_modules']=list(loaded)
    return result


if __name__=='__main__':print(json.dumps(check_local_closure(Path(__file__).parent),sort_keys=True))
