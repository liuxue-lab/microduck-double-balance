"""Original mjlab bridge code with only Warp's CPU storage transport substituted."""
import ast
from pathlib import Path
from types import ModuleType,SimpleNamespace as NS
import typing
import torch

SOURCE=Path(__file__).parent/'fixtures/sim-data.py'


class CpuWarpArray:
    def __init__(self,tensor):
        self.tensor=tensor;self.device=NS(is_cpu=True)
        self.size=tensor.numel();self.shape=tuple(tensor.shape);self.dtype=tensor.dtype


BRIDGE=ModuleType('reviewed_mjlab_bridge_fixture');BRIDGE.__file__=str(SOURCE)
BRIDGE.__dict__.update(torch=torch,wp=NS(array=CpuWarpArray,to_torch=lambda x:x.tensor,dtype_to_torch=lambda x:x))
for name in ('Any','Dict','Generic','Optional','Tuple','TypeVar'):
    BRIDGE.__dict__[name]=getattr(typing,name)
tree=ast.parse(SOURCE.read_bytes())
tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
exec(compile(tree,str(SOURCE),'exec'),BRIDGE.__dict__)


def wrap(tensor):return BRIDGE.TorchArray(CpuWarpArray(tensor))


def bridge_data(data):
    return BRIDGE.WarpBridge(NS(**{k:CpuWarpArray(v) if isinstance(v,torch.Tensor) else v
                                  for k,v in vars(data).items()}))
