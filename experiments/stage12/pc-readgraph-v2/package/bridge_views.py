"""Expose reviewed mjlab TorchArray fields without copying their live storage."""
import importlib
from pathlib import Path

from deploy_contract import require,sha
from bam_graph import storage_signature

BRIDGE_SHA='bc73da4811fc44336f9016d826ef1b2044b8cb0830f2ad01a06b9b84ffa8b4c6'


class BridgeViews:
    def __init__(self,torch):
        self.torch=torch;module=importlib.import_module('mjlab.sim.sim_data')
        path=Path(module.__file__)
        require(sha(path)==BRIDGE_SHA,'Unreviewed mjlab TorchArray bridge source')
        self.wrapper_type=module.TorchArray
        self.source=dict(path=str(path),sha256=BRIDGE_SHA)
        # Keep each original Warp owner alive for the entire captured lifetime,
        # even if the bridge's wrapper cache is later cleared.
        self.owners={}

    def view(self,value,name):
        if isinstance(value,self.torch.Tensor):return value
        require(type(value) is self.wrapper_type,'Unsupported live array type: '+name)
        tensor=value[...]
        require(isinstance(tensor,self.torch.Tensor),'TorchArray did not expose a Tensor: '+name)
        require(storage_signature(tensor)==storage_signature(value._tensor)
                and tensor.untyped_storage().data_ptr()==value._tensor.untyped_storage().data_ptr(),
                'TorchArray conversion copied or changed live storage: '+name)
        self.owners.setdefault(name,value)
        return tensor

    def stats(self):
        return dict(source=self.source,unwrapped_fields=sorted(self.owners),
            conversion='original TorchArray[...] shared-storage view; no clone/as_tensor/device copy',
            original_warp_owners_retained=True)
