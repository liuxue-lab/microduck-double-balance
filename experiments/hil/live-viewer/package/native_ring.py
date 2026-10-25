"""Single-CPU-world ring copies; original DelayBuffer RNG code stays intact."""
import importlib
from pathlib import Path
import torch
from deploy_contract import require,sha

PINS={
 'mjlab.utils.buffers.delay_buffer':'845e68da9e681adee1e93f90b4b1937c282a27f82a147f1fe3c19d83204049b5',
 'mjlab.utils.buffers.circular_buffer':'6a96cb79afbfb2d1234c3ee8e7531db371262673ddc509da7f3e66410ef2ae13'}


class CpuRing:
    def __init__(self,delay):
        for name,pin in PINS.items():
            module=importlib.import_module(name)
            require(sha(Path(module.__file__))==pin,'Unreviewed delay source: '+name)
        from mjlab.utils.buffers import DelayBuffer
        require(type(delay) is DelayBuffer and delay.batch_size==1 and str(delay.device)=='cpu',
                'CPU ring requires original one-world CPU delay')
        self.delay=delay

    def sample(self,data):
        d=self.delay;b=d._buffer
        require(data.shape[0]==1 and data.device.type=='cpu','Invalid CPU ring sample')
        if b._buffer is None:
            b._pointer=-1
            b._buffer=torch.empty((b._max_len,*data.shape),dtype=data.dtype,device='cpu')
        b._pointer=(b._pointer+1)%b._max_len
        b._buffer[b._pointer].copy_(data)
        if int(b._num_pushes[0])==0:b._buffer.copy_(data.unsqueeze(0))
        b._num_pushes.add_(1)
        # Exactly the original random draws and integer delay state updates.
        d._update_lags()
        valid=max(0,min(int(d._current_lags[0]),int(b._num_pushes[0])-1,b._max_len-1))
        value=b._buffer[(b._pointer-valid)%b._max_len]
        return value if valid==0 else value.clone()
