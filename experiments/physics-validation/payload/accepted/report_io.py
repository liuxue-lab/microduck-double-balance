"""Atomic current-run evidence writes; no simulator imports or old-file cleanup."""
from pathlib import Path
import json,os

def encode(value):
    import numpy as np
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError('Unsupported report value: '+type(value).__name__)

def save(path,value):
    text=json.dumps(value,ensure_ascii=False,indent=2,default=encode,allow_nan=False)+'\n'
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_name(path.name+'.writing')
    # A leftover partial write is evidence. Refuse to overwrite it or retry.
    with pending.open('x',encoding='utf-8') as f:
        f.write(text);f.flush();os.fsync(f.fileno())
    os.replace(pending,path)
