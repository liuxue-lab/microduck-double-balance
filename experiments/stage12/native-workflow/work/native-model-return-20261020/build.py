import ast,base64,hashlib,json,zlib
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
digest=lambda b:hashlib.sha256(b).hexdigest()
files={
 'model':dict(remote='native-training-v2/run/checkpoints/update_004500.pt',local='checkpoints/update_004500.pt',bytes=10036060,max_bytes=10036060,sha256='df03202acf010680b63d3277ec879f63e2d4808e5d1286653923a314f125ac6a'),
 'group_20261021':dict(remote='native-evaluation-v1/group-4500-20261021.json',local='evaluation/group-4500-20261021.json',max_bytes=2*1024**2,sha256='7be7e52a27f9192626af8b9f2f7cf94ce06c99e3e86323e2f4055f4e1fdd48bf'),
 'group_20261022':dict(remote='native-evaluation-v1/group-4500-20261022.json',local='evaluation/group-4500-20261022.json',max_bytes=2*1024**2,sha256='5df8d19a0c50869bdd65a06ac18293d458da6f4e46304bebee732e240b9e2edc'),
}
for name in ('native_task.py','source-pins.json'):
 data=(HERE.parent/'native-training-v2/package'/name).read_bytes()
 files[name]=dict(remote='native-training-v2/package/'+name,local='task/'+name,bytes=len(data),max_bytes=len(data),sha256=digest(data))
remote=(HERE/'remote.py').read_text().replace('@@FILES@@',repr(files)).encode()
ssh=(HERE.parent/'native-training-v2/shared_ssh.py').read_bytes()
source=(HERE/'controller.py').read_text()
for name,value in dict(REMOTE_B85=base64.b85encode(zlib.compress(remote,9)).decode(),REMOTE_SHA=digest(remote),SSH_B85=base64.b85encode(zlib.compress(ssh,9)).decode(),SSH_SHA=digest(ssh),FILES=files).items():source=source.replace('@@'+name+'@@',repr(value))
ast.parse(source);ast.parse(remote)
target=ROOT/'microduck-stage12-fetch-4500-20261020.py';target.write_text(source)
(HERE/'files.json').write_text(json.dumps(files,indent=2)+'\n')
receipt=dict(controller=str(target),bytes=target.stat().st_size,sha256=digest(target.read_bytes()),remote_sha256=digest(remote),selected_model='4500',remote_writes=0,PPO=0,simulation_steps=0)
(HERE/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
