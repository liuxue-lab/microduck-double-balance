import ast,base64,hashlib,json,zlib
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
digest=lambda data:hashlib.sha256(data).hexdigest()
package={p.name:p.read_text() for p in sorted((HERE/'package').glob('*.py'))}
for name,source in package.items():ast.parse(source,filename=name)
identity=digest(json.dumps({name:digest(source.encode()) for name,source in package.items()},sort_keys=True,separators=(',',':')).encode())
payload=json.dumps(package,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
source=(HERE/'controller.py').read_text()
for name,value in dict(PACKAGE_B85=base64.b85encode(zlib.compress(payload,9)).decode(),PACKAGE_SHA=digest(payload),PACKAGE_ID=identity).items():
    source=source.replace('@@'+name+'@@',repr(value))
ast.parse(source)
target=ROOT/'microduck-stage12-replay-4500-v1-20261020.py';target.write_text(source)
receipt=dict(controller=str(target),bytes=target.stat().st_size,sha256=digest(target.read_bytes()),package_identity=identity,
    selected_model='4500',maximum_control_steps=1000,maximum_clips=2,maximum_wall_seconds=900,PPO=0,cloud_connections=0)
(HERE/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
