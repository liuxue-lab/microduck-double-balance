import ast
import base64
import hashlib
import json
import re
from pathlib import Path
import zlib

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
PACKAGE=HERE/'package'
digest=lambda b:hashlib.sha256(b).hexdigest()
origin=HERE.parent/'native-training/package'
pins={p.name:digest(p.read_bytes()) for p in sorted(origin.iterdir()) if p.is_file() and p.suffix in ('.py','.json')}
(PACKAGE/'origin-pins.json').write_text(json.dumps(pins,sort_keys=True,indent=2)+'\n')
files={p.name:p.read_bytes() for p in sorted(PACKAGE.iterdir()) if p.is_file() and p.name!='manifest.json'}
for name,value in files.items():
    if name.endswith('.py'):ast.parse(value,filename=name)
identity=digest(json.dumps({k:digest(v) for k,v in sorted(files.items())},sort_keys=True).encode())
manifest=dict(identity=identity,files={k:dict(bytes=len(v),sha256=digest(v)) for k,v in files.items()})
files['manifest.json']=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
(PACKAGE/'manifest.json').write_bytes(files['manifest.json'])
payload=zlib.compress(json.dumps(dict(files={k:base64.b64encode(v).decode() for k,v in files.items()})).encode(),9)
ssh=(HERE/'shared_ssh.py').read_bytes();boot=(HERE/'bootstrap.py').read_bytes();ast.parse(boot)
replacements=dict(SSH_B85=base64.b85encode(zlib.compress(ssh,9)).decode(),SSH_SHA=digest(ssh),
    BOOT_B85=base64.b85encode(zlib.compress(boot,9)).decode(),BOOT_SHA=digest(boot),
    PAYLOAD_B85=base64.b85encode(payload).decode(),PAYLOAD_SHA=digest(payload),PACKAGE_ID=identity)
source=(HERE/'controller.py').read_text()
for name,value in replacements.items():source=source.replace('@@'+name+'@@',repr(value))
assert not re.search(r'@@[A-Z_]+@@',source);ast.parse(source)
target=ROOT/'refine_baseline.py';target.write_text(source)
receipt=dict(package_identity=identity,controller=str(target),controller_bytes=target.stat().st_size,
    controller_sha256=digest(target.read_bytes()),transport_bytes=len(payload),origin_checkpoint=4500,
    num_envs=4096,learning_rate=1e-5,maximum_new_PPO_attempts=1000,maximum_wall_seconds=7200,
    segment_updates=250,maximum_evaluation_segments=4,first_episodes_per_candidate=256,reused_baseline_first_episodes=256,
    consecutive_non_improvement_patience=2,repeated_smoke_runs=0,cloud_run_executed=False)
(HERE/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
