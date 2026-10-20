from pathlib import Path
import base64
import hashlib
import json
import zlib

ROOT=Path(__file__).resolve().parent
ssh=(ROOT.parent/'native-training-v2/shared_ssh.py').read_bytes()
remote=(ROOT/'finish_read.py').read_text()+'''\nimport sys
operation=sys.argv[2]
if operation not in ('status','collect'):
    raise ValueError('Read-only operation required')
value=snapshot(sys.argv[1]) if operation=='status' else collect_finished(sys.argv[1])
print('MD12_FINISH_STATE='+json.dumps(value,ensure_ascii=False,allow_nan=False))
'''
source=(ROOT/'controller.py').read_text()
for key,value in dict(SSH_B85=base64.b85encode(zlib.compress(ssh,9)).decode(),
                     SSH_SHA=hashlib.sha256(ssh).hexdigest(),
                     REMOTE_B85=base64.b85encode(zlib.compress(remote.encode(),9)).decode(),
                     REMOTE_SHA=hashlib.sha256(remote.encode()).hexdigest()).items():
    source=source.replace('@@'+key+'@@',repr(value))
compile(source,'finish-controller','exec')
compile(remote,'finish-remote','exec')
output=ROOT.parent.parent/'microduck-stage12-finish-readonly-20261020.py'
output.write_text(source);output.chmod(0o755)
receipt=dict(controller=str(output),bytes=output.stat().st_size,
             sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
             remote_source_sha256=hashlib.sha256(remote.encode()).hexdigest(),
             upstream_metric_source='https://github.com/mujocolab/mjlab/blob/v1.3.0/src/mjlab/managers/metrics_manager.py',
             upstream_metric_sha256=hashlib.sha256((ROOT/'reference/mjlab-v1.3.0-metrics_manager.py').read_bytes()).hexdigest(),
             cloud_changes=0,PPO_updates=0,simulation_steps=0)
(ROOT/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
