import ast,base64,hashlib,json,zlib
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
digest=lambda data:hashlib.sha256(data).hexdigest()
data=json.loads((HERE/'selection-data.json').read_text())
report=ROOT/'upload/microduck-stage12-speed-refine-20261020T121521.060332Z-12132.json'
assert digest(report.read_bytes())=='48b261c65e6fd92241b480fcb797346bb62353f5513e8b2d185757f40e3f74b6'
files={
 'evaluation':dict(remote='native-speed-refine-v1/evaluations/segment-02/evaluation-result.json',local='evaluation/evaluation-result.json',max_bytes=128*1024,sha256=data['evaluation_sha256']),
 'model':dict(remote='native-speed-refine-v1/run/checkpoints/refine_000500.pt',local='checkpoints/refine_000500.pt',bytes=10036828,max_bytes=10036828,sha256=data['checkpoint']['sha256'])}
for name in ('native_task.py','source-pins.json'):
    body=(HERE.parent/'native-speed-refine-20261020/package'/name).read_bytes()
    files[name]=dict(remote='native-speed-refine-v1/package/'+name,local='task/'+name,bytes=len(body),max_bytes=len(body),sha256=digest(body))
rules=(HERE/'artifact-rules.py').read_text().replace('@@EXPECTED_SUMMARY@@',repr(data['summary']))
def fill(source,values):
    for key,value in values.items():source=source.replace('@@'+key+'@@',repr(value))
    ast.parse(source);return source
remote=fill((HERE/'remote.py').read_text(),dict(FILES=files,ARTIFACT_RULES=rules))
ssh=(HERE/'shared_ssh.py').read_bytes()
fetch=fill((HERE/'fetch.py').read_text(),dict(FILES=files,ARTIFACT_RULES=rules,REMOTE_B85=base64.b85encode(zlib.compress(remote.encode(),9)).decode(),
    REMOTE_SHA=digest(remote.encode()),SSH_B85=base64.b85encode(zlib.compress(ssh,9)).decode(),SSH_SHA=digest(ssh)))
package={p.name:p.read_text() for p in sorted((HERE/'package').glob('*.py'))}
for name,source in package.items():ast.parse(source,filename=name)
identity=digest(json.dumps({name:digest(source.encode()) for name,source in package.items()},sort_keys=True,separators=(',',':')).encode())
payload=json.dumps(package,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
replay=fill((HERE/'replay-controller.py').read_text(),dict(PACKAGE_B85=base64.b85encode(zlib.compress(payload,9)).decode(),PACKAGE_SHA=digest(payload),PACKAGE_ID=identity))
audit=(HERE/'evidence-audit.py').read_text();ast.parse(audit)
sources=json.dumps(dict(fetch=fetch,replay=replay,audit=audit),sort_keys=True,separators=(',',':')).encode()
source=fill((HERE/'controller.py').read_text(),dict(SOURCES_B85=base64.b85encode(zlib.compress(sources,9)).decode(),SOURCES_SHA=digest(sources)))
target=ROOT/'microduck-stage12-selected-speed500-v1-20261020.py';target.write_text(source)
(HERE/'files.json').write_text(json.dumps(files,indent=2)+'\n')
receipt=dict(controller=str(target),bytes=target.stat().st_size,sha256=digest(target.read_bytes()),source_bundle_sha256=digest(sources),
    replay_package_identity=identity,checkpoint_sha256=data['checkpoint']['sha256'],evaluation_sha256=data['evaluation_sha256'],
    selected_model='speed-refine +500; cumulative policy 5000',maximum_local_control_steps=1000,maximum_local_clips=2,
    maximum_local_wall_seconds=900,PPO=0,cloud_GPU_jobs=0,cloud_artifact_writes=0)
(HERE/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
