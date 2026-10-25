from pathlib import Path
import base64,hashlib,json,zlib
ROOT=Path(__file__).resolve().parent
P=ROOT/'package'
files={p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(P.iterdir()) if p.is_file() and p.name!='manifest.json'}
identity=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
manifest={'identity':identity,'files':files,'scope':'native-training-v2; old strict audit remains failed/preserved'}
(P/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
cfg=dict(package_identity=identity,num_envs=2048,max_updates=6000,max_hours=8,seed=20261020,smoke_envs=64,smoke_updates=5,smoke_max_minutes=30,
         historical_validation_PPO_reservations=4,additional_pretraining_validation_runs=1,
         intent='one native smoke then formal joint finetuning; retained Stage08 E; no lower-only phase')
blob=zlib.compress(json.dumps({'config':cfg,'files':{p.name:base64.b85encode(p.read_bytes()).decode() for p in sorted(P.iterdir()) if p.is_file()}},sort_keys=True,separators=(',',':')).encode(),9)
ssh=ROOT/'shared_ssh.py'
source=(ROOT/'controller-template.py').read_text()
values=dict(PACKAGE_B85=base64.b85encode(blob).decode(),PACKAGE_SHA=hashlib.sha256(blob).hexdigest(),
            BOOTSTRAP_B85=base64.b85encode(zlib.compress((ROOT/'bootstrap.py').read_bytes(),9)).decode(),
            SSH_B85=base64.b85encode(zlib.compress(ssh.read_bytes(),9)).decode(),SSH_SHA=hashlib.sha256(ssh.read_bytes()).hexdigest(),PACKAGE_ID=identity,PROBE_B85=base64.b85encode(zlib.compress((P/'common.py').read_bytes()+b'\n'+(P/'probe.py').read_bytes().replace(b'from common import SLOT, LEGACY_SLOT, read, alive, has_campaign, validation_remaining',b''),9)).decode())
for key,val in values.items():source=source.replace('@@'+key+'@@',repr(val))
compile(source,'controller.py','exec')
target=ROOT.parent.parent/'train.py';target.write_text(source);target.chmod(0o755)
receipt=dict(package_identity=identity,package_transport_bytes=len(blob),bootstrap_bytes=(ROOT/'bootstrap.py').stat().st_size,
             controller=str(target),controller_bytes=target.stat().st_size,controller_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
             config=cfg)
(ROOT/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
