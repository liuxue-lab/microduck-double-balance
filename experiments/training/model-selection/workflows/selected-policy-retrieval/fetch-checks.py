"""Exercise chunk transport and resumable disk publication without SSH or GPU."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shlex
from types import SimpleNamespace
import tempfile
from unittest.mock import patch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
source=(ROOT/'fetch_and_replay_selected_policy.py').read_bytes()
outer={'__name__':'test_outer'};exec(compile(source,'<outer>','exec'),outer)
ns=outer['fetch']
remote_bytes=ns['decode'](ns['REMOTE_B85'],ns['REMOTE_SHA'])
remote={'__name__':'test_remote'};exec(compile(remote_bytes,'<remote>','exec'),remote)
checks=[]
def ok(name,condition):
    assert condition,name
    checks.append(name)
digest=lambda b:hashlib.sha256(b).hexdigest()

ok('embedded SSH source matches original shared session helper',ns['decode'](ns['SSH_B85'],ns['SSH_SHA'])==(HERE.parent/'native-training/shared_ssh.py').read_bytes())
ok('model pin matches completed evaluation',ns['FILES']['model']['sha256']=='51821a28880aba74b80e6d5a388dfd7a82098bd3dca04a1e8947b4b2a8d5539a')
ok('remote only opens artifact files in read mode',all(call.args[0].value=='rb' for call in ast.walk(ast.parse(remote_bytes)) if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='open'))
ok('remote imports no GPU, process execution or training modules',all(not any(x in ast.unparse(node) for x in ('torch','mujoco','subprocess','warp')) for node in ast.walk(ast.parse(remote_bytes)) if isinstance(node,(ast.Import,ast.ImportFrom))))

with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);cloud=root/'cloud';cloud.mkdir();local=root/'local';local.mkdir()
    data=b'a'*(2*ns['CHUNK']+17);task=b'# preserved task fixture\n'
    files={}
    for key,body in [('model',data),('task',task)]:
        rel='frozen/'+key;path=cloud/rel;path.parent.mkdir(exist_ok=True);path.write_bytes(body)
        files[key]=dict(remote=rel,local='selected/'+key,max_bytes=len(body),bytes=len(body),sha256=digest(body))
    remote['ROOT']=cloud;remote['FILES']=files
    calls=[]
    def run(argv,**kwargs):
        # The real controller constructs and quotes the remote source/arguments.
        ok_source=argv[:3]==['python3','-B','-c'] and argv[3].encode()==remote_bytes
        assert ok_source
        args=argv[4:];calls.append(args)
        buffer=io.BytesIO();stream=io.TextIOWrapper(buffer,encoding='utf-8')
        try:
            with contextlib.redirect_stdout(stream):remote['main'](args)
            stream.flush();body=buffer.getvalue()
            return SimpleNamespace(returncode=0,stdout=body,stderr=b'')
        finally:stream.detach()
    ssh={'argv':shlex.split}
    real_files=ns['FILES'];ns['FILES']=files
    with patch.object(ns['subprocess'],'run',side_effect=run):
        info=ns['request'](ssh,['inventory'])
        ok('inventory reads fixed bytes and verifies every source hash',info['model']['bytes']==len(data) and info['model']['sha256']==digest(data))
        chunk=ns['request'](ssh,['chunk','model',0,ns['CHUNK']])
        ok('real chunk protocol validates length and checksum',chunk==data[:ns['CHUNK']])
        try:remote['locate']('../outside');rejected=False
        except RuntimeError:rejected=True
        ok('unknown or traversal artifact key rejected',rejected)
    def interrupted(argv,**kwargs):
        if argv[4:6]==['chunk','model'] and int(argv[6])>=ns['CHUNK']:raise TimeoutError('fixture disconnect')
        return run(argv,**kwargs)
    with patch.object(ns['subprocess'],'run',side_effect=interrupted),contextlib.redirect_stdout(io.StringIO()):
        try:ns['download'](local,ssh);stopped=False
        except TimeoutError:stopped=True
    partial=local/'selected/model.partial'
    ok('disconnect preserves only fully verified chunks',stopped and partial.read_bytes()==data[:ns['CHUNK']] and not (local/'selected/model').exists())
    partial.write_bytes(b'x'+data[1:ns['CHUNK']]);bad=partial.read_bytes()
    with patch.object(ns['subprocess'],'run',side_effect=run),contextlib.redirect_stdout(io.StringIO()):
        try:ns['download'](local,ssh);rejected=False
        except RuntimeError:rejected=True
    ok('corrupted partial prefix is rejected without truncation',rejected and partial.read_bytes()==bad)
    partial.write_bytes(data[:ns['CHUNK']]);calls.clear()
    before={str(p):p.read_bytes() for p in cloud.rglob('*') if p.is_file()}
    with patch.object(ns['subprocess'],'run',side_effect=run),contextlib.redirect_stdout(io.StringIO()):found=ns['download'](local,ssh)
    offsets=[int(a[2]) for a in calls if a[:2]==['chunk','model']]
    ok('resumption starts at preserved byte offset',offsets==[ns['CHUNK'],2*ns['CHUNK']])
    ok('verified files publish with full SHA256 and no partial remains',(local/'selected/model').read_bytes()==data and not partial.exists() and len(found)==2)
    ok('remote source bytes remain unchanged',all(Path(p).read_bytes()==body for p,body in before.items()))
    with patch.object(ns['subprocess'],'run',side_effect=AssertionError('must not connect again')),contextlib.redirect_stdout(io.StringIO()):
        again=ns['download'](local,ssh)
    ok('completed repeat requires no SSH and no redownload',again==found)
    (local/'selected/task').write_bytes(b'changed')
    try:ns['local_files'](local);rejected=False
    except RuntimeError:rejected=True
    ok('existing mismatched destination preserved',rejected and (local/'selected/task').read_bytes()==b'changed')
    ns['FILES']=real_files

bad_response=SimpleNamespace(returncode=0,stdout=b'MD12_DATA 3 incorrect\nabc',stderr=b'')
with patch.object(ns['subprocess'],'run',return_value=bad_response):
    try:ns['request']({'argv':lambda x:[]},['chunk','model',0,3]);rejected=False
    except RuntimeError:rejected=True
ok('invalid transport checksum fails before local write',rejected)
receipt=dict(status='PASS',checks=checks,count=len(checks),scope='Offline actual client/remote read functions with isolated files and an SSH subprocess substitute; no real network, GPU, simulation or PPO.')
(HERE/'fetch-offline-review.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status='PASS',checks=len(checks),PPO=0,simulation_steps=0)))
