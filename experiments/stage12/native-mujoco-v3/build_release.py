"""Build the isolated additive Git bundle from the verified v2 parent.

This writes only to the supplied object store, this package's manifest, and the
supplied release directory. It never checks out a branch, trains, or pushes.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zlib

PARENT = '292e8c6a8414d76cd75883e451ef95523c9a4ae3'
PREFIX = 'experiments/stage12/native-mujoco-v3'
NAME = 'microduck-stage12-native-mujoco-v3-20261025.py'


def build(gitdir, source, output):
    objects = set()
    def git(*args, data=None, env=None):
        return subprocess.check_output(['git','--git-dir',str(gitdir),*args],input=data,env=env)
    def put(kind, data):
        value=git('hash-object','-t',kind,'-w','--stdin',data=data).decode().strip()
        objects.add(value)
        return value
    def tree(rows):
        rows.sort(key=lambda r:(r[1]+('/' if r[0]=='40000' else '')).encode())
        return put('tree',b''.join((mode+' '+name).encode()+b'\0'+bytes.fromhex(sha)
                                  for mode,name,sha in rows))
    def entries(sha):
        raw=git('cat-file','tree',sha);rows=[]
        while raw:
            header,raw=raw.split(b'\0',1);mode,name=header.decode().split(' ',1)
            rows.append((mode,name,raw[:20].hex()));raw=raw[20:]
        return rows
    def replace(rows, name, sha, absent=False):
        if absent and any(r[1]==name for r in rows):raise RuntimeError('Existing v3 directory in parent')
        return tree([r for r in rows if r[1]!=name]+[('40000',name,sha)])
    def directory(folder):
        rows=[]
        for path in sorted(folder.iterdir()):
            if path.name.startswith('.') or path.name in ('__pycache__','release','bundle-objects.git'):
                continue
            if path.is_symlink():raise RuntimeError('Source symlink: '+str(path))
            if path.is_dir():rows.append(('40000',path.name,directory(path)))
            elif path.suffix!='.pyc':rows.append(('100644',path.name,put('blob',path.read_bytes())))
        return tree(rows)
    package=source/'package'
    pins={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
          for p in sorted(package.iterdir()) if p.is_file() and p.name!='manifest.json' and p.suffix!='.pyc'}
    identity=hashlib.sha256(json.dumps(pins,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (package/'manifest.json').write_text(json.dumps(dict(identity=identity,files=pins),indent=2)+'\n')
    own=directory(source)
    root=entries(git('show','-s','--format=%T',PARENT).decode().strip())
    experiments=entries(next(r[2] for r in root if r[1]=='experiments'))
    stage12=entries(next(r[2] for r in experiments if r[1]=='stage12'))
    stage=replace(stage12,'native-mujoco-v3',own,absent=True)
    experiment=replace(experiments,'stage12',stage)
    final=replace(root,'experiments',experiment)
    env=dict(os.environ,GIT_AUTHOR_NAME='Codex',GIT_AUTHOR_EMAIL='codex@openai.com',
        GIT_COMMITTER_NAME='Codex',GIT_COMMITTER_EMAIL='codex@openai.com',
        GIT_AUTHOR_DATE='2026-10-25T05:00:00+00:00',GIT_COMMITTER_DATE='2026-10-25T05:00:00+00:00')
    commit=git('commit-tree',final,'-p',PARENT,data=(
        'Warm the actual HIL reference before first release and preserve deadline evidence\n').encode(),env=env).decode().strip()
    objects.add(commit)
    pack=git('pack-objects','--stdout','--delta-base-offset',data=('\n'.join(sorted(objects))+'\n').encode())
    header=('# v2 git bundle\n-'+PARENT+' native-mujoco-v2 prerequisite\n'+commit+' refs/heads/double-balance\n\n').encode()
    bundle=header+pack
    release=dict(commit=commit,package_identity=identity,bundle_sha256=hashlib.sha256(bundle).hexdigest(),
        bundle=base64.b85encode(zlib.compress(bundle,9)).decode())
    output.mkdir(parents=True,exist_ok=True)
    (output/'native-mujoco-v3.bundle').write_bytes(bundle)
    script=output/NAME
    script.write_text('RELEASE_DATA='+repr(release)+'\n'+(source/'controller.py').read_text())
    receipt=dict(parent=PARENT,commit=commit,package_identity=identity,
        bundle_sha256=release['bundle_sha256'],script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        script_bytes=script.stat().st_size,source_prefix=PREFIX,script=str(script.resolve()))
    (output/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--git-dir',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.git_dir.resolve(),args.source.resolve(),args.output.resolve()),indent=2))
