#!/usr/bin/env python3
"""Pack selected checkpoints/evidence; verify and safely extract on the laptop."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import tarfile


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def archive_plan(root,source,final,partial_reason):
    root=root.resolve(strict=True)
    if sha(source)!='c667f96607b68383047f23956ba58805920434465245ef8e32d1148b17fc65b7' or sha(final)!='a5ad0aedb500555b649c5d4b11aded833cbc0f932c1fff1a747e1492534c495c':
        raise ValueError('Stage 07 reference checkpoints differ')
    selected={source.resolve(),final.resolve()}
    reports=list(root.rglob('training.json'))
    if not reports: raise ValueError('No Stage 08 training records')
    for path in reports:
        record=json.loads(path.read_text())
        if record['status'] in ('RUNNING','INITIALIZING'):
            watchdog=path.parent/'watchdog.json'
            confirmed=watchdog.exists() and json.loads(watchdog.read_text()).get('child_exited') is True
            if not partial_reason or not confirmed:
                raise ValueError('A job is running or lacks a confirmed terminal status')
        if record.get('purpose')=='capacity': continue
        if record.get('latest_checkpoint'): selected.add(Path(record['latest_checkpoint']).resolve())
        best=path.parent/'best-development.json'
        if best.exists(): selected.add(Path(json.loads(best.read_text())['checkpoint']).resolve())
    selections=list(root.rglob('*selection*.json'))
    for path in selections:
        value=json.loads(path.read_text())
        if value.get('selection_locked'):
            for model in value['checkpoints']: selected.add(Path(model['path']).resolve())
    final_reports=[p for p in root.rglob('batch-evaluation.json') if json.loads(p.read_text()).get('status')=='PASS']
    if not final_reports and not partial_reason:
        raise ValueError('Final batch evaluation is missing; use an explicit partial-archive reason')
    files={}; omitted=[]
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.is_symlink(): continue
        relative=path.relative_to(root)
        if path.suffix in ('.lock','.tmp') or 'tensorboard' in relative.parts: continue
        if path.suffix=='.pt' and 'checkpoints' in relative.parts and path.resolve() not in selected:
            omitted.append({'path':relative.as_posix(),'size_bytes':path.stat().st_size,
                            'reason':'unselected periodic or discarded capacity model','retained_on_cloud':True})
            continue
        if path.suffix in ('.gz','.tar','.bundle'): continue
        files[relative.as_posix()]=path
    for path in selected:
        if path.is_relative_to(root): name=path.relative_to(root).as_posix()
        elif path==source.resolve(): name='references/update_001000.pt'
        elif path==final.resolve(): name='references/update_006000.pt'
        else: raise ValueError(f'Checkpoint outside this campaign: {path}')
        if not path.is_file(): raise ValueError(f'Checkpoint is missing: {path}')
        files[name]=path
        receipt=path.with_suffix('.json')
        if receipt.exists():
            if json.loads(receipt.read_text())['sha256']!=sha(path): raise ValueError('Checkpoint receipt differs')
            files[str(PurePosixPath(name).with_suffix('.json'))]=receipt
    manifest={'schema_version':1,'stage':8,'stage08_complete':False,
              'created_utc':datetime.now(timezone.utc).isoformat(),'original_cloud_root':str(root),
              'partial_reason':partial_reason,'video_review':'SEPARATE_REQUIRED',
              'omitted_checkpoints':omitted,
              'retention':'source 1000, reference 6000, each segment endpoint, development best and frozen selection; other periodic models remain on cloud',
              'files':[{'path':name,'size_bytes':path.stat().st_size,'sha256':sha(path)} for name,path in sorted(files.items())]}
    return files,manifest


def verify_archive(archive,expected_sha,output):
    if sha(archive)!=expected_sha: raise ValueError('Archive SHA-256 mismatch')
    if output.exists(): raise ValueError('Extraction destination already exists')
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        if any(not m.isfile() or PurePosixPath(m.name).is_absolute() or '..' in PurePosixPath(m.name).parts for m in members):
            raise ValueError('Archive contains a link or unsafe/non-regular member')
        if len({m.name for m in members})!=len(members): raise ValueError('Duplicate archive paths')
        manifest=json.load(tar.extractfile('MANIFEST.json'))
        expected={x['path']:x for x in manifest['files']}
        if set(expected)|{'MANIFEST.json'}!={m.name for m in members}: raise ValueError('Archive manifest membership differs')
        for name,item in expected.items():
            member=tar.getmember(name)
            if member.size!=item['size_bytes']: raise ValueError('Archive member size mismatch')
            h=hashlib.sha256()
            with tar.extractfile(member) as stream:
                for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
            if h.hexdigest()!=item['sha256']: raise ValueError(f'Archive member SHA mismatch: {name}')
        output.mkdir(parents=True)
        tar.extractall(output,filter='data')
    for name,item in expected.items():
        if sha(output/name)!=item['sha256']: raise ValueError('Extracted file verification failed')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='mode',required=True)
    pack=sub.add_parser('pack')
    pack.add_argument('--root',type=Path,required=True)
    pack.add_argument('--source',type=Path,required=True)
    pack.add_argument('--stage07-final',type=Path,required=True)
    pack.add_argument('--output',type=Path,required=True)
    pack.add_argument('--partial-reason')
    verify=sub.add_parser('verify')
    verify.add_argument('--archive',type=Path,required=True)
    verify.add_argument('--sha256',required=True)
    verify.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.mode=='pack':
        files,manifest=archive_plan(args.root,args.source,args.stage07_final,args.partial_reason)
        if args.output.exists(): raise ValueError('Archive already exists')
        with tarfile.open(args.output,'x:gz',compresslevel=3) as tar:
            for name,path in files.items(): tar.add(path,arcname=name,recursive=False)
            content=(json.dumps(manifest,indent=2)+'\n').encode()
            info=tarfile.TarInfo('MANIFEST.json'); info.size=len(content)
            tar.addfile(info,io.BytesIO(content))
        receipt={'path':str(args.output.resolve()),'sha256':sha(args.output),'size_bytes':args.output.stat().st_size}
        args.output.with_suffix(args.output.suffix+'.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt,indent=2))
    else:
        manifest=verify_archive(args.archive,args.sha256,args.output)
        print('Stage08LocalArchive=PASS')
        print('Stage08Artifacts='+str(args.output.resolve()))
        print('Reminder=必要文件已回传并通过哈希校验；确认没有其他任务后请关闭云 GPU 实例。')
        print('ShutdownPerformed=False')


if __name__=='__main__': main()
