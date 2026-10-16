#!/usr/bin/env python3
"""Offline verified cloud input extraction and environment inventory; never PPO.

SSH clone/fetch and uv sync remain explicit shell commands outside this tool.
This file is copied into the isolated transfer branch for a bootstrap entrypoint.
"""
from pathlib import Path
import argparse
import base64
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import zipfile

sys.dont_write_bytecode = True


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def checked_zip(raw, manifest_name, expected_format=None):
    z = zipfile.ZipFile(io.BytesIO(raw))
    names = z.namelist()
    require(len(names) == len(set(names)) and z.testzip() is None, 'ZIP duplicate/CRC error')
    require(sum(i.file_size for i in z.infolist()) <= 100*1024**2, 'ZIP expanded size exceeds scope')
    for name in names:
        p = Path(name)
        require(not p.is_absolute() and '..' not in p.parts and '\\' not in name and not name.endswith('/'),
                'Unsafe ZIP member')
    index = json.loads(z.read(manifest_name))
    if expected_format:
        require(index['format'] == expected_format, 'Input format differs')
        index = index['files']
    require(set(index) | {manifest_name} == set(names), 'ZIP member roster differs')
    for name, row in index.items():
        data = z.read(name)
        require(len(data) == row['bytes'] and sha(data) == row['sha256'], 'ZIP identity differs: '+name)
    return z


def extract_transfer(folder, expected_commit, output):
    require(re.fullmatch('[0-9a-f]{40}', expected_commit) is not None, 'Full expected commit required')
    env = {k:v for k,v in os.environ.items() if not k.startswith('GIT_')}
    env.update(GIT_OPTIONAL_LOCKS='0',GIT_CONFIG_GLOBAL='/dev/null',GIT_CONFIG_NOSYSTEM='1')
    actual = subprocess.run(['git','-C',str(folder),'rev-parse','HEAD'],env=env,check=True,
                            capture_output=True,text=True,timeout=30).stdout.strip()
    require(actual == expected_commit, 'Transfer commit mismatch; no extraction')
    # Also verify worktree files against that immutable commit, including this
    # bootstrap, rather than trusting an edited manifest in the local checkout.
    def committed(name):
        require('/' not in name and '\\' not in name and name not in ('','.','..'), 'Unsafe transfer member')
        raw = subprocess.run(['git','-C',str(folder),'show',expected_commit+':'+name],env=env,
                             check=True,capture_output=True,timeout=30).stdout
        p = folder / name
        require(p.is_file() and not p.is_symlink() and p.read_bytes() == raw, 'Transfer file edited: '+name)
        return raw
    committed('cloud-stage12.py')
    manifest = json.loads(committed('return-manifest.json'))
    require(manifest['format'] == 'microduck-return-ascii-parts-v2' and manifest['bytes'] <= 96*1024**2,
            'Transfer format/size differs')
    blocks = []
    offset = 0
    for i,row in enumerate(manifest['parts']):
        require(row['index'] == i and row['offset'] == offset and row['bytes'] <= 512*1024, 'Part order differs')
        encoded = committed(row['file'])
        require(len(encoded) == row['text_bytes'] and sha(encoded) == row['text_sha256'], 'Encoded SHA differs')
        raw = base64.b64decode(b''.join(encoded.split()),validate=True)
        require(base64.encodebytes(raw) == encoded and len(raw) == row['bytes'] and sha(raw) == row['sha256'],
                'Decoded SHA/representation differs')
        blocks.append(raw)
        offset += len(raw)
    raw = b''.join(blocks)
    require(len(raw) == manifest['bytes'] and sha(raw) == manifest['sha256'], 'Return archive differs')
    with checked_zip(raw,'RETURN-MANIFEST.json') as returned:
        report = json.loads(returned.read('evidence/stage12-report.json'))
        require(report['status'] == 'LOCAL_NATIVE_PILOT_PASS_CLOUD_BUDGET_PENDING' and report['preservation'] == 'PASS',
                'Local pilot not accepted; stop before cloud environment installation')
        package = json.loads(returned.read('program/PACKAGE.json'))
        require({n.removeprefix('program/') for n in returned.namelist() if n.startswith('program/')} ==
                set(package['files']) | {'PACKAGE.json'}, 'Program file roster differs')
        for name,row in package['files'].items():
            data = returned.read('program/'+name)
            require(len(data) == row['bytes'] and sha(data) == row['sha256'], 'Program identity differs')
        raw_inputs = returned.read('evidence/cloud-inputs.zip')
        identity = json.loads(returned.read('evidence/cloud-inputs.json'))
        require(len(raw_inputs) == identity['bytes'] and sha(raw_inputs) == identity['sha256'], 'Cloud inputs differ')
        with checked_zip(raw_inputs,'SOURCE-BUNDLE-MANIFEST.json','stage12-portable-inputs-v2') as inputs:
            require(not output.exists() and not output.is_symlink(), 'Cloud output already exists; preserve it')
            for parent in output.parents:
                require(not parent.is_symlink(),'Cloud output ancestor is symlinked')
            output.mkdir(parents=True)
            (output/'started-input-extraction.json').write_text(json.dumps(dict(commit=actual,archive_sha256=sha(raw),
                    PPO=0,cloud_training=0,no_retry=True),indent=2)+'\n')
            for name in returned.namelist():
                if name.startswith('program/'):
                    dest = output/name
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    with dest.open('xb') as f:
                        f.write(returned.read(name))
            for name in inputs.namelist():
                dest = output/'inputs'/name
                dest.parent.mkdir(parents=True,exist_ok=True)
                with dest.open('xb') as f:
                    f.write(inputs.read(name))
            (output/'local-stage12-report.json').write_bytes(returned.read('evidence/stage12-report.json'))
            (output/'cloud-inputs.json').write_bytes(returned.read('evidence/cloud-inputs.json'))
    print('Stage12CloudInputs=VERIFIED_EXTRACTED; PPO=0 cloud_training=0')
    print('Project='+str(output/'inputs/project'))
    print('InstallOnlyCommand=uv sync --locked --python 3.12 --project '+str(output/'inputs/project'))
    print('No automatic installation, GPU start, training, push, PR, merge or shutdown.')


def environment_check(workspace, hours, spend, currency):
    require(math.isfinite(hours) and math.isfinite(spend) and hours > 0 and spend > 0 and currency,
            'Explicit finite cloud time/spend/currency required')
    project = workspace/'inputs/project'
    require(sys.version_info[:2] == (3,12), 'Pinned Python 3.12 required')
    require(Path(sys.prefix).absolute() == (project/'.venv').absolute(), 'Use the new project .venv interpreter without resolving it')
    marker = workspace/'started-environment-check.json'
    require(not marker.exists(), 'Cloud environment check already started; preserve evidence')
    with marker.open('x') as f:
        json.dump(dict(gpu_hours=hours,spend_limit=spend,currency=currency,PPO=0,cloud_training=0),f)
    result = dict(PPO=0,cloud_training=0,native_integrations=0,native_forward=0,
                  training_ready=False,errors=[])
    try:
        root = workspace/'program'
        sys.path[:0] = [str(root),str(root/'baseline/g11')]
        from worker_g11 import verify_sources
        result['source_pins'] = len(verify_sources(project))
        import importlib.metadata as metadata
        manifest = json.loads((workspace/'inputs/SOURCE-BUNDLE-MANIFEST.json').read_bytes())
        prefixes = {'rsl_rl/':('rsl-rl-lib','rsl_rl/'), 'mjlab/':('mjlab','mjlab/')}
        for name,row in manifest['files'].items():
            if name.startswith('project/'):
                p = workspace/'inputs'/name
            else:
                prefix = next(k for k in prefixes if name.startswith(k))
                dist,_ = prefixes[prefix]
                p = Path(metadata.distribution(dist).locate_file(name))
            require(p.is_file() and not p.is_symlink() and len(p.read_bytes()) == row['bytes'] and
                    sha(p.read_bytes()) == row['sha256'], 'Cloud source differs: '+name)
        require(metadata.version('rsl-rl-lib') == '5.0.1' and metadata.version('tensordict') == '0.10.0',
                'Runner/tensor versions differ')
        import torch
        require(torch.__version__.split('+')[0] == '2.9.1' and torch.version.cuda is not None and
                torch.cuda.is_available(), 'Reviewed Torch CUDA runtime required')
        result.update(torch=torch.__version__,cuda_runtime=torch.version.cuda,
                      devices=[dict(index=i,name=torch.cuda.get_device_name(i),
                                    total_memory=torch.cuda.get_device_properties(i).total_memory)
                               for i in range(torch.cuda.device_count())],
                      status='CLOUD_ENVIRONMENT_IDENTITY_PASS_NATIVE_AND_PPO_PENDING')
    except BaseException as exc:
        import traceback
        result.update(status='CLOUD_ENVIRONMENT_STOPPED_PRESERVE',errors=[dict(error=repr(exc),traceback=traceback.format_exc())])
    with (workspace/'cloud-environment-report.json').open('x') as f:
        json.dump(result,f,indent=2,sort_keys=True)
    print(json.dumps(result,sort_keys=True))
    print('Cloud GPU: if no further authorized cloud work remains, retrieve/verify evidence then shut down manually.')
    return int(bool(result['errors']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--transfer',type=Path)
    g.add_argument('--verify-environment',type=Path)
    p.add_argument('--expected-commit')
    p.add_argument('--output',type=Path)
    p.add_argument('--gpu-hours',type=float,default=0)
    p.add_argument('--spend-limit',type=float,default=0)
    p.add_argument('--currency',default='')
    args = p.parse_args()
    if args.transfer:
        require(args.expected_commit and args.output,'Expected commit and fresh output directory required')
        extract_transfer(args.transfer,args.expected_commit,args.output)
    else:
        return environment_check(args.verify_environment,args.gpu_hours,args.spend_limit,args.currency)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError,ValueError,RuntimeError,KeyError,subprocess.SubprocessError) as exc:
        print('Stage12Cloud=STOP: '+str(exc),file=sys.stderr)
        raise SystemExit(1)
