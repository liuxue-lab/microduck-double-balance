"""Exclusive short compiler/cache paths, with ownership recorded beside evidence.

NVRTC's temporary-directory limit is stricter than Linux PATH_MAX. Evidence stays
in the full run path; a new private /tmp directory belongs to this process only.
No old directory is reused, deleted, shortened by symlink, or made writable.
"""
from pathlib import Path
import json
import os
import shutil
import stat
import tempfile
from integration.durable import require, write_json


def create(output, bootstrap):
    require(shutil.disk_usage('/tmp').free >= 1024**3,
            'Need at least 1 GiB free on /tmp for private compiler scratch')
    short = Path(tempfile.mkdtemp(prefix='md12-',dir='/tmp'))
    require(len(os.fsencode(short)) <= 40 and short.resolve() == short,
            'Compiler scratch must be a short real absolute path')
    row = dict(format='stage12-private-runtime-paths-v1',pid=os.getpid(),uid=os.getuid(),
               output=str(Path(output).absolute()),short_root=str(short),
               temp=str(short/'t'),warp_cache=str(short/'w'),cuda_cache=str(short/'c'),
               cache=str(short/'x'),matplotlib=str(short/'m'),torch=str(short/'h'),
               short_root_mode='0700',automatic_cleanup=False,
               evidence_location=str(bootstrap),native_integrations=0)
    for key in ('temp','warp_cache','cuda_cache','cache','matplotlib','torch'):
        Path(row[key]).mkdir(mode=0o700)
    write_json(short/'owner.json',row)
    write_json(Path(bootstrap)/'runtime-paths.json',row)
    validate(row,output)
    return row


def validate(row,output):
    short = Path(row['short_root'])
    require(row['format']=='stage12-private-runtime-paths-v1' and row['pid']==os.getpid() and
            row['uid']==os.getuid() and row['output']==str(Path(output).absolute()),
            'Compiler scratch owner identity differs')
    require(short.parent==Path('/tmp') and short.name.startswith('md12-') and
            len(os.fsencode(short))<=40 and not short.is_symlink() and short.resolve()==short,
            'Compiler scratch path differs')
    info=short.stat()
    require(stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode)==0o700 and info.st_uid==os.getuid(),
            'Compiler scratch ownership/mode differs')
    for key,leaf in (('temp','t'),('warp_cache','w'),('cuda_cache','c'),('cache','x'),('matplotlib','m'),('torch','h')):
        p=Path(row[key])
        require(p==short/leaf and p.is_dir() and not p.is_symlink() and len(os.fsencode(p))<=48,
                'Compiler cache child differs: '+key)
    require(json.loads((short/'owner.json').read_bytes())==row,'Compiler scratch receipt changed')
    return row


def for_owner(output):
    output=Path(output).absolute()
    path=output.with_name(output.name+'-bootstrap')/'runtime-paths.json'
    require(path.is_file() and not path.is_symlink(),'Compiler scratch receipt missing')
    return validate(json.loads(path.read_bytes()),output)
