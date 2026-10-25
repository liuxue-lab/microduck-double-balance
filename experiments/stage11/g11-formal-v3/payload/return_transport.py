"""Byte-exact ZIP transport through UTF-8-only connectors; <=512KiB raw shards."""
from pathlib import Path
import base64
import hashlib
import json

PART_BYTES = 512*1024
MAX_ARCHIVE_BYTES = 512*1024*1024


def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def decode_part(path,row):
    text=Path(path).read_bytes()
    if row['encoding']!='base64-ascii-lines' or len(text)!=row['text_bytes'] or digest(text)!=row['text_sha256']:
        raise ValueError('Encoded shard validation failed')
    text.decode('ascii')
    raw=base64.b64decode(b''.join(text.split()),validate=True)
    if base64.encodebytes(raw)!=text or len(raw)!=row['bytes'] or not 0<len(raw)<=PART_BYTES or digest(raw)!=row['sha256']:
        raise ValueError('Decoded shard validation failed')
    return raw

def export_parts(archive, destination, branch):
    archive, destination = Path(archive), Path(destination)
    raw = archive.read_bytes()
    if not 0 < len(raw) < MAX_ARCHIVE_BYTES:
        raise ValueError('Return archive must be nonempty and smaller than 512 MiB')
    destination.mkdir(exist_ok=True)
    parts = []
    for index, offset in enumerate(range(0,len(raw),PART_BYTES)):
        block = raw[offset:offset+PART_BYTES]
        encoded=base64.encodebytes(block)
        name = archive.name+'.part'+str(index).zfill(3)+'.base64.txt'
        path = destination/name
        if path.exists():
            if path.read_bytes() != encoded: raise ValueError('Existing shard differs; preserve evidence')
        else:
            with path.open('xb') as f: f.write(encoded)
        row=dict(file=name,index=index,offset=offset,bytes=len(block),sha256=digest(block),
                 encoding='base64-ascii-lines',text_bytes=len(encoded),text_sha256=digest(encoded))
        if decode_part(path,row)!=block:raise ValueError('Encoded shard round trip differs')
        parts.append(row)
    manifest = {'format':'microduck-return-ascii-parts-v2','file':archive.name,'bytes':len(raw),
                'sha256':digest(raw),'part_bytes_max':PART_BYTES,'parts':parts,'branch':branch,
                'scope':'Stage11 G11 real ManagerBasedRlEnv nominal interface matrix; G01-G10 evidence reused; inner raw reduced with live oracle checks retained; review required; independent archive branch; never merge'}
    path = destination/'return-manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest: raise ValueError('Existing shard manifest differs')
    else:
        with path.open('x') as f: json.dump(manifest,f,indent=2);f.write('\n')
    rebuilt = b''.join(decode_part(destination/p['file'],p) for p in parts)
    if rebuilt != raw: raise ValueError('Shard reconstruction differs')
    return manifest
