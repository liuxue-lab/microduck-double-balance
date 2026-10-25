"""Create publication demos from the original double-balance recordings.

Python 3 + FFmpeg. Both camera crops retain native pixels and original speed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

SOURCES = {
    'balance': {'case': 'quiet-2407', 'sha256': '5feb95b1011441723ee74495c9ba5e29f6be967ce84fe638c7597f9931c3ea77', 'bytes': 4617084},
    'disturbance': {'case': 'regression-2500', 'sha256': 'b93968c13e3492fcac34585d8c2b58d14a0360a237c2b9ef4a0a39a3ccac1c8f', 'bytes': 5563728},
}
START, DURATION = 6, 12
CROPS = {'whole_body': [80, 96, 480, 320], 'head_detail': [720, 144, 480, 320]}
GRAPH = '[0:v]{fps}split=2[a][b];[a]crop=480:320:80:96[left];[b]crop=480:320:720:144[right];[left][right]hstack=inputs=2[v]'


def identity(path):
    data = path.read_bytes()
    return {'file': path.name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--manifest-only', action='store_true', help='Record already encoded outputs without re-encoding')
    args = parser.parse_args()
    sources = {}
    for name, pin in SOURCES.items():
        path = args.source_dir / (pin['case'] + '-recorded-hil.mp4')
        actual = identity(path)
        if actual['bytes'] != pin['bytes'] or actual['sha256'] != pin['sha256']:
            raise SystemExit('Unexpected source bytes: ' + str(path))
        sources[name] = path
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name, path in sources.items():
        outputs = []
        for extension, fps, options in [
            ('webp', 20, ['-c:v', 'libwebp_anim', '-quality', '90', '-compression_level', '6', '-loop', '0']),
            ('mp4', 50, ['-c:v', 'libx264', '-preset', 'slow', '-crf', '15', '-pix_fmt', 'yuv420p', '-movflags', '+faststart']),
        ]:
            output = args.output_dir / (name + '-demo.' + extension)
            if not args.manifest_only:
                graph = GRAPH.format(fps='fps=20,' if fps == 20 else '')
                subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-ss', str(START),
                    '-t', str(DURATION), '-i', str(path), '-filter_complex', graph, '-map', '[v]', '-an',
                    *options, str(output)], check=True)
            outputs.append({**identity(output), 'frames_per_second': fps, 'width': 960, 'height': 320})
        records.append({'name': name, 'case_id': SOURCES[name]['case'], 'source': identity(path), 'outputs': outputs})
    manifest = {'schema': 'microduck-public-demos-v1', 'source_kind': 'recorded HIL trajectory rendering',
        'source_run': '20261025T051929Z-dc4ca58d',
        'source_report': '../../../docs/double-balance/evidence/reports/hil-replay-v1-report.json',
        'source_dimensions': [1280, 480], 'source_frames_per_second': 50,
        'start_seconds': START, 'duration_seconds': DURATION, 'playback_speed': 1,
        'crops_xywh': CROPS, 'spatial_resampling': False,
        'webp_quality': 90, 'webp_compression_level': 6, 'h264_crf': 15,
        'ffmpeg': subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0],
        'demos': records}
    (args.output_dir / 'demo-provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
