"""Build small README previews from the two verified Stage12 replay videos.

Requires ffmpeg and the original MP4s in --source-dir. Reads recordings only;
does not load a policy, step physics, or connect to a robot or Orin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

SOURCES = {
    'quiet-2407': ('5feb95b1011441723ee74495c9ba5e29f6be967ce84fe638c7597f9931c3ea77', 4617084),
    'regression-2500': ('b93968c13e3492fcac34585d8c2b58d14a0360a237c2b9ef4a0a39a3ccac1c8f', 5563728),
}
START, DURATION, FPS, WIDTH = 6, 8, 12, 768


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    sources = {}
    for case, (digest, size) in SOURCES.items():
        path = args.source_dir / f'{case}-recorded-hil.mp4'
        data = path.read_bytes()
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            raise SystemExit(f'Refusing unexpected source bytes: {path}')
        sources[case] = path
    args.output_dir.mkdir(parents=True, exist_ok=True)
    version = subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0]
    records = []
    for case, path in sources.items():
        output = args.output_dir / f'{case}-preview.webp'
        graph = f'fps={FPS},scale={WIDTH}:-1:flags=lanczos'
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                        '-ss', str(START), '-t', str(DURATION), '-i', str(path),
                        '-vf', graph, '-c:v', 'libwebp_anim', '-q:v', '55',
                        '-compression_level', '6', '-loop', '0', str(output)], check=True)
        data = output.read_bytes()
        records.append({'case_id': case, 'source_file': path.name,
                        'source_sha256': SOURCES[case][0], 'source_bytes': SOURCES[case][1],
                        'preview_file': output.name, 'preview_sha256': hashlib.sha256(data).hexdigest(),
                        'preview_bytes': len(data), 'start_seconds': START,
                        'duration_seconds': DURATION, 'playback_speed': 1,
                        'frames_per_second': FPS, 'width': WIDTH, 'height': 288})
    manifest = {'schema': 'microduck-double-balance-preview-v1',
                'source_kind': 'offline replay of recorded native-mujoco-v3 HIL states',
                'source_run': '20261025T051929Z-dc4ca58d',
                'source_report': '../../../docs/double-balance/evidence/reports/hil-replay-v1-report.json',
                'live_window_capture': False, 'new_physics_steps': 0,
                'new_policy_inferences': 0, 'new_training_updates': 0,
                'ffmpeg': version, 'previews': records}
    (args.output_dir / 'provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
