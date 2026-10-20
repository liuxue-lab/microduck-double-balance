#!/usr/bin/env python3
"""Read existing training timing logs over SSH. Never starts or alters training."""
import base64
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import zlib

SSH_B85 = @@SSH_B85@@
SSH_SHA = @@SSH_SHA@@
REMOTE_B85 = @@REMOTE_B85@@
REMOTE_SHA = @@REMOTE_SHA@@
SLOT = '/root/autodl-tmp/microduck-double-balance/artifacts/double-balance-stage12/native-training-v2'


def main():
    print('只读统计现有训练日志；不停止、不重启任务，不新增仿真或PPO。', flush=True)
    helper = zlib.decompress(base64.b85decode(SSH_B85))
    remote = zlib.decompress(base64.b85decode(REMOTE_B85))
    if hashlib.sha256(helper).hexdigest() != SSH_SHA or hashlib.sha256(remote).hexdigest() != REMOTE_SHA:
        raise RuntimeError('Embedded read-only code hash differs')
    ssh = {'__name__': 'shared_ssh'}
    exec(compile(helper, '<shared_ssh>', 'exec'), ssh)
    command = ' '.join(map(shlex.quote, ['python3', '-B', '-c', remote.decode(), SLOT]))
    response = subprocess.run(ssh['argv'](command), capture_output=True, timeout=60)
    prefix = 'MD12_PERF_STATE='
    rows = [s[len(prefix):] for s in response.stdout.decode(errors='replace').splitlines() if s.startswith(prefix)]
    if response.returncode or not rows:
        raise RuntimeError('只读查询未完成；未发出任何训练操作。\n' + response.stderr.decode(errors='replace')[-2000:])
    result = json.loads(rows[-1])
    result['local_return_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    folder = Path.home()/'microduck-double-balance/artifacts/double-balance-stage12/native-training-v2-reports'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder/('microduck-stage12-perf-readonly-' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '-' + str(os.getpid()) + '.json')
    data = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    with target.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    log = result.get('log', {})
    for field, label in [('collection_seconds', '采样'), ('learning_seconds', 'PPO更新')]:
        item = log.get(field)
        if item:
            print('%s：最近%d轮中位 %.3f 秒/轮。' % (label, item['samples'], item['median']), flush=True)
    wall = result.get('metrics', {}).get('wall_seconds_per_update')
    if wall:
        print('含日志及保存开销的更新间隔：中位 %.3f 秒，均值 %.3f 秒。' % (wall['median'], wall['mean']), flush=True)
    if not log.get('recent_iterations'):
        print('日志中暂未提取到完整计时块；请以报告中的状态为准。', flush=True)
    print('PERF_READONLY_REPORT_SAVED=' + str(target), flush=True)
    print('报告 %.1f KiB；SHA256=%s' % (len(data)/1024, hashlib.sha256(data).hexdigest()), flush=True)
    print('训练继续按原配置和预算运行。只需回传这个小JSON。', flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n已结束只读查询。')
        raise SystemExit(130)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
