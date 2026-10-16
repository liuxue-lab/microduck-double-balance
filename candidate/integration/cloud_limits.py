"""Cloud-only budget guard. Never changes a physics threshold or retries a call."""
import os
import shutil
import time
from pathlib import Path


def check_boundary(out, force=False):
    # Cleanup writes must remain possible after a limit is reached. Native macro
    # admission calls this with force=False; periodic WAL progress also checks it.
    if force:
        return
    deadline = float(os.environ['MD12_NATIVE_DEADLINE'])
    if time.time() >= deadline:
        raise RuntimeError('CLOUD_NATIVE_TIME_LIMIT: preserve consumed attempts; no retry')
    if shutil.disk_usage(Path(out)).free < 8 * 1024**3:
        raise RuntimeError('CLOUD_EVIDENCE_DISK_LIMIT: keep 8 GiB for return; preserve all evidence')
    if shutil.disk_usage('/tmp').free < 2 * 1024**3:
        raise RuntimeError('CLOUD_RUNTIME_DISK_LIMIT: preserve compiler scratch and existing environment')
