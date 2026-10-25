"""Wrap the unchanged, pinned timeline package; never edit its files/reports."""
import argparse
import importlib
import json
import os
from pathlib import Path
import signal
import sys
import traceback

import probe_helper as helper

PACKAGE = Path('/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-pc-timeline-v1/package')
MANIFEST_SHA = 'c320743d583532893a7d7069770f94cfaa6a0818afcc6dbcaa161dc9a61a3bc1'
PACKAGE_ID = 'ba0b16e7acde7bc8484d9c9519c490110a3556fe528d9276f0cf611732456b54'


def require(ok, message='Pinned package check failed'):
    if not ok:
        raise RuntimeError(message)


def load_package():
    require(PACKAGE.is_dir() and not PACKAGE.is_symlink(), 'Existing timeline package missing')
    require(helper.digest(PACKAGE / 'manifest.json') == MANIFEST_SHA, 'Timeline manifest changed')
    manifest = json.loads((PACKAGE / 'manifest.json').read_text())
    require(manifest['identity'] == PACKAGE_ID)
    # Verify BEFORE importing any of the package's code.
    for name, pin in manifest['files'].items():
        require(Path(name).name == name)
        p = PACKAGE / name
        require(not p.is_symlink() and p.is_file() and p.stat().st_size == pin['bytes'])
        require(helper.digest(p) == pin['sha256'], 'Changed source: ' + name)
    sys.path.insert(0, str(PACKAGE))
    common = importlib.import_module('common')
    require(common.verify_package() == PACKAGE_ID)
    return common


def inspect():
    common = load_package()
    contract = importlib.import_module('timeline_contract')
    deploy = importlib.import_module('deploy_contract')
    loop = importlib.import_module('loop_contract')
    versions = common.verify_runtime()
    inputs = common.verify_inputs()
    chosen = deploy.validate_report(common.SLOT / 'report.json')
    contract.reviewed()
    exported = loop.frozen(common.BASE / 'artifacts/double-balance-stage12/native-ros2-humble-v1/frozen-export')
    return dict(status='FROZEN_INPUTS_VERIFIED', package_identity=PACKAGE_ID,
                versions=versions, inputs=inputs, chosen_checkpoint=chosen, frozen_export=exported)


def compare(run):
    load_package()
    contract = importlib.import_module('timeline_contract')
    first = run / 'inference-only'
    second = run / 'timeline'
    a = json.loads((first / 'worker-report.json').read_text())
    b = json.loads((second / 'worker-report.json').read_text())
    parity = contract.compare_traces(first / 'state-trace.jsonl', second / 'state-trace.jsonl')
    parity.update(contract.compare_audits(a, b))
    parity['all_numeric_state_differences_zero'] = all(v == 0 for v in parity['max_absolute_error_by_field'].values())
    parity['passed'] = parity['passed'] and parity['all_numeric_state_differences_zero'] and all(
        parity[k] for k in ('full_audit_equal', 'trial_equal', 'disturbance_equal'))
    parity['valid_bounded_arms'] = all(contract.valid_partial(x['episode_result']) for x in (a, b))
    parity['passed'] &= parity['valid_bounded_arms']
    return parity


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('inspect', 'compare', 'inference-only', 'timeline'), required=True)
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--run', type=Path)
    args = p.parse_args()
    result = dict(mode=args.mode, status='ENTRY_STARTED')
    code = 2
    def stop(*_):
        raise KeyboardInterrupt('Bounded worker interrupted')
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)
    try:
        if args.mode == 'inspect':
            result = inspect()
            code = 0
        elif args.mode == 'compare':
            result = compare(args.run)
            code = 0 if result['passed'] else 2
        else:
            common = load_package()
            os.chdir(common.PROJECT)
            # The original run() performs its own input, runtime and GPU preflight,
            # 160 control steps, 1600 physics checks and finally restores bindings.
            worker = importlib.import_module('timeline_worker')
            code = worker.run(args.folder, args.mode, args.run / 'STOP.json')
            result = dict(status='TIMELINE_WORKER_EXITED', exit_code=code,
                          loaded_libraries=helper.loaded_libraries())
    except BaseException as exc:
        result.update(status='ENTRY_FAILED_PRESERVE', error=repr(exc), traceback=traceback.format_exc())
    finally:
        helper.write_json(args.folder / 'worker.json', result)
        # Library-version queries do not initialize profiling or alter the run.
        result['loaded_cupti'] = helper.loaded_cupti_versions()
        helper.write_json(args.folder / 'worker.json', result)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
