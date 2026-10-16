"""Re-evaluate retained numeric records only. Never imports simulator or Torch."""
from pathlib import Path
import argparse
import hashlib
import importlib.abc
import json
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))


class NoPhysics(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','warp','mujoco','mujoco_warp','mjlab','rsl_rl','wandb'}:
            raise ImportError('Offline recovery forbids native/training imports: '+fullname)


def no_process_or_network(event, args):
    if event in {'subprocess.Popen','os.system','os.posix_spawn','os.fork','socket.connect','socket.bind'}:
        raise RuntimeError('Offline worker forbids '+event)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.meta_path.insert(0, NoPhysics())
    sys.addaudithook(no_process_or_network)
    import numpy as np
    from recovery.audit import read, verified_index, verify_file, macro_review, resume_review
    from recovery.durable import write_json, require, file_digest
    sys.path.insert(0,str(ROOT/'frozen-v5/baseline/g11/accepted'))
    from ground_scalar_contract import assess
    from batch_position_candidate import assess_cohort
    audit = read(ROOT/'v5-audit.json')
    index = verified_index(ROOT)
    paths = []
    for name, count in audit['accepted_macros'].items():
        paths.extend(name+'/macro-%02d'%i for i in range(count))
    paths.append('two_process_short_pilot/owner-1/macro-27')
    args.output.mkdir(parents=True, exist_ok=True)
    cases, errors, reused = [], [], 0
    for name in paths:
        identity = dict(macro=name, index_sha256=audit['evidence_index_sha256'],
                        worker_sha256=file_digest(__file__),numpy=np.__version__)
        record = args.output/(name.replace('/','--')+'.json')
        try:
            for row in index:
                if row['path'].startswith(name+'/'):
                    verify_file(args.run,row)
            receipt = record.with_suffix('.receipt.json')
            if record.exists() and receipt.exists():
                checked = read(record)
                require(checked['identity'] == identity, 'Completed offline case identity changed')
                require(read(receipt) == dict(identity=identity,sha256=file_digest(record)), 'Offline case receipt differs')
                reused += 1
            else:
                checked = dict(identity=identity, result=macro_review(args.run/name,np,assess,assess_cohort))
                if record.exists():
                    require(read(record) == checked, 'Interrupted offline result differs; preserved')
                else:
                    write_json(record,checked)
                write_json(receipt,dict(identity=identity,sha256=file_digest(record)))
            cases.append(dict(macro=name, report=record.name, sha256=file_digest(record), result=checked['result']))
            print('Stage12OfflineMacro='+name+'; verified',flush=True)
        except Exception as exc:
            errors.append(dict(macro=name,error=repr(exc),traceback=traceback.format_exc()))
    resume = None
    try:
        resume = resume_review(args.run,np)
    except Exception as exc:
        errors.append(dict(scope='resume_snapshot',error=repr(exc),traceback=traceback.format_exc()))
    summary = dict(status='OFFLINE_REVIEW_COMPLETE_NATIVE_REPAIR_PENDING' if not errors else 'OFFLINE_REVIEW_INCOMPLETE_PRESERVE',
        cases=cases,errors=errors,completed_cases=len(cases),expected_cases=len(paths),reused_cases=reused,
        restart=resume,numpy=np.__version__,recorded_numeric_rejection_retained=True,
        N1_reused=True,N2_passed=False,new_integrations=0,new_CPU_forward=0,new_GPU_forward=0,
        environment_constructions=0,PPO=0,cloud_training=0)
    write_json(args.output.parent/'offline-review.json',summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
