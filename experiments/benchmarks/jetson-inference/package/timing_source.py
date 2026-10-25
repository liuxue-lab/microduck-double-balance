"""Load the exact accepted observations; never fabricate or resimulate a trace."""
import json
from pathlib import Path

import common as old
from loop_contract import TOTAL, CASES, EMPTY_CHAIN, chain, validate_request, validate_reply
from deploy_contract import require, sha

REPORT_SHA = '615d9b29e1c85433eebec2a988031ba01f14fc53ffeebf9906ae9355e6ac28e5'
PRIOR_ROOT = old.BASE / 'artifacts/double-balance-stage12/native-ros2-humble-loop-v1'
PACKAGE_ID = 'c105a7b7efdf2b4a96ea9e96d47b5d0a9ddf90e4252f8e97fe1d35bd72818454'


def validate_prior(path):
    require(path.is_file() and not path.is_symlink() and sha(path) == REPORT_SHA,
            'Expected the reviewed complete loop report; preserve all existing results')
    report = old.read(path)
    require(report['status'] == 'CLOSED_LOOP_FUNCTIONAL_PASSED' and report['package_identity'] == PACKAGE_ID
            and report['local_workers_exited'] and report['remote_worker_exit_confirmed']
            and report['original_records_unchanged'] and not report['errors'], 'Prior loop did not complete cleanly')
    rows = report['simulator']['cases']
    require(len(rows) == 2 and report['simulator']['completed_control_steps'] == TOTAL
            and all(r['loop_passed'] and r['acceptance']['passed'] for r in rows), 'Both original cases must pass')
    require([r['case_id'] for r in rows] == [c['case_id'] for c in CASES], 'Original case identities differ')
    return report


def load_trace(path, session, expected_chain):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 32 * 1024**2,
            'Missing or oversized accepted loop transcript')
    values, current = [], EMPTY_CHAIN
    with path.open('rb') as stream:
        for seq in range(TOTAL):
            line = stream.readline(65537)
            require(line and len(line) <= 65536 and line.endswith(b'\n'), 'Incomplete transcript line')
            row = json.loads(line)
            request = validate_request(row['request'], session, seq)
            response = validate_reply(row['response'], request)
            current = chain(current, request, response)
            values.append(dict(obs=request['obs'], actions=response['actions']))
        require(not stream.read(1), 'More than 6000 accepted transcript entries')
    require(current == expected_chain, 'Recorded request/action transcript differs from the accepted report')
    return values


def load_recording():
    report = validate_prior(PRIOR_ROOT / 'report.json')
    folder = Path(report['directory'])
    require(folder.parent.resolve() == (PRIOR_ROOT / 'runs').resolve() and not folder.is_symlink(),
            'Unexpected accepted loop directory')
    path = folder / 'loop-transcript.jsonl'
    expected = report['simulator']['transport']['transcript_sha256']
    require(report['bridge']['transcript_sha256'] == report['orin_server']['transcript_sha256'] == expected,
            'Accepted three-node transcript identities differ')
    values = load_trace(path, report['session'], expected)
    return values, dict(report_sha256=REPORT_SHA, path=str(path), file_sha256=sha(path),
                        accepted_transcript_sha256=expected, observations=len(values),
                        source='accepted PC-MuJoCo/Orin two-case live trace', new_physics_steps=0)
