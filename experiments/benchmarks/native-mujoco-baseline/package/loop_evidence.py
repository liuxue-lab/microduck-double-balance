"""New native-physics evidence, separate from the preserved 2ms Warp run."""
import hashlib
import json
from pathlib import Path

from bootstrap import STAGE, require, sha

HERE = Path(__file__).resolve().parent
REPORT_SHA = 'efc271222eced2901c3574ee6e8421dbe1245933b5d03bd111e359b5416bfee5'
PRIOR_ID = '95d8ac25c6042ceb4ffe6e1464a5ac1a19b9112a0fdeaaf480a6eacfaf207d28'
PRIOR_COMMIT = 'ed53880d049422919d26b07106d6d0e84f19ae29'


def read(path):
    return json.loads(Path(path).read_bytes())


def verify_own_package():
    manifest = read(HERE/'manifest.json')
    identity = hashlib.sha256(json.dumps(manifest['files'], sort_keys=True, separators=(',',':')).encode()).hexdigest()
    require(identity==manifest['identity'],'Native package identity differs')
    for name,pin in manifest['files'].items():
        path = HERE/name
        require(Path(name).name==name and path.is_file() and not path.is_symlink() and
                path.stat().st_size==pin['bytes'] and sha(path)==pin['sha256'], 'Changed source: '+name)
    return identity


def reviewed_evidence():
    root = STAGE/'native-cumulative-loop-v1'
    path = root/'report.json'
    require(not path.is_symlink() and sha(path)==REPORT_SHA,'Reviewed 2ms functional report differs')
    report = read(path)
    require(report.get('status')=='CUMULATIVE_LOOP_FUNCTIONAL_PASSED_REALTIME_PENDING' and
            report.get('errors')==[] and report.get('original_records_unchanged') is True and
            report.get('package_identity')==PRIOR_ID,'Missing completed 2ms functional baseline')
    install = read(root/'code-installed.json')
    require(install.get('commit')==PRIOR_COMMIT and install.get('package_identity')==PRIOR_ID,
            'Prior installed source differs')
    manifest = read(root/'package/manifest.json')
    require(manifest.get('identity')==PRIOR_ID,'Prior package identity differs')
    for name in ('loop_contract.py','loop_ros.py','deploy_contract.py','probe.py','shared_ssh.py'):
        require(sha(HERE/name)==sha(root/'package'/name)==manifest['files'][name]['sha256'],
                'Previously validated communication source changed: '+name)
    return dict(report_path=str(path), report_sha256=REPORT_SHA, source_commit=PRIOR_COMMIT,
        package_identity=PRIOR_ID, physics_backend='MuJoCo Warp', physics_dt=.002,
        functional_passed=True, realtime_passed=False,
        reused_wire_and_FP32_contract=True, old_evidence_relabelled=False)


def verify_reports(simulator, bridge, server):
    from loop_contract import TOTAL, CASES, STEPS
    from deploy_contract import MODEL_SHA
    from native_runtime import adapter_checks
    from native_criteria import accept_case
    require(simulator.get('status')=='LOOP_SIM_FUNCTIONAL_PASSED' and
            simulator.get('original_records_unchanged') is True and
            simulator.get('completed_control_steps')==TOTAL and
            len(simulator.get('cases',[]))==len(CASES),'Native functional run incomplete')
    for row,wanted in zip(simulator['cases'],CASES):
        checks = adapter_checks(row.get('runtime_adapters') or {})
        acceptance = accept_case(row,wanted,MODEL_SHA)
        require(row.get('loop_passed') is True and acceptance==row.get('acceptance') and
                acceptance['passed'] and checks==row.get('adapter_checks') and all(checks.values()),
                'Native case identity, retention or substep audits failed')
        parity = row.get('live_action_parity',{})
        require(parity.get('compared_actions')==STEPS and parity.get('every_applied_action_came_from_Orin') is True
                and parity.get('local_fallback') is False,'Missing every-action Orin comparison')
        timing = row.get('timing',{})
        require(timing.get('completed_ticks')==STEPS and timing.get('incomplete_tick') is False and
                timing.get('all_steps',{}).get('environment_ms',{}).get('count')==STEPS and
                timing.get('release_schedule_anchored') is True and timing.get('dropped_ticks')==0 and
                timing.get('schedule_rebases')==0 and timing.get('first_tick_included') is True,
                'Incomplete fixed-release timing evidence')
        duration = timing.get('all_steps',{}).get('release_to_finish_ms',{})
        require(duration.get('count')==STEPS and isinstance(duration.get('max'),(float,int)) and
                timing.get('deadline_misses')==duration.get('over_20ms'), 'Deadline accounting differs')
        expected_realtime = timing['deadline_misses']==0 and duration['max']<=20.
        require(timing.get('realtime_acceptance') is expected_realtime,
                'Realtime flag does not match every scheduled deadline')
    require(bridge.get('status')=='LOOP_BRIDGE_COMPLETE' and server.get('status')=='LOOP_SERVER_COMPLETE'
            and server.get('model_unchanged') is True,'ROS completion differs')
    records = [simulator['transport'],bridge,server]
    require(all(r.get('completed_requests')==TOTAL for r in records) and
            len({r.get('transcript_sha256') for r in records})==1 and
            isinstance(server.get('transcript_sha256'),str) and len(server['transcript_sha256'])==64,
            'Simulator/bridge/Orin transcript differs')
    require(server.get('recurrent_reset_sequences')==[0,3000],'Incorrect recurrent reset schedule')
    realtime = all(c['timing']['realtime_acceptance'] is True for c in simulator['cases'])
    require(simulator.get('full_loop_realtime_acceptance') is realtime,'Realtime summary differs')
    return dict(functional_closed_loop=True, control_steps=TOTAL, physics_steps=TOTAL*4,
        live_action_comparisons=TOTAL, physics_hz=200, control_hz=50,
        full_loop_realtime_acceptance=realtime, real_time_HIL='PASSED' if realtime else 'PENDING',
        physical_robot_acceptance=False, nominal_period_ms=20.,
        same_policy_new_physics_validation=True, old_500Hz_validation_preserved=True,
        timing_by_case={c['case_id']:c['timing'] for c in simulator['cases']})
