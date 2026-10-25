"""New native-physics evidence, separate from the preserved 2ms Warp run."""
import hashlib
import json
from pathlib import Path

from bootstrap import STAGE, require, sha

HERE = Path(__file__).resolve().parent
REPORT_SHA = 'b6572964f074840b5a00ce495f9ffa575c63f0408b96f3977266ac2198cd5e2e'
PRIOR_ID = '53f61712913533b880bf74cb8cdf7f21dc0ecb16a5a7e51fe82ac2c6903b2216'
PRIOR_COMMIT = '227eba43f06ba0c0a1bc3386a488ec48ce19304e'


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
    root=STAGE/'native-mujoco-v1'
    path=root/'report.json'
    require(not path.is_symlink() and sha(path)==REPORT_SHA,'Reviewed native v1 report differs')
    report=read(path)
    require(report.get('status')=='NATIVE_MUJOCO_FAILED_PRESERVE' and
            report.get('errors')==["ModuleNotFoundError(\"No module named 'torch'\")"] and
            report.get('original_records_unchanged') is True and
            report.get('package_identity')==PRIOR_ID,'Unexpected prior failure; preserve')
    codes=report.get('local_worker_exit_codes',{})
    require(set(codes)=={'preflight','simulator','pc-bridge','orin-server'} and
            all(v==0 for v in codes.values()) and report.get('local_workers_exited') is True,
            'Prior workers did not all complete')
    install=read(root/'code-installed.json')
    require(install.get('commit')==PRIOR_COMMIT and install.get('package_identity')==PRIOR_ID,
            'Prior installed source differs')
    manifest=read(root/'package/manifest.json')
    identity=hashlib.sha256(json.dumps(manifest['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    require(identity==manifest.get('identity')==PRIOR_ID,'Prior package identity differs')
    for name,pin in manifest['files'].items():
        member=root/'package'/name
        require(Path(name).name==name and not member.is_symlink() and
                member.stat().st_size==pin['bytes'] and sha(member)==pin['sha256'],
                'Prior source changed: '+name)
    for name in ('loop_contract.py','loop_ros.py','deploy_contract.py','probe.py','shared_ssh.py'):
        require(sha(HERE/name)==manifest['files'][name]['sha256'],'Validated communication source changed: '+name)
    acceptance=verify_reports(report['simulator'],report['bridge'],report['orin_server'])
    return dict(report_path=str(path),report_sha256=REPORT_SHA,source_commit=PRIOR_COMMIT,
        package_identity=PRIOR_ID,physics_backend='native MuJoCo CPU',physics_dt=.005,
        functional_passed=acceptance['functional_closed_loop'],
        realtime_passed=acceptance['full_loop_realtime_acceptance'],
        deadline_misses=[c['timing']['deadline_misses'] for c in report['simulator']['cases']],
        recovered_summary_failure="system Python imported torch through native_runtime",
        reused_wire_and_FP32_contract=True,old_evidence_relabelled=False)


def verify_reports(simulator, bridge, server):
    from loop_contract import TOTAL, CASES, STEPS
    from deploy_contract import MODEL_SHA
    from native_checks import adapter_checks
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
