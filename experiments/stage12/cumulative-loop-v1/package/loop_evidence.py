"""Freeze successful inputs and keep functional success distinct from real time."""
import json
from pathlib import Path

from bootstrap import STAGE, require, sha

HERE = Path(__file__).resolve().parent
REPORT_SHA = 'a48093e63cce12fa50052f00b425e8eb28542e16aed78210c374e7655505a3c7'
ACTION_ID = '5c7513b74ecfef83be357883e921508a5c61fb402d3dde45fdde2d996c108cca'
ACTION_COMMIT = '25378e7fccbdba8b127b593eeaa47403c56ca866'


def read(path):
    return json.loads(Path(path).read_bytes())


def verify_own_package():
    import hashlib
    manifest = read(HERE / 'manifest.json')
    identity = hashlib.sha256(json.dumps(manifest['files'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(identity == manifest['identity'], 'Cumulative loop package identity differs')
    for name, pin in manifest['files'].items():
        path = HERE / name
        require(Path(name).name == name and path.is_file() and not path.is_symlink() and
                path.stat().st_size == pin['bytes'] and sha(path) == pin['sha256'],
                'Cumulative loop source changed: ' + name)
    return identity


def reviewed_evidence():
    folder = STAGE / 'native-pc-actiongraph-v1'
    path = folder / 'report.json'
    require(not path.is_symlink() and sha(path) == REPORT_SHA, 'Reviewed cumulative GPU report differs')
    row = read(path)
    require(row.get('status') == 'PC_ACTIONGRAPH_MEASURED' and row.get('errors') == [] and
            row.get('package_identity') == ACTION_ID and row.get('original_records_unchanged') is True and
            row.get('recommendation', {}).get('choice') == 'VERIFIED_CUMULATIVE_CANDIDATE' and
            row['recommendation'].get('confirmation_passed') is True,
            'Cumulative candidate has not passed both qualified measurements')
    install = read(folder / 'code-installed.json')
    require(install.get('commit') == ACTION_COMMIT and install.get('package_identity') == ACTION_ID,
            'Qualified actiongraph source installation differs')
    manifest = read(folder / 'package/manifest.json')
    require(manifest.get('identity') == ACTION_ID, 'Qualified actiongraph package differs')
    for name in ('action_graph.py', 'sync_adapters.py', 'static_math.py'):
        pin = manifest['files'][name]
        original, current = folder / 'package' / name, HERE / name
        require(not original.is_symlink() and sha(original) == pin['sha256'] == sha(current),
                'The verified candidate adapter changed: ' + name)
    return dict(report_path=str(path), report_sha256=REPORT_SHA, source_commit=ACTION_COMMIT,
                package_identity=ACTION_ID, choice=row['recommendation']['choice'],
                confirmation_passed=True, prior_production_activated=False,
                adapter_sources='exact byte copies of the qualified candidate')


def verify_reports(simulator, bridge, server):
    from loop_contract import TOTAL, CASES, STEPS
    from cumulative_runtime import adapter_checks
    require(simulator.get('status') == 'LOOP_SIM_FUNCTIONAL_PASSED' and
            simulator.get('original_records_unchanged') is True and
            simulator.get('completed_control_steps') == TOTAL and
            len(simulator.get('cases', [])) == len(CASES), 'Incomplete cumulative simulator run')
    for case, wanted in zip(simulator['cases'], CASES):
        checks = adapter_checks(case.get('runtime_adapters') or {})
        require(case.get('case_id') == wanted['case_id'] and case.get('seed') == wanted['seed'] and
                case.get('force_seed') == wanted['force_seed'] and case.get('loop_passed') is True and
                case.get('acceptance', {}).get('passed') is True and case.get('adapter_checks') == checks and
                all(checks.values()) and case.get('observed_physics_steps') == STEPS * 10,
                'Case identity, physics or adapter validation failed')
        parity = case.get('live_action_parity', {})
        require(parity.get('compared_actions') == STEPS and parity.get('every_applied_action_came_from_Orin') is True
                and parity.get('local_fallback') is False, 'Missing every-step Orin parity')
        timing = case.get('timing', {})
        require(timing.get('completed_ticks') == STEPS and timing.get('incomplete_tick') is False and
                timing.get('all_steps', {}).get('environment_ms', {}).get('count') == STEPS,
                'Incomplete timing evidence')
    require(bridge.get('status') == 'LOOP_BRIDGE_COMPLETE' and server.get('status') == 'LOOP_SERVER_COMPLETE'
            and server.get('model_unchanged') is True, 'ROS worker completion differs')
    records = [simulator['transport'], bridge, server]
    require(all(row.get('completed_requests') == TOTAL for row in records) and
            len({row.get('transcript_sha256') for row in records}) == 1 and
            isinstance(server.get('transcript_sha256'), str) and len(server['transcript_sha256']) == 64,
            'Simulator/bridge/Orin transcript count or identity differs')
    require(server.get('recurrent_reset_sequences') == [0, 3000], 'Incorrect LSTM reset schedule')
    return dict(functional_closed_loop=True, live_action_comparisons=TOTAL,
                control_steps=TOTAL, physics_steps=TOTAL * 10,
                qualified_cumulative_pipeline_validated_for_full_cases=True,
                real_time_HIL='PENDING', full_loop_realtime_acceptance=False,
                physical_robot_acceptance=False, nominal_period_ms=20.,
                original_realtime_requirement_retained=True,
                explanation='This validation uses lockstep progression; observed latency does not certify scheduled 50Hz HIL.',
                timing_by_case={c['case_id']: c['timing'] for c in simulator['cases']})
