"""Run one preserved two-case PC-MuJoCo/Orin loop over the existing wired link."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
import traceback
import uuid

from bootstrap import load_base
old = load_base()
from loop_evidence import reviewed_evidence, verify_own_package, verify_reports
from recover_prior import recover
from review_buffered_loop import review as review_v2
import probe
import shared_ssh
import live_progress
import cpu_placement
import live_support
import live_stream
from deploy_contract import MODEL_SHA, REPORT_SHA, PC_IP, ORIN_IP, require, sha, wired_xml
from loop_contract import ONNX_SHA, TOTAL, CASES, WORKER_LIMIT, SERVER_LIMIT, frozen

HERE = Path(__file__).resolve().parent
ROOT = old.BASE / 'artifacts/double-balance-stage12/native-live-v1'
EXPORT = old.BASE / 'artifacts/double-balance-stage12/native-ros2-humble-v1/frozen-export'
PRIOR = old.BASE / 'artifacts/double-balance-stage12/native-ros2-humble-v4/report.json'
shared_ssh.HOST, shared_ssh.PORT = 'jetson@192.168.50.2', 22


def cpu_environment():
    env = old.clean_environment()
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    return env


def print_deadlines(report):
    cases=report.get('simulator',{}).get('cases',[])
    if not cases:return
    passed=report.get('full_loop_realtime_acceptance') is True
    print('20ms验收：'+('通过，所有周期均按期完成。' if passed else '未通过；功能完成不等于实时通过。'),flush=True)
    for case in cases:
        timing=case.get('timing',{});total=timing.get('completed_ticks',0)
        largest=timing.get('all_steps',{}).get('release_to_finish_ms',{}).get('max')
        print('%s：%d 个周期，超时 %s 次，完整周期最大 %s ms。'%(
            case.get('case_id',''),total,timing.get('deadline_misses','未知'),
            '未知' if largest is None else '%.3f'%largest),flush=True)


def protected_mkdir(path):
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        require(not current.is_symlink(), 'Unexpected directory symlink: ' + str(current))
    path.mkdir(parents=True, exist_ok=True)


def remote(command, stdin=None, timeout=30):
    process = subprocess.run(shared_ssh.argv(command), input=stdin, capture_output=True, timeout=timeout)
    require(process.returncode == 0, 'Orin command failed: ' + process.stderr.decode('utf-8', 'replace')[-3000:])
    return process.stdout


def remote_json(code, args=(), stdin=None):
    return json.loads(remote(shlex.join(['/usr/bin/python3', '-B', '-c', code, *map(str, args)]), stdin))


def stop_owned(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def previous_link():
    prior = old.read(PRIOR)
    require(prior.get('status') == 'ROS_LINK_PASSED'
            and prior.get('package_identity') == '50b02eb752d6fb9f0faceb9113edcbc9ff64cb52a0bfb02ea862f7ef3f91bf8f'
            and prior.get('checkpoint_sha256') == MODEL_SHA
            and prior.get('original_records_unchanged') is True, 'Expected the completed v4 ROS link report')
    link = prior['ros_link']
    require(link.get('status') == 'LINK_PARITY_PASSED' and link.get('completed_samples') == 128
            and link.get('onnx_sha256') == ONNX_SHA, 'Prior ROS model/sample identity differs')
    return dict(path=str(PRIOR), sha256=sha(PRIOR), ros_link=link)


def historical_reports():
    relatives = ('native-mujoco-v1/report.json', 'native-cumulative-loop-v1/report.json', 'native-pc-actiongraph-v1/report.json', 'native-pc-sync-v1/report.json',
        'native-pc-timeline-fixed-v2/fix-report.json', 'native-ros2-humble-loop-v1/report.json',
        'native-ros2-humble-v4/report.json')
    return {name: sha(ROOT.parent / name) for name in relatives}


def preflight_argv(output):
    # This new package reuses the pinned simulator package; it has no common.py.
    # A fresh child must establish the verified import path for itself.
    code = ('import sys;sys.path.insert(0,sys.argv[1]);from bootstrap import load_base;'
            'common=load_base();from native_preflight import preflight;common.write(sys.argv[2],preflight(),exclusive=True)')
    return [str(old.PYTHON), '-B', '-u', '-c', code, str(HERE), str(output)]


def execute(retry=False):
    protected_mkdir(ROOT)
    lock_path = ROOT.parent / 'native-ros2-humble-operation.lock'
    require(not lock_path.is_symlink(), 'Unexpected operation lock symlink')
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('另一个 Humble 任务仍在运行；不并发启动。')
        if (ROOT / 'report.json').exists() and not retry:
            print('本轮已有记录，保留结果，不自动重复回合。需要明确重试时使用 --retry-run。', flush=True)
            return show()
        if retry and (ROOT/'report.json').is_file():
            previous = old.read(ROOT/'report.json')
            pins = [previous.get('controller'), *previous.get('owned_processes', {}).values()]
            require(not any(old.alive(pin) for pin in pins),
                    'Previous owned processes are still alive; stop the previous task before retrying')
        session = uuid.uuid4().hex
        run = ROOT / 'runs' / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + session[:8])
        protected_mkdir(run)
        report = dict(schema='stage12-native-live-v1', status='RUNNING', directory=str(run),
            session=session, started_utc=time.time(), controller=old.identity(os.getpid()),
            added_PPO_updates=0, cloud_connections=0, motor_io=False, hardware_acceptance=False,
            realtime_acceptance=False, full_loop_realtime_acceptance=False, physical_robot_in_scope=False,
            live_hil_acceptance=False,
            cloud_power='OFF_USER_CONFIRMED', profiler_enabled=False, all_200Hz_native_substeps_audited=True,
            prior_training_ledger_modified=False, automatic_retry=False,
            planned_cases=list(CASES), maximum_control_steps=TOTAL,
            mode='native CPU physics; fixed 20ms PC-Orin releases and deadlines; no motor I/O', errors=[])
        def update(**values):
            report.update(values)
            old.write(run / 'report.json', report)
            old.write(ROOT / 'report.json', report)
        def boundary():
            require(not (run / 'STOP.json').exists(), 'Requested stop; preserve partial result')
        processes, streams = {}, []
        destination = None
        temp = None
        before = None
        progress_reader = progress_writer = None
        live_writer = live_reader = None
        def launch(name, argv, env, pass_fds=(), cpu=None):
            stream = (run / (name + '.log')).open('xb')
            streams.append(stream)
            p = subprocess.Popen(argv, cwd=old.PROJECT, stdout=stream, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True,pass_fds=pass_fds,
                                 preexec_fn=(lambda:os.sched_setaffinity(0,{cpu})) if cpu is not None else None)
            processes[name] = p
            update(owned_processes={n: old.identity(x.pid) for n, x in processes.items()})
            return p
        def wait_local(name, limit, required=True):
            process = processes[name]
            begun = last = time.monotonic()
            pending=''
            pipe_pending=''
            def consume(line):
                nonlocal last
                row=live_progress.parse(line) if name=='simulator' else None
                if row is not None:
                    # Once the simulator report exists, its final progress owns
                    # the status. Draining an older live row must not overwrite it.
                    if not (run/'simulator-report.json').exists():
                        old.write(run/'progress.json',row)
                    print(live_progress.display(row),flush=True)
                    last=time.monotonic()
                elif line.startswith('闭环 '):
                    print(line.rstrip(),flush=True)
                    last=time.monotonic()
            def drain(view):
                nonlocal pending,pipe_pending
                lines,pending=live_progress.complete_lines(pending,view.read())
                for line in lines:consume(line)
                if name=='simulator' and progress_reader is not None:
                    while True:
                        try:chunk=os.read(progress_reader,65536)
                        except BlockingIOError:break
                        if not chunk:break
                        lines,pipe_pending=live_progress.complete_lines(pipe_pending,chunk.decode('utf-8'))
                        for line in lines:consume(line)
            with (run / (name + '.log')).open('r', errors='replace') as view:
                while process.poll() is None:
                    boundary()
                    require(time.monotonic() - begun < limit, name + ' wall-clock limit reached')
                    drain(view)
                    if time.monotonic() - last >= 5:
                        print('%s：等待输出 %.0f 秒 | %s' % (name, time.monotonic() - begun,
                                                         run / (name + '.log')), flush=True)
                        last = time.monotonic()
                    time.sleep(.2)
                drain(view)
            if required:
                require(process.returncode == 0, name + ' failed; see ' + str(run / (name + '.log')))
        try:
            update(phase='verify_frozen_inputs', package_identity=verify_own_package(),
                   baseline_package_identity=old.verify_package(), reviewed_candidate=reviewed_evidence(),
                   historical_report_sha256=historical_reports())
            report['previous_ros_link'] = previous_link()
            before = old.verify_inputs()
            require(sha(old.SLOT / 'report.json') == REPORT_SHA, 'Original validation report changed')
            report['frozen_export'] = frozen(EXPORT)
            report['prior_recovery'] = recover(ROOT, old.write)
            report['reviewed_v2'] = review_v2(ROOT, old.write)
            require(shutil.disk_usage(ROOT).free >= 512 * 1024**2, 'Less than 512 MiB free; no files deleted')
            require(not cpu_environment().get('LD_PRELOAD'), 'Run without inherited profiler/library preloads')
            update(phase='local_simulator_preflight')
            launch('preflight', preflight_argv(run / 'preflight.json'), cpu_environment())
            wait_local('preflight', 90)
            report['simulator_preflight'] = old.read(run / 'preflight.json')
            report['cpu_placement']=placement=cpu_placement.plan()
            require(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'),
                    '请在笔记本图形桌面的终端运行；需要可见窗口，不能使用无显示的SSH终端。')
            source = live_support.accepted_source()
            report['accepted_v3_source'] = source
            placement['viewer_cpu'] = live_support.viewer_cpu(placement)
            live_writer, live_reader = live_stream.channel()
            viewer_env = cpu_environment()
            viewer_env.update(MUJOCO_GL='glfw', LP_NUM_THREADS='1')
            viewer_env.pop('PYOPENGL_PLATFORM', None)
            launch('live-viewer', [str(old.PYTHON), '-B', '-u', str(HERE/'live_viewer.py'),
                '--run', str(run), '--model', source['paths']['case-01/native-model.mjb'],
                '--state', source['paths']['case-01/state-data.npz'], '--fd', str(live_reader.fileno())],
                viewer_env, pass_fds=(live_reader.fileno(),), cpu=placement['viewer_cpu'])
            live_reader.close(); live_reader = None
            update(phase='warming_live_display')
            viewer_start = time.monotonic()
            while not (run/'viewer-ready.json').is_file():
                boundary()
                require(processes['live-viewer'].poll() is None, 'Live window failed; see live-viewer.log')
                require(time.monotonic()-viewer_start < 90, 'Live window startup exceeded 90 seconds')
                time.sleep(.1)
            report['live_viewer_ready'] = old.read(run/'viewer-ready.json')
            print('实时窗口已预热。1全身 / 2头部，鼠标转动/平移/缩放。关闭窗口仅结束显示。', flush=True)
            boundary()
            update(phase='verify_wired_hosts')
            pc = probe.probe()
            pc_ready = probe.readiness(pc)
            update(pc_probe=pc, pc_readiness=pc_ready)
            require(pc_ready['ready'], 'PC Humble/link: ' + str(pc_ready['blockers']))
            shared_ssh.connect()
            jetson = json.loads(remote('/usr/bin/python3 -B -', (HERE / 'probe.py').read_bytes(), timeout=80))
            jetson_ready = probe.readiness(jetson, jetson=True)
            require(jetson_ready['ready'], 'Orin Humble/link: ' + str(jetson_ready['blockers']))
            model = (jetson.get('board_model') or '').lower()
            require('orin' in model and 'nano' in model, 'Remote board identity differs')
            update(pc_probe=pc, jetson_probe=jetson, pc_readiness=pc_ready, jetson_readiness=jetson_ready)
            destination = '/home/jetson/microduck-stage12/native-live-v1/runs/' + session
            create = "import pathlib,json,sys\np=pathlib.Path(sys.argv[1]);cur=pathlib.Path('/')\nfor part in p.parts[1:]:\n cur=cur/part\n if cur.is_symlink():raise RuntimeError('Unexpected remote directory symlink')\np.parent.mkdir(parents=True,exist_ok=True);p.mkdir(mode=0o700)\nprint(json.dumps({'path':str(p)}))"
            remote_json(create, [destination])
            for name, address in (('pc', PC_IP), ('orin', ORIN_IP)):
                (run / (name + '-fastdds.xml')).write_text(wired_xml(address))
            files = [EXPORT / n for n in ('final6h_005304.onnx', 'contract.json', 'golden.json', 'export-report.json')]
            files += [HERE / n for n in ('loop_contract.py', 'loop_ros.py', 'deploy_contract.py')]
            files.append(run / 'orin-fastdds.xml')
            update(phase='wired_transfer', remote_directory=destination)
            print('网线传送本次推理文件：%.2f MiB；复用已有 Humble，无需下载软件。' %
                  (sum(p.stat().st_size for p in files) / 1024**2), flush=True)
            control = shared_ssh.connect()
            subprocess.run(['scp', '-P', '22', '-o', 'ControlPath=' + str(control), '-o', 'ControlMaster=no',
                            '-o', 'BatchMode=yes', '-o', 'ProxyCommand=false', *map(str, files),
                            'jetson@192.168.50.2:' + destination + '/'], check=True, timeout=90)
            pins = {p.name: sha(p) for p in files}
            check = "import pathlib,json,hashlib,sys\np=pathlib.Path(sys.argv[1]);pins=json.load(sys.stdin)\nfor n,h in pins.items():\n f=p/n\n if f.is_symlink() or hashlib.sha256(f.read_bytes()).hexdigest()!=h:raise RuntimeError('Transfer SHA mismatch: '+n)\nprint(json.dumps({'verified':len(pins)}))"
            receipt = remote_json(check, [destination], json.dumps(pins).encode())
            update(transfer=dict(destination=destination, files=pins, **receipt))
            boundary()
            temp = tempfile.TemporaryDirectory(prefix='md12-loop-%d-' % os.getuid(), dir='/tmp')
            address = str(Path(temp.name) / 's')
            args = [destination + '/loop_ros.py', 'server', '--folder', destination, '--session', session,
                    '--report', destination + '/server-report.json']
            clears = ['PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH', 'AMENT_PREFIX_PATH', 'CMAKE_PREFIX_PATH',
                      'COLCON_PREFIX_PATH', 'ROS_DISCOVERY_SERVER', 'FASTDDS_DEFAULT_PROFILES_FILE', 'CYCLONEDDS_URI']
            command = shlex.join(['timeout', '--signal=TERM', '--kill-after=5s', str(SERVER_LIMIT + 30) + 's',
                'env', *[k + '=' for k in clears], 'ROS_DOMAIN_ID=62', 'ROS_LOCALHOST_ONLY=0',
                'RMW_IMPLEMENTATION=rmw_fastrtps_cpp', 'RMW_FASTRTPS_USE_QOS_FROM_XML=0',
                'FASTRTPS_DEFAULT_PROFILES_FILE=' + destination + '/orin-fastdds.xml',
                *probe.shell_argv(jetson_ready['python'], args)])
            launch('orin-server', shared_ssh.argv(command), probe.clean_env())
            env = probe.clean_env()
            env['FASTRTPS_DEFAULT_PROFILES_FILE'] = str(run / 'pc-fastdds.xml')
            launch('pc-bridge', probe.shell_argv(pc_ready['python'], [HERE / 'loop_ros.py', 'bridge',
                '--folder', run, '--session', session, '--socket', address, '--report', run / 'bridge-report.json']), env,
                cpu=placement.get('bridge_cpu'))
            update(phase='waiting_ros_discovery')
            begun = time.monotonic()
            while not (run / 'bridge-ready.json').is_file():
                boundary()
                require(time.monotonic() - begun < 65, 'ROS bridge startup timeout')
                require(all(processes[n].poll() is None for n in ('pc-bridge', 'orin-server')),
                        'A ROS worker exited during startup; see logs')
                time.sleep(.2)
            require(processes['live-viewer'].poll() is None, 'Live window closed before formal control')
            update(phase='two_case_mujoco_orin_loop')
            progress_reader,progress_writer=os.pipe2(os.O_NONBLOCK|os.O_CLOEXEC)
            simulator_env=cpu_environment()
            simulator_env['NATIVE_PROGRESS_FD']=str(progress_writer)
            simulator_env['NATIVE_LIVE_FD']=str(live_writer.fileno())
            launch('simulator', [str(old.PYTHON), '-B', '-u', str(HERE / 'loop_sim.py'), '--run', str(run),
                '--session', session, '--socket', address, '--export', str(EXPORT)], simulator_env,
                pass_fds=(progress_writer,live_writer.fileno()),cpu=placement.get('control_cpu'))
            live_writer.close(); live_writer=None
            os.close(progress_writer);progress_writer=None
            wait_local('simulator', WORKER_LIMIT + 60)
            old.write(run/'VIEWER_FINISH.json', dict(reason='formal_cases_finished'), exclusive=True)
            wait_local('live-viewer', 8, required=False)
            wait_local('pc-bridge', 10)
            wait_local('orin-server', 10)
            server = remote_json('import sys;print(open(sys.argv[1]).read())', [destination + '/server-report.json'])
            report['remote_worker_exit_confirmed'] = True
            old.write(run / 'orin-server-report.json', server, exclusive=True)
            simulator = old.read(run / 'simulator-report.json')
            bridge = old.read(run / 'bridge-report.json')
            require(simulator.get('reference_preparation_required') is True and
                    simulator.get('prior_state_comparison',{}).get('all_bit_identical') is True and
                    simulator.get('accepted_v3_comparison',{}).get('all_bit_identical') is True,
                    'New reference preparation/state preservation evidence is missing')
            report['acceptance'] = verify_reports(simulator, bridge, server)
            require(old.verify_inputs() == before and sha(old.SLOT / 'report.json') == REPORT_SHA,
                    'Original model/training evidence changed')
            frozen(EXPORT)
            viewer = old.read(run/'viewer-report.json')
            report['live_display'] = live_support.visual_result(viewer, simulator)
            report['viewer'] = viewer
            finished_status = ('NATIVE_LIVE_REALTIME_PASSED' if report['acceptance']['full_loop_realtime_acceptance']
                               else 'NATIVE_LIVE_FUNCTIONAL_PASSED_REALTIME_PENDING')
            if not report['live_display']['complete']:
                finished_status = 'NATIVE_LIVE_DISPLAY_INCOMPLETE'
                report['errors'].append('Live display checks failed: ' + ', '.join(
                    name for name, ok in report['live_display']['checks'].items() if not ok))
            report['live_hil_acceptance'] = (report['live_display']['complete'] and
                                            report['acceptance']['full_loop_realtime_acceptance'])
            update(status=finished_status, realtime_acceptance=report['acceptance']['full_loop_realtime_acceptance'],
                   full_loop_realtime_acceptance=report['acceptance']['full_loop_realtime_acceptance'], phase='finished', simulator=simulator,
                   bridge=bridge, orin_server=server, original_records_unchanged=True,
                   remote_worker_exit_confirmed=True)
        except BaseException as exc:
            update(status='NATIVE_LIVE_FAILED_PRESERVE', phase='stopping_owned_workers',
                   errors=[repr(exc)], traceback=traceback.format_exc()[-12000:])
        finally:
            # Stop only this invocation's simulator/bridge groups. Never touch
            # package managers, unrelated processes, or the shared SSH master.
            old.write(run / 'STOP.json', dict(reason='controller_finalization', utc=time.time()))
            for name in ('simulator', 'live-viewer', 'pc-bridge', 'preflight'):
                stop_owned(processes.get(name))
            for sock in (live_writer,live_reader):
                if sock is not None:sock.close()
            for fd in (progress_reader,progress_writer):
                if fd is not None:os.close(fd)
            server_process = processes.get('orin-server')
            if server_process is not None and server_process.poll() is None:
                try:
                    remote_json("import pathlib,sys;pathlib.Path(sys.argv[1]).touch();print('{}')", [destination + '/cancel'])
                    server_process.wait(timeout=8)
                    report['remote_stop_command_delivered'] = True
                except Exception as exc:
                    report['remote_stop_error'] = repr(exc)
                    report['remote_worker_exit_confirmed'] = False
                stop_owned(server_process)
            for stream in streams:
                stream.close()
            if temp is not None:
                temp.cleanup()
            for key, name in (('viewer', 'viewer-report.json'), ('simulator', 'simulator-report.json'), ('bridge', 'bridge-report.json'),
                              ('orin_server', 'orin-server-report.json')):
                if (run / name).is_file():
                    report[key] = old.read(run / name)
            if destination is not None and server_process is not None and 'orin_server' not in report:
                try:
                    row = remote_json('import sys;print(open(sys.argv[1]).read())', [destination + '/server-report.json'])
                    report['orin_server'] = row
                    old.write(run / 'orin-server-report.json', row, exclusive=True)
                    report['remote_worker_exit_confirmed'] = bool(server_process.poll() is not None
                                                               and server_process.returncode in (0, 2))
                except Exception as exc:
                    report['remote_report_error'] = repr(exc)
                    report['remote_worker_exit_confirmed'] = False
            if before is not None:
                try:
                    report['original_records_unchanged'] = old.verify_inputs() == before \
                        and sha(old.SLOT / 'report.json') == REPORT_SHA
                    if report.get('accepted_v3_source'):
                        require(live_support.accepted_source() == report['accepted_v3_source'], 'Accepted v3 inputs changed')
                    frozen(EXPORT)
                    require(historical_reports() == report['historical_report_sha256'] and
                            reviewed_evidence() == report['reviewed_candidate'] and
                            verify_own_package() == report['package_identity'], 'Historical evidence or loop sources changed')
                    require(report['original_records_unchanged'], 'Frozen records changed during loop')
                except Exception as exc:
                    report['status'] = 'NATIVE_LIVE_FAILED_PRESERVE'
                    report['errors'].append(repr(exc))
            report['local_worker_exit_codes'] = {n: p.poll() for n, p in processes.items()}
            report['local_workers_exited'] = all(p.poll() is not None for p in processes.values())
            report['log_tails'] = {p.name: p.read_text(errors='replace')[-14000:] for p in run.glob('*.log')}
            if report['status'] in ('NATIVE_LIVE_REALTIME_PASSED', 'NATIVE_LIVE_FUNCTIONAL_PASSED_REALTIME_PENDING', 'NATIVE_LIVE_DISPLAY_INCOMPLETE'):
                if not report['local_workers_exited'] or not report.get('remote_worker_exit_confirmed'):
                    report['status'] = 'NATIVE_LIVE_FAILED_PRESERVE'
                    report['errors'].append('Could not confirm owned worker cleanup')
                else:
                    old.write(run / 'validated-pipeline.json', dict(
                        schema='stage12-native-mujoco-pipeline-validation-v1', package_identity=report['package_identity'],
                        reviewed_candidate=report['reviewed_candidate'], acceptance=report['acceptance'],
                        live_display=report['live_display'], live_hil_acceptance=report['live_hil_acceptance'],
                        source_scope='isolated native-live-v1 runtime; no global activation',
                        checkpoint_sha256=MODEL_SHA, onnx_sha256=ONNX_SHA), exclusive=True)
                    report['validated_pipeline_receipt'] = str(run / 'validated-pipeline.json')
            report['exported_data'] = {str(p.relative_to(run)): dict(bytes=p.stat().st_size, sha256=sha(p))
                for pattern in ('case-*/timing.csv', 'case-*/state-data.npz', 'case-*/state-data.json',
                                'case-*/push10-trace.jsonl', 'case-*/native-model.mjb', 'loop-transcript.jsonl') for p in run.glob(pattern)}
            if report['status']=='NATIVE_LIVE_FAILED_PRESERVE':
                report['realtime_acceptance']=False
                report['full_loop_realtime_acceptance']=False
                report['live_hil_acceptance']=False
            from network_diagnostics import snapshot as network_snapshot
            report['post_run_network'] = network_snapshot()
            update(phase='finished', finished_utc=time.time(), ssh_connection=shared_ssh.DIAGNOSTICS)
            print('NATIVE_LIVE_STATUS=' + report['status'], flush=True)
            print('NATIVE_LIVE_REPORT=' + str(ROOT / 'report.json'), flush=True)
            print_deadlines(report)
            print('实时三维显示：'+('通过' if report.get('live_display',{}).get('complete') else '尚未通过；见报告viewer/live_display')+'；原始v3结果保持独立。', flush=True)
            print('本轮报告包含20ms期限、显示统计和日志摘要；上传本报告即可。', flush=True)
            print('GitHub DNS：'+('已解析；需要发布时单独执行原 Git SSH 推送命令。'
                  if report['post_run_network']['github_dns_resolves'] else
                  '尚未解析；本轮数据已保存。恢复笔记本外网后单独补推送，无需重跑仿真。'),flush=True)
            if report['errors']:
                print('待处理：' + '; '.join(report['errors']), flush=True)
        return 0 if report['status'] == 'NATIVE_LIVE_REALTIME_PASSED' and report.get('live_hil_acceptance') else 2


def show(watch=False):
    while True:
        path = ROOT / 'report.json'
        if not path.is_file():
            print('尚未运行闭环。', flush=True)
            code = 2
        else:
            report = old.read(path)
            status = report['status']
            if status == 'RUNNING' and not old.alive(report.get('controller')):
                status = 'CONTROLLER_EXITED_PRESERVE'
            run = Path(report['directory'])
            progress = old.read(run / 'progress.json') if (run / 'progress.json').is_file() else {}
            print('%s | %s | %d/%d 步 | %.2f%% | %.1f 秒' %
                  (status, report.get('phase', ''), progress.get('completed_control_steps', 0), TOTAL,
                   progress.get('percent', 0.), progress.get('elapsed_seconds', 0.)), flush=True)
            if not watch:
                print('NATIVE_LIVE_REPORT=' + str(path), flush=True)
                print_deadlines(report)
                if report.get('live_display'):
                    print('实时显示：'+('通过' if report['live_display']['complete'] else '未通过')+
                          '；新状态帧数：'+str(report.get('viewer',{}).get('fresh_frames',0)),flush=True)
                if report.get('errors'):
                    print('待处理：' + '; '.join(report['errors']), flush=True)
            code = 0 if status == 'NATIVE_LIVE_REALTIME_PASSED' and report.get('live_hil_acceptance') else 2
        if not watch:
            return code
        time.sleep(2)


def stop():
    report = old.read(ROOT / 'report.json')
    run = Path(report['directory'])
    require(not run.is_symlink() and run.parent.resolve() == (ROOT / 'runs').resolve(), 'Unexpected run directory')
    require(report['status'] == 'RUNNING' and old.alive(report.get('controller')), 'No live owned loop controller')
    old.write(run / 'STOP.json', dict(reason='user_requested_stop', utc=time.time()))
    print('已请求停止本轮闭环；保留报告，不关闭 Orin。', flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('run', 'retry-run', 'status', 'watch', 'stop'))
    args = parser.parse_args()
    if args.operation in ('status', 'watch'):
        return show(args.operation == 'watch')
    if args.operation == 'stop':
        return stop()
    def interrupt(signum, frame):
        raise KeyboardInterrupt('Controller signal ' + str(signum))
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupt)
    return execute(retry=args.operation == 'retry-run')


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('监控/操作已退出。', flush=True)
        raise SystemExit(130)
    except Exception as exc:
        print('NATIVE_LIVE_PRESERVED=' + repr(exc), flush=True)
        raise SystemExit(2)
