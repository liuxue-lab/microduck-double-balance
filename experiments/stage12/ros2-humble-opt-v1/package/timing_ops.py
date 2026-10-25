"""Bounded optimization measurements; never replace the accepted controller/model."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import select
import shlex
import shutil
import signal
import subprocess
import time
import traceback
import uuid

import common as old
import probe
import shared_ssh
import timing_source
from deploy_contract import REPORT_SHA, PC_IP, ORIN_IP, require, sha, wired_xml
from loop_contract import ONNX_SHA, TOTAL, frozen

HERE = Path(__file__).resolve().parent
ROOT = old.BASE / 'artifacts/double-balance-stage12/native-ros2-humble-opt-v1'
EXPORT = old.BASE / 'artifacts/double-balance-stage12/native-ros2-humble-v1/frozen-export'
shared_ssh.HOST, shared_ssh.PORT = 'jetson@192.168.50.2', 22


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


def verify_pair(client, server, scenario):
    require(client['session'] == server['session'] and client['wire'] == server['wire']
            and client['runtime'] == server['runtime'] and server['model_unchanged'] is True,
            'Remote/local identities differ')
    gate = client['receive_gate']
    if scenario == 'replay':
        require(client['status'] == 'SCHEDULE_REPLAY_PASSED' and server['status'] == 'REPLAY_SERVER_COMPLETE'
                and gate['accepted_replies'] == server['completed_requests'] == TOTAL
                and gate['transcript_sha256'] == server['transcript_sha256']
                and server['recurrent_reset_sequences'] == [0, 3000]
                and not gate['fault_latched'] and gate['scheduled_to_validated_ms']['max'] <= 20.,
                'Replay count, transcript, recurrence, or deadline differs')
    else:
        require(client['status'] == 'FAULT_GATE_PASSED' and server['status'] == 'FAULT_SERVER_COMPLETE'
                and server['fault_injected'] and server['completed_requests'] == client['sent_messages'] == 2
                and gate['accepted_replies'] == 1 and gate['fault_latched']
                and all(client['fault_conditions'].values()), 'Fault scenario did not pass')


def recommend(rows, faults):
    passed = [r for r in rows if r.get('passed')]
    baseline = next((r for r in passed if r['name'] == 'json-fp32'), None)
    output = dict(production_activated=False, int8_activated=False, pruning_applied=False,
                  hardware_acceptance=False, complete_control_loop_realtime_acceptance=False,
                  next_required='Review measured report; real full-loop timing remains unverified')
    if baseline is None:
        return dict(output, status='NO_RECOMMENDATION_BASELINE_FAILED')
    base_ms = baseline['client']['receive_gate']['scheduled_to_validated_ms']
    eligible = [r for r in passed if r['client']['receive_gate']['scheduled_to_validated_ms']['p95'] <= .95 * base_ms['p95']
                and r['client']['receive_gate']['scheduled_to_validated_ms']['mean'] < base_ms['mean']]
    chosen = min(eligible,key=lambda r:r['client']['receive_gate']['scheduled_to_validated_ms']['p95']) if eligible else baseline
    fast_ms = chosen['client']['receive_gate']['scheduled_to_validated_ms']
    checked = len(faults) == 3 and all(r.get('passed') and r['wire'] == chosen['wire']
                                    and r['runtime'] == chosen['runtime'] for r in faults)
    return dict(output, status='RECOMMENDED_FOR_NEXT_FULL_LOOP_TEST' if checked else 'FAULT_CHECKS_PENDING_OR_FAILED',
                variant=chosen['name'], wire=chosen['wire'], runtime=chosen['runtime'],
                p95_scheduled_to_validated_reduction_fraction=1-fast_ms['p95']/base_ms['p95'],
                measurement_scope='One bounded run of each named variant, recorded observations, no live plant')


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
            print('已有结果，保留且不自动重试。需要明确重跑时使用 --retry-run。', flush=True)
            return show()
        session = uuid.uuid4().hex
        run = ROOT / 'runs' / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + session[:8])
        protected_mkdir(run)
        report = dict(schema='stage12-humble-optimization-v1', status='RUNNING', directory=str(run),
            session=session, started_utc=time.time(), controller=old.identity(os.getpid()),
            added_PPO_updates=0, cloud_connections=0, motor_io=False, hardware_acceptance=False,
            realtime_acceptance=False, production_model_replaced=False, automatic_retry=False,
            maximum_new_physics_steps=1000, maximum_replay_requests=18000, maximum_fault_requests=6,
            replay_variants=[], fault_cases=[], errors=[])
        processes, streams, remote_jobs = {}, [], {}
        destination, before = None, None
        def update(**values):
            report.update(values)
            old.write(run / 'report.json', report)
            old.write(ROOT / 'report.json', report)
        def boundary():
            require(not (run / 'STOP.json').exists(), 'Requested stop; preserve partial result')
        def launch(name, argv, env, stdin=False):
            stream = (run / (name + '.log')).open('xb')
            streams.append(stream)
            process = subprocess.Popen(argv, cwd=old.PROJECT, stdout=stream, stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE if stdin else subprocess.DEVNULL, env=env, start_new_session=True)
            processes[name] = process
            update(active_log=str(run / (name + '.log')),
                   owned_processes={n: old.identity(p.pid) for n, p in processes.items()})
            return process
        def wait_local(name, limit, required=True):
            process = processes[name]
            begun = last = time.monotonic()
            with (run / (name + '.log')).open('r', errors='replace') as view:
                while process.poll() is None:
                    boundary()
                    require(time.monotonic()-begun < limit, name + ' wall-clock limit reached')
                    for line in view.readlines():
                        if line.startswith(('50Hz ', 'PC_PROFILE=', 'POLICY_BENCHMARK=', 'CANDIDATE_STATUS=')):
                            print(line.rstrip(), flush=True)
                            last=time.monotonic()
                    if time.monotonic()-last >= 5:
                        print('%s：进行中 %.0f 秒 | %s' % (name,time.monotonic()-begun,run/(name+'.log')),flush=True)
                        last=time.monotonic()
                    time.sleep(.1)
                for line in view.readlines():
                    if line.startswith(('50Hz ', 'PC_PROFILE=', 'POLICY_BENCHMARK=', 'CANDIDATE_STATUS=')):
                        print(line.rstrip(),flush=True)
            if required:
                require(process.returncode == 0, name + ' failed; see ' + str(run/(name+'.log')))
            return process.returncode
        def launch_remote(name, args, cancel, timeout_seconds):
            clears = ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','AMENT_PREFIX_PATH','CMAKE_PREFIX_PATH',
                      'COLCON_PREFIX_PATH','ROS_DISCOVERY_SERVER','FASTDDS_DEFAULT_PROFILES_FILE','CYCLONEDDS_URI')
            command = shlex.join(['timeout','--signal=TERM','--kill-after=5s',str(timeout_seconds)+'s',
                'env',*[k+'=' for k in clears], 'ROS_DOMAIN_ID=62','ROS_LOCALHOST_ONLY=0',
                'RMW_IMPLEMENTATION=rmw_fastrtps_cpp','RMW_FASTRTPS_USE_QOS_FROM_XML=0',
                'FASTRTPS_DEFAULT_PROFILES_FILE='+destination+'/orin-fastdds.xml',
                *probe.shell_argv(jetson_ready['python'],args)])
            remote_jobs[name] = cancel
            return launch(name,shared_ssh.argv(command),probe.clean_env())
        def cancel_remote(name):
            process = processes.get(name)
            if process is not None and process.poll() is not None and process.returncode not in (0, 2):
                report.setdefault('remote_exit_unconfirmed',[]).append(name)
                return
            if process is not None and process.poll() is None:
                remote_json("import pathlib,sys;pathlib.Path(sys.argv[1]).touch();print('{}')",[remote_jobs[name]])
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    # The remote timeout remains in force even if SSH is lost.
                    report.setdefault('remote_exit_unconfirmed',[]).append(name)
                stop_owned(process)
                if process.returncode not in (0, 2):
                    report.setdefault('remote_exit_unconfirmed',[]).append(name)
        def transfer(files):
            # A bounded SSH byte stream over the already established wired session.
            # It reports actual bytes sent and never extracts paths from an archive.
            pins={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files}
            receiver='''import pathlib,sys,json,hashlib
p=pathlib.Path(sys.argv[1]);pins=json.loads(sys.stdin.buffer.readline(65537))
for name,pin in pins.items():
 if pathlib.Path(name).name!=name:raise RuntimeError('Unsafe upload member')
 h=hashlib.sha256();left=pin['bytes']
 with (p/name).open('xb') as f:
  while left:
   data=sys.stdin.buffer.read(min(left,65536))
   if not data:raise RuntimeError('Incomplete wired upload')
   f.write(data);h.update(data);left-=len(data)
 if h.hexdigest()!=pin['sha256']:raise RuntimeError('Transfer SHA mismatch: '+name)
print(json.dumps({'verified':len(pins)}))
'''
            command=shlex.join(['timeout','--signal=TERM','--kill-after=5s','100s','/usr/bin/python3','-B','-c',receiver,destination])
            p=launch('wired-transfer',shared_ssh.argv(command),probe.clean_env(),stdin=True)
            total=sum(x['bytes'] for x in pins.values());done=0;started=last=time.monotonic()
            os.set_blocking(p.stdin.fileno(),False)
            def write_block(data):
                offset=0
                while offset<len(data):
                    boundary()
                    require(time.monotonic()-started<90 and p.poll() is None,'Wired transfer timed out or exited')
                    if select.select([], [p.stdin], [], .2)[1]:
                        try:offset+=os.write(p.stdin.fileno(),data[offset:])
                        except BlockingIOError:pass
            write_block(json.dumps(pins).encode()+b'\n')
            for path in files:
                with path.open('rb') as source:
                    while True:
                        block=source.read(65536)
                        if not block:break
                        write_block(block);done+=len(block)
                        if time.monotonic()-last>=.5 or done==total:
                            print('网线传送 %.1f%% | %.2f/%.2f MiB | 平均 %.2f MiB/s' %
                                (done/total*100,done/1024**2,total/1024**2,done/1024**2/max(.001,time.monotonic()-started)),flush=True)
                            last=time.monotonic()
            p.stdin.close();wait_local('wired-transfer',15)
            receipt=json.loads((run/'wired-transfer.log').read_text().splitlines()[-1])
            require(receipt['verified']==len(files),'Incomplete transfer verification')
            return dict(files=pins,bytes=total,seconds=time.monotonic()-started,**receipt)
        def scenario(name, wire, runtime, kind):
            boundary()
            folder=run/name;protected_mkdir(folder)
            sid=uuid.uuid4().hex
            remote_report=destination+'/'+name+'-report.json'
            remote_cancel=destination+'/'+name+'-cancel'
            result=dict(name=name,wire=wire,runtime=runtime,scenario=kind,session=sid,passed=False)
            server_name=name+'-orin';client_name=name+'-pc'
            update(phase=name,active_case_directory=str(folder))
            base_args=['--scenario',kind,'--session',sid,'--wire',wire,'--runtime',runtime]
            try:
                launch_remote(server_name,[destination+'/timing_nodes.py','server',*base_args,'--folder',destination,
                    '--report',remote_report,'--cancel-file',remote_cancel],remote_cancel,220)
                env=probe.clean_env();env['FASTRTPS_DEFAULT_PROFILES_FILE']=str(run/'pc-fastdds.xml')
                launch(client_name,probe.shell_argv(pc_ready['python'],[HERE/'timing_nodes.py','client',*base_args,
                    '--folder',folder,'--report',folder/'client-report.json','--cancel-file',run/'STOP.json']),env)
                client_rc=wait_local(client_name,205,required=False)
                if client_rc!=0:
                    cancel_remote(server_name)
                else:
                    wait_local(server_name,10,required=False)
                result['client']=old.read(folder/'client-report.json')
                result['server']=remote_json('import sys;print(open(sys.argv[1]).read())',[remote_report])
                old.write(folder/'server-report.json',result['server'],exclusive=True)
                verify_pair(result['client'],result['server'],kind)
                require(client_rc==0 and processes[server_name].returncode==0,'Worker exit code differs')
                result['passed']=True
            except Exception as exc:
                result['error']=repr(exc)
            finally:
                stop_owned(processes.get(client_name))
                cancel_remote(server_name)
                if 'client' not in result and (folder/'client-report.json').is_file():
                    result['client']=old.read(folder/'client-report.json')
                old.write(folder/'case-report.json',result,exclusive=True)
            boundary()
            require(not report.get('remote_exit_unconfirmed'),'Remote worker exit unconfirmed; no further scenarios')
            print(name+'：'+('通过' if result['passed'] else '未通过，结果已保留'),flush=True)
            return result
        try:
            update(phase='verify_frozen_inputs',package_identity=old.verify_package())
            prior=timing_source.validate_prior(timing_source.PRIOR_ROOT/'report.json')
            report['accepted_loop_report_sha256']=timing_source.REPORT_SHA
            before=old.verify_inputs()
            require(sha(old.SLOT/'report.json')==REPORT_SHA,'Original validation report changed')
            report['frozen_export']=frozen(EXPORT)
            require(shutil.disk_usage(ROOT).free>=512*1024**2,'Less than 512 MiB free; no files deleted')
            update(phase='profile_100_local_control_steps')
            profile=run/'profile';protected_mkdir(profile)
            launch('pc-profile',[str(old.PYTHON),'-B','-u',str(HERE/'profile_sim.py'),'--folder',str(profile)],old.clean_environment())
            wait_local('pc-profile',210)
            report['pc_profile']=old.read(profile/'profile-report.json')
            require(report['pc_profile']['status']=='PROFILE_COMPLETE','PC profiler incomplete')
            update(phase='prepare_isolated_quantization_candidate')
            candidates=run/'candidates';protected_mkdir(candidates)
            launch('candidate-preparation',[str(old.PYTHON),'-B','-u',str(HERE/'prepare_candidates.py'),
                '--folder',str(candidates),'--export',str(EXPORT)],old.clean_environment())
            wait_local('candidate-preparation',120)
            report['candidate_preparation']=old.read(candidates/'candidate-report.json')
            values,source=timing_source.load_recording()
            old.write(run/'observations.json',dict(source=source,values=values),exclusive=True)
            report['source_recording']=source
            boundary()
            update(phase='verify_wired_hosts')
            pc=probe.probe();pc_ready=probe.readiness(pc)
            require(pc_ready['ready'],'PC Humble/link: '+str(pc_ready['blockers']))
            shared_ssh.connect()
            jetson=json.loads(remote('/usr/bin/python3 -B -',(HERE/'probe.py').read_bytes(),timeout=80))
            jetson_ready=probe.readiness(jetson,jetson=True)
            require(jetson_ready['ready'],'Orin Humble/link: '+str(jetson_ready['blockers']))
            model=(jetson.get('board_model') or '').lower()
            require('orin' in model and 'nano' in model,'Remote board identity differs')
            update(pc_probe=pc,jetson_probe=jetson,pc_readiness=pc_ready,jetson_readiness=jetson_ready)
            destination='/home/jetson/microduck-stage12/ros2-humble-opt-v1/runs/'+session
            create="import pathlib,json,sys\np=pathlib.Path(sys.argv[1]);cur=pathlib.Path('/')\nfor part in p.parts[1:]:\n cur=cur/part\n if cur.is_symlink():raise RuntimeError('Unexpected remote directory symlink')\np.parent.mkdir(parents=True,exist_ok=True);p.mkdir(mode=0o700)\nprint(json.dumps({'path':str(p)}))"
            remote_json(create,[destination])
            for name,address in (('pc',PC_IP),('orin',ORIN_IP)):
                (run/(name+'-fastdds.xml')).write_text(wired_xml(address))
            files=[EXPORT/n for n in ('final6h_005304.onnx','contract.json','golden.json','export-report.json')]
            files += [HERE/n for n in ('loop_contract.py','loop_ros.py','deploy_contract.py','timing_nodes.py',
                                      'timing_contract.py','wire_protocol.py','fast_policy.py','bench_policies.py')]
            files += [run/'orin-fastdds.xml',run/'observations.json',candidates/'candidate-report.json']
            if report['candidate_preparation']['status']=='QUANTIZATION_CANDIDATE_PREPARED':
                files.append(candidates/'mlp-dynamic-int8.onnx')
            update(phase='wired_transfer',remote_directory=destination)
            report['transfer']=transfer(files)
            update(phase='interleaved_orin_policy_benchmark')
            launch_remote('orin-benchmark',[destination+'/bench_policies.py','--folder',destination,
                '--cancel-file',destination+'/benchmark-cancel'],destination+'/benchmark-cancel',220)
            wait_local('orin-benchmark',230,required=False)
            bench=remote_json('import sys;print(open(sys.argv[1]).read())',[destination+'/policy-benchmark.json'])
            old.write(run/'policy-benchmark.json',bench,exclusive=True);report['policy_benchmark']=bench
            require(bench['status']=='POLICY_BENCHMARK_COMPLETE' and processes['orin-benchmark'].returncode==0,
                    'Original-policy benchmark failed')
            variants=[('json-fp32','json','fp32'),('binary-fp32','binary','fp32')]
            if bench['selected_FP32_runtime']=='buffered_fp32':
                variants.append(('binary-buffered-fp32','binary','buffered_fp32'))
            for name,wire,runtime in variants:
                report['replay_variants'].append(scenario(name,wire,runtime,'replay'))
                update()
            preview=recommend(report['replay_variants'],[])
            chosen=next((r for r in report['replay_variants'] if r['name']==preview.get('variant')),None)
            if chosen is not None:
                for kind in ('delay','silence','duplicate'):
                    report['fault_cases'].append(scenario('fault-'+kind,chosen['wire'],chosen['runtime'],kind))
                    update()
            report['recommendation']=recommend(report['replay_variants'],report['fault_cases'])
            complete=all(r['passed'] for r in report['replay_variants']) and len(report['fault_cases'])==3 \
                     and all(r['passed'] for r in report['fault_cases'])
            update(status='OPTIMIZATION_MEASURED' if complete else 'OPTIMIZATION_CHECK_FAILED_PRESERVE')
        except BaseException as exc:
            update(status='OPTIMIZATION_CHECK_FAILED_PRESERVE',errors=[repr(exc)],traceback=traceback.format_exc()[-12000:])
        finally:
            old.write(run/'STOP.json',dict(reason='controller_finalization',utc=time.time()))
            for name,process in processes.items():
                if name not in remote_jobs:
                    stop_owned(process)
            for name in remote_jobs:
                try:cancel_remote(name)
                except Exception as exc:
                    report.setdefault('remote_exit_unconfirmed',[]).append(name)
                    report['errors'].append(repr(exc));stop_owned(processes.get(name))
            for stream in streams:stream.close()
            for field,path in [('pc_profile',run/'profile/profile-report.json'),
                               ('candidate_preparation',run/'candidates/candidate-report.json')]:
                if path.is_file():report[field]=old.read(path)
            if before is not None:
                try:
                    report['original_records_unchanged']=old.verify_inputs()==before and sha(old.SLOT/'report.json')==REPORT_SHA
                    timing_source.validate_prior(timing_source.PRIOR_ROOT/'report.json');frozen(EXPORT)
                    require(report['original_records_unchanged'],'Frozen records changed')
                except Exception as exc:
                    report['status']='OPTIMIZATION_CHECK_FAILED_PRESERVE';report['errors'].append(repr(exc))
            report['local_worker_exit_codes']={n:p.poll() for n,p in processes.items()}
            report['local_workers_exited']=all(p.poll() is not None for p in processes.values())
            report['remote_workers_exited']=all(processes[n].returncode in (0,2) for n in remote_jobs) and not report.get('remote_exit_unconfirmed')
            if not report['remote_workers_exited']:
                report['status']='OPTIMIZATION_CHECK_FAILED_PRESERVE'
            report['log_tails']={p.name:p.read_text(errors='replace')[-12000:] for p in run.glob('*.log')}
            update(phase='finished',finished_utc=time.time(),ssh_connection=shared_ssh.DIAGNOSTICS)
            print('HUMBLE_OPT_STATUS='+report['status'],flush=True)
            print('HUMBLE_OPT_REPORT='+str(ROOT/'report.json'),flush=True)
            if report['errors']:print('待处理：'+'; '.join(report['errors']),flush=True)
        return 0 if report['status']=='OPTIMIZATION_MEASURED' else 2


def show(watch=False):
    while True:
        path=ROOT/'report.json'
        if not path.is_file():
            print('尚未运行优化测量。',flush=True);code=2
        else:
            report=old.read(path);status=report['status']
            if status=='RUNNING' and not old.alive(report.get('controller')):status='CONTROLLER_EXITED_PRESERVE'
            current=Path(report.get('active_case_directory',report['directory']))/'progress.json'
            progress=old.read(current) if current.is_file() else {}
            print('%s | %s | %d/%d | %.1f%% | %.0f 秒' % (status,report.get('phase',''),
                progress.get('completed',0),progress.get('total',0),progress.get('percent',0.),
                report.get('finished_utc',time.time())-report['started_utc']),flush=True)
            if not watch:
                print('HUMBLE_OPT_REPORT='+str(path),flush=True)
                if report.get('errors'):print('待处理：'+'; '.join(report['errors']),flush=True)
            code=0 if status=='OPTIMIZATION_MEASURED' else 2
        if not watch:return code
        time.sleep(2)


def stop():
    report=old.read(ROOT/'report.json');run=Path(report['directory'])
    require(not run.is_symlink() and run.parent.resolve()==(ROOT/'runs').resolve(),'Unexpected run directory')
    require(report['status']=='RUNNING' and old.alive(report.get('controller')),'No live owned optimizer')
    old.write(run/'STOP.json',dict(reason='user_requested_stop',utc=time.time()))
    print('已请求停止本轮测量，保留结果。',flush=True);return 0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=('run','retry-run','status','watch','stop'))
    args=parser.parse_args()
    if args.operation in ('status','watch'):return show(args.operation=='watch')
    if args.operation=='stop':return stop()
    def interrupt(signum,frame):raise KeyboardInterrupt('Controller signal '+str(signum))
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,interrupt)
    return execute(args.operation=='retry-run')


if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:raise SystemExit(130)
    except Exception as exc:
        print('HUMBLE_OPT_PRESERVED='+repr(exc),flush=True);raise SystemExit(2)
