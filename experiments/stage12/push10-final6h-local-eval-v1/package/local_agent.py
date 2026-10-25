"""One detached local candidate-selection and heldout run; no training path."""
import argparse
import fcntl
import signal
import time
import traceback
import zipfile
from common import *
import budget as account
from eval_plan import choose_candidate,case_plan
from eval_criteria import summarize_phase

CLEANUP_SECONDS=90.


def optional(name,default=None):
    path=safe(SLOT,name)
    return read(path) if path.is_file() else default


def process_pin(child):
    until=time.monotonic()+2.
    while child.poll() is None:
        pin=identity(child.pid)
        if pin is not None:return pin
        if time.monotonic()>=until:break
        time.sleep(.01)
    return None


def signal_owned(child,pin,sig):
    if child.poll() is not None:return
    current=identity(child.pid)
    require(pin and current and current['start']==pin['start']
            and current['group']==pin['group']==child.pid,'Owned child identity differs; preserve')
    if current['state'] not in ('Z','X'):os.killpg(pin['group'],sig)


def terminal_saved(path,phase,checkpoint_sha256):
    if not path.is_file() or path.is_symlink():return False
    value=read(path)
    return value.get('phase')==phase and (value.get('checkpoint') or {}).get('sha256')==checkpoint_sha256 \
        and value.get('status') in ('EVALUATION_COMPLETE','EVALUATION_ERROR_PRESERVE','EVALUATION_INCOMPLETE_PRESERVE') \
        and type(value.get('finished_utc')) in (int,float)


def run_worker(phase,receipt,folder,number,fd,deadline,stopped):
    logpath=SLOT/'logs'/('%03d-%s.log'%(number,phase))
    result_path=folder/'evaluation-result.json'
    checkpoint_sha256=read(receipt)['sha256']
    child=pin=None;grace=None;term_sent=False;stop_reason=None;signals=[]
    outcome=None
    try:
        with logpath.open('xb') as log:
            child=subprocess.Popen([str(PYTHON),'-B','-u',str(HERE/'evaluate.py'),
                '--phase',phase,'--checkpoint-json',str(receipt),'--run-dir',str(folder)],
                cwd=PROJECT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,pass_fds=(fd,),env=clean_environment())
        pin=process_pin(child)
        if pin is None and child.poll() is None:
            write(SLOT/'uncertain-worker.json',dict(pid=child.pid,log=str(logpath)),exclusive=True)
            raise RuntimeError('Worker identity unavailable; preserve, do not launch another')
        if pin:write(SLOT/'child.json',pin)
        write(SLOT/'active-worker.json',dict(phase=phase,number=number,pin=pin,
            result_path=str(result_path),log=str(logpath),checkpoint_sha256=checkpoint_sha256))
        while child.poll() is None:
            complete=terminal_saved(result_path,phase,checkpoint_sha256)
            stopped_now=stopped[0] or (SLOT/'STOP.json').exists()
            expired=time.time()>=deadline
            if complete and grace is None:
                grace=time.monotonic()+CLEANUP_SECONDS
            if (stopped_now or expired) and not complete and not term_sent:
                stop_reason='user_stop' if stopped_now else 'evaluation_deadline'
                signal_owned(child,pin,signal.SIGTERM);signals.append('SIGTERM');term_sent=True
                grace=time.monotonic()+CLEANUP_SECONDS
            if grace is not None and time.monotonic()>=grace:
                stop_reason='cleanup_grace_exceeded'
                signal_owned(child,pin,signal.SIGKILL);signals.append('SIGKILL')
            try:child.wait(timeout=.25)
            except subprocess.TimeoutExpired:pass
        outcome=dict(worker_exit_code=child.returncode,stop_reason=stop_reason,
            worker_alive=False,signals=signals,result_saved=terminal_saved(result_path,phase,checkpoint_sha256),
            result_sha256=sha(result_path) if result_path.is_file() else None,
            finished_utc=time.time(),log=str(logpath))
        return outcome
    finally:
        cleanup_error=None
        try:
            if child is not None and child.poll() is None:
                signal_owned(child,pin,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    signal_owned(child,pin,signal.SIGKILL);child.wait(timeout=10)
        except BaseException as exc:
            cleanup_error=repr(exc)
            if not (SLOT/'uncertain-worker.json').exists():
                write(SLOT/'uncertain-worker.json',dict(pid=child.pid,error=cleanup_error),exclusive=True)
        # Preserve the actual exit status even when the caller rejects it.
        if child is not None:
            record=outcome or dict(worker_exit_code=child.returncode,worker_alive=child.poll() is None,
                stop_reason=stop_reason,signals=signals,unexpected_supervisor_error=True,log=str(logpath))
            if cleanup_error:record['cleanup_error']=cleanup_error
            write(SLOT/'logs'/('%03d-finished.json'%number),record,exclusive=True)


def case_summary(row):
    return dict(case_id=row['case_id'],seed=row['seed'],force_seed=row['force_seed'],
        status=row['status'],acceptance=row['acceptance'],
        completed_control_steps=row.get('completed_control_steps',0),
        duration_seconds=(row.get('trial') or {}).get('duration_seconds'),
        completed_pulses=(row.get('disturbance') or {}).get('completed_pulses'),
        first_limit_event=(row.get('model_limits') or {}).get('first_limit_event'),
        physical_terminations=(row.get('trial') or {}).get('terminations'),
        error=row.get('error'),cleanup_error=row.get('cleanup_error'))


def phase_summary(value):
    return dict(phase=value['phase'],status=value['status'],checkpoint=value['checkpoint'],
        summary=value['summary'],cases=[case_summary(row) for row in value['cases']],
        evidence=value['evidence'])


def phase_completed(outcome,value):
    return outcome['worker_exit_code']==0 and value['status']=='EVALUATION_COMPLETE' \
        and outcome['stop_reason'] in (None,'evaluation_deadline')


def supervise(fd):
    os.fstat(fd);started=read(SLOT/'started.json');ledger=read(SLOT/'budget.json');account.validate(ledger)
    require(started['deadline_utc']==ledger['deadline_utc'] and started['package_identity']==verify_package(),
            'Local validation identity changed')
    write(SLOT/'supervisor-started.json',dict(pid=os.getpid(),started_utc=time.time()),exclusive=True)
    stopped=[False]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    selections=[];selection=heldout=None;error=None;status='EVALUATION_ERROR_PRESERVE'
    def boundary():
        require(not stopped[0] and not (SLOT/'STOP.json').exists(),'STOP_REQUESTED')
        require(time.time()<started['deadline_utc'],'LOCAL_EVALUATION_DEADLINE')
    def evaluate_phase(phase,receipt,number,index=None):
        boundary()
        folder=SLOT/'evaluations'/('heldout' if phase=='heldout' else 'selection-%03d'%index)
        write(SLOT/'progress.json',dict(status='EVALUATING',phase=phase,detail=folder.name,
            checkpoint=read(receipt),control_steps=0,completed_pulses=0,
            deadline_utc=started['deadline_utc'],added_PPO_updates=0))
        outcome=run_worker(phase,receipt,folder,number,fd,started['deadline_utc'],stopped)
        result_path=folder/'evaluation-result.json'
        require(result_path.is_file(),'Evaluation worker produced no result; inspect its saved exit record')
        value=read(result_path)
        require(phase_completed(outcome,value),'Evaluation incomplete/runtime failure: '+folder.name)
        require(value['checkpoint']==read(receipt) and value['phase']==phase,'Evaluation candidate identity differs')
        require(value['summary']==summarize_phase(value['cases'],phase,value['checkpoint']['sha256']),
                'Evaluation summary differs from retained cases')
        for item in value['evidence']:
            path=Path(item['path'])
            require(path.resolve().is_relative_to(folder.resolve()) and sha(path)==item['sha256'],
                    'Case evidence changed')
        write(folder/'supervisor-verified.json',dict(process=outcome,sha256=sha(result_path)),exclusive=True)
        return value
    try:
        for index,receipt in enumerate(started['candidates'],1):
            selections.append(evaluate_phase('selection',Path(receipt),index,index))
        selection=choose_candidate(selections)
        write(SLOT/'selection.json',selection,exclusive=True)
        if selection['qualified']:
            locked=SLOT/'selected-candidate.json';write(locked,selection['checkpoint'],exclusive=True)
            heldout=evaluate_phase('heldout',locked,4)
            status='VALIDATION_PASSED' if heldout['summary']['all_passed'] else 'HELDOUT_FAILED'
        else:status='NO_QUALIFIED_CANDIDATE'
    except BaseException as exc:
        error=dict(error=repr(exc),traceback=traceback.format_exc()[-10000:])
        if stopped[0] or (SLOT/'STOP.json').exists():status='VALIDATION_STOPPED_PRESERVE'
        elif time.time()>=started['deadline_utc']:status='VALIDATION_TIME_LIMIT_PRESERVE'
    try:unchanged=verify_inputs()==read(SLOT/'preflight.json')['inputs']
    except BaseException as exc:
        unchanged=False;error=dict(error=repr(exc),earlier_error=error)
    if not unchanged:status='EVALUATION_ERROR_PRESERVE'
    ledger=read(SLOT/'budget.json');account.validate(ledger)
    quiescent=not (SLOT/'uncertain-worker.json').exists() and not alive(optional('child.json'))
    if not quiescent:status='EVALUATION_ERROR_PRESERVE'
    passed=status=='VALIDATION_PASSED' and unchanged and quiescent
    report=dict(schema='stage12-final6h-local-evaluation-report-v1',status=status,
        started_utc=started['started_utc'],finished_utc=time.time(),deadline_utc=started['deadline_utc'],
        package_identity=started['package_identity'],source_return_commit=RETURN_COMMIT,
        input_training_updates=5304,input_total_policy_updates=11314,
        added_PPO_updates=0,training_started=False,cloud_connections=0,power_action='NONE',cloud_power_state='UNCONFIRMED',
        original_training_records_unchanged=unchanged,compute_workers_exited=quiescent,
        evaluation_budget=ledger,maximum_cases=27,evaluated_cases=len(ledger['evaluations']),
        selections=[phase_summary(value) for value in selections],
        chosen_checkpoint=selection['checkpoint'] if selection else None,
        qualified_candidate_found=bool(selection and selection['qualified']),
        heldout=phase_summary(heldout) if heldout else None,
        disturbance_goal_passed=passed,baseline_goal_passed=passed,
        diagnostic_candidate_only=bool(selection and not selection['qualified']),
        all_possible_sequences_proven=False,old_64_case_suite_claimed_passed=False,
        hardware_acceptance=False,strict_contact_acceptance=False,
        automatic_retry=False,automatic_training=False,error=error)
    active=optional('active-worker.json',{})
    if error and active.get('result_path'):
        path=Path(active['result_path'])
        if path.is_file() and path.resolve().is_relative_to(SLOT.resolve()):report['last_worker_result']=read(path)
    write(SLOT/'report.json',report,exclusive=True)
    write(SLOT/'progress.json',dict(status=status,phase='finished',added_PPO_updates=0))
    return report


def launch():
    require(not SLOT.is_symlink(),'Unexpected local validation symlink')
    SLOT.mkdir(parents=True,exist_ok=True)
    lease_path=BASE/'artifacts/stage12-local-evaluation.lock'
    lease_path.parent.mkdir(parents=True,exist_ok=True)
    with lease_path.open('a') as lease:
        try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            require((SLOT/'started.json').exists(),'Another local validation holds the GPU lease')
            return
        if (SLOT/'started.json').exists():return
        require(not any((SLOT/name).exists() for name in ('budget.json','report.json','supervisor.json','child.json')),
                'Orphan local validation records exist; preserve, never reset')
        checks=preflight();candidates=candidate_inputs()
        for name in ('candidates','evaluations','logs'):(SLOT/name).mkdir(exist_ok=True)
        paths=[]
        for cp in candidates:
            path=SLOT/'candidates'/(cp['candidate_label']+'.json');write(path,cp,exclusive=True);paths.append(str(path))
        now=time.time();ledger=account.new(now)
        started=dict(schema='stage12-final6h-local-eval-v1',started_utc=now,deadline_utc=ledger['deadline_utc'],
            package_identity=checks['package_identity'],maximum_cases=27,max_compute_seconds=MAX_SECONDS,
            candidates=paths,selection_plan=case_plan('selection'),heldout_plan=case_plan('heldout'),
            source_return_commit=RETURN_COMMIT,device='local RTX5060',added_PPO_updates=0,
            original_training_budget_reopened=False,automatic_retry=False,training_authorized=False,
            authorization='User continued to frozen-model selection and independent validation after final training')
        write(SLOT/'started.json',started,exclusive=True);write(SLOT/'budget.json',ledger,exclusive=True)
        write(SLOT/'preflight.json',checks,exclusive=True)
        write(SLOT/'progress.json',dict(status='STARTING',phase='selection',added_PPO_updates=0))
        child=None
        try:
            with (SLOT/'logs/supervisor.log').open('xb') as log:
                child=subprocess.Popen([str(PYTHON),'-B','-u',str(HERE/'local_agent.py'),'supervise',
                    '--lock-fd',str(lease.fileno())],cwd=PROJECT,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    pass_fds=(lease.fileno(),),env=clean_environment())
            pin=process_pin(child);require(pin is not None,'Supervisor launch uncertain; preserve reservation')
            write(SLOT/'supervisor.json',pin,exclusive=True)
        except BaseException as exc:
            write(SLOT/'launch-error.json',dict(error=repr(exc),automatic_retry=False,
                  supervisor_may_be_running=child is not None and child.poll() is None),exclusive=True)
            raise


def display():
    report=optional('report.json')
    if report:
        print('验证结束：'+report['status']+'；独立抗扰验收通过：'+str(report['disturbance_goal_passed']),flush=True)
        print('LOCAL_FINAL6H_EVAL_REPORT_SAVED='+str(SLOT/'report.json'),flush=True)
        print('本次新增PPO=0；未连接云主机。',flush=True)
        if report.get('error'):print('错误：'+json.dumps(report['error'],ensure_ascii=False),flush=True)
        return True
    if not (SLOT/'started.json').exists():print('本机验证尚未启动。',flush=True);return True
    progress=optional('progress.json',{});ledger=optional('budget.json',{})
    cp=progress.get('checkpoint') or {};done=len(ledger.get('evaluations',{}))
    print('本机选模/验证：'+str(progress.get('status','STARTING'))+' | '+str(progress.get('phase',''))
          +' | '+str(cp.get('candidate_label',''))+' | 本回合 %.2f/60秒 | 已登记 %d/27 回合'
          %(progress.get('control_steps',0)*.02,done),flush=True)
    if progress.get('completed_pulses') is not None:
        print('已完成扰动脉冲：'+str(progress['completed_pulses'])+'；边界状态：'+str(progress.get('model_bounds_status','待检查')),flush=True)
    if optional('launch-error.json'):print('启动异常记录：'+str(SLOT/'launch-error.json'),flush=True)
    return False


def watch():
    try:
        while True:
            if display():return
            started=read(SLOT/'started.json');active=alive(optional('supervisor.json'))
            if not active and time.time()-started['started_utc']>5:
                if (SLOT/'report.json').exists():continue
                print('后台进程已退出且没有完整报告；保留现场，不重新启动。日志：'+str(SLOT/'logs/supervisor.log'),flush=True)
                return
            time.sleep(2.)
    except KeyboardInterrupt:
        print('已退出监控，本机后台验证继续；本机须保持开机、唤醒。重新查看用 --watch。',flush=True)


def pack():
    require((SLOT/'report.json').is_file() and not alive(optional('child.json')),
            'Evaluation is not complete; inspect --status first')
    require(read(SLOT/'report.json')['compute_workers_exited'],'Worker exit not confirmed')
    output=SLOT/'review.zip'
    if output.exists():print('LOCAL_FINAL6H_EVAL_REVIEW='+str(output),flush=True);return
    names=[]
    for p in SLOT.rglob('*'):
        if not p.is_file() or p.is_symlink():continue
        relative=p.relative_to(SLOT)
        if relative.parts[0] in ('package','candidates','evaluations','logs') or p.name in (
                'report.json','started.json','budget.json','selection.json','selected-candidate.json','preflight.json'):
            names.append(p)
    require(sum(p.stat().st_size for p in names)<200*1024*1024,'Review data unexpectedly large; preserve')
    temp=output.with_suffix('.zip.partial')
    require(not temp.exists(),'Earlier partial review archive exists; preserve')
    with zipfile.ZipFile(temp,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(names):z.write(p,p.relative_to(SLOT).as_posix())
    os.replace(temp,output)
    print('LOCAL_FINAL6H_EVAL_REVIEW='+str(output),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('operation',choices=['run','supervise','watch','status','stop','check','pack'])
    parser.add_argument('--lock-fd',type=int)
    args=parser.parse_args()
    if args.operation=='run':launch();watch()
    elif args.operation=='supervise':supervise(args.lock_fd)
    elif args.operation=='watch':watch()
    elif args.operation=='status':display()
    elif args.operation=='check':print(json.dumps(preflight(),ensure_ascii=False,indent=2))
    elif args.operation=='pack':pack()
    elif args.operation=='stop':
        if (SLOT/'started.json').exists() and not (SLOT/'report.json').exists() and not (SLOT/'STOP.json').exists():
            write(SLOT/'STOP.json',dict(requested_utc=time.time(),reason='user_stop'),exclusive=True)
        display()


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('入口已中断；用 --status 查看现有任务。',flush=True)
    except BaseException as exc:
        if isinstance(exc,SystemExit):raise
        print('LOCAL_FINAL6H_EVAL_PRESERVED='+repr(exc),flush=True)
        raise SystemExit(2)
