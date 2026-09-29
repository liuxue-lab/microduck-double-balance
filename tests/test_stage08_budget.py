"""Persistent clock/restart and independent watchdog tests; CPU subprocesses only."""
from datetime import datetime,timezone
import json
import sys
import time

import pytest
from mjlab_microduck.double_balance_stage08_budget import BudgetLedger,create_ledger,supervise


def stamp(epoch):
    return datetime.fromtimestamp(epoch,timezone.utc).isoformat()


def test_idle_restart_time_and_final_reserve_are_counted(tmp_path):
    path=tmp_path/'budget.json'
    create_ledger(path,'5090',stamp(1000),now=1000)
    clock=[1000.]
    ledger=BudgetLedger(path,wall=lambda:clock[0],monotonic=lambda:0.)
    clock[0]+=10*3600
    assert ledger.remaining(training=True)==42*3600
    restarted=BudgetLedger(path,wall=lambda:clock[0]+20*3600,monotonic=lambda:0.)
    assert restarted.remaining(training=True)==22*3600
    assert restarted.identity==ledger.identity
    with pytest.raises(FileExistsError):
        create_ledger(path,'5090',stamp(clock[0]),now=clock[0])


def test_clock_rollback_and_changed_budget_fail_closed(tmp_path):
    path=tmp_path/'budget.json'
    create_ledger(path,'A800',stamp(1000),now=2000)
    with pytest.raises(ValueError,match='backwards'):
        BudgetLedger(path,wall=lambda:1500,monotonic=lambda:0)
    state=json.loads(path.read_text()); state['maximum_cloud_hours']=48
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError,match='Budget contract'):
        BudgetLedger(path,wall=lambda:2000,monotonic=lambda:0)


def test_no_concurrent_campaign_jobs(tmp_path):
    path=tmp_path/'budget.json'; now=time.time()
    create_ledger(path,'A800',stamp(now),now=now)
    first,second=BudgetLedger(path),BudgetLedger(path)
    with first.job_lock():
        with pytest.raises(ValueError,match='Another'):
            with second.job_lock():
                pass


def test_watchdog_terminates_hung_child_and_retains_ledger(tmp_path):
    path=tmp_path/'budget.json'; now=time.time()
    create_ledger(path,'A800',stamp(now),now=now)
    ledger=BudgetLedger(path)
    result=supervise([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(10)'],
                     ledger,training=True,timeout_seconds=.3,grace_seconds=.2,poll_seconds=.02)
    assert result['exit_code']==-9 and result['watchdog_reason']=='forced_exit_after_grace'
    assert not result['shutdown_performed']
    assert ledger.snapshot()['elapsed_seconds']>=.5
