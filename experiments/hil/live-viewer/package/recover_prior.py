"""Revalidate completed native-v1 evidence without importing or running a simulator."""
import csv
from pathlib import Path

from bootstrap import STAGE,require,sha
from loop_evidence import reviewed_evidence,verify_reports,read


def recover(destination,write):
    reviewed=reviewed_evidence()
    report=read(reviewed['report_path'])
    run=Path(report['directory'])
    require(not run.is_symlink() and run.parent.resolve()==(STAGE/'native-mujoco-v1/runs').resolve(),
            'Unexpected prior run directory')
    files={}
    for name,pin in report['exported_data'].items():
        relative=Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts,'Unexpected export path')
        path=run
        for part in relative.parts:
            path=path/part
            require(not path.is_symlink(),'Unexpected old export symlink')
        require(path.is_file() and path.stat().st_size==pin['bytes'] and sha(path)==pin['sha256'],
                'Prior export changed: '+name)
        files[name]=pin
    cases=[]
    for i,case in enumerate(report['simulator']['cases'],1):
        key='case-%02d/timing.csv'%i
        require(key in files,'Prior timing CSV missing')
        with (run/key).open(newline='') as stream:rows=list(csv.DictReader(stream))
        require(len(rows)==3000 and [int(r['step']) for r in rows]==list(range(1,3001)),
                'Prior timing sequence incomplete')
        release=int(rows[0]['release_ns'])
        require(all(int(r['release_ns'])==release+j*20_000_000 and
                int(r['deadline_ns'])==int(r['release_ns'])+20_000_000 for j,r in enumerate(rows)),
                'Prior release schedule changed')
        late=[r for r in rows if int(r['finished_ns'])>int(r['deadline_ns'])]
        require(len(late)==case['timing']['deadline_misses'],'Prior CSV and summary disagree')
        cases.append(dict(case_id=case['case_id'],deadline_misses=len(late),
            late_progress_ticks=sum(int(r['step'])%25==0 for r in late),
            late_other_ticks=sum(int(r['step'])%25!=0 for r in late),
            late_step_numbers=[int(r['step']) for r in late]))
    acceptance=verify_reports(report['simulator'],report['bridge'],report['orin_server'])
    result=dict(schema='stage12-native-v1-recovered-summary-v1',
        status='FUNCTIONAL_PASSED_REALTIME_PENDING' if not acceptance['full_loop_realtime_acceptance'] else 'REALTIME_PASSED',
        reviewed=reviewed,acceptance=acceptance,prior_timing_analysis=cases,
        verified_exports=files,prior_report_preserved=True,
        prior_auxiliary_audit_clock=dict(recorded_seconds=24.,correct_seconds=60.,
            reason='v1 model-limit duration display used 2ms; actual physics and control counts show 60s',
            old_numbers_preserved=True,physical_limit_checks_unchanged=True),new_physics_steps=0,new_Orin_requests=0,
        dependency_fix='report-only predicates no longer import torch/native runtime')
    path=Path(destination)/'recovered-v1.json'
    if path.exists():require(read(path)==result,'Existing recovery receipt differs; preserve')
    else:write(path,result,exclusive=True)
    return dict(path=str(path),sha256=sha(path),status=result['status'],timing_analysis=cases)
