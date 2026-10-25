"""Recovery validates preserved files and never starts another worker."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from support import ROOT,c,case_row
import loop_evidence
import recover_prior

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value,exclusive=False):
    with Path(p).open('x' if exclusive else 'w') as f:json.dump(value,f)

class Recovery(unittest.TestCase):
    def test_existing_results_recovered_once_and_changed_csv_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            stage=Path(d);prior=stage/'native-mujoco-v1';run=prior/'runs/fixture';run.mkdir(parents=True)
            dest=stage/'native-mujoco-v2';dest.mkdir()
            source=ROOT.parent/'native-mujoco-v1/package'
            if not source.is_dir():self.skipTest('Previous pinned package fixture required')
            shutil.copytree(source,prior/'package',ignore=shutil.ignore_patterns('__pycache__'))
            write(prior/'code-installed.json',dict(commit=loop_evidence.PRIOR_COMMIT,package_identity=loop_evidence.PRIOR_ID))
            shared=dict(completed_requests=6000,transcript_sha256='a'*64)
            cases=[case_row(x) for x in c.CASES];files={}
            for i,case in enumerate(cases,1):
                case['model_limits']['simulated_seconds']=24.
                key='case-%02d/timing.csv'%i;p=run/key;p.parent.mkdir()
                with p.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=['step','release_ns','deadline_ns','finished_ns']);w.writeheader()
                    for n in range(3000):
                        start=1_000_000_000+n*20_000_000
                        w.writerow(dict(step=n+1,release_ns=start,deadline_ns=start+20_000_000,
                            finished_ns=start+(25_000_000 if n==24 else 12_000_000)))
                files[key]=dict(bytes=p.stat().st_size,sha256=digest(p))
            report=dict(directory=str(run),status='NATIVE_MUJOCO_FAILED_PRESERVE',
                errors=["ModuleNotFoundError(\"No module named 'torch'\")"],original_records_unchanged=True,
                package_identity=loop_evidence.PRIOR_ID,local_workers_exited=True,
                local_worker_exit_codes={k:0 for k in ('preflight','simulator','pc-bridge','orin-server')},
                exported_data=files,simulator=dict(status='LOOP_SIM_FUNCTIONAL_PASSED',original_records_unchanged=True,
                    completed_control_steps=6000,cases=cases,transport=shared,full_loop_realtime_acceptance=False),
                bridge=dict(status='LOOP_BRIDGE_COMPLETE',**shared),
                orin_server=dict(status='LOOP_SERVER_COMPLETE',model_unchanged=True,recurrent_reset_sequences=[0,3000],**shared))
            report_path=prior/'report.json';write(report_path,report);before=digest(report_path)
            with patch.object(loop_evidence,'STAGE',stage),patch.object(loop_evidence,'REPORT_SHA',before), \
                    patch.object(recover_prior,'STAGE',stage),patch('subprocess.Popen',side_effect=AssertionError('No workers')):
                a=recover_prior.recover(dest,write);b=recover_prior.recover(dest,write)
                self.assertEqual(a,b);self.assertEqual(a['status'],'FUNCTIONAL_PASSED_REALTIME_PENDING')
                self.assertTrue(all(x['late_progress_ticks']==1 for x in a['timing_analysis']))
                self.assertEqual(digest(report_path),before)
                p=run/'case-01/timing.csv';p.write_bytes(p.read_bytes()+b'changed')
                with self.assertRaisesRegex(RuntimeError,'Prior export changed'):recover_prior.recover(dest,write)
                self.assertEqual(digest(report_path),before)
