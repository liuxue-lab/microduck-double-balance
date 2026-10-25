"""Run the actual supervisor import and report review with site packages disabled."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from support import ROOT,BASE,case_row,c

class SystemPythonReview(unittest.TestCase):
    def test_report_review_and_supervisor_import_need_no_torch(self):
        shared=dict(completed_requests=6000,transcript_sha256='a'*64)
        fixture=dict(simulator=dict(status='LOOP_SIM_FUNCTIONAL_PASSED',original_records_unchanged=True,
            completed_control_steps=6000,cases=[case_row(x) for x in c.CASES],transport=shared,
            full_loop_realtime_acceptance=False),bridge=dict(status='LOOP_BRIDGE_COMPLETE',**shared),
            orin_server=dict(status='LOOP_SERVER_COMPLETE',model_unchanged=True,
                recurrent_reset_sequences=[0,3000],**shared))
        code='''import json,sys
sys.path.insert(0,sys.argv[1])
import bootstrap
old=bootstrap.load_base(__import__('pathlib').Path(sys.argv[2]))
bootstrap.load_base=lambda:old
import loop_ops
from loop_evidence import verify_reports
r=json.load(open(sys.argv[3]))
a=verify_reports(r['simulator'],r['bridge'],r['orin_server'])
assert a['functional_closed_loop'] and not a['full_loop_realtime_acceptance']
assert not ({'torch','numpy','mujoco','mjlab','warp'} & set(sys.modules))
print('STDLIB_REVIEW_PASSED')
'''
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'fixture.json';path.write_text(json.dumps(fixture))
            for report in (path,Path(os.environ.get('MICRODUCK_REVIEW_REPORT',str(path)))):
                result=subprocess.run([sys.executable,'-I','-S','-c',code,str(ROOT/'package'),str(BASE),str(report)],
                    capture_output=True,text=True,timeout=20)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertEqual(result.stdout.strip(),'STDLIB_REVIEW_PASSED')
