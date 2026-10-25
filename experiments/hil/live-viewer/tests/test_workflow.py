"""Actual supervisor subprocesses with synthetic workers; no Orin/network calls."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

HERE=Path(__file__).resolve().parents[1]


class Workflow(unittest.TestCase):
    def test_owned_workers_duplicate_stop_and_no_implicit_retry(self):
        baseline=Path(os.environ['MICRODUCK_REFERENCE_PACKAGE'])
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); package=root/'package'; package.mkdir()
            (package/'live_viewer.py').write_text('''import argparse,json,time,signal
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run');p.add_argument('--model');p.add_argument('--state');p.add_argument('--fd');a=p.parse_args()
r=Path(a.run);(r/'viewer-ready.json').write_text('{"status":"READY"}')
def end(*_):
 (r/'viewer-report.json').write_text('{"status":"LIVE_VIEWER_STOPPED_PRESERVE"}');raise SystemExit(2)
signal.signal(signal.SIGTERM,end)
while True:time.sleep(.02)
''')
            (package/'loop_sim.py').write_text('''import argparse,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run');p.add_argument('--session');p.add_argument('--socket');p.add_argument('--export');a=p.parse_args()
(Path(a.run)/'partial.txt').write_text('keep this partial run')
while True:time.sleep(.02)
''')
            for name in ('loop_ros.py','loop_contract.py','deploy_contract.py','probe.py'):
                (package/name).write_text('synthetic test worker fixture')
            worker=root/'transport.py'
            worker.write_text('''import json,sys,time
from pathlib import Path
r=Path(json.loads(Path(sys.argv[1]).read_text())['directory'])
if sys.argv[2]=='bridge':(r/'bridge-ready.json').write_text('{}')
while not (r/'STOP.json').exists():time.sleep(.02)
''')
            export=root/'export';export.mkdir()
            for name in ('final6h_005304.onnx','contract.json','golden.json','export-report.json'):
                (export/name).write_text('synthetic')
            slot=root/'slot';slot.mkdir();(slot/'report.json').write_text('{}')
            launcher=root/'launch.py'
            launcher.write_text('''import json,os,sys,subprocess
from pathlib import Path
sys.path.insert(0,%r)
import bootstrap
real_load=bootstrap.load_base
bootstrap.load_base=lambda:real_load(Path(%r))
import loop_ops as ops
import network_diagnostics
root=Path(%r)
ops.ROOT=root/'outputs';ops.HERE=root/'package';ops.EXPORT=root/'export'
ops.old.PROJECT=root;ops.old.PYTHON=Path(sys.executable);ops.old.SLOT=root/'slot'
# Some execution hosts expose the outer /proc with inner syscall PIDs.
# The production baseline's /proc implementation is unchanged. Only this
# synthetic-worker fixture supplies namespace-consistent identity tokens.
if int(Path('/proc/self/stat').read_text().split()[0]) != os.getpid():
 import uuid
 identities=root/'test-identities';identities.mkdir(exist_ok=True)
 def identity(pid):
  try:os.kill(int(pid),0)
  except ProcessLookupError:return None
  token=identities/str(pid)
  try:
   with token.open('x') as output:output.write(uuid.uuid4().hex)
  except FileExistsError:pass
  return dict(pid=int(pid),start=token.read_text(),state='R',group=os.getpgid(int(pid)))
 def alive(pin):
  current=identity(pin['pid']) if pin else None
  return bool(current and current['start']==pin['start'])
 ops.old.identity=identity;ops.old.alive=alive
ops.old.verify_inputs=lambda:{'unchanged':True}
ops.old.verify_package=lambda:'test-base'
ops.verify_own_package=lambda:'test-package'
ops.reviewed_evidence=lambda:{}
ops.historical_reports=lambda:{}
ops.previous_link=lambda:{}
ops.frozen=lambda *_:{}
ops.recover=lambda *_:{}
ops.review_v2=lambda *_:{}
ops.sha=lambda *_:'test-sha';ops.REPORT_SHA='test-sha'
ops.cpu_environment=lambda:dict(os.environ,DISPLAY='synthetic-test')
cpus=sorted(os.sched_getaffinity(0))[:3]
ops.cpu_placement.plan=lambda:dict(enabled=True,control_cpu=cpus[0],bridge_cpu=cpus[1],candidates=[dict(cpu=c,core=c,package=0,busy_fraction=0.,maximum_frequency_kHz=1000) for c in cpus])
ops.live_support.accepted_source=lambda:dict(paths={'case-01/native-model.mjb':'synthetic','case-01/state-data.npz':'synthetic'})
ops.preflight_argv=lambda path:[sys.executable,'-c','from pathlib import Path;Path('+repr(str(path))+').write_text("{}")']
ops.probe.probe=lambda:{}
ops.probe.readiness=lambda *_,**kw:dict(ready=True,blockers=[],python=sys.executable)
ops.shared_ssh.connect=lambda:Path('no-network-test')
ops.remote=lambda *_ ,**kw:b'{"board_model":"Jetson Orin Nano synthetic test"}'
ops.remote_json=lambda *_ ,**kw:dict(verified=8)
ops.shared_ssh.argv=lambda *args:[sys.executable,str(root/'transport.py'),str(ops.ROOT/'report.json'),'server']
ops.probe.shell_argv=lambda *args:[sys.executable,str(root/'transport.py'),str(ops.ROOT/'report.json'),'bridge']
original_run=subprocess.run
def local_run(argv,*args,**kwargs):
 if argv[0]=='scp':return subprocess.CompletedProcess(argv,0)
 return original_run(argv,*args,**kwargs)
subprocess.run=local_run
network_diagnostics.snapshot=lambda:dict(github_dns_resolves=False,synthetic_test=True)
if sys.argv[1]=='stop':raise SystemExit(ops.stop())
raise SystemExit(ops.execute(retry=sys.argv[1]=='retry'))
'''%(str(HERE/'package'),str(baseline),str(root)))
            env=dict(os.environ,DISPLAY='synthetic-test')
            log=(root/'parent.log').open('wb')
            process=subprocess.Popen([sys.executable,str(launcher),'run'],stdout=log,stderr=log,env=env)
            try:
                limit=time.monotonic()+15
                while time.monotonic()<limit:
                    path=root/'outputs/report.json'
                    if path.exists():
                        report=json.loads(path.read_text());run=Path(report['directory'])
                        if (run/'partial.txt').exists():break
                    if process.poll() is not None:self.fail((root/'parent.log').read_text())
                    time.sleep(.03)
                else:self.fail('Synthetic supervisor did not launch all workers')
                duplicate=subprocess.run([sys.executable,str(launcher),'run'],capture_output=True,timeout=5,env=env)
                self.assertNotEqual(duplicate.returncode,0)
                self.assertEqual(json.loads(path.read_text())['directory'],str(run))
                stop=subprocess.run([sys.executable,str(launcher),'stop'],capture_output=True,timeout=5,env=env)
                self.assertEqual(stop.returncode,0,stop.stderr.decode()+'\nSUPERVISOR:\n'+(root/'parent.log').read_text()+'\nREPORT:\n'+path.read_text())
                process.wait(timeout=15)
                report=json.loads(path.read_text())
                self.assertEqual(report['status'],'NATIVE_LIVE_FAILED_PRESERVE')
                self.assertTrue(report['local_workers_exited'])
                self.assertFalse(report['live_hil_acceptance'])
                self.assertEqual(set(report['local_worker_exit_codes']),{'preflight','live-viewer','simulator','pc-bridge','orin-server'})
                self.assertEqual((run/'partial.txt').read_text(),'keep this partial run')
                again=subprocess.run([sys.executable,str(launcher),'run'],capture_output=True,timeout=5,env=env)
                self.assertEqual(again.returncode,2)
                self.assertEqual(len(list((root/'outputs/runs').iterdir())),1)
                # Even if a controller died and released its lock, a retained
                # live owned process identity must prevent a new explicit run.
                fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
                if (root/'test-identities').exists():
                    token='synthetic-active-parent'
                    (root/'test-identities'/str(os.getpid())).write_text(token)
                else:token=fields[19]
                report['owned_processes']['synthetic-live-worker']=dict(pid=os.getpid(),start=token)
                path.write_text(json.dumps(report))
                retry=subprocess.run([sys.executable,str(launcher),'retry'],capture_output=True,timeout=5,env=env)
                self.assertNotEqual(retry.returncode,0)
                self.assertIn(b'Previous owned processes',retry.stderr)
                self.assertEqual(len(list((root/'outputs/runs').iterdir())),1)
            finally:
                if process.poll() is None:
                    if 'run' in locals():(run/'STOP.json').write_text('{}')
                    process.wait(timeout=15)
                log.close()


if __name__=='__main__':unittest.main(verbosity=2)
