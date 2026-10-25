"""Repair known Humble blockers, reuse the accepted export, then check the LAN link."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import traceback
import uuid

import common as old
from deploy_contract import MODEL_SHA, REPORT_SHA, sha, verify_export, require
import host_setup
import network_check
import probe
import link_ops as ops

HERE=Path(__file__).resolve().parent
ROOT=old.BASE/'artifacts/double-balance-stage12/native-ros2-humble-v3'
EXPORT=old.BASE/'artifacts/double-balance-stage12/native-ros2-humble-v1/frozen-export'
EXPORT_PINS={
    'final6h_005304.onnx': {'bytes':2501039,'sha256':'8c1c1250b9b88d9a91e3eb095fea8bce21daa73c251b43f3cd8adc1327f03c4e'},
    'contract.json': {'bytes':4041,'sha256':'e0283bf7f00c8e2c7b57128b91ca3ced3abb4698e8f4c4aa11e74a745ad84929'},
    'golden.json': {'bytes':1606400,'sha256':'b7ebe914c3933477d35bc0cc70c6b1bb4c1c83e67bd61dc22cd8f0775b1609e7'}}


def frozen_export():
    result=verify_export(EXPORT)
    require(result['files']==EXPORT_PINS and result['comparisons']==128
            and result['recurrent_resets']==2 and result['added_PPO_updates']==0,
            'Do not replace or re-export the accepted final model')
    return result


def remote_setup(run):
    destination='/home/jetson/microduck-stage12/ros2-humble-v3/setup/'+uuid.uuid4().hex
    create="""import pathlib,json,sys
p=pathlib.Path(sys.argv[1]);cur=pathlib.Path('/')
for part in p.parts[1:]:
 cur=cur/part
 if cur.is_symlink():raise RuntimeError('Unexpected directory symlink')
p.parent.mkdir(parents=True,exist_ok=True);p.mkdir(mode=0o700)
print(json.dumps({'directory':str(p)}))
"""
    ops.remote_json(create,[destination])
    files=[HERE/n for n in ('host_setup.py','apt_scope.py','probe.py',host_setup.BOOTSTRAP)]
    sock=ops.shared_ssh.connect()
    subprocess.run(['scp','-P','22','-o','ControlPath='+str(sock),'-o','ControlMaster=no',
                    '-o','BatchMode=yes','-o','ProxyCommand=false',*map(str,files),
                    'jetson@192.168.50.2:'+destination+'/'],check=True,timeout=60)
    pins={p.name:sha(p) for p in files}
    check="""import pathlib,hashlib,json,sys
p=pathlib.Path(sys.argv[1]);pins=json.load(sys.stdin)
for n,h in pins.items():
 f=p/n
 if f.is_symlink() or hashlib.sha256(f.read_bytes()).hexdigest()!=h:raise RuntimeError('Setup file changed')
print(json.dumps({'verified':len(pins)}))
"""
    ops.remote_json(check,[destination],json.dumps(pins).encode())
    cmd=shlex.join(['env','PYTHONPATH=','PYTHONHOME=','/usr/bin/python3','-B','-u',
                   destination+'/host_setup.py','--role','jetson','--directory',destination])
    argv=ops.shared_ssh.argv(cmd);argv.insert(1,'-tt')
    # A PTY is needed only if missing system dependencies require sudo on Orin.
    # Passwords remain in the ordinary sudo/SSH UI and are never captured.
    rc=subprocess.call(argv)
    result=ops.remote_json('import sys;print(open(sys.argv[1]).read())',[destination+'/host-setup-report.json'])
    result['ssh_returncode'] = rc
    ops.save(run/'jetson-host-setup.json',result)
    return result


def execute(diagnose_only=False):
    ops.protected_mkdir(ROOT)
    with (ROOT/'repair.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run=ROOT/'runs'/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+uuid.uuid4().hex[:8])
        ops.protected_mkdir(run)
        report=dict(schema='stage12-humble-repair-v3',status='RUNNING',directory=str(run),
                    started_utc=time.time(),added_PPO_updates=0,export_repeated=False,
                    cloud_connections=0,hardware_acceptance=False,realtime_acceptance=False,
                    motor_io=False,checkpoint_sha256=MODEL_SHA,blockers=[],errors=[])
        def update(**kw):
            report.update(kw);ops.save(run/'report.json',report);ops.save(ROOT/'report.json',report)
        try:
            update(phase='verify_accepted_export')
            package=old.verify_package();inputs=old.verify_inputs();export=frozen_export()
            require(sha(old.SLOT/'report.json')==REPORT_SHA,'Original validation report changed')
            update(package_identity=package,frozen_export=export,resources_pc=host_setup.resources())
            setup_dir=run/'pc-setup';setup_dir.mkdir()
            setup=host_setup.Setup(setup_dir)
            update(phase='network_diagnosis')
            net=network_check.diagnose(activate=not diagnose_only,sudo=setup.sudo)
            update(network=net,network_activation_actions=setup.actions,pc_probe=probe.probe())
            if diagnose_only:
                update(status='DIAGNOSIS_COMPLETE',phase='finished')
                return
            update(phase='pc_humble_setup')
            # The setup program is kept in the same terminal so sudo can prompt.
            rc=subprocess.call(['/usr/bin/python3','-B','-u',str(HERE/'host_setup.py'),
                                '--role','pc','--directory',str(setup_dir)],env=probe.clean_env())
            pc_result=json.loads((setup_dir/'host-setup-report.json').read_bytes())
            pc=probe.probe();update(pc_setup=pc_result,pc_probe=pc)
            require(rc==0,'PC Humble setup failed; see embedded pc_setup report')
            # Recheck the link after package setup, allowing time for Orin boot.
            net=network_check.diagnose(activate=True,sudo=setup.sudo)
            update(network=net)
            if net['status']!='SSH_PORT_READY':
                update(status='PC_READY_ORIN_CONNECTION_PENDING',phase='finished',blockers=[net['next_step']])
                return
            update(phase='jetson_identity_and_setup')
            jetson=json.loads(ops.remote('/usr/bin/python3 -B -',(HERE/'probe.py').read_bytes(),timeout=75))
            update(jetson_probe_before=jetson)
            model=(jetson.get('board_model') or '').lower()
            require(jetson['architecture']=='aarch64' and 'orin' in model and 'nano' in model,
                    'Remote host does not match the established Orin Nano')
            result=remote_setup(run)
            update(jetson_setup=result)
            require(result['ssh_returncode']==0 and result['status']=='HOST_SETUP_COMPLETE',
                    'Orin environment setup did not complete; see embedded jetson_setup report')
            jetson=result['probe_after']
            pc_ready=probe.readiness(pc);jetson_ready=probe.readiness(jetson,jetson=True)
            blockers=pc_ready['blockers']+jetson_ready['blockers']
            update(jetson_setup=result,pc_readiness=pc_ready,jetson_readiness=jetson_ready,blockers=blockers)
            if blockers:
                update(status='ENVIRONMENT_PENDING',phase='finished');return
            update(phase='ros2_inference_link')
            link=ops.ros_link(run,EXPORT,pc_ready,jetson_ready)
            require(old.verify_inputs()==inputs and sha(old.SLOT/'report.json')==REPORT_SHA,
                    'Original training/validation evidence changed')
            frozen_export()
            update(status='ROS_LINK_PASSED',phase='finished',ros_link=link,
                   original_records_unchanged=True,resources_pc_after=host_setup.resources())
        except BaseException as exc:
            update(status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'REPAIR_FAILED_PRESERVE',
                   phase='finished',errors=[repr(exc)],traceback=traceback.format_exc()[-10000:])
        finally:
            update(finished_utc=time.time())
            print('HUMBLE_REPAIR_STATUS='+report['status'],flush=True)
            print('HUMBLE_REPAIR_REPORT='+str(ROOT/'report.json'),flush=True)
            for msg in report['blockers']+report['errors']:print('待处理：'+msg,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('operation',choices=('repair','diagnose','status'));args=p.parse_args()
    if args.operation=='status':
        print((ROOT/'report.json').read_text() if (ROOT/'report.json').exists() else '尚未运行');return 0
    execute(args.operation=='diagnose')
    return 0 if json.loads((ROOT/'report.json').read_bytes())['status'] in ('ROS_LINK_PASSED','DIAGNOSIS_COMPLETE',
               'PC_READY_ORIN_CONNECTION_PENDING','ENVIRONMENT_PENDING') else 2


if __name__=='__main__':
    raise SystemExit(main())
