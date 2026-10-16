"""Verify the fixed stopped predecessor and reuse its completed CPU/input work."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile
from integration.durable import require, file_digest, publish, write_json


def verify_v1_completed(root):
    frozen = Path(root)/'prior-local-completion-v1'
    audit = json.loads((frozen/'audit.json').read_bytes())
    base = Path(audit['base'])
    run = base/audit['run']
    paths = [base/'started-v1.json',base/audit['archive_filename'],
             base/'return-receipt.json',base/'transfer-receipt.json']
    paths += [run/n for n in audit['retained_reports']]
    source = Path(audit['source_bundle']['path'])
    paths.append(source)
    for p in paths:
        require(p.is_file() and not p.is_symlink(), 'Completed local-v1 evidence missing: '+str(p))
    marker = json.loads(paths[0].read_bytes())
    require(marker['run'] == str(run) and marker['package_manifest_sha256'] == audit['package_manifest_sha256'],
            'Completed local-v1 marker changed')
    require(paths[1].stat().st_size == audit['archive_bytes'] and file_digest(paths[1]) == audit['archive_sha256'],
            'Completed local-v1 return archive changed')
    receipt, transfer = [json.loads(p.read_bytes()) for p in paths[2:4]]
    require(receipt['run'] == str(run) and receipt['archive'] == str(paths[1]) and
            receipt['sha256'] == audit['archive_sha256'] and receipt['bytes'] == audit['archive_bytes'],
            'Completed local-v1 return receipt changed')
    require(transfer['commit'] == audit['transfer_commit'] and transfer['branch'] == audit['transfer_branch'] and
            transfer['archive_sha256'] == audit['archive_sha256'], 'Completed local-v1 transfer identity changed')
    for n in audit['retained_reports']:
        require((run/n).read_bytes() == (frozen/n).read_bytes(), 'Completed local-v1 report changed: '+n)
    report = json.loads((run/'stage12-report.json').read_bytes())
    require(report['status'] == 'LOCAL_COMPLETION_STOPPED_PRESERVE' and report['preservation'] == 'PASS' and
            report['canonical_after'] == audit['canonical_after'] and
            all(v == 0 for v in report['accounting']['native_added'].values()),
            'Predecessor is not the fixed zero-integration startup failure')
    closeout = json.loads((run/'single_owner_lifecycle/owner-closeout.json').read_bytes())
    require(len(closeout['owners']) == 1 and closeout['owners'][0]['exitcode'] == 1 and
            not closeout['owners'][0]['interrupted'] and not closeout['owners'][0]['report_exists'],
            'Predecessor process closeout differs')
    for point,owners in (('single_owner_lifecycle',1),('two_process_short_pilot',2)):
        for owner in range(owners):
            p = run/point/('owner-%d' % owner)
            require(not p.exists() and not p.is_symlink(), 'Old native owner exists; reconcile instead of rerun')
    cpu_path = run/'cpu-regression/real-cpu-report.json'
    cpu = json.loads(cpu_path.read_bytes())
    require(file_digest(cpu_path) == audit['cpu_report_sha256'] and cpu['passed'] is True and
            len(cpu['cases']) == 13 and all(r['passed'] is True for r in cpu['cases']) and
            cpu['native_integrations'] == cpu['PPO'] == cpu['cloud_training'] == 0,
            'Completed CPU evidence cannot be reused')
    require(source.stat().st_size == audit['source_bundle']['bytes'] and
            file_digest(source) == audit['source_bundle']['sha256'], 'Completed portable input changed')
    return {str(p):file_digest(p) for p in paths}


def verify_v2_stopped(root):
    frozen = Path(root)/'prior-local-completion-v2'
    audit = json.loads((frozen/'audit.json').read_bytes())
    base = Path(audit['base']); run = base/audit['run']
    paths = [base/'started-v1.json', base/audit['archive_filename'],
             base/'return-receipt.json', base/'transfer-receipt.json']
    paths += [run/n for n in audit['retained_reports']]
    source = run/audit['source_bundle']['filename']; paths.append(source)
    for p in paths:
        require(p.is_file() and not p.is_symlink(), 'Preserved local-v2 evidence missing: '+str(p))
    marker, receipt, transfer = [json.loads(paths[i].read_bytes()) for i in (0,2,3)]
    require(marker['run'] == str(run) and marker['package_manifest_sha256'] == audit['package_manifest_sha256'],
            'Preserved local-v2 marker changed')
    require(paths[1].stat().st_size == audit['archive_bytes'] and file_digest(paths[1]) == audit['archive_sha256'],
            'Preserved local-v2 archive changed')
    require(receipt['run'] == str(run) and receipt['archive'] == str(paths[1]) and
            receipt['sha256'] == audit['archive_sha256'] and receipt['bytes'] == audit['archive_bytes'],
            'Preserved local-v2 return receipt changed')
    require(transfer['commit'] == audit['transfer_commit'] and transfer['branch'] == audit['transfer_branch'] and
            transfer['archive_sha256'] == audit['archive_sha256'], 'Preserved local-v2 transfer changed')
    for n in audit['retained_reports']:
        require((run/n).read_bytes() == (frozen/n).read_bytes(), 'Preserved local-v2 report changed: '+n)
    report = json.loads((run/'stage12-report.json').read_bytes())
    require(report['status'] == 'LOCAL_COMPLETION_STOPPED_PRESERVE' and report['preservation'] == 'PASS' and
            report['canonical_after'] == audit['canonical_after'] and
            report['accounting']['all_operation_ledgers_clean'] is True and
            all(v == 0 for v in report['accounting']['native_added'].values()),
            'Local-v2 is not the reviewed zero-integration Warp startup failure')
    owner = run/'single_owner_lifecycle/owner-0'
    native = json.loads((owner/'native-report.json').read_bytes())
    require((owner/'native-ledger.jsonl').read_bytes() == b'' and
            native['counts'] == dict(integrations_attempted=0,integrations_completed=0,PPO=0,cloud=0) and
            not native.get('real_environment_constructed',False), 'Local-v2 native consumption requires reconciliation')
    closeout = json.loads((run/'single_owner_lifecycle/owner-closeout.json').read_bytes())
    require(len(closeout['owners']) == 1 and closeout['owners'][0]['exitcode'] == 1 and
            not closeout['owners'][0]['interrupted'] and closeout['owners'][0]['report_exists'],
            'Local-v2 process closeout changed')
    require(not (run/'two_process_short_pilot').exists() and
            not (run/'two_process_short_pilot').is_symlink(), 'Local-v2 N2 exists; reconcile instead of rerun')
    require(source.stat().st_size == audit['source_bundle']['bytes'] and
            file_digest(source) == audit['source_bundle']['sha256'], 'Preserved local-v2 input changed')
    return {str(p):file_digest(p) for p in paths}


def verify_v3_stopped(root):
    frozen = Path(root)/'prior-local-completion-v3'
    audit = json.loads((frozen/'audit.json').read_bytes())
    base = Path(audit['base']); run = base/audit['run']
    paths = [base/'started-v1.json', base/audit['archive_filename'],
             base/'return-receipt.json', base/'transfer-receipt.json']
    paths += [run/n for n in audit['retained_reports']]
    source = run/audit['source_bundle']['filename']; paths.append(source)
    for p in paths:
        require(p.is_file() and not p.is_symlink(), 'Preserved local-v3 evidence missing: '+str(p))
    marker, receipt, transfer = [json.loads(paths[i].read_bytes()) for i in (0,2,3)]
    require(marker['run'] == str(run) and marker['package_manifest_sha256'] == audit['package_manifest_sha256'],
            'Preserved local-v3 marker changed')
    require(paths[1].stat().st_size == audit['archive_bytes'] and file_digest(paths[1]) == audit['archive_sha256'],
            'Preserved local-v3 archive changed')
    require(receipt['run'] == str(run) and receipt['archive'] == str(paths[1]) and
            receipt['sha256'] == audit['archive_sha256'] and receipt['bytes'] == audit['archive_bytes'],
            'Preserved local-v3 return receipt changed')
    require(transfer['commit'] == audit['transfer_commit'] and transfer['branch'] == audit['transfer_branch'] and
            transfer['archive_sha256'] == audit['archive_sha256'], 'Preserved local-v3 transfer changed')
    for n in audit['retained_reports']:
        require((run/n).read_bytes() == (frozen/n).read_bytes(), 'Preserved local-v3 report changed: '+n)
    report = json.loads((run/'stage12-report.json').read_bytes())
    require(report['status'] == 'LOCAL_COMPLETION_STOPPED_PRESERVE' and report['preservation'] == 'PASS' and
            report['canonical_after'] == audit['canonical_after'] and
            report['accounting']['all_operation_ledgers_clean'] is True and
            all(v == 0 for v in report['accounting']['native_added'].values()),
            'Local-v3 integration accounting differs; reconcile before a new attempt')
    owner = run/'single_owner_lifecycle/owner-0'
    native = json.loads((owner/'native-report.json').read_bytes())
    ledger = [json.loads(line) for line in (owner/'native-ledger.jsonl').read_bytes().splitlines()]
    expected = [dict(kind=kind,label='CPU_forward',number=number,seq=seq)
                for seq,(number,kind) in enumerate(((1,'attempt'),(1,'complete'),(2,'attempt'),(2,'complete')),1)]
    require(ledger == expected and native['counts'] == audit['native_counts'] and
            not native.get('real_environment_constructed',False),
            'Local-v3 CPU forward/kinematics/construction consumption must remain recorded')
    closeout = json.loads((run/'single_owner_lifecycle/owner-closeout.json').read_bytes())
    require(len(closeout['owners']) == 1 and closeout['owners'][0]['exitcode'] == 1 and
            not closeout['owners'][0]['interrupted'] and closeout['owners'][0]['report_exists'],
            'Local-v3 process closeout changed')
    require(not (run/'two_process_short_pilot').exists() and
            not (run/'two_process_short_pilot').is_symlink(), 'Local-v3 N2 exists; reconcile instead of rerun')
    require(source.stat().st_size == audit['source_bundle']['bytes'] and
            file_digest(source) == audit['source_bundle']['sha256'], 'Preserved local-v3 input changed')
    return {str(p):file_digest(p) for p in paths}

def verify_v4_stopped(root):
    frozen = Path(root)/'prior-local-completion-v4'
    audit = json.loads((frozen/'audit.json').read_bytes())
    base = Path(audit['base']); run = base/audit['run']
    paths = [base/'started-v1.json', base/audit['archive_filename'],
             base/'return-receipt.json', base/'transfer-receipt.json']
    paths += [run/n for n in audit['retained_reports']]
    source = run/audit['source_bundle']['filename']; paths.append(source)
    for p in paths:
        require(p.is_file() and not p.is_symlink(), 'Preserved local-v4 evidence missing: '+str(p))
    marker, receipt, transfer = [json.loads(paths[i].read_bytes()) for i in (0,2,3)]
    require(marker['run'] == str(run) and marker['package_manifest_sha256'] == audit['package_manifest_sha256'],
            'Preserved local-v4 marker changed')
    require(paths[1].stat().st_size == audit['archive_bytes'] and file_digest(paths[1]) == audit['archive_sha256'],
            'Preserved local-v4 archive changed')
    require(receipt['run'] == str(run) and receipt['archive'] == str(paths[1]) and
            receipt['sha256'] == audit['archive_sha256'] and receipt['bytes'] == audit['archive_bytes'],
            'Preserved local-v4 return receipt changed')
    require(transfer['commit'] == audit['transfer_commit'] and transfer['branch'] == audit['transfer_branch'] and
            transfer['archive_sha256'] == audit['archive_sha256'], 'Preserved local-v4 transfer changed')
    for n in audit['retained_reports']:
        require((run/n).read_bytes() == (frozen/n).read_bytes(), 'Preserved local-v4 report changed: '+n)
    report = json.loads((run/'stage12-report.json').read_bytes())
    require(report['status'] == 'LOCAL_COMPLETION_STOPPED_PRESERVE' and report['preservation'] == 'PASS' and
            report['canonical_after'] == audit['canonical_after'] and
            report['accounting']['all_operation_ledgers_clean'] is True and
            all(v == 0 for v in report['accounting']['native_added'].values()),
            'Local-v4 integration accounting differs; reconcile before a new attempt')
    owner = run/'single_owner_lifecycle/owner-0'
    native = json.loads((owner/'native-report.json').read_bytes())
    ledger = [json.loads(line) for line in (owner/'native-ledger.jsonl').read_bytes().splitlines()]
    expected=[]
    for label,n in (('CPU_forward',3),('GPU_forward',2)):
        for number in range(1,n+1):
            for kind in ('attempt','complete'):
                expected.append(dict(kind=kind,label=label,number=number,seq=len(expected)+1))
    require(ledger == expected and native['counts'] == audit['native_counts'] and
            native.get('real_environment_constructed') is True and
            native['resource_close']['all_released'] is True and native['resource_close']['targets']==42 and
            native['env_close']['status']=='FULL_ENV_CLOSE_RETURNED' and native['contact']['graph_closed'] is True,
            'Local-v4 completed construction/CPU/GPU operations and cleanup must remain recorded')
    closeout = json.loads((run/'single_owner_lifecycle/owner-closeout.json').read_bytes())
    require(len(closeout['owners']) == 1 and closeout['owners'][0]['exitcode'] == 1 and
            not closeout['owners'][0]['interrupted'] and closeout['owners'][0]['report_exists'],
            'Local-v4 process closeout changed')
    require(not (run/'two_process_short_pilot').exists() and
            not (run/'two_process_short_pilot').is_symlink(), 'Local-v4 N2 exists; reconcile instead of rerun')
    require(source.stat().st_size == audit['source_bundle']['bytes'] and
            file_digest(source) == audit['source_bundle']['sha256'], 'Preserved local-v4 input changed')
    return {str(p):file_digest(p) for p in paths}


def verify_completed(root):
    return {**verify_v1_completed(root), **verify_v2_stopped(root), **verify_v3_stopped(root), **verify_v4_stopped(root)}


def reuse_completed(root, out):
    verify_completed(root)
    audit = json.loads((Path(root)/'prior-local-completion-v1/audit.json').read_bytes())
    source = Path(audit['source_bundle']['path'])
    with zipfile.ZipFile(source) as z:
        files = json.loads(z.read('SOURCE-BUNDLE-MANIFEST.json'))['files']
        require(len(files) == 410 and len(z.namelist()) == 411 and
                set(z.namelist()) == set(files)|{'SOURCE-BUNDLE-MANIFEST.json'} and z.testzip() is None,
                'Completed cloud input roster/CRC differs')
        for name,row in files.items():
            rel = Path(name)
            require(not rel.is_absolute() and '..' not in rel.parts and '\\' not in name,'Unsafe completed input path')
            raw = z.read(name)
            require(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'],
                    'Completed cloud input member changed: '+name)
    # Reference the already verified portable input. Never duplicate its capture.
    cloud = dict(path=str(source),bytes=source.stat().st_size,sha256=file_digest(source),members=410,
                 reused_from_commit=audit['transfer_commit'], prior_sources_recaptured=0,
                 newly_collected_build_metadata=[],checkpoints=0,installed_cloud_environment=False)
    cpu = dict(status='REUSED_13_OF_13_PASS',source_commit=audit['transfer_commit'],
               report_sha256=audit['cpu_report_sha256'],reexecuted_cases=0,native_integrations=0,
               PPO=0,cloud_training=0)
    write_json(Path(out)/'cloud-inputs.json',cloud)
    write_json(Path(out)/'cpu-regression-reuse.json',cpu)
    return dict(cloud_inputs=cloud,cpu_regression=cpu)
