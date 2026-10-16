"""Test fixed-predecessor reuse and no-rerun using copies of verified receipts."""
from pathlib import Path
import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))


def run(evidence,archive,evidence_v2,archive_v2,evidence_v3,archive_v3,evidence_v4,archive_v4,result):
    from integration.completed_reuse import verify_completed,reuse_completed
    from integration.durable import file_digest
    checks=[]
    with tempfile.TemporaryDirectory(prefix='stage12-reuse-test-') as temp:
        base=Path(temp);frozen=base/'program/prior-local-completion-v1'
        shutil.copytree(ROOT/'prior-local-completion-v1',frozen)
        audit=json.loads((frozen/'audit.json').read_bytes())
        old=base/'old-artifacts';old.mkdir();oldrun=old/audit['run'];oldrun.mkdir()
        for n in audit['retained_reports']:
            target=oldrun/n;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(evidence/n,target)
        source=oldrun/'cloud-inputs.zip';shutil.copyfile(evidence/'cloud-inputs.zip',source)
        audit['base']=str(old);audit['source_bundle']['path']=str(source)
        (frozen/'audit.json').write_text(json.dumps(audit))
        shutil.copyfile(archive,old/audit['archive_filename'])
        marker=old/'started-v1.json'
        marker.write_text(json.dumps(dict(run=str(oldrun),package_manifest_sha256=audit['package_manifest_sha256'])))
        (old/'return-receipt.json').write_text(json.dumps(dict(run=str(oldrun),archive=str(old/audit['archive_filename']),
            bytes=audit['archive_bytes'],sha256=audit['archive_sha256'])))
        (old/'transfer-receipt.json').write_text(json.dumps(dict(commit=audit['transfer_commit'],branch=audit['transfer_branch'],
            archive_sha256=audit['archive_sha256'])))
        frozen2=base/'program/prior-local-completion-v2'
        shutil.copytree(ROOT/'prior-local-completion-v2',frozen2)
        audit2=json.loads((frozen2/'audit.json').read_bytes())
        old2=base/'old-v2-artifacts';old2.mkdir();run2=old2/audit2['run'];run2.mkdir()
        for n in audit2['retained_reports']:
            target=run2/n;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(evidence_v2/n,target)
        shutil.copyfile(evidence_v2/'cloud-inputs.zip',run2/'cloud-inputs.zip')
        shutil.copyfile(archive_v2,old2/audit2['archive_filename'])
        audit2['base']=str(old2);(frozen2/'audit.json').write_text(json.dumps(audit2))
        (old2/'started-v1.json').write_text(json.dumps(dict(run=str(run2),package_manifest_sha256=audit2['package_manifest_sha256'])))
        (old2/'return-receipt.json').write_text(json.dumps(dict(run=str(run2),archive=str(old2/audit2['archive_filename']),
            bytes=audit2['archive_bytes'],sha256=audit2['archive_sha256'])))
        (old2/'transfer-receipt.json').write_text(json.dumps(dict(commit=audit2['transfer_commit'],branch=audit2['transfer_branch'],
            archive_sha256=audit2['archive_sha256'])))
        frozen3=base/'program/prior-local-completion-v3'
        shutil.copytree(ROOT/'prior-local-completion-v3',frozen3)
        audit3=json.loads((frozen3/'audit.json').read_bytes())
        old3=base/'old-v3-artifacts';old3.mkdir();run3=old3/audit3['run'];run3.mkdir()
        for n in audit3['retained_reports']:
            target=run3/n;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(evidence_v3/n,target)
        shutil.copyfile(evidence_v3/'cloud-inputs.zip',run3/'cloud-inputs.zip')
        shutil.copyfile(archive_v3,old3/audit3['archive_filename'])
        audit3['base']=str(old3);(frozen3/'audit.json').write_text(json.dumps(audit3))
        (old3/'started-v1.json').write_text(json.dumps(dict(run=str(run3),package_manifest_sha256=audit3['package_manifest_sha256'])))
        (old3/'return-receipt.json').write_text(json.dumps(dict(run=str(run3),archive=str(old3/audit3['archive_filename']),
            bytes=audit3['archive_bytes'],sha256=audit3['archive_sha256'])))
        (old3/'transfer-receipt.json').write_text(json.dumps(dict(commit=audit3['transfer_commit'],branch=audit3['transfer_branch'],
            archive_sha256=audit3['archive_sha256'])))
        frozen4=base/'program/prior-local-completion-v4'
        shutil.copytree(ROOT/'prior-local-completion-v4',frozen4)
        audit4=json.loads((frozen4/'audit.json').read_bytes())
        old4=base/'old-v4-artifacts';old4.mkdir();run4=old4/audit4['run'];run4.mkdir()
        for n in audit4['retained_reports']:
            target=run4/n;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(evidence_v4/n,target)
        shutil.copyfile(evidence_v4/'cloud-inputs.zip',run4/'cloud-inputs.zip')
        shutil.copyfile(archive_v4,old4/audit4['archive_filename'])
        audit4['base']=str(old4);(frozen4/'audit.json').write_text(json.dumps(audit4))
        (old4/'started-v1.json').write_text(json.dumps(dict(run=str(run4),package_manifest_sha256=audit4['package_manifest_sha256'])))
        (old4/'return-receipt.json').write_text(json.dumps(dict(run=str(run4),archive=str(old4/audit4['archive_filename']),
            bytes=audit4['archive_bytes'],sha256=audit4['archive_sha256'])))
        (old4/'transfer-receipt.json').write_text(json.dumps(dict(commit=audit4['transfer_commit'],branch=audit4['transfer_branch'],
            archive_sha256=audit4['archive_sha256'])))
        def rejected(name,operation):
            try:operation()
            except RuntimeError:checks.append(dict(name=name,passed=True))
            else:raise AssertionError(name+' was not rejected')
        before=verify_completed(base/'program')
        out=base/'new-run';out.mkdir()
        reused=reuse_completed(base/'program',out)
        assert reused['cpu_regression']['reexecuted_cases']==0
        assert file_digest(out/'cloud-inputs.zip')==audit['source_bundle']['sha256']
        assert verify_completed(base/'program')==before
        ledger=run2/'single_owner_lifecycle/owner-0/native-ledger.jsonl'
        assert ledger.is_file() and ledger.read_bytes()==b''
        checks.append(dict(name='v2-zero-WAL-and-native-started-retained',passed=True))
        ledger3=run3/'single_owner_lifecycle/owner-0/native-ledger.jsonl'
        raw3=ledger3.read_bytes();assert len(raw3.splitlines())==4
        checks.append(dict(name='v3-two-CPU-forwards-kinematics-and-failed-construction-retained',passed=True))
        ledger4=run4/'single_owner_lifecycle/owner-0/native-ledger.jsonl'
        raw4=ledger4.read_bytes();assert len(raw4.splitlines())==10
        checks.append(dict(name='v4-completed-construction-three-CPU-two-GPU-forwards-and-cleanup-retained',passed=True))
        ledger4.write_bytes(b'')
        rejected('cleared-v4-WAL-requires-reconciliation',lambda:verify_completed(base/'program'))
        ledger4.write_bytes(raw4+b'{"unexpected":"work"}\n')
        rejected('added-v4-WAL-requires-reconciliation',lambda:verify_completed(base/'program'))
        ledger4.write_bytes(raw4)
        unexpected4=run4/'two_process_short_pilot';unexpected4.mkdir()
        rejected('unexpected-v4-N2-blocks-new-run',lambda:verify_completed(base/'program'))
        unexpected4.rmdir()
        assert verify_completed(base/'program')==before

        ledger3.write_bytes(b'')
        rejected('cleared-v3-WAL-requires-reconciliation',lambda:verify_completed(base/'program'))
        ledger3.write_bytes(raw3+b'{"unexpected":"work"}\n')
        rejected('added-v3-WAL-requires-reconciliation',lambda:verify_completed(base/'program'))
        ledger3.write_bytes(raw3)
        unexpected3=run3/'two_process_short_pilot';unexpected3.mkdir()
        rejected('unexpected-v3-N2-blocks-new-run',lambda:verify_completed(base/'program'))
        unexpected3.rmdir()
        assert verify_completed(base/'program')==before
        ledger.write_bytes(b'{"unexpected":"work"}\n')
        rejected('changed-v2-WAL-requires-reconciliation',lambda:verify_completed(base/'program'))
        ledger.write_bytes(b'')  # Restore this synthetic fixture only.
        unexpected=run2/'two_process_short_pilot';unexpected.mkdir()
        rejected('unexpected-v2-N2-blocks-new-run',lambda:verify_completed(base/'program'))
        unexpected.rmdir()
        assert verify_completed(base/'program')==before
        checks.append(dict(name='reuse-13-completed-CPU-cases-and-410-inputs-no-recapture',passed=True))
        rejected('existing-output-preserved',lambda:reuse_completed(base/'program',out))
        raw=marker.read_bytes();marker.write_text('{}')
        try:
            verify_completed(base/'program')
        except (RuntimeError,KeyError):checks.append(dict(name='changed-started-marker-rejected',passed=True))
        else:raise AssertionError('Changed started accepted')
        marker.write_bytes(raw)
        oldowner=oldrun/'single_owner_lifecycle/owner-0';oldowner.mkdir()
        rejected('unexpected-old-native-owner-blocks-new-run',lambda:verify_completed(base/'program'))
        oldowner.rmdir()  # This is an empty synthetic test folder only.
        cpu=oldrun/'cpu-regression/real-cpu-report.json';raw=cpu.read_bytes();cpu.write_bytes(raw+b' ')
        rejected('changed-completed-CPU-report-rejected',lambda:verify_completed(base/'program'));cpu.write_bytes(raw)
        raw=source.read_bytes();source.write_bytes(raw+b'bad')
        rejected('changed-completed-portable-input-rejected',lambda:verify_completed(base/'program'));source.write_bytes(raw)
        assert verify_completed(base/'program')==before
        spec=importlib.util.spec_from_file_location('stage12_local_test',ROOT/'microduck-stage12-local.py')
        launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
        newbase=base/'new-started';newbase.mkdir();runpath=newbase/'existing';runpath.mkdir()
        mark=newbase/'started-v1.json';mark.write_text(json.dumps(dict(run=str(runpath))))
        (runpath/'stage12-report.json').write_text(json.dumps(dict(status='FIXTURE_STOPPED')))
        snapshot={str(p):p.read_bytes() for p in newbase.rglob('*') if p.is_file()}
        with patch.object(launcher,'BASE',newbase):assert launcher.run(38720,43200)==2
        assert snapshot=={str(p):p.read_bytes() for p in newbase.rglob('*') if p.is_file()}
        checks.append(dict(name='existing-v5-started-only-status-no-execution',passed=True))
    payload=dict(passed=True,checks=checks,native_integrations=0,PPO=0,cloud_training=0,fixtures_only=True)
    result.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--result',type=Path,required=True)
    p.add_argument('--evidence-v2',type=Path,required=True);p.add_argument('--archive-v2',type=Path,required=True)
    p.add_argument('--evidence-v3',type=Path,required=True);p.add_argument('--archive-v3',type=Path,required=True)
    p.add_argument('--evidence-v4',type=Path,required=True);p.add_argument('--archive-v4',type=Path,required=True)
    a=p.parse_args();run(a.evidence,a.archive,a.evidence_v2,a.archive_v2,a.evidence_v3,a.archive_v3,a.evidence_v4,a.archive_v4,a.result)
