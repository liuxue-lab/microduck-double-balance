"""End-to-end artifact checks with explicitly synthetic acceptance receipts.

No native/PPO runs. Uses the real retained source archive and real local Git.
Run against a package whose PACKAGE.json has already been regenerated.
"""
from pathlib import Path
import argparse
import copy
import importlib.util
import json
import shutil
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0,str(ROOT))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-archive',type=Path,required=True)
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--result',type=Path,required=True)
    args = p.parse_args()
    from integration.cloud_inputs import prepare_inputs
    from stage12.transfer import prepare_transfer, verify_return
    launcher = module('local_launcher',ROOT/'microduck-stage12-local.py')
    cloud = module('cloud_bootstrap',ROOT/'cloud-stage12.py')
    launcher.verify_package()
    checks = []
    def blocked(name, fn):
        try:
            fn()
        except RuntimeError:
            checks.append(dict(name=name,passed=True))
            return
        raise AssertionError(name+' was not blocked')
    with tempfile.TemporaryDirectory(prefix='stage12-delivery-fixture-') as temp:
        base = Path(temp)
        stub = base/'input-reference'
        (stub/'prior-integration-v1').mkdir(parents=True)
        prior = json.loads((ROOT/'prior-integration-v1/audit.json').read_bytes())
        prior['source_bundle']['path'] = str(args.source_archive)
        (stub/'prior-integration-v1/audit.json').write_text(json.dumps(prior))
        shutil.copyfile(ROOT/'cloud-build-metadata-pins.json',stub/'cloud-build-metadata-pins.json')
        run = base/'20261016T020000Z-aabbcc'
        run.mkdir()
        inputs = prepare_inputs(stub,run,args.project)
        assert inputs['members'] == 410 and inputs['prior_sources_recaptured'] == 0
        checks.append(dict(name='real-source-reuse-plus-four-pinned-build-files',passed=True))
        report = dict(status='LOCAL_NATIVE_PILOT_PASS_CLOUD_BUDGET_PENDING',preservation='PASS',
                      canonical_before=launcher.CANONICAL,canonical_after=launcher.CANONICAL,
                      budget=launcher.budget(43200),training_ready=False,
                      fixture=True,actual_native_integrations=0,actual_PPO=0)
        (run/'stage12-report.json').write_text(json.dumps(report))
        mapping=run/'single_owner_lifecycle/owner-0-bootstrap/runtime-paths.json'
        mapping.parent.mkdir(parents=True)
        mapping.write_text(json.dumps(dict(fixture=True,short_root='/tmp/md12-fixture')))
        diagnostics=[]
        for name in ('manager-interface-preflight.json','owner-inventory.json','macro-00/owner-inventory-failure.json'):
            target=run/'single_owner_lifecycle/owner-0'/name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text('{"fixture":true}');diagnostics.append(target)
        with patch.object(launcher,'BASE',base):
            archive = launcher.package_return(run)
        import zipfile
        with zipfile.ZipFile(archive) as z:
            assert z.read('evidence/single_owner_lifecycle/owner-0-bootstrap/runtime-paths.json')==mapping.read_bytes()
        checks.append(dict(name='short-runtime-path-receipt-in-return-archive',passed=True))
        with zipfile.ZipFile(archive) as z:
            assert all(z.read('evidence/'+str(p.relative_to(run)))==p.read_bytes() for p in diagnostics)
        checks.append(dict(name='complete-manager-interface-diagnostics-in-return-archive',passed=True))
        verified = verify_return(archive)
        summary = dict(archive_sha256=verified['sha256'],archive_bytes=verified['bytes'],fixture=True)
        receipt = prepare_transfer(archive,base/'transfer','stage12-local-completion-v5-transfer-20261016t020000z-aabbcc',summary)
        from integration.cloud_commands import render
        import subprocess
        commands=render(receipt)
        assert receipt['branch'] in commands and receipt['commit'] in commands
        subprocess.run(['bash','-n'],input=commands.encode(),check=True)
        checks.append(dict(name='v5-cloud-commands-pinned-and-shell-syntax-valid',passed=True))
        cloud.extract_transfer(base/'transfer',receipt['commit'],base/'cloud-input-fixture')
        assert (base/'cloud-input-fixture/inputs/project/README.md').read_bytes() == (args.project/'README.md').read_bytes()
        checks.append(dict(name='real-Git-parted-return-and-cloud-extraction-roundtrip',passed=True,
                           fixture_native_acceptance=True,actual_native_integrations=0))
        blocked('ref-mismatch-before-extraction',lambda:cloud.extract_transfer(base/'transfer','0'*40,base/'wrong-ref'))
        assert not (base/'wrong-ref').exists()
        blocked('existing-cloud-output-preserved',lambda:cloud.extract_transfer(base/'transfer',receipt['commit'],base/'cloud-input-fixture'))
        part = next((base/'transfer').glob('*.base64.txt'))
        original = part.read_bytes()
        part.write_bytes(original+b'corrupt')
        blocked('edited-worktree-part-blocked',lambda:cloud.extract_transfer(base/'transfer',receipt['commit'],base/'edited-part'))
        assert not (base/'edited-part').exists()
        blocked('cloud-budget-missing-before-marker',lambda:cloud.environment_check(base/'cloud-input-fixture',0,0,''))
        assert not (base/'cloud-input-fixture/started-environment-check.json').exists()
        bad = copy.deepcopy(prior)
        bad['source_bundle']['sha256'] = '0'*64
        (stub/'prior-integration-v1/audit.json').write_text(json.dumps(bad))
        blocked('changed-old-source-package-blocked',lambda:prepare_inputs(stub,run,args.project))
    result = dict(passed=True,checks=checks,native_integrations=0,PPO=0,cloud_training=0,
                  fixture_receipts_are_not_physical_acceptance=True)
    with args.result.open('x') as f:
        json.dump(result,f,indent=2,sort_keys=True)
    print(json.dumps(result,sort_keys=True))


if __name__ == '__main__':
    main()
