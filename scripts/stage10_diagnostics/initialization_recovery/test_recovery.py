"""Offline regressions: lifecycle guard and evidence-bound, idempotent recovery."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import contextlib,hashlib,io,json,os,shutil,tempfile,unittest
import numpy as np
import stage10_core as core
import stage10_recovery_common as common
import stage10_initialization_worker as worker
from stage10_initialization_assistance import assert_unprocessed_zero_assistance
import stage10_init_compare as compare

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('recovery_driver',HERE/'microduck-stage10-initialization-recovery.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
FIXTURE=HERE.parents[1]/'stage10-initialization-c-failure-review'

class RecoveryTests(unittest.TestCase):
    def fixture(self):
        term=NS(_force=None,_raw=np.zeros((128,0)),cfg=NS(levels=(0.,)))
        raw=NS(num_envs=128,action_manager=NS(get_term=lambda _:term),
               _basketball_state=NS(hold=np.zeros(128)),sim=NS(wp_data=NS(
                   xfrc_applied=np.zeros((128,25,6)),qfrc_applied=np.zeros((128,32)))))
        return raw,term,NS(numpy=lambda a:a)

    def test_never_processed_state_is_read_only(self):
        raw,term,capture=self.fixture();keys=set(vars(term));force=raw.sim.wp_data.xfrc_applied.copy()
        result=assert_unprocessed_zero_assistance(raw,capture)
        self.assertEqual(result['status'],'PASS_UNPROCESSED_INITIALIZATION_ONLY')
        self.assertEqual(set(vars(term)),keys);self.assertIsNone(term._force)
        np.testing.assert_array_equal(force,raw.sim.wp_data.xfrc_applied)
        self.assertFalse(result['caches_created_or_cleared'])

    def test_materialized_even_zero_caches_are_rejected(self):
        for name in ('_force','_torque','_duck_force','_duck_torque'):
            with self.subTest(name=name):
                raw,term,cap=self.fixture();setattr(term,name,np.zeros((128,1,3)))
                with self.assertRaises(RuntimeError):assert_unprocessed_zero_assistance(raw,cap)
        raw,term,cap=self.fixture();del term._force
        with self.assertRaises(RuntimeError):assert_unprocessed_zero_assistance(raw,cap)

    def test_nonzero_or_nonfinite_hold_and_applied_forces_are_rejected(self):
        for name in ('hold','xfrc_applied','qfrc_applied'):
            for value in (1e-30,np.nan,np.inf):
                with self.subTest(name=name,value=value):
                    raw,term,cap=self.fixture()
                    arr=raw._basketball_state.hold if name=='hold' else getattr(raw.sim.wp_data,name)
                    arr.flat[-1]=value
                    with self.assertRaises(RuntimeError):assert_unprocessed_zero_assistance(raw,cap)
                    self.assertEqual(np.isnan(arr.flat[-1]),np.isnan(value))

    def test_configuration_scope_is_not_relaxed(self):
        raw,term,cap=self.fixture();term.cfg.levels=(1.,)
        with self.assertRaises(RuntimeError):assert_unprocessed_zero_assistance(raw,cap)
        raw,term,cap=self.fixture();term._raw=np.zeros((128,1))
        with self.assertRaises(RuntimeError):assert_unprocessed_zero_assistance(raw,cap)

    def test_real_reviewed_first_trace_remains_failed_and_pinned(self):
        if not FIXTURE.exists():self.skipTest('returned review fixture not installed on laptop')
        with patch.object(common,'expected_folder',lambda:FIXTURE):
            report=common.validate_approved_first(FIXTURE/common.REFERENCE['approved_attempt'])
            self.assertEqual(report['status'],'FAILED_REVIEW_REQUIRED')
            self.assertNotIn('backend_after',report)
            with self.assertRaisesRegex(RuntimeError,'explicitly reviewed'):
                common.validate_approved_first(FIXTURE/'01-initialization/attempt-unreviewed')

    def test_new_partial_construction_cannot_be_retried(self):
        with tempfile.TemporaryDirectory() as t:
            parent=Path(t);output=parent/'attempt-x';output.mkdir()
            core.write(output/'construction-started.json',{'repeat_requires_review':True})
            with self.assertRaisesRegex(RuntimeError,'construction was already attempted'):
                worker.choose_existing(parent)

    def completed(self):
        result=core.read(FIXTURE/common.REFERENCE['approved_attempt']/'result.json')
        result.update(status=worker.COMPLETE,backend_after=result['backend_before'],
                      source_recheck_after='PASS_101_FROZEN_FILES',
                      initialization_assistance_check={'status':'PASS_UNPROCESSED_INITIALIZATION_ONLY'})
        result.pop('error');return result

    def test_completed_receipt_tamper_blocks_reuse(self):
        if not FIXTURE.exists():self.skipTest('review fixture missing')
        with tempfile.TemporaryDirectory() as t:
            parent=Path(t);out=parent/'attempt-x';out.mkdir();core.write(out/'result.json',self.completed())
            (out/'trace.dat').write_bytes(b'exact')
            core.write(parent/'completed.json',{'attempt':out.name,'files':core.inventory(out)})
            self.assertEqual(worker.choose_existing(parent),out)
            (out/'trace.dat').write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError,'SHA256'):worker.choose_existing(parent)

    @contextlib.contextmanager
    def mock_laptop(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);repo=root/'repo';repo.mkdir();art=root/'art';home=art/'initialization'
            folder=home/common.REFERENCE['batch_directory']
            # Independent copy; the uploaded evidence is never edited.
            shutil.copytree(FIXTURE,folder)
            core.write(home/'active-batch.json',{'folder':str(folder),'tools':common.REFERENCE['original_plan']['tools']})
            checked=common.checked
            def check_inputs(path,digest):
                if str(path)==common.REFERENCE['original_plan']['checkpoint'] or str(path).endswith('/initial-states/dev.pt'):
                    return Path(path)
                return checked(path,digest)
            with patch.object(core,'ART',art),patch.object(core,'REPO',repo),\
                 patch.object(core,'verify_source',lambda:{}),patch.object(common,'checked',check_inputs),\
                 patch.object(worker,'prior_review',lambda:{}),\
                 patch.object(driver.shutil,'disk_usage',lambda p:NS(free=20*1024**3)):
                yield folder

    def test_controller_runs_only_two_then_reuses_and_preserves_all_original_files(self):
        if not FIXTURE.exists():self.skipTest('review fixture missing')
        calls=[]
        with self.mock_laptop() as folder:
            def supervise(command,log,timeout_s):
                output=Path(command[-1]);calls.append(output.parent.name)
                output.mkdir();core.write(output/'result.json',self.completed())
                Path(log).write_text('MOCK ONLY; no simulator constructed')
            def comparison(outputs,path):
                self.assertEqual(outputs[0],folder/common.REFERENCE['approved_attempt'])
                result={'status':common.GAP_STATUS};core.write(path,result);return result
            with patch.object(core,'supervised',supervise),patch.object(compare,'compare_all',comparison),\
                 patch.object(driver,'package_review',lambda _:[]):
                driver.controller();driver.controller()
                self.assertEqual(calls,['02-initialization','03-initialization'])
                run=core.read(folder/driver.RECOVERY_DIR/'run.json')
                self.assertEqual(run['status'],common.GAP_STATUS)
                self.assertTrue(run['first_run_postcheck_gap_preserved'])
                core.verify_inventory(folder,common.REFERENCE['original_inventory'])
                self.assertFalse((folder/'01-initialization/completed.json').exists())
                self.assertEqual(core.read(folder/'run.json')['status'],'STOPPED_REVIEW_REQUIRED')
                # An unreviewed failed construction cannot trigger a third fresh process.
                done=folder/'03-initialization/completed.json';done.unlink()
                out=next((folder/'03-initialization').glob('attempt-*/result.json')).parent
                core.write(out/'result.json',{'status':'FAILED_REVIEW_REQUIRED','environment_construction_started':True})
                with self.assertRaisesRegex(RuntimeError,'construction was already attempted'):driver.controller()
                self.assertEqual(calls,['02-initialization','03-initialization'])

    def test_actual_comparison_records_gap_and_rejects_missing_new_postchecks(self):
        if not FIXTURE.exists():self.skipTest('review fixture missing')
        with self.mock_laptop() as folder:
            first=folder/common.REFERENCE['approved_attempt'];outputs=[first]
            for job in worker.JOBS[1:]:
                target=folder/job/'attempt-mocked'
                shutil.copytree(first,target)
                core.write(target/'result.json',self.completed());outputs.append(target)
            result=compare.compare_all(outputs,folder/'mock-comparison.json')
            self.assertEqual(result['status'],common.GAP_STATUS)
            self.assertTrue(result['first_run_postchecks_incomplete'])
            self.assertEqual(len(result['pairs']),3)
            self.assertFalse(result['stage10_complete'])
            data=core.read(outputs[1]/'result.json');data['backend_after']={}
            core.write(outputs[1]/'result.json',data)
            with self.assertRaisesRegex(RuntimeError,'Precision'):compare.compare_all(outputs,folder/'no.json')

    def test_original_file_tamper_is_rejected_before_any_job(self):
        if not FIXTURE.exists():self.skipTest('review fixture missing')
        with self.mock_laptop() as folder:
            (folder/common.REFERENCE['approved_attempt']/'result.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'SHA256'):common.validate_saved_plan(folder)

if __name__=='__main__':unittest.main(verbosity=2)
