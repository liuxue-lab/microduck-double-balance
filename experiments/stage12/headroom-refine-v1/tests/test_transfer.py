"""Exercise the actual Git/object/ZIP return path against temporary local repos."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'package'))
import common
import agent
import transfer
spec=importlib.util.spec_from_file_location('return_client_test',ROOT/'controller.py')
client=importlib.util.module_from_spec(spec);spec.loader.exec_module(client)


def digest(data):return hashlib.sha256(data).hexdigest()


class ReturnTests(unittest.TestCase):
    def exercise(self,name='run/checkpoints/candidate.pt',wrong_file_hash=False):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);repo=base/'remote';repo.mkdir();local=base/'local';local.mkdir()
            branch='stage12-headroom-v1-return-test'
            subprocess.run(['git','init','-q','-b',branch,str(repo)],check=True)
            data=b'CPU fixture only, not a real policy'
            archive=repo/'part-0001.zip'
            with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:z.writestr(name,data)
            index=dict(package_identity='a'*64,branch=branch,
                archives={'part-0001.zip':dict(bytes=archive.stat().st_size,
                    sha256=digest(archive.read_bytes()),members=[name])},
                inventory=dict(included={name:dict(bytes=len(data),sha256='0'*64 if wrong_file_hash else digest(data))}))
            raw=(json.dumps(index)+'\n').encode();(repo/'return-index.json').write_bytes(raw)
            subprocess.run(['git','-C',str(repo),'add','.'],check=True)
            subprocess.run(['git','-C',str(repo),'-c','user.name=CPU test','-c','user.email=test@example.invalid',
                '-c','commit.gpgsign=false','commit','-qm','Temporary return fixture'],check=True)
            commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
            client.write(local/'cloud-return.json',dict(status='CLOUD_ARTIFACT_PUSH_COMPLETE',branch=branch,
                commit=commit,index_sha256=digest(raw)))
            with patch.multiple(client,LOCAL=local,REMOTE=str(repo)),patch.object(client,'RELEASE_DATA',{'package_identity':'a'*64},create=True):
                if name.startswith('../') or wrong_file_hash:
                    with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(RuntimeError):client.download_return()
                    self.assertFalse((local/'return-verified.json').exists())
                    self.assertFalse((local/'returned'/commit/'escape.pt').exists())
                else:
                    with contextlib.redirect_stdout(io.StringIO()):
                        client.download_return()
                        client.download_return()  # retry reuses the same pinned bytes; no cloud or PPO call
                    receipt=client.read(local/'return-verified.json')
                    self.assertEqual(receipt['verified_files'],1)
                    self.assertEqual((Path(receipt['path'])/'files'/name).read_bytes(),data)
                    self.assertEqual(client.read(local/'shutdown-reminder.json')['actual_shutdown_confirmation'],'UNCONFIRMED')

    def test_git_return_and_retry_verify_exact_model_bytes(self):self.exercise()
    def test_traversal_member_is_rejected_before_extraction(self):self.exercise('../escape.pt')
    def test_wrong_unpacked_hash_never_marks_return_complete(self):self.exercise(wrong_file_hash=True)

    def test_inventory_keeps_candidates_latest_and_lists_omissions(self):
        with tempfile.TemporaryDirectory() as directory:
            slot=Path(directory)
            for rel in ('run/checkpoints','evaluations/selection-1','cache'):(slot/rel).mkdir(parents=True,exist_ok=True)
            latest=slot/'run/checkpoints/headroom_000505.pt';latest.write_bytes(b'latest')
            candidate=slot/'run/checkpoints/headroom_000255.pt';candidate.write_bytes(b'candidate')
            omitted=slot/'run/checkpoints/headroom_000025.pt';omitted.write_bytes(b'old intermediate')
            (slot/'cache/not-returned').write_bytes(b'cache')
            common.write(slot/'evaluations/selection-1/candidate.json',dict(path=str(candidate)))
            state=dict(status='CAMPAIGN_COMPLETE',exit=dict(checked_baseline_goal_passed=False),
                       processes_alive={'supervisor':False,'child':False})
            with patch.multiple(transfer,SLOT=slot,snapshot=lambda:state,input_checkpoint=lambda:dict(path=str(latest))):
                inventory=transfer.make_inventory()
            self.assertIn('run/checkpoints/headroom_000505.pt',inventory['included'])
            self.assertIn('run/checkpoints/headroom_000255.pt',inventory['included'])
            self.assertIn('run/checkpoints/headroom_000025.pt',inventory['omitted'])
            self.assertNotIn('cache/not-returned',inventory['included'])

    def test_repeated_start_returns_existing_state_without_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);(base/'artifacts').mkdir();slot=base/'slot';slot.mkdir()
            common.write(slot/'started.json',{'already_started':True})
            with patch.multiple(agent,BASE=base,SLOT=slot,snapshot=lambda:{'status':'EXISTING'}),\
                    patch.object(agent,'preflight') as preflight:
                self.assertEqual(agent.launch(False),{'status':'EXISTING'})
                preflight.assert_not_called()


if __name__=='__main__':unittest.main(verbosity=2)
