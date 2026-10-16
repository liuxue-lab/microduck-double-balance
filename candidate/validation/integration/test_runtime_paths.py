from pathlib import Path
import copy
import json
import os
import tempfile
import unittest
from integration.runtime_paths import create,validate,for_owner


class RuntimePaths(unittest.TestCase):
    def test_separate_private_roots_and_long_evidence(self):
        with tempfile.TemporaryDirectory(prefix='stage12-runtime-path-check-') as temp:
            base=Path(temp)/('long-evidence-'*12);base.mkdir()
            rows=[]
            for i in range(2):
                out=base/('owner-'+str(i));boot=out.with_name(out.name+'-bootstrap');boot.mkdir()
                row=create(out,boot);rows.append(row)
                self.assertEqual(for_owner(out),row)
                self.assertLessEqual(len(os.fsencode(row['temp'])),48)
                self.assertEqual(Path(row['short_root']).stat().st_mode & 0o777,0o700)
                self.assertTrue((boot/'runtime-paths.json').is_file())
            self.assertNotEqual(rows[0]['short_root'],rows[1]['short_root'])

    def test_owner_and_receipt_changes_rejected(self):
        with tempfile.TemporaryDirectory(prefix='stage12-runtime-path-check-') as temp:
            base=Path(temp);out=base/'owner-0';boot=base/'owner-0-bootstrap';boot.mkdir()
            row=create(out,boot)
            for key,value in (('pid',-1),('uid',-1),('output',str(base/'other')),
                              ('warp_cache','/tmp'),('temp',str(base))):
                bad=copy.deepcopy(row);bad[key]=value
                with self.assertRaises(RuntimeError):validate(bad,out)
            receipt=Path(row['short_root'])/'owner.json'
            raw=receipt.read_bytes();receipt.write_text('{}')
            with self.assertRaises(RuntimeError):validate(row,out)
            receipt.write_bytes(raw)
            self.assertEqual(validate(row,out),row)

    def test_symlink_and_permission_changes_rejected(self):
        with tempfile.TemporaryDirectory(prefix='stage12-runtime-path-check-') as temp:
            base=Path(temp);out=base/'owner-0';boot=base/'owner-0-bootstrap';boot.mkdir()
            row=create(out,boot);root=Path(row['short_root'])
            root.chmod(0o755)
            with self.assertRaises(RuntimeError):validate(row,out)
            root.chmod(0o700)
            leaf=Path(row['warp_cache']);leaf.rmdir();leaf.symlink_to(row['cache'])
            with self.assertRaises(RuntimeError):validate(row,out)
            leaf.unlink();leaf.mkdir(mode=0o700)
            self.assertEqual(validate(row,out),row)


if __name__=='__main__':unittest.main()
