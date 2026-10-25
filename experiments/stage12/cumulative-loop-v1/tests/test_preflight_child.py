"""The actual child command must load the verified base, without inherited imports."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from support import loop_ops


class PreflightChild(unittest.TestCase):
    def test_fresh_interpreter_uses_bootstrap_without_a_local_common_module(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            (folder/'bootstrap.py').write_text('''import json
from pathlib import Path
from types import SimpleNamespace
def load_base():
    def write(path,value,exclusive=False):
        assert exclusive
        with Path(path).open('x') as stream:json.dump(value,stream)
    return SimpleNamespace(write=write,preflight=lambda:dict(verified_base_loaded=True))
''')
            output=folder/'preflight.json'
            with patch.object(loop_ops,'HERE',folder), patch.object(loop_ops.old,'PYTHON',Path(sys.executable)):
                argv=loop_ops.preflight_argv(output)
            env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME')}
            result=subprocess.run(argv,cwd=folder,env=env,capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(output.read_bytes()),dict(verified_base_loaded=True))
