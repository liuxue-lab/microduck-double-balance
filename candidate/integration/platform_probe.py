"""Deny the optional CPU-name subprocess via CPython's own OSError fallback.

No subprocess permission is added. Warp uses the resulting empty label as CPU;
the CUDA device name/capability still comes from the native CUDA driver.
"""
from pathlib import Path
import errno
import hashlib
import inspect
import platform
import sys
from integration.durable import require, file_digest

REVIEWED_FUNCTION_SHA256 = '53e1da5ebc4b4a0848cbd2770311ada42e82eeae00e89966d859406d5a45f2fc'


class ProcessorFallback:
    def __init__(self):
        self.source = None
        self.identity = None
        self.code = None
        self.report = dict(status='NOT_CONFIGURED', external_processes_allowed=0,
                           optional_processor_denials=[], installed_sources_modified=False,
                           semantics='Optional CPU display label only; CUDA device checks unchanged')

    def configure(self):
        require(sys.platform == 'linux' and sys.version_info[:3] == (3, 12, 14),
                'Reviewed processor fallback requires the pinned Linux Python 3.12.14')
        function = platform._Processor.from_subprocess
        raw = inspect.getsource(function).encode('utf-8')
        require(hashlib.sha256(raw).hexdigest() == REVIEWED_FUNCTION_SHA256,
                'CPython optional processor function differs from reviewed source')
        self.source = Path(inspect.getsourcefile(function)).resolve()
        self.identity = file_digest(self.source)
        self.code = function.__code__
        self.report.update(status='VERIFIED_OPTIONAL_PROCESSOR_FALLBACK_READY',
                           source=str(self.source), source_sha256=self.identity,
                           function_sha256=REVIEWED_FUNCTION_SHA256)

    def handle(self, event, args):
        if event != 'subprocess.Popen' or self.source is None:
            return
        executable, argv, _cwd, _env = args
        if executable != 'uname' or not isinstance(argv, (list, tuple)) or list(argv) != ['uname', '-p']:
            return
        frame = sys._getframe(1)
        while frame is not None:
            if frame.f_code is self.code and frame.f_globals.get('__name__') == 'platform':
                require(file_digest(self.source) == self.identity, 'Processor source changed during query')
                require(not self.report['optional_processor_denials'], 'Repeated optional processor query')
                self.report['optional_processor_denials'].append(dict(argv=list(argv), spawned=False,
                    exception='FileNotFoundError', handling='CPython except OSError -> empty label; Warp -> CPU'))
                self.report['status'] = 'OPTIONAL_PROCESSOR_UNAVAILABLE_NATIVE_FALLBACK'
                raise FileNotFoundError(errno.ENOENT, 'Stage12 disables optional external CPU-name discovery', 'uname')
            frame = frame.f_back

    def verify_unchanged(self):
        if self.source is not None:
            require(file_digest(self.source) == self.identity and
                    platform._Processor.from_subprocess.__code__ is self.code,
                    'CPython processor source/callable changed')
