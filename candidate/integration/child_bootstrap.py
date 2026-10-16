"""Production owner bootstrap shared with isolated, zero-physics import checks."""
from contextlib import contextmanager
from pathlib import Path
import importlib.metadata as meta
import os
import sys
import tempfile
import traceback
from integration.durable import require, write_json


@contextmanager
def owner_bootstrap(root, output):
    root, output = Path(root).absolute(), Path(output).absolute()
    bootstrap = output.with_name(output.name + '-bootstrap')
    for p in (output, bootstrap, *bootstrap.parents):
        require(not p.is_symlink(), 'Owner bootstrap symlink refused: '+str(p))
    require(not output.exists() and not bootstrap.exists(), 'Existing owner/bootstrap is never reused')
    bootstrap.mkdir()
    report = dict(status='STARTED', original_cwd=str(Path.cwd()),
                  native_session_entered=False, installed_sources_modified=False,
                  automatic_retry=False, PPO=0, cloud_training=0)
    font = git = processor = None
    try:
        from integration.runtime_paths import create
        paths = create(output,bootstrap)
        short = Path(paths['short_root'])
        scratch = Path(paths['temp'])
        report.update(cwd=str(scratch),runtime_paths=paths)
        # Each spawned process inherits the campaign cwd. GitPython must run
        # from THIS owner's new scratch directory, before its guarded import.
        os.chdir(scratch)
        os.environ.update(CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1',
                          WANDB_MODE='disabled', WANDB_DISABLED='true', HF_HUB_OFFLINE='1',
                          MUJOCO_GL='disable', MPLBACKEND='Agg', MJLAB_WARP_QUIET='1',
                          TMPDIR=str(scratch),TMP=str(scratch),TEMP=str(scratch),
                          WARP_CACHE_PATH=paths['warp_cache'],CUDA_CACHE_PATH=paths['cuda_cache'],
                          XDG_CACHE_HOME=paths['cache'],MPLCONFIGDIR=paths['matplotlib'],TORCH_HOME=paths['torch'])
        tempfile.tempdir = None
        sys.dont_write_bytecode = True
        from stage12.headless import prepare
        from stage12.optional_imports import FontconfigFallback
        from stage12.git_import import GitImportProbe, is_null_device
        from integration.platform_probe import ProcessorFallback
        report['headless'] = prepare(root, scratch)
        font, git = FontconfigFallback(), GitImportProbe()
        font.configure(root)
        git.configure(root, scratch)
        processor = ProcessorFallback()
        processor.configure()

        def boundary(event, args):
            font.handle(event, args)
            processor.handle(event, args)
            if git.handle(event, args):
                return
            if event in ('socket.connect', 'socket.connect_ex', 'socket.getaddrinfo', 'subprocess.Popen', 'os.system'):
                raise RuntimeError('Native child forbids network/unreviewed subprocess: '+event)
            if event == 'open':
                path, mode, flags = args
                writing = isinstance(mode, str) and any(c in mode for c in 'wax+')
                writing = writing or isinstance(flags, int) and bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
                if writing and not isinstance(path, int):
                    p = Path(os.fsdecode(path)).resolve()
                    require(is_null_device(path) or p.is_relative_to(output) or p.is_relative_to(bootstrap) or p.is_relative_to(short),
                            'Native child write outside its new output: '+str(p))

        sys.addaudithook(boundary)
        git.preload()
        dist = meta.distribution('imageio-ffmpeg')
        media = [Path(dist.locate_file(f)) for f in dist.files or []
                 if str(f).startswith('imageio_ffmpeg/binaries/ffmpeg-') and not str(f).endswith(('.py','.pyc'))]
        require(len(media) == 1 and media[0].is_file(), 'ffmpeg identity differs')
        os.environ['IMAGEIO_FFMPEG_EXE'] = str(media[0])
        report['status'] = 'GUARDED_BOOTSTRAP_READY'
        yield report
        report['status'] = 'GUARDED_BOOTSTRAP_CLOSED'
    except BaseException as exc:
        report.update(status='BOOTSTRAP_OR_SESSION_STOPPED_PRESERVE', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        for name, probe in (('font',font), ('git',git), ('processor',processor)):
            if probe is not None:
                report[name] = probe.report
                try:
                    probe.verify_unchanged()
                except BaseException as exc:
                    report.setdefault('cleanup_errors',[]).append(repr(exc))
                    report['status'] = 'BOOTSTRAP_OR_SESSION_STOPPED_PRESERVE'
        write_json(bootstrap/'bootstrap-report.json',report)
        if report.get('cleanup_errors') and sys.exc_info()[0] is None:
            raise RuntimeError('Bootstrap source protection failed; preserve report')
