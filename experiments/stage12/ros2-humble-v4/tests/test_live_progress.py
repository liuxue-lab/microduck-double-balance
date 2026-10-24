import contextlib
import hashlib
import http.server
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'package'))
import live_progress as live
import operation_guard as guard
import shared_ssh as ssh


class ProgressTests(unittest.TestCase):
    def test_child_has_terminal_and_output_is_logged(self):
        with tempfile.TemporaryDirectory() as d:
            log=Path(d)/'output.log'; display=io.StringIO()
            with contextlib.redirect_stdout(display):
                rc=live.run([sys.executable,'-u','-c',
                             "import os; print('TTY='+str(os.isatty(1))); print('42% 123 kB/s 2min')"],
                            os.environ,log,5,'测试下载')
            self.assertEqual(rc,0)
            self.assertIn('TTY=True',log.read_text())
            self.assertIn('42% 123 kB/s',display.getvalue())

    def test_download_timeout_retains_partial_log(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            log=Path(d)/'partial.log'
            with self.assertRaises(TimeoutError):
                live.run([sys.executable,'-u','-c',"import time; print('partial download'); time.sleep(3)"],
                         os.environ,log,.3,'测试下载')
            self.assertIn('partial download',log.read_text())

    def test_pip_percent_handles_split_chunks_and_final_without_newline(self):
        display=io.StringIO()
        with contextlib.redirect_stdout(display):
            p=live.PipText()
            p.feed('\x1b[2K\r 5.0/');p.feed('20.0 MB 1.0 MB/s eta 0:00:15\r')
            p.feed('20.0/20.0 MB 1.0 MB/s eta 0:00:00',final=True)
        self.assertIn('25.0%',display.getvalue())
        self.assertIn('100.0%',display.getvalue())
        self.assertIn('1.0 MB/s',display.getvalue())

    def test_package_lock_detection_uses_device_and_inode(self):
        with tempfile.TemporaryDirectory() as d:
            lock=Path(d)/'lock';lock.touch();st=lock.stat()
            locks=Path(d)/'proc-locks'
            locks.write_text('5: POSIX ADVISORY WRITE 1234 %02x:%02x:%d 0 EOF\n' %
                             (os.major(st.st_dev),os.minor(st.st_dev),st.st_ino))
            self.assertEqual(guard.held_package_locks([lock],locks),
                             [dict(pid=1234,path=str(lock),kind='POSIX')])

    def test_ssh_failure_evidence_does_not_assume_bad_password(self):
        with tempfile.TemporaryDirectory() as d,patch.object(ssh,'DIAGNOSTICS',{'attempts':[]}):
            log=Path(d)/'ssh.log'
            log.write_text('Connection closed by 192.168.50.2 port 22\n')
            r=ssh.record_attempt(log,255,'test')
            self.assertEqual(r['category'],'CONNECTION_FAILED_WITHOUT_CONFIRMED_AUTHENTICATION')
            log.write_text('Authenticated to 192.168.50.2 using password.\nConnection closed\n')
            r=ssh.record_attempt(log,255,'test')
            self.assertEqual(r['category'],'FAILED_AFTER_AUTHENTICATION')

    def test_ssh_first_session_is_wired_command_and_single_authentication(self):
        with tempfile.TemporaryDirectory() as d,patch.object(ssh,'_directory',return_value=Path(d)),\
             patch.object(ssh,'HOST','jetson@192.168.50.2'),patch.object(ssh,'PORT',22),\
             patch.object(ssh,'_SOCKET',None),patch.object(ssh,'_live',side_effect=[False,True]),\
             patch.object(ssh,'DIAGNOSTICS',{'attempts':[]}),\
             patch.object(ssh.subprocess,'run',return_value=types.SimpleNamespace(returncode=0)) as run,\
             contextlib.redirect_stdout(io.StringIO()):
            ssh.connect()
            argv=run.call_args.args[0]
            self.assertNotIn('-N',argv);self.assertNotIn('-f',argv)
            self.assertEqual(argv[-1],'/usr/bin/true')
            self.assertIn('ProxyCommand=none',argv);self.assertIn('ProxyJump=none',argv)
            self.assertIn('192.168.50.1',argv)
            self.assertIn('NumberOfPasswordPrompts=1',argv)
            self.assertEqual(run.call_count,1)

    @unittest.skipUnless(shutil.which('apt-get'), 'APT unavailable')
    def test_real_apt_download_shows_speed_percentage_and_reuses_cache(self):
        # A dedicated local HTTP repository and isolated APT state. Its synthetic
        # package is only downloaded, never installed. trusted=yes is confined
        # to this disposable fixture; deployed commands retain signature checks.
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);root.chmod(0o755)
            repo=root/'repo';repo.mkdir();pool=repo/'pool';pool.mkdir()
            archive=pool/'microduck-progress-fixture_1.0_all.deb'
            data=bytes(range(256))*8192;archive.write_bytes(data)
            index=repo/'dists/jammy/main/binary-amd64/Packages';index.parent.mkdir(parents=True)
            index.write_text('Package: microduck-progress-fixture\nVersion: 1.0\nArchitecture: all\n'
                             'Maintainer: Fixture <fixture@example.invalid>\nDescription: Download-only fixture\n'
                             'Filename: pool/'+archive.name+'\nSize: '+str(len(data))+'\nSHA256: '+hashlib.sha256(data).hexdigest()+'\n\n')
            raw=index.read_bytes()
            (repo/'dists/jammy/Release').write_text(
                'Origin: MicroduckTest\nLabel: MicroduckTest\nSuite: jammy\nCodename: jammy\n'
                'Architectures: amd64 all\nComponents: main\nSHA256:\n '+hashlib.sha256(raw).hexdigest()+
                ' '+str(len(raw))+' main/binary-amd64/Packages\n')
            counts={'deb':0}
            class Handler(http.server.SimpleHTTPRequestHandler):
                def __init__(self,*a,**kw):super().__init__(*a,directory=str(repo),**kw)
                def log_message(self,*a):pass
                def copyfile(self,source,outputfile):
                    if self.path.endswith('.deb'):
                        counts['deb']+=1
                        while True:
                            chunk=source.read(65536)
                            if not chunk:break
                            outputfile.write(chunk);outputfile.flush();time.sleep(.25)
                    else:super().copyfile(source,outputfile)
            server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                empty=root/'empty';empty.mkdir()
                lists=root/'lists';lists.mkdir();(lists/'partial').mkdir()
                archives=root/'archives';archives.mkdir();(archives/'partial').mkdir()
                status=root/'status';status.touch()
                sources=root/'sources.list'
                sources.write_text('deb [trusted=yes] http://127.0.0.1:%d jammy main\n'%server.server_port)
                config=root/'apt.conf'
                config.write_text('Dir::Etc::parts "'+str(empty)+'";\nDir::Etc::main "/dev/null";\n')
                env=dict(os.environ,APT_CONFIG=str(config),LC_ALL='C',DEBIAN_FRONTEND='noninteractive')
                values={'Dir::Etc::sourcelist':sources,'Dir::Etc::sourceparts':empty,
                        'Dir::State::lists':lists,'Dir::State::status':status,'Dir::Cache::archives':archives,
                        'Dir::Cache::pkgcache':'','Dir::Cache::srcpkgcache':'','quiet':'0',
                        # The build container has no mapped _apt uid. This
                        # local fixture stays under the current uid; production
                        # APT commands do not change APT::Sandbox::User.
                        'APT::Sandbox::User':'root',
                        'APT::Color':'0','Acquire::Languages':'none'}
                options=[arg for key,value in values.items() for arg in ('-o',key+'='+str(value))]
                update=subprocess.run(['apt-get',*options,'update'],env=env,capture_output=True,text=True,timeout=30)
                self.assertEqual(update.returncode,0,update.stdout+update.stderr)
                argv=['apt-get',*options,'--download-only','--yes','--no-install-recommends',
                      'install','microduck-progress-fixture']
                out=io.StringIO()
                with contextlib.redirect_stdout(out):
                    rc=live.run(argv,env,root/'download.log',30,'APT真实下载测试')
                self.assertEqual(rc,0,(root/'download.log').read_text())
                text=out.getvalue()
                self.assertRegex(text,r'\d+%')
                self.assertRegex(text,r'\d+(?:\.\d+)?\s*[kMG]?B/s')
                before=counts['deb']
                with contextlib.redirect_stdout(io.StringIO()):
                    rc=live.run(argv,env,root/'reuse.log',15,'APT缓存复用测试')
                self.assertEqual(rc,0)
                self.assertEqual(counts['deb'],before)
                self.assertEqual(hashlib.sha256((archives/archive.name).read_bytes()).hexdigest(),
                                 hashlib.sha256(data).hexdigest())
                (ROOT/'apt-progress-fixture.log').write_text(text)
            finally:
                server.shutdown();server.server_close();thread.join(timeout=2)


if __name__=='__main__':unittest.main()
