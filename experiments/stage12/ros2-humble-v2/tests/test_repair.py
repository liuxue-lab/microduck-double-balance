import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'package'))
import host_setup as h
import network_check as n
import repair


class SetupGuardTests(unittest.TestCase):
    def test_official_source_package_pin(self):
        path=ROOT/'package'/h.BOOTSTRAP
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),h.BOOTSTRAP_SHA)

    def test_apt_allows_only_needed_unprotected_plan(self):
        self.assertEqual(h.check_apt_plan('Inst ros-humble-rclpy (3.3.17 jammy)\nInst libfoo (1.2 jammy)\n'),
                         ['ros-humble-rclpy','libfoo'])

    def test_apt_rejects_removal_and_gpu_kernel_changes(self):
        for text in ['Remv systemd [1.0]\n','Inst nvidia-l4t-core [36.4.3] (36.4.4)\n',
                     'Inst cuda-toolkit-12-6 (12.6)\n','Inst linux-image-5.15.0 (1)\n',
                     'Inst python3.12 (3.12.15)\n','1 downgraded\n']:
            with self.subTest(text=text),self.assertRaises(RuntimeError):h.check_apt_plan(text)

    def resource(self,disk=4*h.GIB,ram=h.GIB):
        return dict(memory=dict(MemAvailable=ram),disks={'/':dict(free_bytes=disk,free_inodes=20000)})

    def test_ram_pressure_does_not_delete_disk_files(self):
        with tempfile.TemporaryDirectory() as d,patch.object(h,'resources',return_value=self.resource(ram=1)),\
             patch.object(h.Setup,'command') as command:
            with self.assertRaisesRegex(RuntimeError,'RAM_AVAILABLE_LOW'):h.Setup(d).room(install=True)
            command.assert_not_called()

    def test_only_low_disk_triggers_apt_cache_clean(self):
        with tempfile.TemporaryDirectory() as d,patch.object(h,'resources',side_effect=[self.resource(disk=1),self.resource()]),\
             patch.object(h.Setup,'sudo',return_value=['sudo','-n']),patch.object(h.Setup,'command') as command:
            h.Setup(d).room(install=True)
            command.assert_called_once_with(['sudo','-n','apt-get','clean'])

    def test_sufficient_space_preserves_every_cache(self):
        with tempfile.TemporaryDirectory() as d,patch.object(h,'resources',return_value=self.resource()),\
             patch.object(h.Setup,'command') as command:
            h.Setup(d).room(install=True)
            command.assert_not_called()

    def test_frozen_export_accepts_only_reported_bytes(self):
        source=json.loads((ROOT/'reference/export-passed-environment-pending.json').read_bytes())['export']
        with patch.object(repair,'verify_export',return_value=source):self.assertEqual(repair.frozen_export(),source)
        changed=copy.deepcopy(source);changed['files']['final6h_005304.onnx']['sha256']='a'*64
        with patch.object(repair,'verify_export',return_value=changed),self.assertRaises(RuntimeError):repair.frozen_export()


class NetworkTests(unittest.TestCase):
    def profile(self):
        return dict(zip(n.PROPERTIES,['enp4s0','manual','192.168.50.1/24','','','yes','disabled']))

    def snapshot(self,active=True,carrier='1'):
        return dict(carrier=carrier,profile=self.profile(),profile_matches_verified_guide=True,
                    active_connection=dict(stdout='jetson-direct\n' if active else '--\n'),
                    addresses=dict(stdout=json.dumps([dict(addr_info=[dict(local='192.168.50.1',prefixlen=24)])]) if active else '[]'),
                    route=dict(stdout='[{"dev":"enp4s0","prefsrc":"192.168.50.1"}]' if active else '[]'))

    def test_verified_profile_rejects_gateway_dns_or_other_net(self):
        self.assertTrue(n.profile_matches(self.profile()))
        for key,value in [('ipv4.gateway','192.168.50.2'),('ipv4.dns','8.8.8.8'),
                          ('ipv4.addresses','192.168.8.1/24'),('ipv6.method','auto')]:
            values=self.profile();values[key]=value
            self.assertFalse(n.profile_matches(values))

    def test_inactive_profile_is_activated_even_if_carrier_unknown(self):
        before=self.snapshot(False,None);after=self.snapshot()
        with patch.object(n,'snapshot',side_effect=[before,after]),patch.object(n,'tcp_ready',return_value={'ready':True}),\
             patch.object(n,'command',return_value={'returncode':0,'stdout':'[]'}),patch.object(n.subprocess,'run') as run:
            run.return_value.returncode=0;run.return_value.stdout='ok';run.return_value.stderr=''
            r=n.diagnose(True,lambda:['sudo','-n'])
            self.assertEqual(r['status'],'SSH_PORT_READY')
            argv=run.call_args.args[0]
            self.assertIn('up',argv);self.assertNotIn('modify',argv);self.assertNotIn('down',argv)

    def test_active_correct_profile_never_restarted(self):
        s=self.snapshot()
        with patch.object(n,'snapshot',return_value=s),patch.object(n,'tcp_ready',return_value={'ready':True}),\
             patch.object(n,'command',return_value={}),patch.object(n.subprocess,'run') as run:
            self.assertEqual(n.diagnose(True,lambda:['sudo'])['status'],'SSH_PORT_READY')
            run.assert_not_called()

    def test_no_route_not_treated_as_missing_ros(self):
        with patch.object(n,'snapshot',return_value=self.snapshot()),\
             patch.object(n,'tcp_ready',return_value={'ready':False,'errno':113,'error':'No route to host'}),\
             patch.object(n,'command',return_value={'stdout':'[{"state":["FAILED"]}]'}):
            r=n.diagnose(False)
            self.assertEqual(r['status'],'ORIN_UNREACHABLE')
            self.assertIn('jetson-wired',r['next_step'])

    def test_physical_and_service_failures_are_distinguished(self):
        for carrier,port,expected in [('0',{'ready':False},'PHYSICAL_LINK_DOWN'),
                                     ('1',{'ready':False,'errno':111},'SSH_SERVICE_REFUSED')]:
            with self.subTest(expected=expected),patch.object(n,'snapshot',return_value=self.snapshot(carrier=carrier)),\
                 patch.object(n,'tcp_ready',return_value=port),patch.object(n,'command',return_value={}):
                self.assertEqual(n.diagnose(False)['status'],expected)


if __name__=='__main__':unittest.main()
