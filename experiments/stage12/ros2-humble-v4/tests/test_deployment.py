import ast
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'package'))
import deploy_contract as c
import probe


class ContractTests(unittest.TestCase):
    def test_reviewed_final_report(self):
        cp = c.validate_report(ROOT / 'reference/local-validation-passed.json')
        self.assertEqual(cp['policy_updates'], 11314)
        self.assertEqual(cp['sha256'], c.MODEL_SHA)

    def packet(self, **changes):
        p = dict(schema=c.SCHEMA, session='a' * 32, model='b' * 64, seq=0, reset=True, obs=[0.] * 61)
        p.update(changes)
        return json.dumps(p)

    def test_sequence_identity_and_first_reset(self):
        c.request(self.packet(), 'a' * 32, 'b' * 64, 0)
        for changes in [dict(seq=1), dict(seq=True), dict(reset=False), dict(reset=1),
                        dict(model='c' * 64), dict(session='d' * 32), dict(obs=[0.] * 60),
                        dict(obs=[float('nan')] * 61), dict(obs=[True] * 61)]:
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                c.request(self.packet(**changes), 'a' * 32, 'b' * 64, 0)
        with self.assertRaises(RuntimeError):
            c.request(self.packet(), 'a' * 32, 'b' * 64, 1)

    def test_inference_tolerance_and_nonfinite(self):
        self.assertLess(c.comparison([1. + 1e-6], [1.]), 1e-5)
        for value in ([1.1], [float('nan')], [float('inf')], [], [True]):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                c.comparison(value, [1.])

    def test_interface_only_xml(self):
        import xml.etree.ElementTree as ET
        tree = ET.fromstring(c.wired_xml(c.PC_IP))
        ns = {'d': 'http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles'}
        self.assertEqual(tree.find('.//d:interfaceWhiteList/d:address', ns).text, c.PC_IP)
        self.assertEqual(tree.find('.//d:useBuiltinTransports', ns).text, 'false')
        with self.assertRaises(RuntimeError):
            c.wired_xml('192.168.8.88')

    def host(self, jetson=False):
        ip = c.ORIN_IP if jetson else c.PC_IP
        iface = 'enP8p1s0' if jetson else 'enp4s0'
        return dict(os=dict(ID='ubuntu', VERSION_ID='22.04'), architecture='aarch64' if jetson else 'x86_64',
                    wired_addresses=dict(stdout=json.dumps([dict(ifname=iface, addr_info=[dict(local=ip,prefixlen=24)])])),
                    route_to_peer=dict(stdout=json.dumps([dict(dev=iface,prefsrc=ip)])),
                    ros_candidates=[dict(requested_python='/usr/bin/python3',
                        details=dict(ros_ok=True,ros_distro='humble',rmw='rmw_fastrtps_cpp',
                                     ort_providers=['CPUExecutionProvider']))])

    def test_existing_two_host_readiness(self):
        self.assertTrue(probe.readiness(self.host())['ready'])
        self.assertTrue(probe.readiness(self.host(True), True)['ready'])

    def test_wrong_os_rmw_or_route_blocks(self):
        for mutate in [lambda h:h['os'].update(VERSION_ID='24.04'),
                       lambda h:h.update(architecture='aarch64'),
                       lambda h:h['ros_candidates'][0]['details'].update(ros_distro='jazzy'),
                       lambda h:h['ros_candidates'][0]['details'].update(rmw='rmw_cyclonedds_cpp'),
                       lambda h:h['route_to_peer'].update(stdout='[{"dev":"wlan0"}]')]:
            h = self.host(); mutate(h)
            self.assertFalse(probe.readiness(h)['ready'])

    def test_no_ort_blocks_jetson_but_not_pc(self):
        h = self.host(True); h['ros_candidates'][0]['details'].pop('ort_providers')
        self.assertFalse(probe.readiness(h, True)['ready'])
        h = self.host(); h['ros_candidates'][0]['details'].pop('ort_providers')
        self.assertTrue(probe.readiness(h)['ready'])

    def test_clean_env_keeps_system_unchanged(self):
        with patch.dict(os.environ, {'PYTHONPATH':'bad', 'ROS_DISTRO':'jazzy', 'CONDA_PREFIX':'bad',
                                      'ROS_DISCOVERY_SERVER':'bad', 'LD_LIBRARY_PATH':'bad'}):
            e = probe.clean_env()
            self.assertNotIn('PYTHONPATH', e)
            self.assertNotIn('LD_LIBRARY_PATH', e)
            self.assertNotIn('ROS_DISCOVERY_SERVER', e)
            self.assertEqual(e['ROS_DOMAIN_ID'], '62')
            self.assertEqual(os.environ['ROS_DISTRO'], 'jazzy')

    def test_export_manifest_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d); model=folder/'test.onnx'; model.write_bytes(b'fixture only')
            r=dict(status='EXPORT_PARITY_PASSED',checkpoint_sha256=c.MODEL_SHA,
                   validation_report_sha256=c.REPORT_SHA,policy_and_normalizer_unchanged=True,
                   files={model.name:dict(bytes=model.stat().st_size,sha256=c.sha(model))})
            (folder/'export-report.json').write_text(json.dumps(r))
            c.verify_export(folder)
            model.write_bytes(b'modified model')
            with self.assertRaises(RuntimeError):c.verify_export(folder)


class CpuOnnxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import numpy as np
        cls.torch, cls.np = torch, np
        torch.manual_seed(43)
        torch.set_num_threads(1)
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls.directory.name) / 'fixture.onnx'
        class Exportable(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('mean',torch.linspace(-.3,.3,61))
                self.register_buffer('std',torch.linspace(.2,2.,61))
                self.lstm=torch.nn.LSTM(61,256,1)
                self.mlp=torch.nn.Sequential(torch.nn.Linear(256,32),torch.nn.ELU(),torch.nn.Linear(32,14))
                self.input_names=['obs','h_in','c_in'];self.output_names=['actions','h_out','c_out']
            def forward(self,obs,h,c):
                out,(h,c)=self.lstm(((obs-self.mean)/self.std).unsqueeze(0),(h,c))
                return self.mlp(out.squeeze(0)),h,c
            def get_dummy_inputs(self):
                return torch.zeros(1,61),torch.zeros(1,1,256),torch.zeros(1,1,256)
        cls.network=Exportable().eval()
        # Execute the exact pinned mjlab export method, without constructing a
        # GPU simulator. This fixture is explicitly not the user's final model.
        source=ast.parse((ROOT/'reference/upstream-mjlab-runner.py').read_text())
        method=next(n for n in next(n for n in source.body if isinstance(n,ast.ClassDef)).body
                    if isinstance(n,ast.FunctionDef) and n.name=='export_policy_to_onnx')
        namespace={'torch':torch,'os':os}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),
                     '<pinned mjlab export method>','exec'),namespace)
        actor=SimpleNamespace(as_onnx=lambda verbose:cls.network)
        runner=SimpleNamespace(alg=SimpleNamespace(get_policy=lambda:actor))
        namespace['export_policy_to_onnx'](runner,cls.directory.name,cls.path.name)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_128_sequential_action_and_state_parity(self):
        t,np=self.torch,self.np
        rng=np.random.default_rng(174)
        p=c.OnnxPolicy(self.path)
        h=t.zeros(1,1,256);cell=t.zeros_like(h)
        for i in range(128):
            reset=i in (0,64)
            if reset:h=t.zeros(1,1,256);cell=t.zeros_like(h)
            obs=rng.normal(0,.2,61).astype(np.float32)
            with t.inference_mode():
                a,h,cell=self.network(t.from_numpy(obs[None]),h,cell)
            actual=p.step(obs.tolist(),reset=reset)
            self.assertTrue(np.allclose(actual,a[0].numpy(),atol=c.ATOL,rtol=c.RTOL))
            self.assertTrue(np.allclose(p.h,h.numpy(),atol=c.ATOL,rtol=c.RTOL))
            self.assertTrue(np.allclose(p.c,cell.numpy(),atol=c.ATOL,rtol=c.RTOL))

    def test_hidden_state_carried_and_reset(self):
        p=c.OnnxPolicy(self.path)
        a=p.step([0.]*61,reset=True)
        b=p.step([0.]*61)
        self.assertNotEqual(a,b)
        self.assertEqual(a,p.step([0.]*61,reset=True))

    def test_invalid_observation_does_not_advance_state(self):
        p=c.OnnxPolicy(self.path);p.step([0.]*61)
        h=p.h.copy();cell=p.c.copy()
        with self.assertRaises(RuntimeError):p.step([float('nan')]*61)
        self.assertTrue(self.np.array_equal(h,p.h))
        self.assertTrue(self.np.array_equal(cell,p.c))

    def test_two_nodes_128_requests_with_in_memory_ros_api(self):
        # A transport-independent integration check, not a claim of ROS/DDS or
        # Jetson execution. Match only the verified Humble Node API surface.
        import concurrent.futures
        import queue
        import shutil
        import threading
        import types
        import ros_check
        guard=threading.Lock();publishers={};subscribers={}
        class Message:
            def __init__(self,data):self.data=data
        class Node:
            def __init__(self,name):self.queue=queue.Queue();self.owned_p=[];self.owned_s=[]
            def create_publisher(self,kind,topic,qos):
                with guard:publishers.setdefault(topic,[]).append(self)
                self.owned_p.append(topic)
                def publish(message):
                    with guard:targets=list(subscribers.get(topic,[]))
                    for node,callback in targets:node.queue.put((callback,Message(message.data)))
                return SimpleNamespace(publish=publish)
            def create_subscription(self,kind,topic,callback,qos):
                with guard:subscribers.setdefault(topic,[]).append((self,callback))
                self.owned_s.append(topic)
                return object()
            def count_subscribers(self,topic):
                with guard:return len(subscribers.get(topic,[]))
            def count_publishers(self,topic):
                with guard:return len(publishers.get(topic,[]))
            def destroy_node(self):
                with guard:
                    for topic in self.owned_p:publishers[topic].remove(self)
                    for topic in self.owned_s:subscribers[topic]=[(n,c) for n,c in subscribers[topic] if n is not self]
        def spin_once(node,timeout_sec):
            try:callback,message=node.queue.get(timeout=timeout_sec)
            except queue.Empty:return
            callback(message)
        rclpy=types.ModuleType('rclpy');rclpy.init=lambda:None;rclpy.shutdown=lambda:None;rclpy.spin_once=spin_once
        node_module=types.ModuleType('rclpy.node');node_module.Node=Node
        qos=types.ModuleType('rclpy.qos');qos.QoSProfile=lambda **kw:kw
        qos.HistoryPolicy=SimpleNamespace(KEEP_LAST=1);qos.ReliabilityPolicy=SimpleNamespace(RELIABLE=1)
        qos.DurabilityPolicy=SimpleNamespace(VOLATILE=1)
        msg=types.ModuleType('std_msgs.msg');msg.String=Message
        modules={'rclpy':rclpy,'rclpy.node':node_module,'rclpy.qos':qos,
                 'std_msgs':types.ModuleType('std_msgs'),'std_msgs.msg':msg}
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            shutil.copyfile(self.path,folder/'final6h_005304.onnx')
            policy=c.OnnxPolicy(self.path);cases=[]
            for seq in range(128):
                reset=seq in (0,64);obs=[.01*(seq%5)]*61
                actions=policy.step(obs,reset=reset)
                cases.append(dict(seq=seq,reset=reset,obs=obs,actions=actions,
                                  h=policy.h.reshape(-1).tolist(),c=policy.c.reshape(-1).tolist()))
            (folder/'golden.json').write_text(json.dumps(dict(onnx_sha256=c.sha(self.path),cases=cases)))
            pins={p.name:dict(bytes=p.stat().st_size,sha256=c.sha(p)) for p in folder.iterdir()}
            (folder/'export-report.json').write_text(json.dumps(dict(status='EXPORT_PARITY_PASSED',
                checkpoint_sha256=c.MODEL_SHA,validation_report_sha256=c.REPORT_SHA,
                policy_and_normalizer_unchanged=True,files=pins)))
            def args(role):return SimpleNamespace(role=role,folder=str(folder),session='a'*32,
                                                  report=str(folder/(role+'.json')))
            with patch.dict(sys.modules,modules), concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                server=pool.submit(ros_check.run,args('server'))
                client=pool.submit(ros_check.run,args('client'))
                self.assertEqual(client.result(timeout=12),0)
                self.assertEqual(server.result(timeout=12),0)
            result=json.loads((folder/'client.json').read_bytes())
            self.assertEqual(result['completed_samples'],128)
            self.assertEqual(result['status'],'LINK_PARITY_PASSED')
            self.assertFalse(result['hardware_acceptance'])
            self.assertFalse(result['realtime_acceptance'])


if __name__=='__main__':
    unittest.main()
