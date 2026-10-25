"""Full 6000-packet CPU orchestration. Synthetic recurrence is not real ONNX/DDS proof."""
import io
from pathlib import Path
import socket
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from support import c, loop_sim
from telemetry import Timing as RealTiming

class Timing(RealTiming):
    # Transport only, unpaced. Fixed-release scheduling is tested separately.
    def begin(self):
        self.epoch=self.clock()-len(self.rows)*self.PERIOD_NS
        return super().begin()
import loop_ros
from bus_fixture import FakeBus

class SyntheticRecurrent:
    def __init__(self):self.reset()
    def reset(self):self.state=np.zeros(14,dtype=np.float32)
    def step(self,obs,reset=False):
        if reset:self.reset()
        values=np.array(obs,dtype=np.float32)
        self.state=self.state*np.float32(.5)+values[:14]*np.float32(.01)+values[55:].sum()*np.float32(.001)
        return self.state.tolist()

class Transport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch
        torch.set_num_threads(1)
    def reference(self):
        torch=self.torch
        class Reference:
            def __init__(self):self.network=SyntheticRecurrent()
            def reset(self):self.network.reset()
            def __call__(self,observations,stochastic_output=False):
                return torch.tensor([self.network.step(observations['actor'][0].tolist())])
        return Reference()
    def observations(self,seq):
        torch=self.torch
        class Observations(dict):
            def to(self,device):return Observations({k:v.to(device) for k,v in self.items()})
        return Observations(actor=(torch.linspace(-.1,.1,61)+.003*(seq%17)).reshape(1,61))

    def test_6000_recurrent_requests_socket_bridge_and_live_reference(self):
        torch = self.torch
        bus, errors = FakeBus(), []
        with tempfile.TemporaryDirectory(prefix='md12-test-', dir='/tmp') as d:
            folder = Path(d)
            address = folder / 's'
            server_args = SimpleNamespace(role='server', folder=d, session='a' * 32)
            bridge_args = SimpleNamespace(role='bridge', folder=d, socket=str(address), session='a' * 32)
            server_result, bridge_result = {}, {}
            def work(fn, args, result):
                try:
                    fn(args, result)
                except BaseException as exc:
                    errors.append(repr(exc))
            # This build host permits socketpair but forbids socket(AF_UNIX).
            # Exercise real byte transport/peer credentials through socketpair;
            # pathname bind/listen/connect are fixture adapters, not LAN proof.
            client_end, server_end = socket.socketpair()
            notify_send, notify_receive = socket.socketpair()
            class Listener:
                def bind(self, value):
                    assert value == str(address)
                def listen(self, backlog):
                    assert backlog == 1
                def fileno(self):
                    return notify_receive.fileno()
                def accept(self):
                    notify_receive.recv(1)
                    return server_end, None
                def close(self):
                    notify_receive.close()
                    notify_send.close()
            class Connector:
                def __getattr__(self, name):
                    return getattr(client_end, name)
                def connect(self, value):
                    assert value == str(address)
                    notify_send.sendall(b'1')
            with patch.object(loop_ros, 'endpoints', side_effect=bus.endpoints), patch.object(loop_ros, 'frozen'), \
                    patch.object(socket, 'socket', side_effect=[Listener(), Connector()]), \
                    patch.object(loop_ros, 'OnnxPolicy', side_effect=lambda path: SyntheticRecurrent()), \
                    patch.object(torch.cuda, 'synchronize', side_effect=AssertionError('Native CPU path touched CUDA')):
                server = threading.Thread(target=work, args=(loop_ros.server, server_args, server_result), daemon=True)
                bridge = threading.Thread(target=work, args=(loop_ros.bridge, bridge_args, bridge_result), daemon=True)
                server.start()
                bridge.start()
                deadline = time.monotonic() + 8
                while not (folder / 'bridge-ready.json').exists():
                    self.assertFalse(errors, errors)
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(.01)
                trace = io.StringIO()
                client = c.LoopClient(address, 'a' * 32, trace)
                try:
                    with torch.inference_mode():
                        for case in range(2):
                            stats = dict(compared_actions=0, max_action_absolute_error=0., timing=Timing(folder))
                            reference = self.reference()
                            proxy = loop_sim.RemotePolicy(reference, client, torch, case, stats)
                            proxy.reset()
                            for step in range(3000):
                                actions = proxy(self.observations(case * 3000 + step))
                                stats['timing'].environment_started()
                                stats['timing'].add(environment_ms=0.,state_capture_ms=0.)
                                stats['timing'].end()
                                self.assertEqual(tuple(actions.shape), (1, 14))
                                self.assertTrue(bool(torch.isfinite(actions).all()))
                            self.assertEqual(stats['timing'].summary()['completed_ticks'], 3000)
                            self.assertEqual(stats['compared_actions'], 3000)
                            self.assertLess(stats['max_action_absolute_error'], 2e-5)
                    client.finish()
                finally:
                    client.close()
                server.join(5)
                bridge.join(5)
                self.assertFalse(server.is_alive())
                self.assertFalse(bridge.is_alive())
            self.assertFalse(errors, errors)
            self.assertEqual(server_result['recurrent_reset_sequences'], [0, 3000])
            self.assertEqual(server_result['transcript_sha256'], client.chain)
            self.assertEqual(bridge_result['transcript_sha256'], client.chain)
            self.assertEqual(server_result['completed_requests'], 6000)
            self.assertEqual(client.seq, 6000)
            self.assertEqual(len(trace.getvalue().splitlines()), 6000)


    def test_parity_error_does_not_return_a_fallback_action(self):
        client=SimpleNamespace(seq=0,step=lambda obs:[100.]*14,ipc_ms=[1.],ros_ms=[.8],inference_ms=[.3])
        with tempfile.TemporaryDirectory() as d:
            stats=dict(compared_actions=0,max_action_absolute_error=0.,timing=Timing(d))
            proxy=loop_sim.RemotePolicy(self.reference(),client,self.torch,0,stats)
            proxy.reset()
            with self.torch.inference_mode(),self.assertRaises(RuntimeError):proxy(self.observations(0))
            self.assertEqual(stats['compared_actions'],0)
            self.assertTrue(stats['timing'].summary()['incomplete_tick'])
            self.assertEqual(stats['timing'].summary()['completed_ticks'],0)
