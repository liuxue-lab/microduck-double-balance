"""CPU/Unix-socket tests. In-memory ROS is explicitly not real DDS or hardware."""
import copy
from contextlib import ExitStack
import io
import json
import os
from pathlib import Path
import queue
import socket
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'package'))
import loop_contract as c
import loop_ros
import loop_sim
import loop_ops
from deploy_contract import OnnxPolicy


class ProtocolTests(unittest.TestCase):
    def test_only_two_cases_reset_at_exact_boundaries(self):
        for seq in (0, 1, 2999, 3000, 3001, 5999):
            p = c.packet('a' * 32, seq, [0.] * 61)
            c.validate_request(p, 'a' * 32, seq)
            self.assertEqual(p['reset'], seq in (0, 3000))
            for key, value in [('seq', True), ('case', 2), ('step', -1), ('reset', not p['reset']),
                               ('model', 'bad'), ('session', 'b' * 32), ('obs', [float('nan')] * 61),
                               ('obs', [True] * 61), ('obs', [0.] * 60)]:
                with self.subTest(seq=seq, key=key), self.assertRaises(RuntimeError):
                    c.validate_request(dict(p, **{key: value}), 'a' * 32, seq)
        for seq in (-1, 6000, True):
            with self.assertRaises(RuntimeError):
                c.packet('a' * 32, seq, [0.] * 61)

    def test_response_binds_full_observation_and_session(self):
        request = c.packet('a' * 32, 0, [0.] * 61)
        reply = c.reply(request, [2.] * 14, .5)
        c.validate_reply(reply, request)
        for key, value in [('seq', True), ('seq', 1), ('request_sha256', 'bad'), ('model', 'bad'),
                           ('actions', [float('inf')] * 14), ('inference_ms', -1), ('inference_ms', True)]:
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                c.validate_reply(dict(reply, **{key: value}), request)
        changed = copy.deepcopy(request)
        changed['obs'][55] = .1
        with self.assertRaises(RuntimeError):
            c.validate_reply(reply, changed)

    def test_fragmentation_eof_and_packet_size(self):
        a, b = socket.socketpair()
        try:
            transport = c.JsonSocket(a)
            b.sendall(b'{"v":')
            def tail():
                time.sleep(.01)
                b.sendall(b'3}\n{"v":4}\n')
            t = threading.Thread(target=tail)
            t.start()
            self.assertEqual(transport.receive(1), {'v': 3})
            self.assertEqual(transport.receive(1), {'v': 4})
            t.join()
            b.close()
            with self.assertRaises(RuntimeError):
                transport.receive(.1)
        finally:
            a.close()
            b.close()
        a, b = socket.socketpair()
        try:
            b.sendall(b'x' * c.MAX_PACKET)
            with self.assertRaises(RuntimeError):
                c.JsonSocket(a).receive(.1)
        finally:
            a.close()
            b.close()

    def test_expired_response_does_not_advance_or_return_action(self):
        client = c.LoopClient.__new__(c.LoopClient)
        client.session, client.seq, client.chain = 'a' * 32, 0, c.EMPTY_CHAIN
        client.trace = io.StringIO()
        client.ipc_ms, client.ros_ms, client.inference_ms = [], [], []
        req = c.packet(client.session, 0, [0.] * 61)
        response = dict(reply=c.reply(req, [0.] * 14, .5), ros_rtt_ms=1.)
        client.io = SimpleNamespace(send=lambda x: None, receive=lambda timeout: response)
        client.sock = SimpleNamespace(settimeout=lambda timeout: None)
        with patch.object(c.time, 'monotonic', side_effect=[0., 0., 1.01]):
            with self.assertRaises(RuntimeError):
                client.step([0.] * 61)
        self.assertEqual(client.seq, 0)
        self.assertEqual(client.trace.getvalue(), '')

    def test_completion_requires_all_three_matching_transcripts(self):
        sim = dict(status='LOOP_SIM_FUNCTIONAL_PASSED', original_records_unchanged=True,
                   completed_control_steps=6000, cases=[dict(loop_passed=True)] * 2,
                   transport=dict(completed_requests=6000, transcript_sha256='a'))
        bridge = dict(status='LOOP_BRIDGE_COMPLETE', completed_requests=6000, transcript_sha256='a')
        server = dict(status='LOOP_SERVER_COMPLETE', completed_requests=6000, transcript_sha256='a',
                      model_unchanged=True, recurrent_reset_sequences=[0, 3000])
        loop_ops.verify_reports(sim, bridge, server)
        for field, value in [('transcript_sha256', 'b'), ('recurrent_reset_sequences', [0]),
                             ('completed_requests', 5999), ('model_unchanged', False)]:
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                loop_ops.verify_reports(sim, bridge, dict(server, **{field: value}))

    def test_watch_continues_after_completion_until_interrupted(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            report = dict(status='CLOSED_LOOP_FUNCTIONAL_PASSED', directory=d, phase='finished')
            (root / 'report.json').write_text(json.dumps(report))
            with patch.object(loop_ops, 'ROOT', root), patch.object(loop_ops.time, 'sleep', side_effect=KeyboardInterrupt), \
                    patch('builtins.print') as output:
                with self.assertRaises(KeyboardInterrupt):
                    loop_ops.show(watch=True)
                self.assertTrue(output.called)


class FakeBus:
    def __init__(self):
        self.guard = threading.Lock()
        self.publishers, self.subscribers = {}, {}

    def endpoints(self, args):
        bus = self
        class Message:
            def __init__(self, data):
                self.data = data
        class Node:
            def __init__(self):
                self.queue = queue.Queue()
                self.owned_p, self.owned_s = [], []
            def create_publisher(self, kind, topic, qos):
                with bus.guard:
                    bus.publishers.setdefault(topic, []).append(self)
                self.owned_p.append(topic)
                def publish(message):
                    with bus.guard:
                        targets = list(bus.subscribers.get(topic, []))
                    for node, callback in targets:
                        node.queue.put((callback, Message(message.data)))
                return SimpleNamespace(publish=publish)
            def create_subscription(self, kind, topic, callback, qos):
                with bus.guard:
                    bus.subscribers.setdefault(topic, []).append((self, callback))
                self.owned_s.append(topic)
                return object()
            def count_publishers(self, topic):
                with bus.guard:
                    return len(bus.publishers.get(topic, []))
            def count_subscribers(self, topic):
                with bus.guard:
                    return len(bus.subscribers.get(topic, []))
            def destroy_node(self):
                with bus.guard:
                    for topic in self.owned_p:
                        bus.publishers[topic].remove(self)
                    for topic in self.owned_s:
                        bus.subscribers[topic] = [(n, c) for n, c in bus.subscribers[topic] if n is not self]
        def spin_once(node, timeout_sec):
            try:
                callback, message = node.queue.get(timeout=timeout_sec)
            except queue.Empty:
                return
            callback(message)
        ros = SimpleNamespace(spin_once=spin_once, shutdown=lambda: None)
        return ros, Node(), {}, Message, '/fixture/' + args.session


class WorkerLifecycle(unittest.TestCase):
    def scenario(self, failing):
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            run = Path(d)
            export = run / 'export'
            export.mkdir()
            (export / 'contract.json').write_text('{}')
            checkpoint = dict(path=str(loop_sim.old.INPUTS / 'run/checkpoints/final6h_005304.pt'))
            calls, finished = [], []
            client = SimpleNamespace(seq=0, close=lambda: None, snapshot=lambda: {},
                                     finish=lambda: finished.append(True))
            def evaluate_one(context, folder, case, cp, identity, boundary, progress):
                boundary()
                calls.append(case)
                context.stats['compared_actions'] = 3000
                context.stats['max_action_absolute_error'] = 0.
                client.seq += 3000
                return dict(completed_control_steps=3000, acceptance=dict(passed=not failing))
            stack.enter_context(patch.object(loop_sim.old, 'preflight', return_value=dict(package_identity='test')))
            stack.enter_context(patch.object(loop_sim.old, 'verify_inputs', return_value={'unchanged': True}))
            stack.enter_context(patch.object(loop_sim, 'frozen', return_value={}))
            stack.enter_context(patch.object(loop_sim, 'validate_report', return_value=checkpoint))
            stack.enter_context(patch.object(loop_sim.evaluate, 'load_runtime', return_value=SimpleNamespace()))
            stack.enter_context(patch.object(loop_sim.evaluate, 'evaluate_one', side_effect=evaluate_one))
            stack.enter_context(patch.object(loop_sim, 'linked_context',
                side_effect=lambda base, client, index, stats, contract: SimpleNamespace(stats=stats)))
            stack.enter_context(patch.object(loop_sim, 'LoopClient', return_value=client))
            stack.enter_context(patch.object(loop_sim.signal, 'signal'))
            stack.enter_context(patch.object(loop_sim.signal, 'alarm'))
            args = SimpleNamespace(run=d, export=str(export), socket='/fixture', session='a' * 32)
            rc = loop_sim.run(args)
            report = json.loads((run / 'simulator-report.json').read_bytes())
            return rc, report, calls, finished

    def test_failed_first_case_is_preserved_without_second_case(self):
        rc, report, calls, finished = self.scenario(True)
        self.assertEqual(rc, 2)
        self.assertEqual(len(calls), 1)
        self.assertFalse(finished)
        self.assertEqual(report['status'], 'LOOP_SIM_FAILED_PRESERVE')
        self.assertTrue(report['original_records_unchanged'])

    def test_two_cases_finish_once_with_full_progress(self):
        rc, report, calls, finished = self.scenario(False)
        self.assertEqual(rc, 0)
        self.assertEqual(calls, list(c.CASES))
        self.assertEqual(finished, [True])
        self.assertEqual(report['completed_control_steps'], 6000)
        self.assertFalse(report['realtime_acceptance'])
        self.assertFalse(report['motor_io'])


class SyntheticLstmLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)
        torch.manual_seed(87)
        class Network(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('mean', torch.linspace(-.3, .3, 61))
                self.register_buffer('scale', torch.linspace(.3, 2., 61))
                self.lstm = torch.nn.LSTM(61, 256)
                self.mlp = torch.nn.Sequential(torch.nn.Linear(256, 32), torch.nn.ELU(), torch.nn.Linear(32, 14))
            def forward(self, obs, h, cell):
                out, (h, cell) = self.lstm(((obs - self.mean) / self.scale).unsqueeze(0), (h, cell))
                return self.mlp(out.squeeze(0)), h, cell
        cls.network = Network().eval()
        cls.folder = tempfile.TemporaryDirectory()
        cls.path = Path(cls.folder.name) / 'final6h_005304.onnx'
        torch.onnx.export(cls.network, (torch.zeros(1, 61), torch.zeros(1, 1, 256), torch.zeros(1, 1, 256)),
            cls.path, opset_version=18, dynamo=False, input_names=['obs', 'h_in', 'c_in'],
            output_names=['actions', 'h_out', 'c_out'], dynamic_axes={})

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def reference(self):
        torch, network = self.torch, self.network
        class Reference:
            def reset(self):
                self.h = torch.zeros(1, 1, 256)
                self.c = torch.zeros_like(self.h)
            def __call__(self, obs, stochastic_output=False):
                a, self.h, self.c = network(obs['actor'], self.h, self.c)
                return a
        return Reference()

    def observations(self, seq):
        torch = self.torch
        class Observations(dict):
            def to(self, device):
                return Observations({k: v.to(device) for k, v in self.items()})
        # Exercise the full vector, including the six head-ball slots.
        return Observations(actor=(torch.linspace(-.1, .1, 61) + .003 * (seq % 17)).reshape(1, 61))

    def test_6000_onnx_requests_socket_bridge_and_live_reference(self):
        torch = self.torch
        bus, errors = FakeBus(), []
        with tempfile.TemporaryDirectory(prefix='md12-test-', dir='/tmp') as d:
            folder = Path(d)
            address = folder / 's'
            server_args = SimpleNamespace(role='server', folder=self.folder.name, session='a' * 32)
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
                    patch.object(socket, 'socket', side_effect=[Listener(), Connector()]):
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
                            stats = dict(compared_actions=0, max_action_absolute_error=0.)
                            reference = self.reference()
                            proxy = loop_sim.RemotePolicy(reference, client, torch, case, stats)
                            proxy.reset()
                            for step in range(3000):
                                actions = proxy(self.observations(case * 3000 + step))
                                self.assertEqual(tuple(actions.shape), (1, 14))
                                self.assertTrue(bool(torch.isfinite(actions).all()))
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

    def test_parity_error_never_returns_local_fallback_action(self):
        client = SimpleNamespace(seq=0, step=lambda obs: [100.] * 14)
        stats = dict(compared_actions=0, max_action_absolute_error=0.)
        proxy = loop_sim.RemotePolicy(self.reference(), client, self.torch, 0, stats)
        proxy.reset()
        with self.torch.inference_mode(), self.assertRaises(RuntimeError):
            proxy(self.observations(0))
        self.assertEqual(stats['compared_actions'], 0)


if __name__ == '__main__':
    unittest.main()
