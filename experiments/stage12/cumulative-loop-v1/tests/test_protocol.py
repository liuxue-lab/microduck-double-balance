"""Unchanged bounded JSON protocol rejects stale, malformed and expired actions."""
import copy
import io
import socket
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from support import c

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

