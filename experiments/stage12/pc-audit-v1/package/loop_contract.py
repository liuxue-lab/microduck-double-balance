"""Bounded simulation-only wire protocol. No fallback actions or motor access."""
import hashlib
import json
import math
from pathlib import Path
import socket
import time

from deploy_contract import require, vector, valid_session, verify_export, sha

SCHEMA = 'stage12-humble-plant-loop-v1'
ONNX_SHA = '8c1c1250b9b88d9a91e3eb095fea8bce21daa73c251b43f3cd8adc1327f03c4e'
EXPORT_PINS = {
    'final6h_005304.onnx': ONNX_SHA,
    'contract.json': 'e0283bf7f00c8e2c7b57128b91ca3ced3abb4698e8f4c4aa11e74a745ad84929',
    'golden.json': 'b7ebe914c3933477d35bc0cc70c6b1bb4c1c83e67bd61dc22cd8f0775b1609e7',
}
# Regression cases already accepted for this exact frozen checkpoint. This is
# an integration test, not a new random-sequence generalization claim.
CASES = (
    dict(case_id='quiet-2407', seed=2026102407, force_seed=None),
    dict(case_id='regression-2500', seed=2026102407, force_seed=2026102500),
)
STEPS = 3000
TOTAL = STEPS * len(CASES)
RPC_TIMEOUT = 1.0
WORKER_LIMIT = 2000
SERVER_LIMIT = 2200
MAX_PACKET = 24000
EMPTY_CHAIN = '0' * 64


def encode(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def chain(previous, request, reply):
    return hashlib.sha256(bytes.fromhex(previous) + encode(request) + b'\n' + encode(reply)).hexdigest()


def frozen(folder):
    result = verify_export(folder)
    for name, expected in EXPORT_PINS.items():
        require(sha(Path(folder) / name) == expected, 'Reviewed frozen export differs: ' + name)
    require(result['comparisons'] == 128 and result['recurrent_resets'] == 2,
            'Expected the reviewed recurrent export parity evidence')
    return result


def packet(session, seq, obs):
    require(type(seq) is int and 0 <= seq < TOTAL, 'Control sample limit exceeded')
    case, step = divmod(seq, STEPS)
    return dict(schema=SCHEMA, session=valid_session(session), model=ONNX_SHA,
                seq=seq, case=case, step=step, reset=step == 0, obs=vector(obs, 61))


def validate_request(value, session, seq):
    require(isinstance(value, dict), 'Request must be an object')
    require(set(value) == {'schema', 'session', 'model', 'seq', 'case', 'step', 'reset', 'obs'},
            'Unexpected request fields')
    for key in ('seq', 'case', 'step'):
        require(type(value[key]) is int, 'Sequence fields must be integers')
    require(type(value['reset']) is bool, 'Reset flag must be boolean')
    require(value == packet(session, seq, value['obs']), 'Stale/reordered request or invalid reset')
    return value


def reply(request, actions, inference_ms):
    return dict(schema=SCHEMA, session=request['session'], model=ONNX_SHA,
                seq=request['seq'], request_sha256=digest(request),
                actions=vector(actions, 14), inference_ms=inference_ms)


def validate_reply(value, request):
    require(isinstance(value, dict), 'Reply must be an object')
    require('error' not in value, 'Orin: ' + str(value.get('error')))
    require(set(value) == {'schema', 'session', 'model', 'seq', 'request_sha256', 'actions', 'inference_ms'},
            'Unexpected reply fields')
    require(value['schema'] == SCHEMA and value['session'] == request['session']
            and value['model'] == ONNX_SHA and type(value['seq']) is int
            and value['seq'] == request['seq'] and value['request_sha256'] == digest(request),
            'Reply belongs to another observation/session/model')
    vector(value['actions'], 14)
    duration = value['inference_ms']
    require(type(duration) in (int, float) and math.isfinite(duration) and duration >= 0,
            'Invalid inference timing')
    return value


def measured(values):
    if not values:
        return dict(count=0, mean=None, p95=None, p99=None, max=None, over_20ms=0)
    ordered = sorted(values)
    return dict(count=len(values), mean=sum(values) / len(values),
                p95=ordered[math.ceil(.95 * len(values)) - 1],
                p99=ordered[math.ceil(.99 * len(values)) - 1], max=ordered[-1],
                over_20ms=sum(v > 20. for v in values))


class JsonSocket:
    """One bounded JSON line at a time; fragmented reads and EOF are explicit."""
    def __init__(self, sock):
        self.sock = sock
        self.buffer = bytearray()

    def send(self, value):
        data = encode(value) + b'\n'
        require(len(data) <= MAX_PACKET, 'IPC packet exceeds size limit')
        self.sock.sendall(data)

    def receive(self, timeout):
        deadline = time.monotonic() + timeout
        while b'\n' not in self.buffer:
            remaining = deadline - time.monotonic()
            require(remaining > 0, 'IPC response deadline exceeded')
            self.sock.settimeout(remaining)
            block = self.sock.recv(min(4096, MAX_PACKET - len(self.buffer)))
            require(bool(block), 'IPC closed before a complete reply')
            self.buffer.extend(block)
            require(len(self.buffer) < MAX_PACKET or b'\n' in self.buffer, 'Oversized IPC packet')
        line, _, rest = self.buffer.partition(b'\n')
        self.buffer = bytearray(rest)
        require(bool(line), 'Empty IPC packet')
        return json.loads(line)


class LoopClient:
    def __init__(self, address, session, trace):
        self.session, self.trace = valid_session(session), trace
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(5)
        try:
            self.sock.connect(str(address))
        except BaseException:
            self.sock.close()
            raise
        self.io = JsonSocket(self.sock)
        self.seq, self.chain = 0, EMPTY_CHAIN
        self.ipc_ms, self.ros_ms, self.inference_ms = [], [], []

    def step(self, obs):
        req = packet(self.session, self.seq, obs)
        begun = time.monotonic()
        self.sock.settimeout(RPC_TIMEOUT)
        self.io.send(req)
        remaining = RPC_TIMEOUT - (time.monotonic() - begun)
        require(remaining > 0, 'IPC send exhausted response deadline')
        envelope = self.io.receive(remaining)
        elapsed = (time.monotonic() - begun) * 1000
        require(elapsed < RPC_TIMEOUT * 1000, 'Late action rejected before simulator advance')
        require(isinstance(envelope, dict) and set(envelope) == {'reply', 'ros_rtt_ms'}, 'Invalid bridge response')
        response = validate_reply(envelope['reply'], req)
        rtt = envelope['ros_rtt_ms']
        require(type(rtt) in (int, float) and math.isfinite(rtt) and 0 <= rtt < RPC_TIMEOUT * 1000,
                'Invalid or expired ROS round trip')
        require(rtt <= elapsed + 1., 'Bridge timing exceeds measured end-to-end time')
        self.chain = chain(self.chain, req, response)
        self.trace.write(json.dumps(dict(request=req, response=response, ipc_roundtrip_ms=elapsed,
                                        ros_roundtrip_ms=rtt), allow_nan=False) + '\n')
        self.seq += 1
        if self.seq % 25 == 0:
            self.trace.flush()
        self.ipc_ms.append(elapsed)
        self.ros_ms.append(rtt)
        self.inference_ms.append(response['inference_ms'])
        return response['actions']

    def finish(self):
        require(self.seq == TOTAL, 'Cannot complete a partial two-case run')
        self.io.send(dict(command='finish', session=self.session, count=self.seq, chain=self.chain))
        ack = self.io.receive(5)
        require(ack == dict(status='BRIDGE_COMPLETE', count=self.seq, chain=self.chain),
                'Bridge completion disagrees with simulator transcript')

    def close(self):
        self.sock.close()

    def snapshot(self):
        return dict(completed_requests=self.seq, transcript_sha256=self.chain,
                    ipc_roundtrip_ms=measured(self.ipc_ms), ros_roundtrip_ms=measured(self.ros_ms),
                    orin_inference_ms=measured(self.inference_ms), nominal_period_ms=20.,
                    hard_response_timeout_ms=RPC_TIMEOUT * 1000,
                    mode='lockstep simulation, no wall-clock actuator scheduling',
                    realtime_acceptance=False, hardware_acceptance=False)
