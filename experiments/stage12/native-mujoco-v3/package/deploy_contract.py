"""Frozen policy identity and inference-only deployment contracts."""
import hashlib
import json
import math
from pathlib import Path
import re

MODEL_SHA = 'cf7ce9297672943687019afe071d94918d3eb2620c18a1e8fe689512f8c08111'
REPORT_SHA = 'd052f3cf8031be5ec5685fc1aa02c906301932d2a4849ac34fd450206b359228'
MODEL_BYTES = 10041821
DOMAIN = 62
PC_IP = '192.168.50.1'
ORIN_IP = '192.168.50.2'
ATOL, RTOL = 2e-5, 2e-4
SCHEMA = 'microduck-stage12-humble-inference-v1'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_report(path):
    require(sha(path) == REPORT_SHA, 'Local validation report differs from the reviewed report')
    r = json.loads(Path(path).read_bytes())
    require(r['status'] == 'VALIDATION_PASSED' and r['disturbance_goal_passed'] is True
            and r['baseline_goal_passed'] is True and r['added_PPO_updates'] == 0
            and r['compute_workers_exited'] is True and r['original_training_records_unchanged'] is True,
            'Frozen simulation validation did not pass cleanly')
    cp = r['chosen_checkpoint']
    require(cp['sha256'] == MODEL_SHA and cp['bytes'] == MODEL_BYTES
            and cp['completed_updates'] == 5304 and cp['policy_updates'] == 11314,
            'Do not deploy a different candidate')
    return cp


def verify_export(folder):
    folder = Path(folder)
    require(not folder.is_symlink(), 'Unexpected export symlink')
    r = json.loads((folder / 'export-report.json').read_bytes())
    require(r['status'] == 'EXPORT_PARITY_PASSED' and r['checkpoint_sha256'] == MODEL_SHA
            and r['validation_report_sha256'] == REPORT_SHA
            and r['policy_and_normalizer_unchanged'] is True, 'Export is not accepted')
    for name, pin in r['files'].items():
        require(Path(name).name == name, 'Invalid export member')
        p = folder / name
        require(p.is_file() and not p.is_symlink() and p.stat().st_size == pin['bytes']
                and sha(p) == pin['sha256'], 'Export changed: ' + name)
    return r


def vector(value, count):
    require(isinstance(value, list) and len(value) == count, 'Wrong vector size')
    require(all(type(x) in (int, float) and math.isfinite(x) for x in value), 'Invalid vector value')
    return value


def request(payload, session, model, expected_seq):
    require(isinstance(payload, str) and len(payload) <= 16000, 'Invalid request size')
    p = json.loads(payload)
    require(p['schema'] == SCHEMA and p['session'] == session and p['model'] == model,
            'Session/model mismatch')
    require(type(p['seq']) is int and p['seq'] == expected_seq, 'Repeated, missing or reordered request')
    require(type(p['reset']) is bool, 'Reset must be boolean')
    if expected_seq == 0:
        require(p['reset'], 'First observation must reset recurrent state')
    vector(p['obs'], 61)
    return p


def valid_session(value):
    require(re.fullmatch(r'[0-9a-f]{32}', value) is not None, 'Invalid session ID')
    return value


def comparison(actual, expected, atol=ATOL, rtol=RTOL):
    vector(actual, len(expected)); vector(expected, len(actual))
    errors = [abs(a - b) for a, b in zip(actual, expected)]
    require(all(e <= atol + rtol * abs(b) for e, b in zip(errors, expected)), 'Inference parity exceeded tolerance')
    return max(errors, default=0.)


class OnnxPolicy:
    """Original normalized ONNX; state is carried, never normalized twice."""
    def __init__(self, path):
        import numpy as np
        import onnxruntime as ort
        self.np = np
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        self.inputs = {x.name: x for x in self.session.get_inputs()}
        self.outputs = {x.name: x for x in self.session.get_outputs()}
        require(set(self.inputs) == {'obs', 'h_in', 'c_in'}
                and set(self.outputs) == {'actions', 'h_out', 'c_out'}, 'Expected the frozen LSTM policy')
        expected = {'obs': [1, 61], 'h_in': [1, 1, 256], 'c_in': [1, 1, 256],
                    'actions': [1, 14], 'h_out': [1, 1, 256], 'c_out': [1, 1, 256]}
        for name, item in {**self.inputs, **self.outputs}.items():
            require(item.shape == expected[name] and item.type == 'tensor(float)', 'ONNX schema differs: ' + name)
        self.reset()

    def reset(self):
        self.h = self.np.zeros((1, 1, 256), dtype=self.np.float32)
        self.c = self.np.zeros_like(self.h)

    def step(self, obs, reset=False):
        vector(obs, 61)
        if reset:
            self.reset()
        values = self.session.run(['actions', 'h_out', 'c_out'],
                                  {'obs': self.np.asarray([obs], dtype=self.np.float32),
                                   'h_in': self.h, 'c_in': self.c})
        require(all(self.np.isfinite(x).all() for x in values), 'Nonfinite ONNX result')
        actions, self.h, self.c = values
        return actions[0].tolist()


def wired_xml(address):
    require(address in (PC_IP, ORIN_IP), 'Unapproved wired interface')
    return '''<?xml version="1.0" encoding="UTF-8" ?>
<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
  <transport_descriptors><transport_descriptor>
    <transport_id>microduck_wired</transport_id><type>UDPv4</type>
    <interfaceWhiteList><address>%s</address></interfaceWhiteList>
  </transport_descriptor></transport_descriptors>
  <participant profile_name="microduck_wired" is_default_profile="true"><rtps>
    <userTransports><transport_id>microduck_wired</transport_id></userTransports>
    <useBuiltinTransports>false</useBuiltinTransports>
  </rtps></participant>
</profiles>
''' % address
