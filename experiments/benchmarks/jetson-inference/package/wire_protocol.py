"""Versioned little-endian benchmark frames on an existing Humble byte carrier.

A future public robot API should use a dedicated semantic ROS message.
"""
from array import array
import struct
from deploy_contract import require
from loop_contract import ONNX_SHA, SCHEMA, encode, packet, validate_request

HEADER = struct.Struct('<4sBBH16s32sIB3x')
OBS = struct.Struct('<61f')
ACTION = struct.Struct('<32sd14f')


class Wire:
    def __init__(self, kind):
        require(kind in ('json', 'binary'), 'Unknown wire format')
        self.kind = kind
        if kind == 'json':
            from std_msgs.msg import String
            self.Message = String
        else:
            from std_msgs.msg import UInt8MultiArray
            self.Message = UInt8MultiArray

    def request(self, value):
        if self.kind == 'json':
            return self.Message(data=encode(value).decode())
        data = HEADER.pack(b'MD12', 1, 1, 0, bytes.fromhex(value['session']), bytes.fromhex(value['model']),
                           value['seq'], int(value['reset'])) + OBS.pack(*value['obs'])
        return self.Message(data=array('B', data))

    def response(self, value):
        if self.kind == 'json':
            return self.Message(data=encode(value).decode())
        data = HEADER.pack(b'MD12', 1, 2, 0, bytes.fromhex(value['session']), bytes.fromhex(value['model']),
                           value['seq'], int(value['seq'] in (0, 3000)))
        data += ACTION.pack(bytes.fromhex(value['request_sha256']), value['inference_ms'], *value['actions'])
        return self.Message(data=array('B', data))

    def header(self, message, kind, size):
        require(not message.layout.dim and message.layout.data_offset == 0, 'Unexpected binary layout')
        require(len(message.data) == size, 'Binary frame length differs')
        raw = bytes(message.data)
        require(len(raw) == size, 'Binary frame length differs')
        magic, version, actual_kind, reserved, session, model, seq, reset = HEADER.unpack_from(raw)
        require(magic == b'MD12' and version == 1 and actual_kind == kind and reserved == 0
                and reset in (0, 1) and raw[61:64] == b'\0\0\0', 'Invalid binary header')
        return raw, session.hex(), model.hex(), seq, bool(reset)

    def decode_request(self, message, session, seq):
        import json
        if self.kind == 'json':
            require(len(message.data.encode()) <= 24000, 'Oversized JSON request')
            value = json.loads(message.data)
        else:
            raw, sid, model, number, reset = self.header(message, 1, HEADER.size + OBS.size)
            value = packet(sid, number, list(OBS.unpack_from(raw, HEADER.size)))
            require(model == ONNX_SHA and value['reset'] == reset, 'Binary model/reset differs')
        return validate_request(value, session, seq)

    def decode_response(self, message):
        if self.kind == 'json':
            return message.data
        raw, session, model, seq, reset = self.header(message, 2, HEADER.size + ACTION.size)
        require(reset == (seq in (0, 3000)), 'Binary response reset differs')
        request_sha, duration, *actions = ACTION.unpack_from(raw, HEADER.size)
        # Retain the original semantic gate and full observation digest checks.
        # This removes JSON on the wire, not every JSON operation in Python.
        return encode(dict(schema=SCHEMA, session=session, model=model, seq=seq,
            request_sha256=request_sha.hex(), inference_ms=duration, actions=actions)).decode()
