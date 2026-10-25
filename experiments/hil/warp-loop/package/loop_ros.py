"""ROS Humble inference server and Python-3.10 bridge to the PC simulator."""
import argparse
import json
import os
from pathlib import Path
import select
import signal
import socket
import struct
import time
import traceback

from deploy_contract import OnnxPolicy, require, valid_session
from loop_contract import (SCHEMA, ONNX_SHA, TOTAL, SERVER_LIMIT, RPC_TIMEOUT,
    MAX_PACKET, EMPTY_CHAIN, JsonSocket, encode, frozen, validate_request,
    validate_reply, reply, chain, measured)


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def endpoints(args):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
    from std_msgs.msg import String
    valid_session(args.session)
    rclpy.init()
    node = Node('stage12_loop_' + args.role + '_' + args.session)
    qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                     reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
    root = '/microduck/stage12/plant_' + args.session
    return rclpy, node, qos, String, root


def server(args, result):
    frozen(args.folder)
    policy = OnnxPolicy(Path(args.folder) / 'final6h_005304.onnx')
    # Warm only the runtime. The first live observation still resets h/c.
    policy.step([0.] * 61, reset=True)
    policy.reset()
    ros, node, qos, String, root = endpoints(args)
    publisher = node.create_publisher(String, root + '/actions', qos)
    state = dict(seq=0, chain=EMPTY_CHAIN, resets=[], error=None, finished=None)
    timings = []
    def receive(message):
        try:
            require(state['error'] is None, 'Server already failed')
            require(len(message.data.encode()) <= MAX_PACKET, 'Oversized ROS request')
            p = validate_request(json.loads(message.data), args.session, state['seq'])
            begun = time.monotonic()
            actions = policy.step(p['obs'], reset=p['reset'])
            elapsed = (time.monotonic() - begun) * 1000
            response = validate_reply(reply(p, actions, elapsed), p)
            publisher.publish(String(data=encode(response).decode()))
            state['chain'] = chain(state['chain'], p, response)
            if p['reset']:
                state['resets'].append(p['seq'])
            timings.append(elapsed)
            state['seq'] += 1
            if state['seq'] == TOTAL:
                state['finished'] = time.monotonic()
        except Exception as exc:
            state['error'] = repr(exc)
            publisher.publish(String(data=encode(dict(schema=SCHEMA, session=args.session,
                model=ONNX_SHA, seq=state['seq'], error=state['error'])).decode()))
    subscription = node.create_subscription(String, root + '/observations', receive, qos)
    started = time.monotonic()
    try:
        print('LOOP_ORIN_READY', flush=True)
        while time.monotonic() - started < SERVER_LIMIT:
            require(not (Path(args.folder) / 'cancel').exists(), 'Controller requested stop')
            ros.spin_once(node, timeout_sec=.05)
            require(state['error'] is None, str(state['error']))
            if state['finished'] is not None and time.monotonic() - state['finished'] > 1:
                break
        require(state['seq'] == TOTAL, 'Orin server reached its bounded deadline')
        require(state['resets'] == [0, 3000], 'Recurrent reset schedule differs')
        frozen(args.folder)
        result.update(status='LOOP_SERVER_COMPLETE', model_unchanged=True)
    finally:
        result.update(completed_requests=state['seq'], transcript_sha256=state['chain'],
                      recurrent_reset_sequences=state['resets'], inference_ms=measured(timings),
                      warmup_inferences=1, warmup_actions_applied=False)
        node.destroy_node()
        ros.shutdown()


def bridge(args, result):
    ros, node, qos, String, root = endpoints(args)
    publisher = node.create_publisher(String, root + '/observations', qos)
    state = dict(pending=None, received=None, error=None)
    listener = peer = None
    count, transcript, timings = 0, EMPTY_CHAIN, []
    address = Path(args.socket)
    def receive(message):
        try:
            require(state['error'] is None, 'Bridge already failed')
            require(state['pending'] is not None and state['received'] is None,
                    'Unsolicited or repeated ROS reply')
            require(len(message.data.encode()) <= MAX_PACKET, 'Oversized ROS reply')
            state['received'] = validate_reply(json.loads(message.data), state['pending'])
        except Exception as exc:
            state['error'] = repr(exc)
    subscription = node.create_subscription(String, root + '/actions', receive, qos)
    started = time.monotonic()
    def check():
        require(not (Path(args.folder) / 'STOP.json').exists(), 'Controller requested stop')
        require(time.monotonic() - started < SERVER_LIMIT, 'Bridge bounded deadline exceeded')
        require(state['error'] is None, str(state['error']))
    try:
        while node.count_subscribers(root + '/observations') != 1 or node.count_publishers(root + '/actions') != 1:
            check()
            require(time.monotonic() - started < 40, 'ROS wired discovery timed out')
            ros.spin_once(node, timeout_sec=.02)
        require(not address.exists() and not address.is_symlink(), 'IPC path already exists')
        require(address.parent.stat().st_uid == os.getuid() and address.parent.stat().st_mode & 0o077 == 0,
                'IPC parent must be private to the current user')
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(address))
        listener.listen(1)
        save(Path(args.folder) / 'bridge-ready.json', dict(status='READY', session=args.session, socket=str(address)))
        print('LOOP_BRIDGE_READY', flush=True)
        while not select.select([listener], [], [], .05)[0]:
            check()
            ros.spin_once(node, timeout_sec=0.)
        peer, _ = listener.accept()
        pid, uid, gid = struct.unpack('3i', peer.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        require(uid == os.getuid(), 'Unexpected IPC peer user')
        result['simulator_peer_pid'] = pid
        io = JsonSocket(peer)
        while True:
            check()
            while not io.buffer and not select.select([peer], [], [], .02)[0]:
                check()
                ros.spin_once(node, timeout_sec=0.)
            incoming = io.receive(5)
            if isinstance(incoming, dict) and incoming.get('command') == 'finish':
                require(incoming == dict(command='finish', session=args.session, count=TOTAL, chain=transcript)
                        and count == TOTAL, 'Premature/mismatched loop completion')
                ros.spin_once(node, timeout_sec=0.)
                check()
                io.send(dict(status='BRIDGE_COMPLETE', count=count, chain=transcript))
                result['status'] = 'LOOP_BRIDGE_COMPLETE'
                break
            p = validate_request(incoming, args.session, count)
            ros.spin_once(node, timeout_sec=0.)
            check()
            state.update(pending=p, received=None)
            begun = time.monotonic()
            publisher.publish(String(data=encode(p).decode()))
            while state['received'] is None:
                check()
                require(time.monotonic() - begun < RPC_TIMEOUT, 'ROS response timeout; no retry or cached action')
                ros.spin_once(node, timeout_sec=.001)
            elapsed = (time.monotonic() - begun) * 1000
            check()
            require(elapsed < RPC_TIMEOUT * 1000, 'Expired ROS action rejected')
            response = state['received']
            io.send(dict(reply=response, ros_rtt_ms=elapsed))
            transcript = chain(transcript, p, response)
            count += 1
            timings.append(elapsed)
            state.update(pending=None, received=None)
    finally:
        result.update(completed_requests=count, transcript_sha256=transcript, ros_roundtrip_ms=measured(timings))
        if peer is not None:
            peer.close()
        if listener is not None:
            listener.close()
            address.unlink(missing_ok=True)
        node.destroy_node()
        ros.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('role', choices=('server', 'bridge'))
    parser.add_argument('--folder', required=True)
    parser.add_argument('--session', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--socket')
    args = parser.parse_args()
    result = dict(status='LOOP_ROS_FAILED_PRESERVE', role=args.role, session=args.session,
                  onnx_sha256=ONNX_SHA, motor_io=False, added_PPO_updates=0,
                  hardware_acceptance=False, realtime_acceptance=False, expected_requests=TOTAL)
    def stop(signum, frame):
        raise RuntimeError('ROS process stop signal: ' + str(signum))
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    begun = time.monotonic()
    try:
        (server if args.role == 'server' else bridge)(args, result)
    except BaseException as exc:
        result.update(status='LOOP_ROS_FAILED_PRESERVE', error=repr(exc), traceback=traceback.format_exc()[-10000:])
    finally:
        result['elapsed_seconds'] = time.monotonic() - begun
        save(args.report, result)
    print('LOOP_ROS_STATUS=' + result['status'], flush=True)
    return 0 if result['status'] in ('LOOP_SERVER_COMPLETE', 'LOOP_BRIDGE_COMPLETE') else 2


if __name__ == '__main__':
    raise SystemExit(main())
