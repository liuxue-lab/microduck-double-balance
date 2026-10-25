"""CPU fixtures; not real DDS, Jetson, final-model, or hardware acceptance."""
from contextlib import ExitStack
import copy
import io
import json
from pathlib import Path
import queue
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'package'))
import loop_contract as c
import timing_contract as tc
import timing_source
import timing_nodes as nodes
import timing_ops as ops
import wire_protocol as wp
import prepare_candidates as prep
import bench_policies as bench
from deploy_contract import OnnxPolicy,comparison,sha
from fast_policy import BufferedPolicy


class Message:
    def __init__(self,data):
        self.data=data
        self.layout=SimpleNamespace(dim=[],data_offset=0)


def wire(kind):
    result=wp.Wire.__new__(wp.Wire)
    result.kind=kind;result.Message=Message
    return result


class GateAndWire(unittest.TestCase):
    def test_6000_frames_preserve_semantics_chain_and_recording(self):
        now=[0.];session='a'*32;trace=io.StringIO()
        gate=tc.DeadlineGate(session,clock=lambda:now[0],trace=trace)
        w=wire('binary');current=c.EMPTY_CHAIN
        for seq in range(6000):
            # All values, including head-ball slots, are exactly representable.
            request=c.packet(session,seq,[float(i-30)/32+float(seq%16)/16 for i in range(61)])
            frame=w.request(request)
            self.assertEqual(len(frame.data),308)
            self.assertEqual(c.encode(request),c.encode(w.decode_request(frame,session,seq)))
            response=c.reply(request,[float(i-7)/16 for i in range(14)],.75)
            self.assertEqual(len(w.response(response).data),160)
            decoded=w.decode_response(w.response(response))
            self.assertEqual(json.loads(decoded),response)
            now[0]=seq*.02
            gate.begin(request,response['actions'],now[0]);now[0]+=.003
            gate.receive(decoded)
            current=c.chain(current,request,response)
        self.assertEqual(gate.seq,6000);self.assertIsNone(gate.fault)
        self.assertEqual(current,gate.chain)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'trace.jsonl';path.write_text(trace.getvalue())
            values=timing_source.load_trace(path,session,current)
            self.assertEqual(len(values),6000)
            corrupted=trace.getvalue().replace('"inference_ms": 0.75','"inference_ms": 0.76',1)
            self.assertNotEqual(corrupted,trace.getvalue());path.write_text(corrupted)
            with self.assertRaises(RuntimeError):timing_source.load_trace(path,session,current)
            path.write_text(trace.getvalue()+'{}\n')
            with self.assertRaises(RuntimeError):timing_source.load_trace(path,session,current)
            path.write_text(trace.getvalue()[:-10])
            with self.assertRaises(RuntimeError):timing_source.load_trace(path,session,current)

    def gate(self):
        now=[1.];request=c.packet('a'*32,0,[0.]*61)
        gate=tc.DeadlineGate('a'*32,clock=lambda:now[0])
        gate.begin(request,[0.]*14,1.)
        return now,request,gate

    def test_delay_and_silence_latch_without_advancing(self):
        for use_poll in (True,False):
            now,request,gate=self.gate();now[0]=1.021
            if use_poll:gate.poll()
            gate.receive(c.encode(c.reply(request,[0.]*14,.5)).decode())
            self.assertEqual(gate.seq,0);self.assertEqual(gate.discarded,1)
            self.assertIn(gate.fault['reason'],('RESPONSE_DEADLINE','LATE_REPLY'))
            with self.assertRaisesRegex(RuntimeError,'FAULT_LATCHED'):
                gate.begin(request,[0.]*14,now[0])
            self.assertEqual(gate.snapshot()['actions_applied_to_plant'],0)

    def test_duplicate_identity_error_and_no_cached_action(self):
        now,request,gate=self.gate();payload=c.encode(c.reply(request,[0.]*14,.5)).decode()
        now[0]=1.003;gate.receive(payload)
        now[0]=1.02;gate.begin(c.packet('a'*32,1,[0.]*61),[0.]*14,1.02)
        gate.receive(payload)
        self.assertEqual(gate.seq,1);self.assertEqual(gate.fault['reason'],'INVALID_REPLY')
        self.assertFalse(gate.snapshot()['cached_action_fallback'])

    def test_validation_time_counts_toward_deadline(self):
        now,request,gate=self.gate()
        def expensive(*_):now[0]=1.021;return 0.
        with patch.object(tc,'comparison',side_effect=expensive):
            gate.receive(c.encode(c.reply(request,[0.]*14,.5)).decode())
        self.assertEqual(gate.seq,0);self.assertEqual(gate.fault['reason'],'LATE_REPLY')

    def test_missed_dispatch_never_catches_up(self):
        gate=tc.DeadlineGate('a'*32,clock=lambda:1.021)
        with self.assertRaises(RuntimeError):gate.begin(c.packet('a'*32,0,[0.]*61),[0.]*14,1.)
        self.assertEqual(gate.sent,0);self.assertEqual(gate.fault['reason'],'DISPATCH_DEADLINE')

    def test_wire_rejects_malformed_header_nonfinite_and_layout(self):
        w=wire('binary');p=c.packet('a'*32,0,[0.]*61);good=w.request(p)
        for offset in (0,4,5,6,8,24,56,60,61,62,63):
            value=bytearray(good.data);value[offset]^=1
            with self.subTest(offset=offset),self.assertRaises(RuntimeError):
                w.decode_request(Message(value),'a'*32,0)
        for data in (bytes(good.data)[:-1],bytes(good.data)+b'\0'):
            with self.assertRaises(RuntimeError):w.decode_request(Message(data),'a'*32,0)
        bad=w.request(p);bad.layout.dim=[1]
        with self.assertRaises(RuntimeError):w.decode_request(bad,'a'*32,0)
        bad=Message(bytes(good.data[:wp.HEADER.size])+wp.OBS.pack(*([float('nan')]*61)))
        with self.assertRaises(RuntimeError):w.decode_request(bad,'a'*32,0)
        frame=w.response(c.reply(p,[0.]*14,.5));frame.data[60]=0
        with self.assertRaises(RuntimeError):w.decode_response(frame)

    def test_json_and_binary_use_identical_semantic_validation(self):
        for kind in ('json','binary'):
            w=wire(kind);p=c.packet('a'*32,3000,[-0.]*61)
            self.assertEqual(c.encode(p),c.encode(w.decode_request(w.request(p),'a'*32,3000)))
            response=c.reply(p,[0.]*14,.5)
            self.assertEqual(c.validate_reply(json.loads(w.decode_response(w.response(response))),p),response)

    def test_reviewed_uploaded_report_is_valid(self):
        result=timing_source.validate_prior(ROOT/'reference/closed-loop-functional-passed.json')
        self.assertEqual(result['status'],'CLOSED_LOOP_FUNCTIONAL_PASSED')


class FakeBus:
    def __init__(self):
        self.lock=threading.Lock();self.publishers={};self.subscribers={}
    def endpoints(self,args):
        bus=self
        class Node:
            def __init__(self):self.queue=queue.Queue();self.p=[];self.s=[]
            def create_publisher(self,kind,topic,qos):
                with bus.lock:bus.publishers.setdefault(topic,[]).append(self)
                self.p.append(topic)
                def publish(message):
                    with bus.lock:targets=list(bus.subscribers.get(topic,[]))
                    for node,callback in targets:node.queue.put((callback,copy.deepcopy(message)))
                return SimpleNamespace(publish=publish)
            def create_subscription(self,kind,topic,callback,qos):
                with bus.lock:bus.subscribers.setdefault(topic,[]).append((self,callback))
                self.s.append(topic);return object()
            def count_publishers(self,topic):
                with bus.lock:return len(bus.publishers.get(topic,[]))
            def count_subscribers(self,topic):
                with bus.lock:return len(bus.subscribers.get(topic,[]))
            def destroy_node(self):
                with bus.lock:
                    for topic in self.p:bus.publishers[topic].remove(self)
                    for topic in self.s:bus.subscribers[topic]=[(n,c) for n,c in bus.subscribers[topic] if n is not self]
        def spin_once(node,timeout_sec):
            try:callback,message=node.queue.get(timeout=timeout_sec)
            except queue.Empty:return
            callback(message)
        return SimpleNamespace(spin_once=spin_once,shutdown=lambda:None),Node(),{},Message,'/fixture/'+args.session


class NodeFlow(unittest.TestCase):
    def exercise(self,kind,transport):
        bus=FakeBus();errors=[];server_result={};client_result={}
        policy=SimpleNamespace(step=lambda obs,reset=False:[0.]*14,reset=lambda:None)
        with tempfile.TemporaryDirectory() as d,ExitStack() as stack:
            folder=Path(d);args=SimpleNamespace(folder=d,session='b'*32,scenario=kind,
                wire=transport,runtime='fp32',cancel_file=str(folder/'stop'),role='server')
            values=[dict(obs=[0.]*61,actions=[0.]*14) for _ in range(4)]
            stack.enter_context(patch.object(nodes,'TOTAL',4))
            stack.enter_context(patch.object(nodes,'OnnxPolicy',return_value=policy))
            stack.enter_context(patch.object(nodes,'frozen'))
            stack.enter_context(patch.object(nodes,'Wire',side_effect=wire))
            stack.enter_context(patch.object(nodes.loop_ros,'endpoints',side_effect=bus.endpoints))
            stack.enter_context(patch.object(timing_source,'load_recording',return_value=(values,{})))
            def server():
                try:nodes.server(args,server_result)
                except BaseException as exc:errors.append(repr(exc))
            thread=threading.Thread(target=server,daemon=True);thread.start()
            try:
                nodes.client(SimpleNamespace(**dict(vars(args),role='client')),client_result)
            finally:
                if client_result.get('status') not in ('SCHEDULE_REPLAY_PASSED','FAULT_GATE_PASSED'):
                    (folder/'stop').touch()
                thread.join(4)
            self.assertFalse(thread.is_alive());self.assertFalse(errors,errors)
            self.assertFalse(client_result['receive_gate']['motor_io'])
            self.assertEqual(client_result['status'],'SCHEDULE_REPLAY_PASSED' if kind=='replay' else 'FAULT_GATE_PASSED')
            if kind=='replay':
                self.assertEqual(client_result['receive_gate']['transcript_sha256'],server_result['transcript_sha256'])
            else:
                self.assertEqual(client_result['sent_messages'],2)
                self.assertTrue(all(client_result['fault_conditions'].values()))
    def test_real_clock_json_replay_in_memory(self):self.exercise('replay','json')
    def test_real_clock_binary_replay_in_memory(self):self.exercise('replay','binary')
    def test_delayed_reply_is_discarded(self):self.exercise('delay','binary')
    def test_missing_reply_latches(self):self.exercise('silence','binary')
    def test_repeated_reply_latches(self):self.exercise('duplicate','binary')


class SelectionAndStatus(unittest.TestCase):
    def row(self,name,mean,p95,passed=True):
        return dict(name=name,wire='json' if name=='json-fp32' else 'binary',runtime='fp32',passed=passed,
                    client=dict(receive_gate=dict(scheduled_to_validated_ms=dict(mean=mean,p95=p95))))
    def test_selection_needs_measured_gain_and_matching_fault_checks(self):
        rows=[self.row('json-fp32',3.,4.),self.row('binary-fp32',2.,3.)]
        faults=[dict(passed=True,wire='binary',runtime='fp32')]*3
        result=ops.recommend(rows,faults)
        self.assertEqual(result['variant'],'binary-fp32')
        self.assertEqual(result['status'],'RECOMMENDED_FOR_NEXT_FULL_LOOP_TEST')
        self.assertFalse(result['production_activated']);self.assertFalse(result['int8_activated'])
        self.assertFalse(result['complete_control_loop_realtime_acceptance'])
        self.assertEqual(ops.recommend(rows,faults[:2])['status'],'FAULT_CHECKS_PENDING_OR_FAILED')
        rows[1]['passed']=False
        self.assertEqual(ops.recommend(rows,[])['variant'],'json-fp32')
        rows[0]['passed']=False
        self.assertEqual(ops.recommend(rows,[])['status'],'NO_RECOMMENDATION_BASELINE_FAILED')
    def test_tiny_gain_is_not_selected(self):
        result=ops.recommend([self.row('json-fp32',3.,4.),self.row('binary-fp32',2.99,3.99)],[])
        self.assertEqual(result['variant'],'json-fp32')
        self.assertEqual(result['p95_scheduled_to_validated_reduction_fraction'],0.)
    def test_lowest_p95_cannot_hide_mean_regression(self):
        rows=[self.row('json-fp32',3.,4.),self.row('binary-fp32',3.1,3.2),
              self.row('binary-buffered-fp32',2.8,3.4)]
        self.assertEqual(ops.recommend(rows,[])['variant'],'binary-buffered-fp32')
    def test_watch_keeps_refreshing_even_after_finished(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'report.json').write_text(json.dumps(dict(status='OPTIMIZATION_MEASURED',
                directory=d,phase='finished',started_utc=1,finished_utc=2)))
            with patch.object(ops,'ROOT',root),patch.object(ops.time,'sleep',side_effect=KeyboardInterrupt),patch('builtins.print') as output:
                with self.assertRaises(KeyboardInterrupt):ops.show(watch=True)
                self.assertTrue(output.called)


class SyntheticModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch;torch.set_num_threads(1);torch.manual_seed(87)
        class Network(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('mean',torch.linspace(-.3,.3,61))
                self.register_buffer('scale',torch.linspace(.3,2.,61))
                self.lstm=torch.nn.LSTM(61,256)
                self.mlp=torch.nn.Sequential(torch.nn.Linear(256,32),torch.nn.ELU(),torch.nn.Linear(32,14))
            def forward(self,obs,h,cell):
                out,(h,cell)=self.lstm(((obs-self.mean)/self.scale).unsqueeze(0),(h,cell))
                return self.mlp(out.squeeze(0)),h,cell
        cls.network=Network().eval();cls.tmp=tempfile.TemporaryDirectory();cls.folder=Path(cls.tmp.name)
        cls.path=cls.folder/'final6h_005304.onnx'
        torch.onnx.export(cls.network,(torch.zeros(1,61),torch.zeros(1,1,256),torch.zeros(1,1,256)),
            cls.path,opset_version=18,dynamo=False,input_names=['obs','h_in','c_in'],
            output_names=['actions','h_out','c_out'],dynamic_axes={})
        policy=OnnxPolicy(cls.path);golden=[]
        for i in range(128):
            obs=cls.obs(i);action=policy.step(obs,reset=i in (0,64))
            golden.append(dict(obs=obs,reset=i in (0,64),actions=action,h=policy.h.reshape(-1).tolist(),c=policy.c.reshape(-1).tolist()))
        (cls.folder/'golden.json').write_text(json.dumps(dict(cases=golden)))
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    @staticmethod
    def obs(i):return [float((j-30)+(i%17))/64 for j in range(61)]
    def test_buffered_fp32_6000_steps_and_exact_reset(self):
        original=OnnxPolicy(self.path);fast=BufferedPolicy(self.path)
        for i in range(6000):
            a=original.step(self.obs(i),reset=i in (0,3000));b=fast.step(self.obs(i),reset=i in (0,3000))
            comparison(a,b);comparison(original.h.reshape(-1).tolist(),fast.h.reshape(-1).tolist())
            comparison(original.c.reshape(-1).tolist(),fast.c.reshape(-1).tolist())
        before=fast.h.copy();index=fast.index
        with self.assertRaises(RuntimeError):fast.step([float('nan')]*61,reset=True)
        self.assertTrue((fast.h==before).all());self.assertEqual(index,fast.index)
        fast.reset();self.assertEqual(float(abs(fast.h).sum()+abs(fast.c).sum()),0.)
    def test_quantization_candidate_and_interleaved_benchmark(self):
        import onnx
        before=sha(self.path)
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            with patch.object(prep,'frozen'):
                result=prep.prepare(folder,self.folder)
            self.assertEqual(result['status'],'QUANTIZATION_CANDIDATE_PREPARED',result.get('error'))
            self.assertFalse(result['pruning_applied']);self.assertTrue(result['recurrent_weights_unchanged'])
            q=onnx.load(folder/'mlp-dynamic-int8.onnx');base=onnx.load(self.path)
            self.assertEqual([n.SerializeToString() for n in base.graph.node if n.op_type=='LSTM'],
                             [n.SerializeToString() for n in q.graph.node if n.op_type=='LSTM'])
            (folder/self.path.name).write_bytes(self.path.read_bytes())
            original=OnnxPolicy(self.path);values=[]
            for i in range(6000):
                obs=self.obs(i);values.append(dict(obs=obs,actions=original.step(obs,reset=i in (0,3000))))
            (folder/'observations.json').write_text(json.dumps(dict(values=values,source=dict(report_sha256=timing_source.REPORT_SHA))))
            with patch.object(bench,'frozen'):
                measured=bench.benchmark(folder)
            self.assertEqual(measured['status'],'POLICY_BENCHMARK_COMPLETE')
            for name in ('fp32','buffered_fp32','mlp_int8'):
                self.assertEqual(measured['variants'][name]['compared_steps'],6000)
            self.assertTrue(measured['variants']['buffered_fp32']['strict_numerical_parity'])
            self.assertFalse(measured['int8_deployment_selected'])
            self.assertTrue(measured['variants']['mlp_int8']['needs_new_physical_closed_loop_validation'])
            self.assertEqual(sha(self.path),before)
    def test_unsupported_gemm_does_not_modify_original(self):
        import numpy as np
        import onnx
        model=onnx.load(self.path)
        target=next(n for n in model.graph.node if n.op_type=='Gemm')
        for attr in target.attribute:
            if attr.name=='alpha':attr.f=.25;break
        else:target.attribute.append(onnx.helper.make_attribute('alpha',.25))
        before=model.SerializeToString()
        with self.assertRaises(RuntimeError):prep.lower_dense(model,onnx,np)
        self.assertEqual(model.SerializeToString(),before)


if __name__=='__main__':unittest.main()
