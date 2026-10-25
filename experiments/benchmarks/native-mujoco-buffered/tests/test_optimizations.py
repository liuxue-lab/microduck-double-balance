"""Reference ordering, deadline budgets, RNG parity and cache invalidation."""
import copy
import io
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import torch
from support import c
from reference_overlap import LoopClient
from native_ring import CpuRing
from mjlab.utils.buffers import DelayBuffer
from native_backend import DataView
import native_properties
from cpu_placement import select


class Optimizations(unittest.TestCase):
    def client(self):
        client=LoopClient.__new__(LoopClient)
        client.seq=0;client.session='a'*32;client.chain=c.EMPTY_CHAIN
        client.trace=io.StringIO();client.ipc_ms=[];client.ros_ms=[];client.inference_ms=[]
        client.sock=NS(settimeout=lambda value:None)
        events=[];requests=[];timeouts=[]
        def send(req):events.append('send');requests.append(req)
        def receive(timeout):
            events.append('receive');timeouts.append(timeout)
            return dict(reply=c.reply(requests[-1],[.1]*14,0.),ros_rtt_ms=0.)
        client.io=NS(send=send,receive=receive)
        return client,events,timeouts

    def test_reference_after_send_before_receive_consumes_original_timeout(self):
        client,events,timeouts=self.client();original=client.io.receive
        now=[100.]
        def reference():events.append('reference');now[0]+=.003
        with patch('time.monotonic',side_effect=lambda:now[0]):
            self.assertEqual(client.step_with_reference([0.]*61,reference),[.1]*14)
        self.assertEqual(events,['send','reference','receive'])
        self.assertAlmostEqual(timeouts[0],c.RPC_TIMEOUT-.003)
        self.assertIs(client.io.receive,original);self.assertEqual(client.seq,1)

    def test_reference_failure_or_budget_exhaustion_never_returns_an_action(self):
        for failure in (True,False):
            client,events,timeouts=self.client();original=client.io.receive;now=[100.]
            def reference():
                if failure:raise ValueError('reference failed')
                now[0]+=c.RPC_TIMEOUT+.1
            with patch('time.monotonic',side_effect=lambda:now[0]):
                with self.assertRaises((ValueError,RuntimeError)):
                    client.step_with_reference([0.]*61,reference)
            self.assertEqual(events,['send']);self.assertEqual(client.seq,0)
            self.assertEqual(client.trace.getvalue(),'');self.assertIs(client.io.receive,original)

    def test_cpu_ring_matches_full_original_state_and_RNG_across_reset(self):
        torch.manual_seed(712)
        original=DelayBuffer(min_lag=3,max_lag=6,batch_size=1,device='cpu')
        candidate=copy.deepcopy(original);ring=CpuRing(candidate)
        generator=torch.Generator().manual_seed(28)
        for step in range(500):
            if step in (7,83,419):original.reset();candidate.reset()
            value=torch.randn(1,14,generator=generator)
            before=torch.get_rng_state();original.append(value);expected=original.compute().clone()
            after=torch.get_rng_state();torch.set_rng_state(before);actual=ring.sample(value)
            self.assertTrue(torch.equal(expected.view(torch.int32),actual.view(torch.int32)))
            self.assertTrue(torch.equal(after,torch.get_rng_state()))
            for name in ('_current_lags','_step_count','_phase_offsets'):
                self.assertTrue(torch.equal(getattr(original,name),getattr(candidate,name)))
            self.assertEqual(original._buffer._pointer,candidate._buffer._pointer)
            for name in ('_buffer','_num_pushes'):
                self.assertTrue(torch.equal(getattr(original._buffer,name),getattr(candidate._buffer,name)))

    def test_property_cache_invalidation_copy_semantics_and_restoration(self):
        class Entity:
            def __init__(self):self.value=torch.tensor([1.])
        for name in native_properties.FIELDS:
            setattr(Entity,name,property(lambda owner:owner.value*2))
        original=Entity.root_link_pose_w;owner=Entity();foreign=Entity()
        raw=NS(sim=NS(data=DataView(NS())),scene=NS(entities={'robot':NS(data=owner)}))
        with patch.object(native_properties,'verify_sources',return_value=({'mjlab.entity.data':NS(EntityData=Entity)},{})):
            cache=native_properties.PublishedProperties(raw,torch)
        try:
            owner.root_link_pose_w.fill_(100.)
            self.assertEqual(owner.root_link_pose_w.item(),2.)
            owner.value.fill_(3.);raw.sim.data.pull()
            self.assertEqual(owner.root_link_pose_w.item(),6.)
            foreign.value.fill_(5.);self.assertEqual(foreign.root_link_pose_w.item(),10.)
        finally:cache.close()
        self.assertIs(Entity.root_link_pose_w,original)
        self.assertTrue(cache.stats()['closed'])

    def test_cpu_plan_respects_capacity_and_separates_physical_cores(self):
        rows=[dict(cpu=c,capacity=cap,maximum_frequency_kHz=f,busy_fraction=busy,package=0,core=core)
              for c,cap,f,busy,core in ((2,100,4000,.5,1),(3,100,4000,.2,1),(6,100,4000,.3,3),(8,50,3000,0.,4))]
        self.assertEqual(select(rows),(3,6))
        self.assertEqual(select(rows[:1]),(2,None))
        self.assertEqual(select([]),(None,None))
