"""Compute the local reference after send while the original Orin RPC is in flight.

Packet construction, deadlines, reply validation, transcript and applied actions
remain the original LoopClient implementation. There is one request in flight.
"""
import time
from loop_contract import LoopClient as OriginalClient
from deploy_contract import require

class LoopClient(OriginalClient):
    def step_with_reference(self,observation,reference):
        original=self.io.receive
        called=False
        def receive(timeout):
            nonlocal called
            require(not called,'Reference callback invoked more than once')
            called=True
            begun=time.monotonic()
            reference()
            remaining=timeout-(time.monotonic()-begun)
            require(remaining>0,'Reference exhausted the original RPC deadline')
            return original(remaining)
        self.io.receive=receive
        try:
            result=super().step(observation)
            require(called,'Reference callback was skipped')
            return result
        finally:self.io.receive=original
