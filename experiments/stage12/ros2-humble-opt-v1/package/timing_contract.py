"""Monotonic deadlines and a latched receive gate; no automatic rearming."""
import json
import time

from deploy_contract import require, comparison, vector
from loop_contract import (EMPTY_CHAIN, MAX_PACKET, chain, measured,
                           validate_request, validate_reply)

PERIOD = .020
SCENARIOS = ('replay', 'delay', 'silence', 'duplicate')


class DeadlineGate:
    def __init__(self, session, clock=time.monotonic, trace=None):
        self.session, self.clock, self.trace = session, clock, trace
        self.pending, self.fault = None, None
        self.seq = self.sent = self.discarded = 0
        self.chain, self.max_error = EMPTY_CHAIN, 0.
        self.dispatch_ms, self.response_ms, self.age_ms = [], [], []
        self.inference_ms, self.discard_examples = [], []

    def fail(self, reason, detail=None):
        if self.fault is None:
            self.fault = dict(reason=reason, seq=self.seq, monotonic_seconds=self.clock(), detail=detail)

    def begin(self, request, expected, due):
        require(self.fault is None, 'FAULT_LATCHED: fresh process/session and reset required')
        require(self.pending is None, 'Only one outstanding request is permitted')
        validate_request(request, self.session, self.seq)
        vector(expected, 14)
        sent = self.clock()
        if sent > due + PERIOD:
            self.fail('DISPATCH_DEADLINE')
            raise RuntimeError('Dispatch deadline missed; no catch-up burst')
        require(sent >= due, 'Do not publish before the scheduled observation time')
        self.pending = dict(request=request, expected=expected, due=due, sent=sent, deadline=due + PERIOD)
        self.sent += 1

    def poll(self):
        if self.fault is None and self.pending is not None and self.clock() > self.pending['deadline']:
            self.fail('RESPONSE_DEADLINE')

    def discard(self, reason, payload):
        self.discarded += 1
        if len(self.discard_examples) < 8:
            self.discard_examples.append(dict(reason=reason, payload_prefix=str(payload)[:160]))

    def receive(self, payload):
        if self.fault is not None:
            self.discard('FAULT_LATCHED', payload)
            return
        if self.pending is None:
            self.fail('UNEXPECTED_REPLY')
            self.discard('UNEXPECTED_REPLY', payload)
            return
        p = self.pending
        try:
            require(isinstance(payload, str) and len(payload.encode()) <= MAX_PACKET, 'Oversized response')
            response = validate_reply(json.loads(payload), p['request'])
            error = comparison(response['actions'], p['expected'])
        except Exception as exc:
            self.fail('INVALID_REPLY', repr(exc))
            self.discard('INVALID_REPLY', payload)
            return
        received = self.clock()
        if received > p['deadline']:
            self.fail('LATE_REPLY')
            self.discard('LATE_REPLY', payload)
            return
        rtt = (received - p['sent']) * 1000
        age = (received - p['due']) * 1000
        dispatch = (p['sent'] - p['due']) * 1000
        self.chain = chain(self.chain, p['request'], response)
        self.max_error = max(error, self.max_error)
        self.response_ms.append(rtt)
        self.age_ms.append(age)
        self.dispatch_ms.append(dispatch)
        self.inference_ms.append(response['inference_ms'])
        if self.trace is not None:
            self.trace.write(json.dumps(dict(seq=self.seq, request=p['request'], response=response,
                dispatch_lateness_ms=dispatch, response_ms=rtt, scheduled_to_validated_ms=age), allow_nan=False) + '\n')
        self.seq += 1
        self.pending = None

    def snapshot(self):
        return dict(sent_requests=self.sent, accepted_replies=self.seq, fault_latched=self.fault is not None,
            fault=self.fault, discarded_replies=self.discarded, discard_examples=self.discard_examples,
            transcript_sha256=self.chain, max_action_absolute_error=self.max_error,
            response_ms=measured(self.response_ms), dispatch_lateness_ms=measured(self.dispatch_ms),
            scheduled_to_validated_ms=measured(self.age_ms), orin_inference_ms=measured(self.inference_ms),
            deadline_ms=PERIOD * 1000, automatic_rearm=False, cached_action_fallback=False,
            actions_applied_to_plant=0, motor_io=False)
