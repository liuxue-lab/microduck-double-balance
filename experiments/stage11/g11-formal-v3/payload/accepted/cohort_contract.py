"""Offline transaction/cadence contract, not a Simulation implementation."""
from dataclasses import dataclass
import hashlib
import json

def require(ok,message):
    if not ok:raise RuntimeError(message)

def shared_tiers(world_tiers):
    require(len(world_tiers)==1 and all(t and all(type(n) is int and n>0 for n in t)
            and list(t)==sorted(set(t)) for t in world_tiers),'One ordered explicit tier roster required')
    common=sorted(set(world_tiers[0]))
    require(common,'No shared predeclared tier; cannot silently add a resolution')
    return tuple(common)

def cohort_tiers(world_tiers,floors):
    require(len(floors)==1 and all(type(n) is int and n>=0 for n in floors),'One predeclared floor required')
    common=shared_tiers(world_tiers)
    acceptable=tuple(n for n in common if all(n>=f for f in floors))
    require(acceptable,'No common tier satisfies both original floors; do not invent a finer tier')
    return dict(common_trial_tiers=common,possible_acceptance_tiers=acceptable)

def rk_route(route,callback_names):
    require(route=='native_step_rk4','Split step1/step2 is not a pinned RK4 route')
    require(not callback_names,'Unbudgeted native callbacks would run inside every forward')
    return dict(forwards_per_step=4,internal_stage_forwards=3,physical_stages=4,
                applies_to='Pinned native step RK4 with callbacks absent; endpoint forward is additional')

@dataclass(frozen=True)
class CommitTicket:
    serial:int
    endpoint_digest:str

class CohortTransaction:
    """No public commit before whole-cohort acceptance and required replay.

    Callers retain Data/owner/RNG/Model snapshots and verify them externally.
    Failed restoration poisons this coordinator; it cannot authorize publication.
    """
    def __init__(self):
        self.phase='idle';self.serial=0;self.public_ticks=0
        self.state_generation=0;self.solved_generation=None;self.sensed_generation=None
        self.candidate=None;self.ticket=None;self.stage_digest=None

    def begin(self):
        require(self.phase=='idle','Transaction already active or poisoned')
        self.serial+=1;self.phase='trial';self.candidate=None;self.ticket=None

    def invalidate(self):
        require(self.phase in ('trial','replay'),'Native state mutation outside private trial')
        self.state_generation+=1;self.solved_generation=None;self.sensed_generation=None

    def forward_complete(self,generation):
        require(self.phase in ('trial','replay') and generation==self.state_generation,'Stale or external forward completion')
        self.solved_generation=generation;self.sensed_generation=None

    def contacts_ready(self):
        require(self.phase in ('trial','replay') and self.solved_generation==self.state_generation,'Contacts require a fresh completed forward')

    def sense_complete(self,generation):
        self.contacts_ready();require(generation==self.state_generation,'Stale sense completion')
        self.sensed_generation=generation

    def decide(self,decision,endpoint_bytes,*,replay_required,stage_trace_digest):
        require(self.phase=='trial' and type(replay_required) is bool,'Unexpected decision boundary')
        self.contacts_ready()
        require(self.sensed_generation==self.state_generation,'Final sensor cache not refreshed')
        require(isinstance(endpoint_bytes,bytes) and endpoint_bytes,'Explicit private endpoint bytes required')
        require(isinstance(stage_trace_digest,str) and len(stage_trace_digest)==64 and
                all(c in '0123456789abcdef' for c in stage_trace_digest),'Explicit physical-stage SHA256 required')
        require(len(decision['worlds'])==1 and all(type(r['accepted']) is bool for r in decision['worlds']), 'One explicit world decision required')
        accepted=all(r['accepted'] for r in decision['worlds'])
        require(decision['accepted'] is accepted and decision['action']==('commit_whole_cohort' if accepted else 'rollback_whole_cohort'), 'Aggregate decision differs')
        if not accepted:self.phase='restore_required';return None
        self.candidate=hashlib.sha256(endpoint_bytes).hexdigest()
        self.stage_digest=stage_trace_digest
        self.phase='restore_for_replay' if replay_required else 'ready'
        if not replay_required:self.ticket=CommitTicket(self.serial,self.candidate)
        return self.ticket

    def restored(self,*,data_exact,owners_exact,rng_exact,model_exact,bindings_exact):
        require(self.phase in ('restore_required','restore_for_replay','abort_required'),'Unexpected rollback boundary')
        checks=(data_exact,owners_exact,rng_exact,model_exact,bindings_exact)
        if not all(v is True for v in checks):
            self.phase='poisoned';self.ticket=None
            raise RuntimeError('Rollback evidence failed; publication permanently blocked')
        replay=self.phase=='restore_for_replay'
        self.phase='replay' if replay else 'idle'
        self.state_generation+=1;self.solved_generation=None;self.sensed_generation=None

    def replay_complete(self,endpoint_bytes,*,stage_trace_digest):
        require(self.phase=='replay','Replay not scheduled')
        self.contacts_ready()
        require(self.sensed_generation==self.state_generation,'Replay sensor caches stale')
        if hashlib.sha256(endpoint_bytes).hexdigest()!=self.candidate or stage_trace_digest!=self.stage_digest:
            self.phase='abort_required';self.ticket=None
            raise RuntimeError('Exact endpoint replay failed; rollback required')
        self.phase='ready';self.ticket=CommitTicket(self.serial,self.candidate)
        return self.ticket

    def abort(self):
        require(self.phase not in ('idle','poisoned'),'Nothing safe to abort')
        self.phase='abort_required';self.ticket=None

    def publish(self,ticket):
        require(self.phase=='ready' and ticket is self.ticket and isinstance(ticket,CommitTicket),'Valid one-use commit ticket required')
        self.public_ticks+=1;self.phase='idle';self.ticket=None
        return self.public_ticks

    def permit_owner_consumer(self):
        require(self.phase=='idle','Owner/BAM/RNG/cadence work during unpublished trial refused')

class GraphLifecycle:
    """Pure guard model: binds complete caller-provided allocation/handle tokens.

    Tokens must include shapes/strides/pointers, all opaque owner identities,
    and static Model options/callback identities captured by Python branches.
    This class neither dereferences GPU pointers nor clones native handles.
    """
    def __init__(self):
        self.phase='empty';self.epoch=0;self.graphs={};self.bindings=None;self.pending=None

    @staticmethod
    def fingerprint(bindings):
        require(isinstance(bindings,dict) and bindings,'Complete binding manifest required')
        require('static_options' in bindings and bool(bindings['static_options']),
                'Static Model/callback configuration fingerprint required')
        return hashlib.sha256(json.dumps(bindings,sort_keys=True,allow_nan=False).encode()).hexdigest()

    def begin_capture(self,bindings,*,synchronized,in_transaction,gc_suspended):
        require(self.phase!='capturing' and not in_transaction and synchronized is True and gc_suspended is True,
                'Capture needs a synchronized boundary and suspended GC outside a trial')
        self.pending=self.fingerprint(bindings);self.phase='capturing'
        # The owner must keep previous native graphs alive until capture exits.
        return tuple(self.graphs.values())

    def finish_capture(self,bindings,graphs,*,has_context):
        require(self.phase=='capturing','No capture active')
        expected={'step','forward','reset'}|({'sense'} if has_context else set())
        if self.fingerprint(bindings)!=self.pending or set(graphs)!=expected or any(v is None for v in graphs.values()):
            self.phase='invalid';self.pending=None
            raise RuntimeError('Incomplete capture or changed bindings; launch refused')
        self.graphs=dict(graphs);self.bindings=self.pending;self.pending=None;self.epoch+=1;self.phase='ready'

    def invalidate(self):
        require(self.phase!='capturing','Cannot change allocation/handle during capture')
        self.phase='invalid'

    def launch(self,name,bindings):
        require(self.phase=='ready' and self.fingerprint(bindings)==self.bindings and name in self.graphs,
                'Stale graph, array binding or opaque handle; recapture required')
        return self.graphs[name]

    def release(self,*,synchronized,in_capture):
        require(self.phase!='capturing' and not in_capture and synchronized is True,'Do not destroy graph roots inside capture or while work is outstanding')
        self.graphs={};self.bindings=None;self.phase='empty'
