"""New v7 candidate: predeclared finite refinement; frozen thresholds and native arithmetic.
Original v5/v6 failure stays rejected. Not physically accepted yet.
"""
import numpy as np
from lifecycle_support import require
from cohort_contract import CohortTransaction,cohort_tiers
from ground_pair_driver import GroundPairDriver
from integration.refinement_contract import macro_cost, TIERS
from ground_scalar_contract import assess
from batch_position_candidate import assess_cohort
from continuation_driver import FixedContinuation
from clock_contract import TickClock
from replay_diagnostics import compare as compare_replay

CASES = tuple((name,mask,mode) for name,mask in
    (('control',None),('reset-0',(0,)),('reset-1',(1,)),('empty',()))
    for mode in ('adaptive','reference'))

class RefinedAdaptive:
    _trial=GroundPairDriver._trial
    _rollback=GroundPairDriver._rollback
    _commit=GroundPairDriver._commit
    def __init__(self,bridge,engine,ticks,emit=lambda kind,row:None):
        require(ticks in ((7,),(14,),(8,),(15,),(0,)),'Unreviewed adaptive continuation clocks')
        require(engine.budget is bridge.budget and engine.data is bridge.sim.wp_data and
                engine.model is bridge.sim.wp_model,'Adaptive continuation owner mismatch')
        self.bridge,self.engine,self.ticks,self.emit=bridge,engine,ticks,emit
        self.transaction=CohortTransaction();self.phase='idle';self.trial_serial=0
        self.rejections=[];self.commits=0;self.last_failure=None;self.accepted_trial=None
    @property
    def published(self):return np.array([TickClock(t).published for t in self.ticks],np.float32)
    def record(self,kind,**row):self.emit(kind,dict(ticks=list(self.ticks),**row))
    def advance(self):
        require(self.phase=='idle' and self.bridge.phase=='idle' and self.commits==0,
                'Adaptive continuation is single-use, including after failure')
        self.bridge.budget.reserve(**macro_cost())
        require(self.bridge.sim.wp_data.time.numpy().tobytes()==self.published.tobytes(),
                'Adaptive continuation public clocks differ')
        cohort_tiers((TIERS,),(0,))
        self.phase='running';tx=self.transaction
        publication_before=self.bridge.budget.counts.get('public_clock_write_attempted',0)
        try:
            self.bridge.checkpoint();tx.begin()
            self.record('macro_start',numeric_Data_hashes=self.bridge.saved.data.fingerprint(),
                scope='post-reset adaptive candidate',mandatory_exact_replay=True)
            previous=self._trial(32,'estimate');self._rollback('discard_estimator')
            for n in TIERS[1:]:
                tx.begin();candidate=self._trial(n,'candidate' if n==64 else 'retry')
                decision=assess_cohort(previous['worlds'],candidate['worlds'],(0,),assess)
                self.record('decision',resolution=n,**decision)
                ticket=tx.decide(decision,candidate['endpoint'],replay_required=True,
                    stage_trace_digest=candidate['stage_sha256'])
                if decision['accepted']:
                    self._rollback('mandatory_accepted_replay')
                    replay=self._trial(n,'replay')
                    self.last_replay_comparison=compare_replay(candidate,replay)
                    self.record('replay_comparison',resolution=n,**self.last_replay_comparison)
                    ticket=tx.replay_complete(replay['endpoint'],stage_trace_digest=replay['stage_sha256'])
                    self.record('exact_replay',resolution=n,full_numeric_Data_and_held_owners=True,
                        sensor_values_and_schema=True,physical_stage_hashes=True)
                    self.accepted_trial=replay
                    result=self._commit(ticket,replay,decision)
                    result.update(replayed=True,mandatory_exact_replay=True,
                        actual_trial_count=self.trial_serial,
                        actual_integrations=sum(t for t in TIERS if t <= n)+n,
                        accepted_stage_sha256=replay['stage_sha256'],
                        post_reset_adaptive_candidate=True)
                    return result
                self.rejections.append(decision);self._rollback('reject_whole_cohort')
                previous=candidate
            raise RuntimeError('Maximum declared tier512 failed; initial cohort restored; no extra tier or retry')
        except BaseException as exc:
            self.last_failure=dict(error=repr(exc),transaction_phase=tx.phase,
                bridge_phase=self.bridge.phase,commits=self.commits,replay_comparison=getattr(self,'last_replay_comparison',None))
            if self.bridge.phase=='trial' and tx.phase!='idle' and self.bridge.budget.counts.get('public_clock_write_attempted',0)==publication_before:
                try:self.last_failure['rollback_proof']=self._rollback('exception_abort')
                except BaseException as error:self.last_failure['restore_error']=repr(error)
            self.phase='stopped';tx.ticket=None
            if self.bridge.phase=='trial':self.bridge.close()
            raise
