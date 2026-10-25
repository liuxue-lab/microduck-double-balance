"""One fixed-tier 2 ms diagnostic continuation, with no adaptive acceptance credit.

The reviewed RK trial implementation is reused unchanged. Its transaction only
tracks private forward/sense freshness; no acceptance decision or publish ticket
is issued. Whole-checkpoint control/reset replay is performed by the worker.
"""
import numpy as np
from ground_pair_driver import GroundPairDriver
from clock_contract import TickClock,publication_bound
from native_state_bridge import require
from macro_state_bridge import MACRO_LIMITS

BRANCHES=(('control',None),('control-replay',None),('reset-0',(0,)),
          ('reset-0-replay',(0,)),('reset-1',(1,)),('reset-1-replay',(1,)),('empty',()))

def continuation_cost():
    cost={n:0 for n in MACRO_LIMITS}
    cost.update(state_snapshot=1,state_restore=1,integrations=128,RK4_integrator=128,
        GPU_forward=514,position_write=128,sense=2,endpoint_sensor_read=2,
        private_clock_write=1,model_option_write=2,public_clock_write=1)
    return cost

class PrivateFreshness:
    """Fixed diagnostic freshness only; cannot issue an adaptive commit ticket."""
    def __init__(self):
        self.phase='idle';self.state_generation=0;self.solved=None;self.sensed=None
    def begin(self):
        require(self.phase=='idle','Nested diagnostic trial');self.phase='trial'
    def invalidate(self):
        require(self.phase=='trial','Mutation outside diagnostic trial')
        self.state_generation+=1;self.solved=self.sensed=None
    def forward_complete(self,generation):
        require(self.phase=='trial' and generation==self.state_generation,'Stale diagnostic forward')
        self.solved=generation;self.sensed=None
    def sense_complete(self,generation):
        require(self.phase=='trial' and generation==self.state_generation==self.solved,'Stale diagnostic sense')
        self.sensed=generation
    def close(self):
        require(self.phase=='trial' and self.sensed==self.state_generation,'Diagnostic endpoint not sensed')
        self.phase='finished'

class FixedContinuation:
    _trial=GroundPairDriver._trial
    def __init__(self,bridge,engine,ticks,emit=lambda kind,row:None):
        require(ticks in ((8,),(15,),(0,)), 'Undeclared continuation clocks')
        require(engine.budget is bridge.budget and engine.data is bridge.sim.wp_data and
                engine.model is bridge.sim.wp_model,'Continuation owner mismatch')
        self.bridge,self.engine,self.ticks,self.emit=bridge,engine,ticks,emit
        self.transaction=PrivateFreshness();self.phase='idle';self.trial_serial=0
        self.rejections=[];self.commits=0;self.last_failure=None
    @property
    def published(self):return np.array([TickClock(t).published for t in self.ticks],np.float32)
    def record(self,kind,**row):self.emit(kind,dict(ticks=list(self.ticks),**row))
    def advance(self):
        require(self.phase=='idle' and self.bridge.phase=='idle' and self.commits==0,
                'Continuation is single-use, including after failure')
        self.bridge.budget.reserve(**continuation_cost())
        d=self.bridge.sim.wp_data
        require(d.time.numpy().tobytes()==self.published.tobytes(),'Continuation public clocks differ')
        self.phase='running';b=self.bridge;tx=self.transaction
        publication_before=b.budget.counts.get('public_clock_write_attempted',0)
        try:
            b.checkpoint();tx.begin()
            self.record('macro_start',numeric_Data_hashes=b.saved.data.fingerprint(),
                scope='fixed-tier diagnostic only',adaptive_acceptance=False)
            result=self._trial(128,'fixed_continuation')
            require(b.endpoint_bytes()==result['endpoint'],'Continuation endpoint changed')
            next_ticks=tuple(t+1 for t in self.ticks)
            times=np.array([TickClock(t).published for t in next_ticks],np.float32)
            pose=(d.qpos.numpy().tobytes(),d.qvel.numpy().tobytes())
            b.budget.run('public_clock_write',lambda:d.time.assign(times))
            b.forward(self.engine.native_forward);b.sense()
            require(pose==(d.qpos.numpy().tobytes(),d.qvel.numpy().tobytes()) and
                    d.time.numpy().tobytes()==times.tobytes(),'Continuation refresh changed primary state')
            self.record('commit_prepared',next_ticks=list(next_ticks),adaptive_acceptance=False,
                diagnostic_clock_publication_only=True,contact_ordering=self.engine.contact_order.last)
            tx.close();b.close();self.ticks=next_ticks;self.commits=1;self.phase='finished'
            return dict(resolution=128,integrations=128,dt_s=.002,public_ticks=list(next_ticks),
                publication=[publication_bound(t,float(v)) for t,v in zip(next_ticks,times)],
                adaptive_acceptance=False,second_physical_trajectory=True,
                stage_sha256=result['stage_sha256'])
        except BaseException as exc:
            self.last_failure=dict(error=repr(exc),bridge_phase=b.phase,commits=self.commits)
            # A public refresh failure must never rewind published clocks.
            if b.phase=='trial' and b.budget.counts.get('public_clock_write_attempted',0)==publication_before:
                try:
                    proof=b.restore()
                    require(d.time.numpy().tobytes()==self.published.tobytes(),'Abort public clocks differ')
                    self.record('rollback',reason='continuation_exception_abort',**proof)
                except BaseException as err:self.last_failure['restore_error']=repr(err)
                if b.phase=='trial':b.close()
            self.phase='stopped';raise
