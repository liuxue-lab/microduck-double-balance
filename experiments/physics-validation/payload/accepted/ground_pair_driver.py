"""Separate one-macro ground pair pilot; original ground criteria unchanged.

Two fresh complete diagnostic worlds seeded at archived ground ticks7 and14.
Only original ground tiers32/64/128; stop at128, no higher tier or retry.
Both real event types and original numerical/event rejection are required.
This is not the old head/ground schedule or a completed G09 acceptance claim.
"""
import hashlib,warnings
import numpy as np
from ground_scalar_contract import assess,brackets
from convergence_contract import PAIRS,ContactIntegral
from clock_contract import TickClock,publication_bound
from cohort_contract import CohortTransaction,cohort_tiers
from batch_position_candidate import BatchPositionAccumulator,assess_cohort
from packed_contact_observer import observe,occupancy
from native_state_bridge import canonical,require
from macro_state_bridge import MACRO_LIMITS

def macro_cost(tiers=(32,64,128)):
    require(tiers==(32,64,128),'No undeclared tier or budget expansion')
    trials=len(tiers)+1;steps=sum(tiers)+tiers[-1]
    cost=dict(MACRO_LIMITS)
    cost.update(state_snapshot=1,state_restore=trials+1,integrations=steps,RK4_integrator=steps,
        GPU_forward=4*steps+trials+1,position_write=steps,sense=trials+1,endpoint_sensor_read=trials+1,
        private_clock_write=trials,model_option_write=2*trials,public_clock_write=1)
    return cost

class GroundPairDriver:
    def __init__(self,bridge,engine,ticks,emit=lambda kind,row:None):
        require(ticks in ((7,),(14,)),'Explicit late-window integer clocks required; kinematic seeds are not owner checkpoints')
        require(engine.budget is bridge.budget and engine.data is bridge.sim.wp_data and engine.model is bridge.sim.wp_model,'Engine/state owner mismatch')
        self.bridge,self.engine,self.ticks,self.emit=bridge,engine,ticks,emit
        self.transaction=CohortTransaction();self.phase='idle';self.trial_serial=0
        self.rejections=[];self.commits=0;self.last_failure=None
    @property
    def published(self):return np.array([TickClock(t).published for t in self.ticks],np.float32)
    def record(self,kind,**row):self.emit(kind,dict(ticks=list(self.ticks),**row))
    def _rollback(self,reason):
        tx=self.transaction
        if tx.phase not in ('restore_required','restore_for_replay','abort_required'):tx.abort()
        proof=self.bridge.restore_into(tx)
        require(self.bridge.sim.wp_data.time.numpy().tobytes()==self.published.tobytes(),'Rollback public clocks differ')
        self.record('rollback',reason=reason,**proof)
        return proof
    def _trial(self,n,role):
        b=self.bridge;tx=self.transaction;d=b.sim.wp_data
        require(tx.phase in ('trial','replay') and d.time.numpy().tobytes()==self.published.tobytes(),'Trial did not start from public checkpoint')
        self.trial_serial+=1;self.record('trial_start',trial=self.trial_serial,resolution=n,role=role)
        if hasattr(self.engine,'solver_order'):
            self.engine.solver_order.begin_trial(self.ticks,n,role,self.trial_serial)
        b.options(n);b.private_origin();tx.invalidate()
        acc=BatchPositionAccumulator(d.qpos.numpy(),np.ones(1,np.bool_))
        integrals=[ContactIntegral()];samples=[[]];stage_hash=hashlib.sha256()
        with self.engine.hooks():
            for _ in range(n):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always');row=b.microstep(self.engine,acc,(None,))
                require(not caught,'Native warning stops macro; no silent continuation')
                stage_hash.update(bytes.fromhex(row['stage_sha256']))
                self.record('microstep',trial=self.trial_serial,resolution=n,micro_index=int(acc.steps[0])-1,
                    micro_dt_s=float(np.float32(.002/n)),**row)
                for stage in row['stages']:
                    for w in range(1):
                        contacts=stage['contacts'][w]
                        if stage['stage']==0:samples[w].append(occupancy(contacts,PAIRS))
                        converted={'contact_force_rows':[dict(geoms=c['geom'],raw_solver_forces=c['raw_solver_rows'],distance_m=c['distance_m']) for c in contacts]}
                        integrals[w].observe(converted,stage['stage'],float(np.float32(.002/n)))
        # Endpoint physics uses the trial options; formal Model options are restored
        # before endpoint fingerprints and before any public manager consumer.
        b.forward(self.engine.native_forward);tx.forward_complete(tx.state_generation)
        raw={name:a.numpy() for name,a in b.saved.data.arrays.items()}
        contacts=observe(raw,state_generation=b.generation,solved_generation=b.solved)
        qpos,qvel=d.qpos.numpy(),d.qvel.numpy();kernel_times=d.time.numpy().tolist()
        b.options()
        self.record('trial_physics_complete',trial=self.trial_serial,resolution=n,role=role,world_qpos=qpos,world_qvel=qvel,endpoint_contacts=contacts,kernel_times=kernel_times,numeric_Data_hashes=b.saved.data.fingerprint(),contact_ordering=getattr(self.engine,'contact_order',None).last if hasattr(self.engine,'contact_order') else None)
        b.sense();tx.sense_complete(tx.state_generation)
        worlds=[]
        for w in range(1):
            samples[w].append(occupancy(contacts[w],PAIRS))
            worlds.append(dict(resolution=n,endpoint=dict(qpos=qpos[w].copy(),qvel=qvel[w].copy()),event_samples=samples[w],brackets=brackets(samples[w],n),integral=integrals[w].result()))
        result=dict(id=self.trial_serial,resolution=n,worlds=worlds,endpoint=b.endpoint_bytes(),stage_sha256=stage_hash.hexdigest(),kernel_times=kernel_times)
        self.record('trial_complete',trial=self.trial_serial,resolution=n,role=role,stage_sha256=result['stage_sha256'],endpoint_sha256=hashlib.sha256(result['endpoint']).hexdigest(),kernel_times=kernel_times,
            worlds=worlds,endpoint_contacts=contacts,endpoint_record=result['endpoint'].decode(),numeric_Data_hashes=b.saved.data.fingerprint())
        return result
    def check_required_witness(self,accepted):
        require(any(not r['numerical_accepted'] for d in self.rejections for r in d['worlds']),
            'Planned original-criterion rejection not observed; no acceptance credit')
        expected=((('lower_floor',),('left_foot_floor','right_foot_floor'))[self.bridge.bank.logical_world],)
        for world,names in zip(accepted['worlds'],expected):
            require(all(not world['event_samples'][0][p] and world['event_samples'][-1][p] for p in names),
                'Planned distinct physical contact entry not observed; stop without retuning')

    def _commit(self,ticket,accepted,decision):
        b=self.bridge;tx=self.transaction;d=b.sim.wp_data
        require(tx.phase=='ready' and ticket is tx.ticket,'Single-use replay-checked ticket required')
        require(b.endpoint_bytes()==accepted['endpoint'],'Accepted endpoint changed before public commit')
        next_ticks=tuple(t+1 for t in self.ticks)
        times=np.array([TickClock(t).published for t in next_ticks],np.float32)
        pose=(d.qpos.numpy().tobytes(),d.qvel.numpy().tobytes())
        b.budget.run('public_clock_write',lambda:d.time.assign(times))
        require(d.time.numpy().tobytes()==times.tobytes(),'Public time write failed or partial')
        # Time-dependent derived fields/BVH are refreshed at the public clocks.
        b.forward(self.engine.native_forward);b.sense()
        require(pose==(d.qpos.numpy().tobytes(),d.qvel.numpy().tobytes()) and d.time.numpy().tobytes()==times.tobytes(),'Publication refresh changed primary state/time')
        publication=[publication_bound(t,float(v)) for t,v in zip(next_ticks,times)]
        committed_endpoint=hashlib.sha256(b.endpoint_bytes()).hexdigest()
        # A precommit record is emitted before the final in-memory ticket
        # consumption. An observer failure still permits whole-checkpoint restore.
        self.record('commit_prepared',next_ticks=list(next_ticks),endpoint_sha256=committed_endpoint,publication=publication,contact_ordering=getattr(self.engine,'contact_order',None).last if hasattr(self.engine,'contact_order') else None)
        tx.publish(ticket);self.ticks=next_ticks;self.commits+=1;b.close();self.phase='idle'
        return dict(accepted_resolution=accepted['resolution'],decision=decision,rejections=list(self.rejections),
            replayed=any(not r['numerical_accepted'] for d0 in self.rejections for r in d0['worlds']),
            public_ticks=list(self.ticks),publication=publication,accepted_integrals=[w['integral'] for w in accepted['worlds']],
            committed_endpoint_sha256=committed_endpoint,actual_native_evidence_required=True)
    def advance(self):
        require(self.phase=='idle' and self.bridge.phase=='idle','Driver stopped or macro already active')
        require(self.commits==0 and self.ticks in ((7,),(14,)),'Exactly one declared ground-pair macro; no second run')
        # Includes maximum-tier replay, publication refresh, and recovery reserve.
        # This is a cost check, never an authorization or increase of any cap.
        self.bridge.budget.reserve(**macro_cost())
        require(self.bridge.sim.wp_data.time.numpy().tobytes()==self.published.tobytes(),'Native/public clock mismatch before checkpoint')
        cohort_tiers(((32,64,128),),(0,))
        self.phase='running';self.rejections=[];tx=self.transaction
        try:
            self.bridge.checkpoint();tx.begin();self.record('macro_start',numeric_Data_hashes=self.bridge.saved.data.fingerprint())
            previous=self._trial(32,'estimate');self._rollback('discard_estimator')
            accepted=decision=None
            for n in (64,128):
                tx.begin();candidate=self._trial(n,'candidate' if n==64 else 'retry')
                decision=assess_cohort(previous['worlds'],candidate['worlds'],(0,0),assess)
                self.record('decision',resolution=n,**decision)
                numerical=any(not r['numerical_accepted'] for r0 in self.rejections for r in r0['worlds'])
                ticket=tx.decide(decision,candidate['endpoint'],replay_required=numerical,stage_trace_digest=candidate['stage_sha256'])
                if decision['accepted']:
                    accepted=candidate
                    if numerical:
                        self._rollback('original_criterion_replay')
                        replay=self._trial(n,'replay')
                        ticket=tx.replay_complete(replay['endpoint'],stage_trace_digest=replay['stage_sha256'])
                        self.record('exact_replay',resolution=n,full_numeric_Data_and_held_owners=True,sensor_values_and_schema=True,physical_stage_hashes=True)
                        accepted=replay
                    self.check_required_witness(accepted)
                    return self._commit(ticket,accepted,decision)
                self.rejections.append(decision);self._rollback('reject_whole_cohort')
                previous=candidate
            raise RuntimeError('Maximum declared tier failed; starting cohort restored; no extra tier or retry')
        except Exception as exc:
            self.last_failure=dict(error=str(exc),transaction_phase=tx.phase,bridge_phase=self.bridge.phase,commits=self.commits)
            if self.bridge.phase=='trial' and tx.phase!='idle':
                try:self._rollback('exception_abort')
                except Exception as restore_error:self.last_failure['restore_error']=str(restore_error)
            self.phase='stopped';tx.ticket=None
            if self.bridge.phase=='trial':self.bridge.close()
            raise
