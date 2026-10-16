"""Reuse accepted RK4 arithmetic, transactions, sorting and solver reductions.

New scope: consecutive nominal macro ticks owned by a real RL environment.
Evidence retains all macro/trial decisions and full boundary snapshots; inner
contact/constraint checks run unchanged in strength but have hash summaries.
No temporary numerical CUDA graph is captured. The single contact-sort graph
is retained with its binding checks and native destruction record.
"""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from runtime_accounting import require, save, jsonable
from trajectory_candidate import TrajectoryContactGraph, TrajectoryReductions
from solver_order_candidate import SolverOrder, expected_ids, verify_sorted
from ground_pair_driver import GroundPairDriver
from integration.refinement_contract import macro_cost, TIERS
from integration.refined_adaptive import RefinedAdaptive as AdaptiveContinuation
from evidence_journal import EvidenceJournal, encoded
from sensor_state_snapshot import digest


class BoundaryContactArchive:
    def __init__(self, out):
        self.out = Path(out)
        self.rows = 0
        self.chain = hashlib.sha256()
        self.sample_next = True
        self.samples = 0
    def write(self, before_metadata, before_sort, after_sort, proof):
        require(proof['serial'] == self.rows+1, 'Contact evidence sequence differs')
        row = dict(serial=proof['serial'], proof=proof)
        for label, raw in (('before_metadata', before_metadata), ('before_sort', before_sort), ('after_sort', after_sort)):
            row[label] = {n: digest(a) for n, a in raw.items()}
        self.chain.update(encoded(row)); self.rows += 1
        if self.sample_next:
            name = 'contact-sample-%03d' % self.samples
            np.savez_compressed(self.out/(name+'.npz'),
                **{label+'.'+n: a for label, raw in (('before_metadata', before_metadata),
                   ('before_sort', before_sort), ('after_sort', after_sort)) for n, a in raw.items()})
            save(self.out/(name+'.json'), row)
            self.samples += 1; self.sample_next = False
    def summary(self):
        return dict(records=self.rows, full_byte_checks_per_record=True,
            digest=self.chain.hexdigest(), raw_samples=self.samples,
            all_inner_raw_blobs_retained=False, scope='G11 interface audit; G10 raw evidence reused')


class FormalContact(TrajectoryContactGraph):
    CALL_CAP = 28340


class FormalSolverOrder(SolverOrder):
    def __init__(self, out, world, d, wp, solver, gate, caps, limits):
        super().__init__(out, world, d, wp, solver, gate, caps, limits)
        self.compact_digest = hashlib.sha256()
        self.sample_next = True
        self.samples = 0
    def begin_trial(self, ticks, resolution, role, trial):
        self.target = None
    def after_constraint(self, ctx):
        # Same preconditions, true changed-state oracle, canonical kernel and
        # postchecks as accepted SolverOrder.after_constraint; only sink differs.
        require(self.pending is not None and self.pending[0] is ctx, 'Missing changed-row input')
        self.gate.sync(); _, old, nefc, done = self.pending
        new = self.d.efc.state.numpy().copy()
        before = ctx.changed_efc_ids.numpy().copy()
        count_raw = ctx.changed_efc_count.numpy().copy(); count = int(count_raw[0])
        expected = expected_ids(old, new, before, count, nefc, done, self.quadratic)
        self.call('SolverOrder_kernel', self.wp.launch, self.k.canonical_changes, dim=1,
                  inputs=[ctx.changed_efc_count, ctx.changed_efc_ids])
        after = ctx.changed_efc_ids.numpy().copy()
        verify_sorted(before, after, count, expected)
        require(ctx.changed_efc_count.numpy().tobytes() == count_raw.tobytes() and
                self.d.efc.state.numpy().tobytes() == new.tobytes(), 'Canonicalization changed count/state')
        require(self.rows < self.limits['records'], 'Changed-row evidence budget exhausted')
        row = dict(serial=self.rows+1, forward=self.gate.counts['GPU_forward_attempted'],
            count=count, nefc=nefc, done=done, old=digest(old), new=digest(new),
            before=digest(before), after=digest(after), expected=expected)
        self.compact_digest.update(encoded(row))
        if self.sample_next:
            name = 'sample-%03d' % self.samples
            np.savez_compressed(self.root/(name+'.npz'), before_state=old, after_state=new,
                                before=before, after=after, count=count_raw)
            save(self.root/(name+'.json'), row)
            self.samples += 1; self.sample_next = False
        self.rows += 1
        self.reordered += int(before.tobytes() != after.tobytes()); self.pending = None
    def summary(self):
        return dict(records=self.rows, reordered_lists=self.reordered,
            sha256=self.compact_digest.hexdigest(), raw_samples=self.samples,
            all_live_oracle_checks_retained=True, all_inner_raw_blobs_retained=False,
            pending=self.pending is not None, tolerance_changes=0)


class FormalReductions(TrajectoryReductions):
    def __init__(self, *args, **kwargs):
        # The inherited numeric kernels and install method are byte-identical.
        # Substitute only the audit sink before its constructor is called.
        import solver_order_candidate as source
        original = source.SolverOrder
        source.SolverOrder = FormalSolverOrder
        try:
            super().__init__(*args, **kwargs)
        finally:
            source.SolverOrder = original


class FormalAdaptive(AdaptiveContinuation):
    def __init__(self, bridge, engine, ticks, emit):
        require(isinstance(ticks, tuple) and len(ticks) == 1 and
                type(ticks[0]) is int and 0 <= ticks[0] < 20,
                'Only this batch\'s nominal ticks0..19 are released')
        # Reuse the verified constructor's owner/transaction checks, then install
        # the explicitly declared formal clock. Trial/decision/replay/commit
        # methods and original 32/64/128 thresholds are unchanged.
        super().__init__(bridge, engine, (0,), emit)
        self.ticks = ticks


class InterfaceEvidence(EvidenceJournal):
    def __init__(self, path, bridge):
        super().__init__(path, bridge, byte_limit=48*1024*1024,
                         file_limit=100, record_limit=1700)
        self.micro_rows = []
        self.stage_digest = hashlib.sha256()
    def __call__(self, kind, row):
        require(not self.stopped and not self.closed, 'Interface journal unavailable')
        self.attempted_records += 1
        require(self.attempted_records <= self.record_limit, 'Interface record budget exhausted')
        try:
            prefix = 'record-%05d' % self.attempted_records
            if kind == 'microstep':
                state = row['post_integrator']
                self.micro_rows.append(dict(trial=row['trial'], resolution=row['resolution'],
                    micro_index=row['micro_index'], qpos=state['qpos'], qvel=state['qvel'],
                    time=state['time'], stage_sha256=row['stage_sha256']))
                self.stage_digest.update(bytes.fromhex(row['stage_sha256']))
                return
            parts = self._snapshot(prefix) if kind in ('macro_start','commit_prepared') else {}
            event = dict(sequence=self.attempted_records, kind=kind, row=row, snapshots=sorted(parts))
            parts[prefix+'.json'] = encoded(event)
            self._write(parts); self.events.append(prefix+'.json')
        except BaseException:
            self.stopped = True
            raise
    def finalize(self):
        import io
        if self.micro_rows:
            buf = io.BytesIO()
            np.savez_compressed(buf,
                **{k: np.asarray([r[k] for r in self.micro_rows]) for k in self.micro_rows[0]})
            self._write({'microsteps.npz': buf.getvalue(), 'inner-scope.json': encoded(dict(
                microsteps=len(self.micro_rows), stage_digest=self.stage_digest.hexdigest(),
                raw_every_micro_qpos_qvel_time=True, all_trial_endpoint_records=True,
                full_boundary_Data_owners=True, all_inner_full_Data_raw=False,
                previous_G09_G10_reused=True))})
        return super().finalize()


class BudgetedRuntime:
    def __init__(self, env, torch, wp, fw, mw, bank, gate, budget, out, contact, reductions, raw_step, raw_rk, *, max_macros, max_episode_ticks):
        self.env, self.torch, self.wp, self.fw, self.mw = env, torch, wp, fw, mw
        self.bank, self.gate, self.budget, self.out = bank, gate, budget, Path(out)
        self.contact, self.reductions = contact, reductions
        self.raw_step, self.raw_rk = raw_step, raw_rk
        require(type(max_macros) is int and max_macros > 0, 'Explicit macro budget required')
        require(type(max_episode_ticks) is int and 1 <= max_episode_ticks <= 5000, 'Episode tick bound differs')
        self.max_macros, self.max_episode_ticks = max_macros, max_episode_ticks
        self.counter_origin = env._sim_step_counter
        self.macros = 0; self.stopped = False; self.results = []
        self.last_endpoint = None

    def advance(self, env):
        from integration.cloud_limits import check_boundary
        check_boundary(self.out)
        from integration.owner_roster import roots_for
        from integration.refined_state import RefinedStateBridge as GroundPairStateBridge
        from native_rk4_bridge import NativeRK4Bridge
        from formal_task import measured_surface
        from scene_evidence import owner_state, save_arrays
        require(env is self.env and not self.stopped and self.macros < self.max_macros, 'Session budget exhausted or stopped')
        require(type(env._g11_tick) is int and 0 <= env._g11_tick < self.max_episode_ticks, 'Episode tick budget exhausted')
        require(not bool(env._manual_reset_pending.any()), 'Pending manual reset')
        cost = macro_cost()
        for k in ('Env_substep_counter','Action_apply','Scene_write','Scene_update','Metrics_substep'):
            cost[k] += 1
        self.budget.reserve(**cost)
        index = self.macros
        macro_out = self.out/('macro-%02d' % index); macro_out.mkdir()
        self.gate.journal.checkpoint('macro-%02d-prepare' % index)
        started = time.monotonic()
        driver = evidence = None
        try:
            self.budget.run('Env_substep_counter', lambda: setattr(env, '_sim_step_counter', env._sim_step_counter+1))
            self.budget.run('Action_apply', env.action_manager.apply_action)
            self.budget.run('Scene_write', env.scene.write_data_to_sim)
            require(bool(self.torch.isfinite(env.sim.data.qpos).all() and self.torch.isfinite(env.sim.data.qvel).all()), 'Nonfinite macro input')
            require(bool((env.sim.data.xfrc_applied == 0).all() and (env.sim.data.qfrc_applied == 0).all()), 'Unexpected external assistance')
            roots, roster = roots_for(env, self.bank, self.torch, self.wp)
            save(macro_out/'owner-roster.json', roster)
            bridge = GroundPairStateBridge(env.sim, env.scene, roots, self.bank, self.torch, self.wp, self.budget)
            engine = NativeRK4Bridge(self.fw, self.mw, env.sim.wp_model, env.sim.wp_data, self.wp, self.budget)
            engine.native_step, engine.native_rk = self.raw_step, self.raw_rk
            engine.contact_order, engine.solver_order = self.contact, self.reductions.order
            self.contact.archive.sample_next = True
            self.reductions.order.sample_next = True
            evidence = InterfaceEvidence(macro_out/'raw', bridge)
            driver = EpisodeAdaptive(bridge, engine, (env._g11_tick,), evidence, self.max_episode_ticks)
            self.last_driver = driver
            result = driver.advance()
            evidence.finalize()
            require(driver.commits == 1 and result['replayed'], 'Missing accepted/replayed macro')
            env._g11_tick = result['public_ticks'][0]
            endpoint_fields = ('qpos','qvel','time','xpos','xquat','site_xpos','site_xmat','cvel','sensordata')
            self.last_endpoint = {n: digest(getattr(env.sim.wp_data,n).numpy()) for n in endpoint_fields}
            self.budget.run('Scene_update', lambda: env.scene.update(dt=env.physics_dt))
            self.budget.run('Metrics_substep', env.metrics_manager.compute_substep)
            require(self.last_endpoint == {n: digest(getattr(env.sim.wp_data,n).numpy()) for n in endpoint_fields},
                    'Public Scene/metrics altered accepted native endpoint')
            require(env._sim_step_counter == self.counter_origin+index+1, 'Duplicated public substep counter')
            measured_surface(env)
            self.macros += 1
            self.gate.journal.macros = self.macros
            result['elapsed_s'] = time.monotonic()-started
            save(macro_out/'macro-result.json', result, durable=True)
            self.results[:] = [result]  # Full prior results remain on disk; bounded RAM.
            self.gate.journal.checkpoint('macro-%02d-accepted' % index)
            return result
        except BaseException:
            self.stopped = True
            try:
                from integration.owner_inventory import record
                record(macro_out/'owner-inventory-failure.json',env,self.bank,self.torch,self.wp)
            except Exception as inventory_error:
                save(macro_out/'owner-inventory-error.json',dict(error=repr(inventory_error)))
            if evidence is not None and evidence.started and not evidence.closed and not evidence.stopped:
                try:
                    evidence.finalize()
                    save(macro_out/'partial-evidence.json', dict(complete_macro=False,
                        completed_microstep_rows_retained=True, no_retry=True))
                except BaseException as error:
                    save(macro_out/'partial-evidence-error.json', dict(error=repr(error), no_retry=True))
            if driver is not None:
                save(macro_out/'macro-failure.json', dict(
                    stopped=True, no_retry=True, failure=driver.last_failure,
                    decision_rejections=list(driver.rejections), declared_tiers=list(TIERS),
                    raw_evidence='raw/evidence-inventory.json', numerical_thresholds_changed=False,
                    driver_commits=driver.commits, private_phase=driver.phase, public_tick=env._g11_tick))
            raise


class EpisodeAdaptive(AdaptiveContinuation):
    """Episode scope for the separately declared v7 finite refinement candidate."""
    def __init__(self, bridge, engine, ticks, emit, episode_limit):
        require(type(ticks) is tuple and len(ticks) == 1 and type(ticks[0]) is int and
                0 <= ticks[0] < episode_limit <= 5000, 'Invalid declared episode tick')
        super().__init__(bridge, engine, (0,), emit)
        self.ticks = ticks
