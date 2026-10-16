"""Native owner factory candidate, for explicit future bounded measurement only.

Executed only by the new local campaign after explicit native-limit admission.
Source/setup arithmetic is retained from frozen G11;
new lifecycle/clock/reset changes require native acceptance. No training release.
"""
from pathlib import Path
from contextlib import contextmanager
import hashlib
import json
import os
import sys
import time
import traceback
from types import MethodType
from integration.durable import require, write_json, file_digest
from integration.planning import validate_owner, fingerprint

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'baseline/g11'
_ACTIVE = False


def admission(plan, authorization):
    validate_owner(plan)
    require(authorization.get('scope') == 'STAGE12_CLOUD_5090_EXECUTION_V1', 'Cloud scope token missing')
    require(authorization.get('authorization') == 'USER_AUTHORIZED_NATIVE_MEASUREMENT' and
            authorization.get('owner_plan_sha256') == fingerprint(plan) and
            authorization.get('PPO') == 0 and authorization.get('cloud_training') == 0,
            'Native budget remains zero until the complete owner plan is explicitly authorized')
    require(authorization.get('canonical_before') == {'completed': 704664, 'attempted': 704665},
            'Fresh ledger reconciliation is required before a later native batch')


@contextmanager
def owner_session(repo, out, plan, authorization):
    global _ACTIVE
    admission(plan, authorization)  # Before imports, files, models, CUDA or native calls.
    require(not _ACTIVE, 'One native owner per process; global patches cannot be shared')
    _ACTIVE = True
    repo, out = Path(repo), Path(out)
    require(not out.exists(), 'Existing native output/started state is never reused')
    out.mkdir(parents=False)
    write_json(out / 'started-v1.json', dict(plan=plan, authorization=authorization, no_retry=True))
    sys.path[:0] = [str(HERE), str(HERE/'accepted'), str(HERE/'host-candidate')]
    from worker_g11 import verify_sources
    from runtime_accounting import save, make_classes, review_ledger
    from integration.native_accounting import SessionJournal
    caps = plan['caps']
    counts = dict(integrations_attempted=0, integrations_completed=0, PPO=0, cloud=0)
    journal = SessionJournal(out, counts, caps, plan)
    report = dict(status='STAGE12_NATIVE_CANDIDATE_STARTED', training_ready=False,
                  canonical_before={'completed': 704664, 'attempted': 704665},
                  PPO=0, cloud_training=0, physical_acceptance=False)
    gate = env = contact = reductions = resources = runtime = None
    env_initialized = False
    pins = None
    errors = []
    try:
        pins = verify_sources(repo); save(out/'source-pins-before.json', pins)
        sys.path.insert(0, str(repo/'src'))
        import numpy as np
        import torch
        import warp as wp
        import mujoco
        import mujoco_warp as mw
        from mujoco_warp._src import forward as fw, io as mwio, solver
        from mjlab.envs import manager_based_rl_env as envmod
        from mjlab.scene import Scene
        from mjlab.sim import Simulation
        from mjlab.sensor import raycast_sensor as raymod
        from rsl_rl.algorithms.ppo import PPO
        from formal_task import make_cfg, measured_surface
        from integration.owner_roster import roots_for
        from integration.rng_policy import install_public_rng
        from integration.native_runtime import BudgetedRuntime, FormalContact, FormalReductions, BoundaryContactArchive
        from checkpoint_review import reuse_checkpoint_review, config_manifest
        from formal_config import preflight
        from mjlab.utils.noise import UniformNoiseCfg
        from source_contracts import build_step_candidate, patched_event_reset
        from isolation_policy import StreamBank, canonical_ids
        from native_state_bridge import model_tree
        from formal_owner_keys import FormalOwnerSnapshot as StrictOwnerSnapshot, owner_state, install as install_owner_keys, preflight_cache
        from sensor_state_snapshot import DataSnapshot, digest
        from scene_evidence import save_arrays, array_meta
        from macro_state_bridge import MACRO_LIMITS
        from resource_lifetime import Resources
        from support_material_contract import candidate_solref
        from mjlab.managers import event_manager as eventmod
        from mjlab_microduck.tasks.microduck_double_balance_env_cfg import make_microduck_double_balance_env_cfg
        from mjlab_microduck.actuator.friction_dr_bam import FrictionDRBamActuator
        from integration.manager_contracts import preflight as manager_preflight
        report['manager_interface_preflight'] = manager_preflight(out)
        require(torch.__version__.split('+')[0] == '2.9.1', 'Torch core differs')
        require(torch.cuda.is_available(), 'Local CUDA required')
        from integration.runtime_paths import for_owner
        paths = for_owner(out)
        wp.config.kernel_cache_dir = paths['warp_cache']
        report['runtime_paths'] = paths
        device = wp.get_device('cuda:0')
        require(device.is_cuda and '5090' in device.name, 'Only the inventoried RTX5090 cloud candidate is supported')
        require(mujoco.get_mjcb_control() is None, 'CPU control callback present')
        wp.set_device(device)
        def sync():
            wp.synchronize(); torch.cuda.synchronize()
        Gate, Budget = make_classes()
        gate = Gate(counts, journal, sync, caps)
        for obj, name in ((torch, 'load'), (torch, 'save'), (torch.jit, 'load')):
            gate.patch(obj, name, gate.blocked('checkpoint.'+name))
        install_owner_keys(gate)
        raw_step, raw_rk = fw.step, fw.rungekutta4
        graph_apis = {n: getattr(wp, n) for n in ('capture_begin','capture_end','capture_launch')}
        for obj, names in (
            (mujoco, ('mj_step','mj_step1','mj_step2','mj_Euler','mj_RungeKutta','mj_implicit','mj_geomDistance')),
            (mw, ('step','step1','step2','euler','implicit','rungekutta4','render')),
            (fw, ('step','step1','step2','euler','implicit','rungekutta4')),
            (wp, ('ScopedCapture','capture_begin','capture_end','capture_launch')),
            (Simulation, ('step',)), (PPO, ('__init__',))):
            for name in names:
                if hasattr(obj, name):
                    gate.patch(obj, name, gate.blocked(type(obj).__name__+'.'+name))
        def limit(obj, name, label):
            gate.patch(obj, name, gate.limited(label, caps[label], getattr(obj, name)))
        limit(mujoco, 'mj_forward', 'CPU_forward')
        limit(mujoco, 'mj_kinematics', 'CPU_kinematics')
        forward = gate.limited('GPU_forward', caps['GPU_forward'], fw.forward)
        gate.patch(mw, 'forward', forward); gate.patch(fw, 'forward', forward)
        reset_native = gate.limited('GPU_reset_data', caps['GPU_reset_data'], mwio.reset_data)
        gate.patch(mw, 'reset_data', reset_native); gate.patch(mwio, 'reset_data', reset_native)
        for obj, name, label in (
            (Simulation,'reset','Simulation_reset'), (Scene,'initialize','Scene_initialize'),
            (Scene,'write_data_to_sim','Scene_write'), (Scene,'reset','Scene_reset'),
            (Simulation,'expand_model_fields','Model_expansion'),
            (Simulation,'set_sensor_context','Sensor_context_set'),
            (Simulation,'sense','sense'), (mw,'refit_bvh','BVH_refit'),
            (mw,'create_render_context','context_create'), (raymod,'rays','ray_batch')):
            limit(obj, name, label)
        def scene_dt(args, kwargs):
            require((args[1] if len(args)>1 else kwargs.get('dt')) in (0., .002), 'Unexpected Scene timing')
        gate.patch(Scene, 'update', gate.limited('Scene_update', caps['Scene_update'], Scene.update, scene_dt))
        class EagerSimulation(Simulation):
            def _should_use_cuda_graph(self):
                return False
            def create_graph(self):
                counts['graph_setup_suppressed'] = counts.get('graph_setup_suppressed', 0)+1
                self.step_graph = self.forward_graph = self.reset_graph = self.sense_graph = None
        gate.patch(envmod, 'Simulation', EagerSimulation)
        report['checkpoint'] = dict(old_checkpoint_loads=0, mode='fresh-proposal-only')
        cfg = make_cfg()
        report['configuration_preflight'] = preflight(cfg, out, UniformNoiseCfg)
        for name in ('g11_step_probe', 'g11_local_probe', 'g11_class_probe', 'g11_global_probe'):
            del cfg.events[name]
        cfg.recorders = {}
        del cfg.terminations['g11_fault_injection']
        require(cfg.auto_reset is False, 'Terminal bridge owns reset; native auto_reset must be false')
        save(out/'semantic-config-before.json', dict(
            old_tray_config=config_manifest(make_microduck_double_balance_env_cfg(play=True)),
            target_head_config=config_manifest(cfg),
            checkpoint_training_config_reconstructed=False,
            comparison='Known task/feature/timing mismatch blocks direct resume; actual v1 checkpoint inspection reused'))
        env = envmod.ManagerBasedRlEnv.__new__(envmod.ManagerBasedRlEnv)
        envmod.ManagerBasedRlEnv.__init__(env, cfg, device='cuda:0', render_mode=None)
        env_initialized = True
        report['real_environment_constructed'] = True
        counts['full_RL_environment_constructed'] = 1
        require(counts.get('integrations_attempted', 0) == 0 and counts.get('GPU_forward_attempted', 0) == 0,
                'Unexpected constructor GPU physics')
        sim, scene = env.sim, env.scene
        require(all(getattr(sim.wp_model.callback, n) is None for n in
                    ('passive','control','act_dyn','act_gain','act_bias','sensor','contactfilter')), 'GPU callback present')
        require(len(scene.sensors) == 14 and not sim.use_cuda_graph, 'Sensor/eager scope differs')
        ctx = scene.sensor_context
        require(ctx is sim._sensor_context and not ctx.has_cameras and len(ctx.raycast_sensors) == 1,
                'Only non-camera single-world context released')
        resources = Resources(wp, ctx, out, dict(pid=os.getpid(), scope='final environment context only'))
        report['resource_scope'] = 'Final context only; constructor transient context contents are not cloned'
        import rigid_metadata_kernel
        gate.limited('Rigid_module_load', 1, lambda: wp.load_module(rigid_metadata_kernel, device=device))()
        reductions = FormalReductions(sim.wp_model, sim.wp_data, wp, solver, gate, caps,
            out=out, world=plan['logical_world'], order_limits=plan['solver_order_evidence'])
        class PlannedContact(FormalContact):
            CALL_CAP = caps['GPU_forward']
        contact = PlannedContact(sim.wp_model, sim.wp_data, wp, gate, counts,
            plan['contact_layout'], caps['GPU_forward'], plan['graph_limits'], graph_apis, out)
        contact.configure(BoundaryContactArchive(out), rigid_metadata_kernel.rigid_metadata)
        contact.prepare(); reductions.install()
        gate.patch(fw.collision_driver, 'collision', contact.wrap(fw.collision_driver.collision))
        original_material = sim.wp_model.geom_solref.numpy().copy()
        require(digest(original_material) == plan['original_solref_sha256'], 'Original material differs')
        gate.limited('Model_material_setup', 1, lambda: sim.wp_model.geom_solref.assign(candidate_solref(original_material, .0025)))()
        save_arrays(out, 'material', dict(before=original_material, after=sim.wp_model.geom_solref.numpy()))
        act = scene['robot'].actuators[0]
        require(len(scene['robot'].actuators) == 1 and type(act) is FrictionDRBamActuator, 'BAM roster differs')
        require(act._dt == .002 and act._delay_buffer.min_lag == 3 and act._delay_buffer.max_lag == 6,
                'BAM cadence/delay differs')
        limit(act, 'apply_delay', 'BAM_apply_delay'); limit(act, 'compute', 'BAM_compute')
        em = env.event_manager
        bank = StreamBank(torch, gate, counts, out, caps, plan['logical_world'])
        report['rng_policy'] = install_public_rng(env, act, bank, gate, torch)
        save(out/'private-rng-policy.json', report['rng_policy'])
        budget = Budget({k: caps[k] for k in MACRO_LIMITS}, counts, journal, sync)
        runtime = BudgetedRuntime(env, torch, wp, fw, mw, bank, gate, budget, out, contact, reductions, raw_step, raw_rk,
            max_macros=plan['max_macros'], max_episode_ticks=plan['max_episode_ticks'])
        step = build_step_candidate(vars(envmod), runtime.advance, derived_state_policy='accepted_endpoint')
        gate.patch(env, 'step', MethodType(step, env))
        native_reset = env.reset
        def reset(*, seed=None, env_ids=None, options=None):
            require(seed is None and options is None, 'Reseeding/custom reset options not released')
            ids = canonical_ids(env_ids, 1)
            if not ids: return env.obs_buf, env.extras
            return native_reset(env_ids=torch.tensor(ids, dtype=torch.long, device=env.device))
        gate.patch(env, 'reset', reset)

        # Count all public manager entry points, including explicit terminal resets.
        roster = {
            'action': [('process_action','Action_process_action'), ('apply_action','Action_apply'), ('reset','Action_reset')],
            'observation': [('compute','Observation_compute'), ('reset','Observation_reset')],
            'event': [('apply','Event_apply'), ('reset','Event_reset')],
            'reward': [('compute','Reward_compute'), ('reset','Reward_reset')],
            'termination': [('compute','Termination_compute'), ('reset','Termination_reset')],
            'metrics': [('compute','Metrics_compute'), ('compute_substep','Metrics_substep'), ('reset','Metrics_reset')],
            'command': [('compute','Command_compute'), ('reset','Command_reset')],
            'curriculum': [('compute','Curriculum_compute'), ('reset','Curriculum_reset')],
            'recorder': [('record_pre_reset','Recorder_pre_reset'), ('record_post_reset','Recorder_post_reset'),
                         ('record_post_step','Recorder_post_step'), ('close','Recorder_close')]}
        for manager, entries in roster.items():
            obj = getattr(env, manager+'_manager')
            for method, label in entries:
                limit(obj, method, label)
        from tensordict import TensorDict
        from integration.transitions import TerminalBridge
        bridge = TerminalBridge(env, torch, TensorDict, plan['max_resets'])
        from mjlab_microduck.tasks import mdp
        mdp._double_balance_tray_site_id(env, tray_site_name='stage11_head_material_frame')
        from integration.owner_inventory import record as owner_inventory
        report['owner_inventory'] = owner_inventory(out/'owner-inventory.json',env,bank,torch,wp)
        roots, roster = roots_for(env, bank, torch, wp)
        snapshot = StrictOwnerSnapshot(roots, torch, wp)
        snapshot.assert_restored()
        save(out/'owner-preflight.json', dict(manifest=snapshot.manifest(), roster=roster))
        torch.cuda.reset_peak_memory_stats()
        def inspect():
            from integration.process_vector import tensor_wire
            from integration.durable import encoded, digest as byte_digest
            sync()
            free, total = torch.cuda.mem_get_info()
            rng = {str(k): v.cpu().tolist() for k, v in bank.states.items()}
            return dict(logical_world=plan['logical_world'], public_steps=journal.formal_steps,
                        macros=runtime.macros, resets=bridge.resets, counts=dict(counts),
                        rng_sha256=byte_digest(encoded(rng)), rng_draw_counts=dict(bank.draw_counts),
                        observation_sha256=byte_digest(encoded(tensor_wire(bridge.get_observations()))),
                        episode_length=env.episode_length_buf.cpu().tolist(),
                        memory=dict(torch_peak_allocated=torch.cuda.max_memory_allocated(),
                                    torch_peak_reserved=torch.cuda.max_memory_reserved(),
                                    driver_free=free, driver_total=total,
                                    scope='per-owner Torch peak; simultaneous driver snapshot includes all processes, not a driver peak'))
        journal.checkpoint('candidate-owner-ready')
        yield dict(env=env, bridge=bridge, runtime=runtime, counts=counts, journal=journal,
                   torch=torch, wp=wp, bank=bank, gate=gate, inspect=inspect)
        report['status'] = 'STAGE12_NATIVE_CANDIDATE_FINISHED_REVIEW_REQUIRED'
    except BaseException as exc:
        errors.append(dict(stage='session', error=repr(exc), traceback=traceback.format_exc()))
        raise
    finally:
        from integration.environment_cleanup import close_environment
        for label, operation in (
            ('env_close', (lambda: close_environment(env,env_initialized)) if env is not None else None),
            ('contact_close', (lambda: contact.close()) if contact is not None else None),
            ('resource_close', (lambda: resources.close(env.scene.sensor_context, env.sim))
                if resources is not None and env is not None else None)):
            if operation:
                try:
                    report[label] = operation()
                except BaseException as exc:
                    errors.append(dict(stage=label, error=repr(exc)))
        # Restore destructor hooks even when witness verification fails.
        if resources is not None:
            resources.unpatch()
        if contact is not None:
            report['contact'] = contact.summary()
        if reductions is not None:
            report['reductions'] = reductions.summary()
        if gate is not None:
            save(out/'operation-profile.json', dict(timings=gate.timings, inclusive_nested=True))
            gate.close()
        try:
            after = verify_sources(repo)
            require(pins is not None and after == pins, 'Source identity changed/initial verification incomplete')
            save(out/'source-pins-after.json', after)
        except BaseException as exc:
            errors.append(dict(stage='source_exit', error=repr(exc)))
        report.update(counts=counts, errors=errors, elapsed_s=time.monotonic()-journal.start,
                      macro_completed=0 if runtime is None else runtime.macros,
                      public_steps_completed=journal.formal_steps,
                      native_added_completed=counts.get('integrations_completed',0),
                      native_added_attempted=counts.get('integrations_attempted',0),
                      aggregate_canonical_not_recomputed_per_owner=True, no_automatic_retry=True)
        if errors:
            report['status'] = 'STAGE12_NATIVE_CANDIDATE_STOPPED_PRESERVE'
        journal.close()
        report['native_ledger'] = review_ledger(out/'native-ledger.jsonl')
        save(out/'native-report.json', report, durable=True)
        # Process is intentionally not reusable even after orderly cleanup.
        if errors and sys.exc_info()[0] is None:
            raise RuntimeError('Native cleanup/identity failed; preserve all evidence')
