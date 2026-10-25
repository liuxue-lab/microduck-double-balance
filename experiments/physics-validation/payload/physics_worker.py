"""Single real ManagerBasedRlEnv, two remaining bounded formal steps, no policy runner."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
import traceback
from types import MethodType

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE/'accepted'), str(HERE/'host-candidate')]
from runtime_accounting import require, save, Journal, make_classes, review_ledger


def verify_sources(repo):
    plan = json.loads((HERE/'accepted/plan.json').read_bytes())
    paths = {repo/n: h for n, h in plan['project_source_hashes'].items()}
    for distribution, entries in plan['installed_source_hashes'].items():
        dist = importlib.metadata.distribution(distribution)
        paths.update({Path(dist.locate_file(n)): h for n, h in entries.items()})
    for n, v in plan['versions'].items():
        require(importlib.metadata.version(n) == v, 'Version differs: '+n)
    for row in json.loads((HERE/'accepted/integration-source-pins.json').read_bytes())+[plan['new_source_pin']]:
        p = repo/row['relative_path'] if row['distribution'] == 'project' else Path(
            importlib.metadata.distribution(row['distribution']).locate_file(row['relative_path']))
        paths[p] = row['sha256']
    require(len(paths) == 219, 'Source roster differs')
    for p, h in paths.items():
        require(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == h, 'Pinned source differs: '+str(p))
    return {str(p): h for p, h in paths.items()}


def run(repo, out):
    plan = json.loads((HERE/'plan.json').read_bytes()); caps = plan['caps']
    counts = dict(integrations_attempted=0, integrations_completed=0, PPO=0, cloud=0)
    journal = Journal(out, counts, caps)
    report = dict(status='G11_STARTED', G11='OPEN', gate_completed=10, gate_total=11,
                  canonical_before=dict(completed=670872, attempted=670873),
                  real_environment_constructed=False, full_matrix_completed=False,
                  PPO=0, cloud_training=0, no_automatic_gate_closure=True)
    report['manager_counter_scope'] = 'Public real reset/step/close after construction; constructor shape probes/startup precede this manager trace; native CPU/GPU entry points counted before environment construction'
    gate = env = contact = reductions = resources = runtime = None
    trace = []; errors = []; ok = False
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
        from formal_owners import roots_for, install_public_rng
        from formal_runtime import Runtime, FormalContact, FormalReductions, BoundaryContactArchive
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
        require(torch.__version__.split('+')[0] == '2.9.1', 'Torch core differs')
        require(torch.cuda.is_available(), 'Local CUDA required')
        wp.config.kernel_cache_dir = str(out/'kernel-cache')
        device = wp.get_device('cuda:0')
        require(device.is_cuda and '5060' in device.name, 'Only local RTX5060 released')
        require(mujoco.get_mjcb_control() is None, 'CPU control callback present')
        wp.set_device(device)
        def sync():
            wp.synchronize(); torch.cuda.synchronize()
        Gate, Budget = make_classes()
        gate = Gate(counts, journal, sync, caps)
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
        report['checkpoint'] = reuse_checkpoint_review(out)
        cfg = make_cfg()
        report['configuration_preflight'] = preflight(cfg, out, UniformNoiseCfg)
        save(out/'semantic-config-before.json', dict(
            old_tray_config=config_manifest(make_microduck_double_balance_env_cfg(play=True)),
            target_head_config=config_manifest(cfg),
            checkpoint_training_config_reconstructed=False,
            comparison='Known task/feature/timing mismatch blocks direct resume; actual v1 checkpoint inspection reused'))
        env = envmod.ManagerBasedRlEnv(cfg, device='cuda:0', render_mode=None)
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
            out=out, world=0, order_limits=plan['solver_order_evidence'])
        contact = FormalContact(sim.wp_model, sim.wp_data, wp, gate, counts,
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
        report['reused_v2'] = json.loads((HERE/'nominal-rollout/audit.json').read_bytes())
        require(report['reused_v2']['canonical_after']['completed']==670872 and
                report['reused_v2']['normal_step_completed_scoped'], 'V2 reuse identity differs')
        bank = StreamBank(torch, gate, counts, out, caps, 0)
        report['rng_policy'] = install_public_rng(env, act, bank, gate, torch)
        save(out/'private-rng-policy.json', report['rng_policy'])
        budget = Budget({k: caps[k] for k in MACRO_LIMITS}, counts, journal, sync)
        runtime = Runtime(env, torch, wp, fw, mw, bank, gate, budget, out, contact, reductions, raw_step, raw_rk)
        step = build_step_candidate(vars(envmod), runtime.advance, derived_state_policy='accepted_endpoint')
        gate.patch(env, 'step', MethodType(step, env))
        native_reset = env.reset
        def reset(*, seed=None, env_ids=None, options=None):
            require(seed is None and options is None, 'Reseeding/custom reset options not released')
            ids = canonical_ids(env_ids, 1)
            if not ids: return env.obs_buf, env.extras
            return native_reset(env_ids=torch.tensor(ids, dtype=torch.long, device=env.device))
        gate.patch(env, 'reset', reset)

        def history():
            result = {}
            for group, buffers in env.observation_manager._group_obs_term_history_buffer.items():
                for name, b in buffers.items(): result[group+'/'+name] = b._num_pushes.cpu().tolist()
            for group, buffers in env.observation_manager._group_obs_term_delay_buffer.items():
                for name, b in buffers.items(): result['delay/'+group+'/'+name] = b._buffer._num_pushes.cpu().tolist()
            return result
        def scalar_state():
            return dict(sim_counter=env._sim_step_counter, control_counter=env.common_step_counter,
                episode=env.episode_length_buf.cpu().tolist(), tick=env._g11_tick,
                time=sim.data.time.cpu().tolist(), history=history(),
                bam_pushes=act._delay_buffer._buffer._num_pushes.cpu().tolist(),
                pending=env._manual_reset_pending.cpu().tolist(), events=env._g11_events.cpu().tolist())
        def boundary(label):
            sync(); roots, roster = roots_for(env, bank, torch, wp)
            snapshot = DataSnapshot(sim.wp_data, wp)
            arrays = {n: a.numpy().copy() for n, a in snapshot.arrays.items()}
            save_arrays(out, label+'-data', arrays)
            owners, scalars, topology = owner_state(roots, torch, wp)
            save_arrays(out, label+'-owners', owners, scalars=scalars, topology=topology)
            row = dict(state=scalar_state(), data_hashes=array_meta(arrays), owner_roster=roster,
                       surface=measured_surface(env)[2])
            save(out/(label+'.json'), row); return row
        def rng():
            return dict(cpu=digest(torch.get_rng_state().numpy()),
                        cuda=digest(torch.cuda.get_rng_state('cuda:0').cpu().numpy()))
        baseline_rng = rng()
        save(out/'default-rng-boundary.json', baseline_rng)
        # Trace the actual nine-manager calls, without replaying their stateful computations.
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
                original = getattr(obj, method)
                def observed(*args, _original=original, _label=label, **kwargs):
                    row = dict(sequence=len(trace)+1, method=_label, before=scalar_state(),
                               dt=kwargs.get('dt'), update_history=kwargs.get('update_history'))
                    trace.append(row)
                    if _label in ('Termination_compute','Reward_compute'):
                        require(runtime.macros % 10 == 0 and runtime.macros > 0, 'Manager entered before accepted tenth macro')
                        require(bool(torch.isfinite(sim.data.qpos).all() and torch.isfinite(sim.data.qvel).all()), 'Raw endpoint nonfinite')
                        require(runtime.last_endpoint == {n: digest(getattr(sim.wp_data,n).numpy()) for n in runtime.last_endpoint},
                                'Reward/termination no longer reads accepted endpoint')
                        row['accepted_derived_endpoint_hashes'] = dict(runtime.last_endpoint)
                        row['endpoint_qpos'] = sim.data.qpos.cpu().tolist()
                        row['endpoint_surface'] = measured_surface(env)[2]
                    result = _original(*args, **kwargs)
                    row['after'] = scalar_state(); row['completed'] = True
                    if _label == 'Reward_compute':
                        require(kwargs['dt'] == .020 and bool(torch.isfinite(result).all()), 'Invalid reward/cadence')
                        row['reward'] = result.cpu().tolist()
                        row['weighted_terms'] = env.reward_manager._step_reward.cpu().tolist()
                    return result
                gate.patch(obj, method, gate.limited(label, caps[label], observed))

        journal.checkpoint('real-reset-and-owner-preflight')
        env.reset()
        require(rng() == baseline_rng, 'Default RNG advanced during real reset')
        require({k: tuple(v.shape) for k, v in env.obs_buf.items()} == {'actor': (1,61), 'critic': (1,85)}, 'Formal observation layout differs')
        require(all(bool(torch.isfinite(v).all()) for v in env.obs_buf.values()), 'Nonfinite initial observations')
        # Resolve only the source-owned site lookup, before any new integration.
        from mjlab_microduck.tasks import mdp
        mdp._double_balance_tray_site_id(env, tray_site_name='stage11_head_material_frame')
        roots, roster = roots_for(env, bank, torch, wp)
        owners = StrictOwnerSnapshot(roots, torch, wp); owners.assert_restored()
        save(out/'owner-preflight.json', dict(manifest=owners.manifest(), roster=roster))
        save(out/'formal-task-startup.json', env._g11_startup_report)
        save_arrays(out, 'actual-head-surface', dict(triangles=env._g11_surface.triangles,
            spawn=env._g11_surface.spawn, witness=env._g11_surface.witness, normal=env._g11_surface.normal))
        save(out/'resolved-target-semantics.json', config_manifest(env.cfg, env))
        guard_counts=dict(counts)
        data_guard=DataSnapshot(sim.wp_data,wp)
        owner_guard=StrictOwnerSnapshot(roots,torch,wp)
        report['tuple_cache_preflight']=preflight_cache(env,torch,wp,mdp)
        data_guard.assert_restored();owner_guard.assert_restored()
        require(counts==guard_counts and rng()==baseline_rng,
                'Owner preflight changed native/manager counters or default RNG')
        save(out/'tuple-cache-preflight.json',report['tuple_cache_preflight'])
        report['reset_checks_reused']=report['reused_v2']['reset_checks']
        report['fresh_initial_reset_this_batch']=True
        report['continuation_scope']='Fresh isolated reset; not bitwise recovery of missing v2 post-step owner state'
        boundary('initial-reset')
        baseline_events = env._g11_events.clone()
        recorder = env.recorder_manager._terms[0]
        for index, case in enumerate(('auto-timeout','manual-failure')):
            journal.checkpoint('formal-step-'+case)
            env.cfg.auto_reset = case == 'auto-timeout'
            if case == 'auto-timeout': env.episode_length_buf[:] = env.max_episode_length-1
            if case == 'manual-failure': env._g11_forced_failure[:] = True
            action = torch.zeros((1,14), device=env.device)
            action[:] = torch.linspace(-.001, .001, 14, device=env.device)*(1 if case == 'auto-timeout' else -1)
            pre = boundary('step-%d-before'%index)
            obs_before = {k: v.clone() for k, v in env.obs_buf.items()}
            recorder_start = len(recorder.entries); trace_start = len(trace); counts_before = dict(counts)
            with torch.no_grad(): obs, reward, terminated, timed_out, extras = env.step(action)
            require(runtime.macros == (index+1)*10 and env._sim_step_counter == (index+1)*10, 'Formal decimation/counter differs')
            require(all(bool(torch.isfinite(v).all()) for v in obs.values()) and bool(torch.isfinite(reward).all()), 'Nonfinite formal outputs')
            require(rng() == baseline_rng, 'Default RNG advanced across formal step')
            require(counts['Action_apply_completed']-counts_before.get('Action_apply_completed',0) == 10 and
                    counts['Metrics_substep_completed']-counts_before.get('Metrics_substep_completed',0) == 10,
                    'Duplicate/missing public substep application')
            extra_reset = int(case == 'auto-timeout')
            require(counts['BAM_compute_completed']-counts_before.get('BAM_compute_completed',0) == 10+extra_reset and
                    counts['BAM_apply_delay_completed']-counts_before.get('BAM_apply_delay_completed',0) == 10+extra_reset,
                    'BAM must advance once per real Scene.write, including automatic reset')
            if case == 'auto-timeout':
                require(bool(timed_out.all()) and not bool(terminated.any()) and int(env.episode_length_buf[0]) == 0,
                        'Auto timeout/reset contract differs')
                hooks = recorder.entries[recorder_start:]
                require([e['hook'] for e in hooks] == ['pre_reset','post_reset','post_step'], 'Auto recorder order differs')
                require(torch.equal(hooks[0]['action'], action) and
                        all(torch.equal(hooks[0]['observations'][k], obs_before[k]) for k in obs_before), 'Pre-reset terminal input/action lost')
                require(bool((hooks[1]['action'] == 0).all()), 'Post-reset action not zero')
            else:
                require(bool(terminated.all()) and not bool(timed_out.any()) and bool(env._manual_reset_pending.all()), 'Manual termination contract differs')
                before_pending = dict(counts); action_pending = env.action_manager.action.clone()
                try: env.step(action)
                except RuntimeError as exc: require('must be reset via' in str(exc), 'Wrong pending-reset rejection')
                else: raise RuntimeError('Pending manual environment stepped twice')
                require(counts == before_pending and torch.equal(action_pending, env.action_manager.action), 'Pending rejection mutated state')
            history_after = history()
            for key, before in pre['state']['history'].items():
                require(history_after[key] == [1] if extra_reset else history_after[key] == [before[0]+1],
                        'Observation history advanced outside public compute: '+key)
            require(torch.equal(env._g11_events-baseline_events, torch.full_like(baseline_events,index+1)), 'Events escaped public once-per-step boundary')
            row = dict(case=case, reward=reward.cpu().tolist(), terminated=terminated.cpu().tolist(),
                timed_out=timed_out.cpu().tolist(), actual_calls={k: v-counts_before.get(k,0) for k,v in counts.items()},
                trace=trace[trace_start:], state=scalar_state(), observation_history_checked=True,
                terminal_fault_is_test_injection=case=='manual-failure', numerical_thresholds_changed=False)
            require(sum(r['method']=='Reward_compute' for r in row['trace']) == 1 and
                    sum(r['method']=='Termination_compute' for r in row['trace']) == 1 and
                    sum(r['method']=='Observation_compute' for r in row['trace']) == 1,
                    'Public reward/termination/observation computed more than once')
            save(out/('formal-step-%d.json'%index), row, durable=True)
            save_arrays(out, 'formal-step-%d-outputs'%index,
                        {**{k: v.cpu().numpy() for k,v in obs.items()}, 'reward': reward.cpu().numpy()})
            boundary('step-%d-after'%index)
            journal.formal_steps += 1; journal.checkpoint('formal-step-'+case+'-completed')
        env.reset(env_ids=[0])
        require(not bool(env._manual_reset_pending.any()) and rng() == baseline_rng, 'Explicit manual reset failed')
        require(counts.get('Observation_noise_completed',0)==4*counts.get('Observation_compute_completed',0),
                'Noise must execute once per original noisy term per public observation compute')
        report['noise_execution'] = dict(calls=counts['Observation_noise_completed'],
            observation_computes=counts['Observation_compute_completed'], original_terms_per_compute=4,
            default_RNG_unchanged=True, cache_owners_captured=True)
        boundary('final-reset')
        report['remaining_matrix_completed'] = True
        report['full_matrix_completed'] = False
        report['combined_coverage_review_required'] = dict(normal='reused v2 scoped result; post-step full owner boundary absent', automatic_timeout='v3 measured', manual_failure='v3 measured', contiguous_three_step_run=False)
        report['real_manager_classes'] = {n: type(getattr(env,n+'_manager')).__name__ for n in roster_names()}
        report['null_curriculum_is_native_nominal_config'] = True
        report['physical_scope'] = 'One fresh native world, two remaining 20 ms control steps, plus reused v2 normal result; no long rollout or multiworld formal regression'
        ok = True
    except BaseException as exc:
        errors.append(dict(stage='worker', error=repr(exc), traceback=traceback.format_exc()))
    finally:
        for label, operation in (
            ('env_close', (lambda: env.close()) if env is not None else None),
            ('contact_close', (lambda: contact.close()) if contact is not None else None),
            ('resource_close', (lambda: resources.close(env.scene.sensor_context, env.sim)) if resources is not None and env is not None else None)):
            if operation:
                try: report[label] = operation()
                except BaseException as exc:
                    errors.append(dict(stage=label, error=repr(exc), traceback=traceback.format_exc())); ok=False
        if contact is not None: report['contact'] = contact.summary()
        if reductions is not None: report['reductions'] = reductions.summary()
        if gate is not None:
            save(out/'operation-profile.json', dict(timings=gate.timings, inclusive_nested=True,
                prior_runtime_seconds=5285, cross_run_speedup_claimed=False))
            gate.close()
        try:
            after = verify_sources(repo); save(out/'source-pins-after.json', after)
            require(after == pins, 'Sources changed during run')
        except BaseException as exc:
            errors.append(dict(stage='source_exit', error=repr(exc))); ok=False
        report.update(status='G11_REMAINING_MATRIX_MEASURED_REVIEW_REQUIRED' if ok else 'G11_STOPPED_PRESERVE_NO_RETRY',
            counts=counts, errors=errors, elapsed_s=time.monotonic()-journal.start,
            macro_completed=0 if runtime is None else runtime.macros,
            cumulative_completed=670872+counts.get('integrations_completed',0),
            cumulative_attempted=670873+counts.get('integrations_attempted',0),
            no_automatic_retry=True)
        save(out/'manager-trace.json', trace)
        journal.close()
        report['native_ledger'] = review_ledger(out/'native-ledger.jsonl')
        save(out/'report.json', report, durable=True)
    print('Stage11G11Result='+report['status'], flush=True)
    print('Stage11G11Report='+str(out/'report.json'), flush=True)
    return 0 if ok else 2


def roster_names():
    return ('action','observation','event','reward','termination','metrics','command','curriculum','recorder')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.repo, args.out))
