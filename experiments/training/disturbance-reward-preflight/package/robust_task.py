"""Reward-only overlay; pulse physics and episode curricula have separate owners."""
import ast
import hashlib
import inspect
import textwrap

from robust_costs import (costs, joint_importance, SERVO_JOINTS, OTHER_JOINTS,
                          MARGIN_RAD, POSITION_WEIGHT, OUTWARD_WEIGHT,
                          FAILURE_MAGNITUDE, FAILURE_WEIGHT)

COST_WEIGHTS = dict(other_joint_margin=POSITION_WEIGHT,
                    other_joint_outward=OUTWARD_WEIGHT,
                    physical_failure=FAILURE_WEIGHT)


def _measured_other_joints(env):
    import torch
    robot = env.scene['robot']
    cache = getattr(env, '_s12_robust_joint_mapping', None)
    if cache is None or cache['robot'] is not robot:
        names = tuple(robot.joint_names)
        if len(names) != 14 or set(names) != set(SERVO_JOINTS):
            raise RuntimeError('Final fine-tune requires all fourteen original named servos')
        local = [names.index(name) for name in OTHER_JOINTS]
        global_ids = robot.indexing.joint_ids.detach().cpu().tolist()
        if len(global_ids) != len(names) or len(set(global_ids)) != len(global_ids):
            raise RuntimeError('Original named joint mapping differs')
        model = env.sim.mj_model
        gids = [global_ids[index] for index in local]
        if not all(int(model.jnt_type[index]) == 3 for index in gids):
            raise RuntimeError('Expected the original scalar hinge joints')
        device = env.sim.data.qpos.device
        qids = torch.tensor([int(model.jnt_qposadr[index]) for index in gids],
                            dtype=torch.long, device=device)
        vids = torch.tensor([int(model.jnt_dofadr[index]) for index in gids],
                            dtype=torch.long, device=device)
        local_ids = torch.tensor(local, dtype=torch.long, device=device)
        bounds = robot.data.joint_pos_limits[:, local_ids]
        if not bool((torch.isfinite(bounds).all()
                     & ((bounds[..., 1]-bounds[..., 0]) > 2*MARGIN_RAD).all())):
            raise RuntimeError('Nonfinite or overlapping nominal joint proximity bands')
        cache = dict(robot=robot, local_ids=local_ids, qids=qids, vids=vids,
                     importance=torch.tensor(joint_importance(OTHER_JOINTS),
                            dtype=env.sim.data.qpos.dtype, device=device))
        env._s12_robust_joint_mapping = cache
    # Direct state reads avoid depending on an entity-data cache after integration.
    q = env.sim.data.qpos[:, cache['qids']]
    velocity = env.sim.data.qvel[:, cache['vids']]
    bounds = robot.data.joint_pos_limits[:, cache['local_ids']]
    if bounds.shape[0] == 1 and q.shape[0] != 1:
        bounds = bounds.expand(q.shape[0], -1, -1)
    return q, velocity, bounds, cache['importance']


def other_joint_margin_cost(env):
    return costs(*_measured_other_joints(env))[0]


def other_joint_outward_cost(env):
    return costs(*_measured_other_joints(env))[1]


def physical_failure_cost(env):
    import torch
    # mjlab computes termination terms before rewards and auto-reset. The caller
    # verifies that installed ordering before the first PPO update.
    manager = env.termination_manager
    lost = manager.get_term('top_ball_lost')
    fell = manager.get_term('fell')
    if lost.dtype is not torch.bool or fell.dtype is not torch.bool or lost.shape != fell.shape:
        raise RuntimeError('Physical termination terms must be matching boolean vectors')
    # One penalty even if both physical failures occur together; never time_out.
    return (lost | fell).to(dtype=torch.float32)*FAILURE_MAGNITUDE


def reward_costs(env):
    position, outward = costs(*_measured_other_joints(env))
    return dict(other_joint_margin=position, other_joint_outward=outward,
                physical_failure=physical_failure_cost(env))


def validate_cost_signs(env):
    import torch
    from headroom_task import head_roll_position_cost, head_roll_outward_cost
    from headroom_costs import POSITION_WEIGHT as head_position_weight
    from headroom_costs import OUTWARD_WEIGHT as head_outward_weight
    weighted = {name: value*COST_WEIGHTS[name] for name, value in reward_costs(env).items()}
    weighted['head_roll_margin'] = head_position_weight*head_roll_position_cost(env)
    weighted['head_roll_outward'] = head_outward_weight*head_roll_outward_cost(env)
    checks = [test for value in weighted.values()
              for test in (torch.isfinite(value).all(), (value <= 0.).all())]
    if not bool(torch.stack(checks).all()):
        raise RuntimeError('Final fine-tune cost became nonfinite or a positive reward')
    return weighted


def make_cfg(num_envs=4096, seed=2026102601):
    from mjlab.managers import RewardTermCfg
    from headroom_task import make_cfg as headroom_make_cfg
    cfg = headroom_make_cfg(num_envs, seed)
    for name in ('top_ball_lost', 'fell'):
        if name not in cfg.terminations or cfg.terminations[name].time_out:
            raise RuntimeError('Missing original non-timeout physical termination: '+name)
    functions = dict(other_joint_margin=other_joint_margin_cost,
                     other_joint_outward=other_joint_outward_cost,
                     physical_failure=physical_failure_cost)
    for name, function in functions.items():
        if name in cfg.rewards:
            raise RuntimeError('Refuse to overwrite an existing reward: '+name)
        cfg.rewards[name] = RewardTermCfg(func=function, weight=COST_WEIGHTS[name])
    return cfg


def verify_runtime_reward_scaling(manager, control_dt):
    """Exercise the installed method on isolated CPU buffers, never the live env.

    mjlab 1.3.0 uses ``scale = dt if self._scale_by_dt else 1.0``.
    Check numerical behavior rather than guessing multiplication variable names.
    """
    import torch
    from types import SimpleNamespace
    if abs(float(control_dt)-.02) > 1e-12 or getattr(manager, '_scale_by_dt', None) is not True:
        raise RuntimeError('Installed reward scaling must enable weight * dt at 50Hz')
    compute = inspect.unwrap(type(manager).compute)
    rates = (torch.tensor([100., 0., 100.]), torch.tensor([0., 4., -2.]))
    terms = [SimpleNamespace(func=lambda env, value=value:value.clone(),
                             weight=weight, params={})
             for value, weight in zip(rates, (-1., .5))]
    terms.append(SimpleNamespace(func=lambda env:torch.full((3,), 999.), weight=0., params={}))
    rate = -rates[0]+.5*rates[1]
    observed = []
    for dt in (float(control_dt), 2.*float(control_dt)):
        probe = SimpleNamespace(_scale_by_dt=manager._scale_by_dt, _env=object(),
            _term_names=['failure', 'signed_probe', 'zero_weight'], _term_cfgs=terms,
            _reward_buf=torch.full((3,), 37.), _step_reward=torch.full((3, 3), 37.),
            _episode_sums={name:torch.zeros(3) for name in ('failure', 'signed_probe', 'zero_weight')})
        with torch.inference_mode():
            actual = compute(probe, dt).clone()
            repeated = compute(probe, dt).clone()
        if not (torch.allclose(actual, rate*dt, rtol=1e-6, atol=1e-7)
                and torch.equal(actual, repeated)
                and torch.allclose(probe._episode_sums['failure'], -rates[0]*dt*2)
                and torch.allclose(probe._step_reward[:, 0], -rates[0])
                and bool((probe._step_reward[:, 2] == 0.).all())):
            raise RuntimeError('Installed reward scaling differs from weight * dt at 50Hz')
        observed.append(dict(dt_seconds=dt, rewards=actual.tolist()))
    return dict(method='isolated_CPU_installed_compute', live_manager_modified=False,
                physics_steps=0, PPO_updates=0, scale_by_dt=True, cases=observed)


def verify_runtime_reward_order(raw):
    """Read installed code before learning; record the terminal-cost dt contract.

    This deliberately rejects a changed runtime instead of silently applying a
    per-second failure weight before terminations or after auto-reset.
    """
    step_source = textwrap.dedent(inspect.getsource(inspect.unwrap(type(raw).step)))
    compute_source = textwrap.dedent(inspect.getsource(
        inspect.unwrap(type(raw.reward_manager).compute)))
    tree = ast.parse(step_source)
    calls = {name: [] for name in ('termination_manager', 'reward_manager')}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == 'compute' and isinstance(node.func.value, ast.Attribute):
            owner = node.func.value.attr
            if owner in calls:
                calls[owner].append(node)
    if len(calls['termination_manager']) != 1 or len(calls['reward_manager']) != 1:
        raise RuntimeError('Cannot verify unique installed termination/reward compute calls')
    terminal, reward = calls['termination_manager'][0], calls['reward_manager'][0]
    if terminal.lineno >= reward.lineno:
        raise RuntimeError('Installed runtime computes physical-failure reward before terminations')
    resets = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute) and node.func.attr == '_reset_idx']
    if not resets or any(node.lineno <= reward.lineno for node in resets):
        raise RuntimeError('Cannot verify physical-failure reward precedes automatic reset')
    if not any(isinstance(node, ast.Attribute) and node.attr == 'step_dt'
               for node in ast.walk(reward)):
        raise RuntimeError('Installed reward call does not use the control-step duration')
    scaling = verify_runtime_reward_scaling(raw.reward_manager, raw.step_dt)
    # The original term flags must exist before a rollout and be boolean.
    physical_failure_cost(raw)
    return dict(termination_before_reward=True, reward_before_auto_reset=True,
                reward_uses_control_step_dt=True,
                weight_times_dt_verified=True, control_dt_seconds=float(raw.step_dt),
                scaling_probe=scaling,
                failure_penalty_per_terminal_step=FAILURE_MAGNITUDE*FAILURE_WEIGHT*float(raw.step_dt),
                timeout_penalized=False, duplicate_failure_penalized_once=True,
                env_step_source_sha256=hashlib.sha256(step_source.encode()).hexdigest(),
                reward_compute_source_sha256=hashlib.sha256(compute_source.encode()).hexdigest())
