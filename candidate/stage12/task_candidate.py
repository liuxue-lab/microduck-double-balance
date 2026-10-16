"""Build and temporarily register config-only candidates; never install a task.

The separate physical factory is integration.native_factory; this registry
placeholder remains fail-closed. The native
registry has no env_cls slot, so registering this config cannot release physics.
"""
from copy import deepcopy
from .contracts import TASK_ID, PreparedRunner, deny_training, require


def make_candidates():
    from formal_task import make_cfg
    from checkpoint_review import config_manifest
    from mjlab.managers import EventTermCfg
    from mjlab_microduck.tasks.microduck_double_balance_env_cfg import MicroduckDoubleBalanceRlCfg

    base = make_cfg()
    cfg = deepcopy(base)
    for name in ("g11_step_probe", "g11_local_probe", "g11_class_probe", "g11_global_probe"):
        require(name in cfg.events, "Missing diagnostic event " + name)
        del cfg.events[name]
    require(set(cfg.recorders) == {"g11_record"}, "Unexpected recorder; review first")
    cfg.recorders = {}
    require("g11_fault_injection" in cfg.terminations, "Diagnostic termination missing")
    del cfg.terminations["g11_fault_injection"]
    cfg.auto_reset = True
    cfg.events = {"stage12_release_block": EventTermCfg(func=deny_training, mode="startup"), **cfg.events}
    cfg.scene.num_envs = 1
    # Everything affecting task values remains inherited, including nominal spawn,
    # actor noise, rewards, stability threshold, float32 success comparison and hold=0.
    before, after = config_manifest(base), config_manifest(cfg)
    for key in ("actor_observations", "critic_observations", "actions", "rewards",
                "physics_dt_s", "decimation", "step_dt_s", "observation_term_order"):
        require(before[key] == after[key], "Protected task semantics changed: " + key)
    expected = deepcopy(before["terminations"])
    expected.pop("g11_fault_injection")
    require(after["terminations"] == expected, "Non-diagnostic termination changed")
    agent = deepcopy(MicroduckDoubleBalanceRlCfg)
    require(agent.num_steps_per_env == 24 and agent.actor.class_name == "RNNModel" and
            agent.actor.rnn_type == "lstm" and agent.actor.rnn_hidden_dim == 256 and
            agent.actor.rnn_num_layers == 1 and agent.actor.obs_normalization is True and
            agent.critic.obs_normalization is True, "Runner family differs from inspected source")
    require(hasattr(agent, "resume") and hasattr(agent, "upload_model"), "Installed runner config requires review")
    agent.experiment_name = "double_balance_head_stage12_prepared"
    agent.run_name = "fresh_nominal_not_released"
    agent.max_iterations = 0
    agent.resume = False
    agent.upload_model = False
    # No runner is constructed; logger selection is made only if supported.
    if hasattr(agent, "logger"):
        agent.logger = "tensorboard"
    return base, cfg, agent


def register_prepared(cfg, agent):
    from mjlab.tasks.registry import register_mjlab_task, list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
    require(TASK_ID not in list_tasks(), "Prepared task ID already registered")
    register_mjlab_task(task_id=TASK_ID, env_cfg=cfg, play_env_cfg=deepcopy(cfg),
                        rl_cfg=agent, runner_cls=PreparedRunner)
    loaded, loaded_agent = load_env_cfg(TASK_ID), load_rl_cfg(TASK_ID)
    require(loaded is not cfg and loaded.auto_reset is True and loaded.scene.num_envs == 1,
            "Registry deepcopy contract differs")
    require(loaded_agent.max_iterations == 0 and loaded_agent.resume is False and
            load_runner_cls(TASK_ID) is PreparedRunner, "Prepared runner guard missing")
    loaded.scene.num_envs = 999
    require(load_env_cfg(TASK_ID).scene.num_envs == 1 and cfg.scene.num_envs == 1,
            "Registry shared mutable config")
    return {"task_id": TASK_ID, "in_process_registration": True, "persistent_installation": False,
            "env_factory": "stage12.contracts.create_environment (blocked)",
            "registry_has_env_cls_slot": False, "stock_train_entrypoint_released": False,
            "max_iterations": 0, "resume": False}
