"""Scoped, inference-only removal of reviewed reward and privileged-observation work.

The original actor compute_group path is untouched. The zero reward returned to
RslRlVecEnvWrapper is explicitly a compatibility placeholder, not a measurement.
This adapter is never used for learning, reset, or production activation.
"""
import importlib
import inspect
import json
from pathlib import Path

from deploy_contract import require,sha
from phase_cache import Bindings

SOURCES=json.loads((__import__('bootstrap').BASE_PACKAGE / 'runtime_sources.json').read_bytes())
BASE_OBS='mjlab.envs.mdp.observations'
VEL_OBS='mjlab.tasks.velocity.mdp.observations'
BASE_REWARD='mjlab.envs.mdp.rewards'
VEL_REWARD='mjlab.tasks.velocity.mdp.rewards'
MDP='mjlab_microduck.tasks.mdp'
CRITIC={
    'base_lin_vel':(BASE_OBS,'base_lin_vel'),
    'base_ang_vel':(BASE_OBS,'builtin_sensor'),
    'projected_gravity':(BASE_OBS,'projected_gravity'),
    'joint_pos':(BASE_OBS,'joint_pos_rel'),'joint_vel':(BASE_OBS,'joint_vel_rel'),
    'actions':(BASE_OBS,'last_action'),'command':(BASE_OBS,'generated_commands'),
    'foot_height':(MDP,'foot_height_safe'),'foot_air_time':(MDP,'foot_air_time_safe'),
    'foot_contact':(VEL_OBS,'foot_contact'),
    'foot_contact_forces':(MDP,'foot_contact_forces_safe'),
    'head_command':(MDP,'basketball_head_pad'),'body_command':(MDP,'basketball_state'),
    'top_ball_state':('native_task','head_critic_observation'),
}
REWARDS={
    'track_linear_velocity':(VEL_REWARD,'track_linear_velocity'),
    'track_angular_velocity':(VEL_REWARD,'track_angular_velocity'),
    'pose':(VEL_REWARD,'variable_posture'),
    'body_ang_vel':(VEL_REWARD,'body_angular_velocity_penalty'),
    'angular_momentum':(VEL_REWARD,'angular_momentum_penalty'),
    'dof_pos_limits':(BASE_REWARD,'joint_pos_limits'),
    'action_rate_l2':(BASE_REWARD,'action_rate_l2'),
    'self_collisions':(VEL_REWARD,'self_collision_cost'),
    'upright':(MDP,'wrestle_upright'),'centered':(MDP,'basketball_centered'),
    'feet_on_ball':(MDP,'basketball_feet_on_ball'),'height':(MDP,'basketball_height'),
    'ball_speed':(MDP,'basketball_ball_speed_l2'),
    'double_balance':('native_task','head_reward'),
    'top_ball_speed':('native_task','head_speed_cost'),
    'head_surface_normal_tilt':(MDP,'stage11_head_normal_tilt_l2'),
}


def verify_sources():
    modules={};receipts={}
    for name,pin in SOURCES.items():
        module=importlib.import_module(name);path=Path(inspect.getsourcefile(module))
        require(path.is_file() and sha(path)==pin['sha256'],'Unreviewed inference source: '+name)
        modules[name]=module;receipts[name]=dict(path=str(path),sha256=pin['sha256'])
    return modules,receipts


def validate_terms(raw,modules):
    obs=raw.observation_manager;reward=raw.reward_manager
    require(type(obs) is modules['mjlab.managers.observation_manager'].ObservationManager,
            'Unreviewed observation manager type')
    require(type(reward) is modules['mjlab.managers.reward_manager'].RewardManager,
            'Unreviewed reward manager type')
    require(set(obs._group_obs_term_names)=={'actor','critic'},'Unexpected observation groups')
    require(tuple(obs.group_obs_dim['actor'])==(61,) and tuple(obs.group_obs_dim['critic'])==(85,),
            'Expected full 61D actor / 85D critic before adapter')
    require(obs.cfg['critic'].enable_corruption is False,'Critic corruption cannot be skipped')
    names=obs._group_obs_term_names['critic'];cfgs=obs._group_obs_term_cfgs['critic']
    require(set(names)==set(CRITIC) and len(names)==len(cfgs)==len(CRITIC),'Unreviewed critic terms')
    for name,cfg in zip(names,cfgs):
        module,attr=CRITIC[name]
        require(cfg.func is getattr(modules[module],attr),'Unreviewed critic function: '+name)
        require(cfg.noise is None and cfg.delay_max_lag==0 and cfg.history_length==0,
                'Skipping critic would change random/history state: '+name)
    require(not obs._group_obs_class_term_cfgs['critic']
            and not obs._group_obs_term_delay_buffer['critic']
            and not obs._group_obs_term_history_buffer['critic'],'Critic has stateful buffers')
    require(set(reward._term_names)==set(REWARDS)
            and len(reward._term_names)==len(reward._term_cfgs)==len(REWARDS),'Unreviewed reward terms')
    for name,cfg in zip(reward._term_names,reward._term_cfgs):
        module,attr=REWARDS[name];original=getattr(modules[module],attr)
        actual=type(cfg.func) if inspect.isclass(original) else cfg.func
        require(actual is original,'Unreviewed reward function: '+name)
    return dict(critic_terms=list(names),reward_terms=list(reward._term_names))


class InferenceOnly(Bindings):
    def __init__(self,raw,torch):
        super().__init__();self.raw=raw;self.torch=torch;self.active=False
        self.reward_calls=0;self.actor_calls=0;self.critic_calls=0
        require(torch.is_inference_mode_enabled(),'Inference-only adapter requires torch.inference_mode')
        require(raw.num_envs==1 and raw.cfg.auto_reset is False and not raw.cfg.curriculum,
                'Adapter only supports one frozen-policy episode without auto-reset/curriculum')
        require(abs(raw.step_dt-.02)<1e-12 and abs(raw.physics_dt-.005)<1e-12
                and raw.cfg.decimation==4,'Unexpected control/physics rates')
        modules,self.sources=verify_sources();self.terms=validate_terms(raw,modules)
        obs=raw.observation_manager;reward=raw.reward_manager
        require(obs._obs_buffer is not None and set(obs._obs_buffer)=={'actor','critic'},
                'Install only after the full actor/critic reset')
        self.placeholder=torch.zeros(raw.num_envs,device=raw.device,dtype=torch.float32)
        original_group=obs.compute_group
        def actor_group(group_name,update_history=False):
            if group_name!='actor':
                self.critic_calls+=1;raise RuntimeError('Privileged observations requested during inference-only step')
            self.actor_calls+=1
            return original_group(group_name,update_history)
        def omitted_reward(dt):
            require(self.active and torch.is_inference_mode_enabled(),'Reward omission outside inference scope')
            require(abs(float(dt)-.02)<1e-12,'Unexpected reward timestep')
            self.reward_calls+=1
            logs=raw.extras.get('log',{})
            for key in tuple(logs):
                if key.startswith('Episode_Reward/') or key=='Metrics/angular_momentum_mean':logs.pop(key)
            raw.extras['runtime_reward_omitted']=True
            return self.placeholder
        def refuse_reset(*args,**kwargs):
            raise RuntimeError('Close the inference-only adapter before resetting the environment')
        try:
            self.set(obs,'_group_obs_term_names',{'actor':obs._group_obs_term_names['actor']})
            self.set(obs,'compute_group',actor_group)
            self.set(obs,'reset',refuse_reset)
            self.set(reward,'compute',omitted_reward)
            # Do not mutate the full reset dictionary already held by policy.
            obs._obs_buffer={'actor':obs._obs_buffer['actor']}
            self.active=True
        except BaseException:self.close();raise

    def stats(self):
        return dict(reward_not_computed=True,zero_reward_is_wrapper_placeholder=True,
            critic_not_computed_after_reset=True,omitted_reward_calls=self.reward_calls,
            actor_group_calls=self.actor_calls,unexpected_critic_calls=self.critic_calls,
            omitted_logging_fields=['Episode_Reward/*','Metrics/angular_momentum_mean'],
            actor_compute_group='original mjlab 1.3.0, including noise/delay/history',
            eligibility=self.terms,verified_sources=self.sources)

    def close(self):
        was_installed=bool(self.saved);super().close();self.active=False
        if was_installed:
            self.raw.observation_manager._obs_buffer=None
            self.raw.extras.pop('runtime_reward_omitted',None)
