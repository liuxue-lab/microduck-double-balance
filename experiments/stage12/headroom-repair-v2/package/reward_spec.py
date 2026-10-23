"""Reward identity available to the controller without importing Torch."""
REWARD_SPEC=dict(version='speed_alignment_v1',changed_term='double_balance',weight=4.,
    original_upper_speed_std_m_s=.12,upper_speed_std_m_s=.08,lower_horizontal_speed_std_m_s=.15,
    formula='original position/height/upright factors * exp(-upper_speed^2/0.08^2) * exp(-lower_xy_speed^2/0.15^2)',
    smooth_dense_reward=True,consecutive_success_bonus=False,other_rewards_unchanged=True,
    observation_and_physics_unchanged=True,evaluation_uses_original_task=True)
