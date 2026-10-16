"""Pure source-derived worksheets; these functions cannot construct or run PPO."""
from .contracts import require


def recurrent_partition(num_envs, num_mini_batches):
    require(type(num_envs) is int and type(num_mini_batches) is int and
            num_envs > 0 and 0 < num_mini_batches <= num_envs,
            "Recurrent minibatches need 1 <= num_mini_batches <= num_envs")
    require(num_envs % num_mini_batches == 0,
            "Pinned RSL recurrent generator silently omits remainder envs; exact division required")
    width = num_envs // num_mini_batches
    return [[i * width, (i + 1) * width] for i in range(num_mini_batches)]


def rollout_cost(num_envs, num_mini_batches):
    ranges = recurrent_partition(num_envs, num_mini_batches)
    transitions = num_envs * 24
    macros = transitions * 10
    return dict(status="DRAFT_NOT_AUTHORIZED", num_envs=num_envs, num_mini_batches=num_mini_batches,
                recurrent_environment_slices=ranges, public_steps_per_env=24,
                transitions_per_rollout=transitions, macro_advances=macros,
                max_rollout_native_integrations=macros * 352,
                macro_forward_reservation=macros * 1413,
                setup_reset_sense_forward_budget=None, authorized_PPO_updates=0,
                full_execution_budget_released=False)


def next_iteration(last_completed_index, completed_updates):
    require(type(last_completed_index) is int and type(completed_updates) is int and completed_updates >= 1,
            "Checkpoint must have an explicitly committed completed-update ledger")
    require(last_completed_index >= 0 and last_completed_index + 1 == completed_updates,
            "Pinned RSL iter is a zero-based last update index, not a completed-update count")
    return completed_updates


def plan(inherited_minibatches=4):
    return dict(status="STATIC_RUNNER_REVIEW_ONLY", training_ready=False, executed=False,
                stock_train_allowed=False, init_at_random_ep_len=False,
                inherited_minibatches=inherited_minibatches,
                inherited_minibatches_valid_for_prepared_N1=inherited_minibatches == 1,
                minibatches_changed_in_candidate=False,
                draft_small_scale=[rollout_cost(1, 1), rollout_cost(2, 1), rollout_cost(2, 2)],
                timeout_resolution="UNRELEASED: select and test terminal-observation bootstrap; current RSL uses V(s_t)",
                continuation="Update-boundary reset continuation only after explicit RNG/RNN/ledger policy; no exact mid-rollout claim",
                zero_iteration_constructor_is_dry_run=False)
