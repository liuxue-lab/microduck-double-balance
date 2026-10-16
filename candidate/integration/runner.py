"""Dedicated update loop candidate; never construct PPO in CPU preparation.

The controller accepts explicit dependencies so bookkeeping and failure paths
can be tested without a policy or optimizer. Physical acceptance and cloud
budget must be provided by a future released composition before construction.
"""
import math
from .durable import require, save_checkpoint, verify_resume
from .transitions import process_transition
from .planning import recurrent_batches


def restore_learning_rate(algorithm):
    rates = [group["lr"] for group in algorithm.optimizer.param_groups]
    require(rates and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in rates)
            and len(set(rates)) == 1, "Optimizer LR state is inconsistent")
    algorithm.learning_rate = float(rates[0])


def construction_gate(release, budget):
    # No stub may silently route to the stock ManagerBasedRlEnv/stock train.py.
    require(release.get("physical_acceptance") is True and
            release.get("full_episode_and_reset_accepted") is True and
            release.get("throughput_measured") is True and
            release.get("cloud_package_verified") is True,
            "Physical lifecycle/throughput/cloud package acceptance remains pending")
    require(budget.get("authorization") == "USER_AUTHORIZED_CLOUD_PPO" and
            type(budget.get("PPO_updates")) is int and budget["PPO_updates"] > 0 and
            budget.get("cloud_training") == 1 and budget.get("policy_constructions") == 2 and
            budget.get("optimizer_constructions") == 1,
            "Explicit cloud PPO/model/optimizer budget required before any construction")
    require(budget.get("gpu_hours", 0) > 0 and budget.get("spend_limit", 0) > 0 and
            isinstance(budget.get("currency"), str) and budget["currency"],
            "Cloud time/spend limit missing")


def run_updates(env, algorithm, torch, ledger, checkpoint_dir, updates, mini_batches,
                counters, on_update=None):
    batches = recurrent_batches(env.num_envs, mini_batches)
    require(type(updates) is int and 0 < updates <= ledger.max_updates - ledger.attempted,
            "PPO update budget exceeded")
    require(algorithm.num_mini_batches == mini_batches, "Constructed PPO minibatches differ")
    require(getattr(algorithm, "rnd", None) is None, "RND extension is not part of this candidate")
    require(str(getattr(algorithm, "device", env.device)) == str(env.device),
            "Construct the vector observations on the policy device before running updates")
    algorithm.train_mode()
    obs = env.get_observations()
    for _ in range(updates):
        index = ledger.begin(batches["transitions_per_update"])
        with torch.inference_mode():
            for _step in range(24):
                actions = algorithm.act(obs)
                obs, rewards, dones, extras = env.step(actions.to(env.device))
                process_transition(algorithm, obs, rewards, dones, extras)
            algorithm.compute_returns(obs)
        losses = algorithm.update()
        # Save every update in this initial correctness-first controller.
        # No stock final-save repeats, zero-based duplicate index, or overwrite.
        metadata = dict(counters(), next_iteration=index + 1, completed_updates=index + 1)
        def serialize(f):
            state = algorithm.save()
            state.update(iter=index, completed_updates=index + 1, next_iteration=index + 1,
                         stage12_identity=ledger.identity, counters=metadata,
                         cpu_rng_state=torch.get_rng_state(),
                         cuda_rng_state=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])
            torch.save(state, f)
        receipt = save_checkpoint(checkpoint_dir, ledger, serialize, metadata)
        if on_update is not None:
            on_update(index + 1, losses, receipt)
    return ledger.completed


def load_committed_update(algorithm, torch, ledger_path, receipt_path, identity, device):
    row = verify_resume(ledger_path, receipt_path, identity)
    from pathlib import Path
    state = torch.load(Path(receipt_path).parent / row["payload"], weights_only=True, map_location=device)
    require(state["stage12_identity"] == identity and state["completed_updates"] == row["completed_updates"]
            and state["next_iteration"] == row["next_iteration"], "Payload/receipt identity mismatch")
    require(algorithm.load(state, load_cfg=None, strict=True), "Algorithm did not load its iteration state")
    restore_learning_rate(algorithm)
    # Physics, streams, RNN and rollout are fresh and explicitly reset by caller.
    # Policy sampler RNG is restored, but this is not exact trajectory continuation.
    torch.set_rng_state(state["cpu_rng_state"].cpu())
    if state["cuda_rng_state"]:
        require(torch.cuda.is_initialized(), "CUDA RNG restore requires the authorized device")
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda_rng_state"]])
    return row["next_iteration"]
