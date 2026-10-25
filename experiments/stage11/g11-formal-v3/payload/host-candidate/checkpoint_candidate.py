"""Lossless-or-error key migration and explicit checkpoint semantic comparison.

Input is an already inspected mapping. This file does not deserialize, load,
save or alter a checkpoint, instantiate a runner, or migrate physical state.
"""
from collections.abc import Mapping


class MigrationError(ValueError):
    pass


def migrate_checkpoint_keys(checkpoint):
    if not isinstance(checkpoint, Mapping):
        raise MigrationError("Checkpoint must be a mapping")
    out = dict(checkpoint)
    changes = []
    if "model_state_dict" in out:
        if "actor_state_dict" in out or "critic_state_dict" in out:
            raise MigrationError("Mixed legacy and current model dictionaries")
        legacy = out.pop("model_state_dict")
        if not isinstance(legacy, Mapping):
            raise MigrationError("Legacy state must be a mapping")
        actor, critic = {}, {}
        routes = (("actor.", actor, "mlp."), ("critic.", critic, "mlp."),
                  ("actor_obs_normalizer.", actor, "obs_normalizer."),
                  ("critic_obs_normalizer.", critic, "obs_normalizer."))
        for key, value in legacy.items():
            if not isinstance(key, str):
                raise MigrationError("Non-string state key")
            if key in ("std", "log_std"):
                destination, name = actor, key
            else:
                candidates = [(target, prefix + key[len(old):]) for old, target, prefix in routes if key.startswith(old)]
                if len(candidates) != 1:
                    raise MigrationError("Unmapped legacy state key: " + key)
                destination, name = candidates[0]
            if name in destination:
                raise MigrationError("Migration would overwrite: " + name)
            destination[name] = value
            changes.append({"from": "model_state_dict." + key,
                            "to": ("actor_state_dict." if destination is actor else "critic_state_dict.") + name})
        out.update(actor_state_dict=actor, critic_state_dict=critic)

    for name in ("actor_state_dict", "critic_state_dict"):
        if name not in out or not isinstance(out[name], Mapping) or not out[name]:
            raise MigrationError("Complete nonempty actor and critic state dictionaries required")
        out[name] = dict(out[name])
    actor = out["actor_state_dict"]
    for old, new in (("std", "distribution.std_param"), ("log_std", "distribution.log_std_param")):
        if old in actor:
            if new in actor:
                raise MigrationError("Distribution migration would overwrite: " + new)
            actor[new] = actor.pop(old)
            changes.append({"from": "actor_state_dict." + old, "to": "actor_state_dict." + new})
    if "distribution.std_param" in actor and "distribution.log_std_param" in actor:
        raise MigrationError("Ambiguous simultaneous std and log_std parameterizations")
    return out, {"status": "KEY_MAPPING_ONLY", "changes": changes,
                 "checkpoint_loaded": False, "semantic_compatibility": "UNVERIFIED"}


SEMANTIC_FIELDS = {
    "actor_observations": {"term_order", "terms", "history_layout", "flatten_order", "normalizer"},
    "critic_observations": {"term_order", "terms", "history_layout", "flatten_order", "normalizer"},
    "actions": {"joint_order", "scale", "offset", "clip", "actuator_model", "delay"},
    "timing": {"physics_dt_s", "decimation", "step_dt_s", "derived_state_policy"},
    "task": {"id", "reward_terms", "termination_terms", "source_sha256"},
}


def compare_semantics(previous, target):
    """Same dimensions are insufficient: ordered feature meaning is required.

    The comparison does not establish provenance of a supplied manifest. Both
    manifests must be extracted/reviewed against the actual checkpoint/task.
    """
    missing, differences = [], []
    def compare(a, b, path):
        if type(a) is not type(b):
            differences.append(path + ":type")
        elif isinstance(a, dict):
            if set(a) != set(b):
                differences.append(path + ":keys")
            for key in sorted(set(a) & set(b)):
                compare(a[key], b[key], path + "." + key)
        elif isinstance(a, list):
            if len(a) != len(b):
                differences.append(path + ":length")
            for i, (x, y) in enumerate(zip(a, b)):
                compare(x, y, f"{path}[{i}]")
        elif a != b:
            differences.append(path)
    for side, record in (("previous", previous), ("target", target)):
        if not isinstance(record, dict):
            missing.append(side + ":manifest")
            continue
        for group, fields in SEMANTIC_FIELDS.items():
            if not isinstance(record.get(group), dict):
                missing.append(side + "." + group)
                continue
            missing.extend(side + "." + group + "." + key for key in sorted(fields - set(record[group])))
    if not missing:
        compare(previous, target, "manifest")
    return {"status": "MANIFESTS_MATCH_REQUIRE_PROVENANCE" if not missing and not differences else "BLOCKED",
            "missing": missing, "differences": differences, "checkpoint_loaded": False,
            "automatic_semantic_migration": False, "real_checkpoint_compatible": False}
