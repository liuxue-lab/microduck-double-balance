from pathlib import Path
import base64
import difflib
import hashlib
import json
import zlib

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
OLD = ROOT.parent/'native-training'
original = (OLD/'package/train.py').read_text()
candidate = original


def change(old, new):
    global candidate
    if candidate.count(old) != 1:
        raise RuntimeError('Candidate source anchor changed: '+old[:100])
    candidate = candidate.replace(old, new, 1)


helpers = '''def require_finite_group(torch, values, message):
    """One host read on the success path; still check every input each call."""
    checks = [torch.isfinite(v).all() for v in values]
    if checks:
        require(bool(torch.stack(checks).all().item()), message)


def require_step_outputs(torch, observations, rewards, dones, nan_state):
    """Check before returning to PPO; never delay detection to batch end."""
    checks = [~nan_state.any()]
    checks.extend(torch.isfinite(v).all() for v in observations.values())
    checks.extend((torch.isfinite(rewards).all(), torch.isfinite(dones).all()))
    if not bool(torch.stack(checks).all().item()):
        # Additional reads only on the exceptional path, for a useful reason.
        require(not bool(nan_state.any().item()), 'Nonfinite simulator state detected; preserve, no automatic retry')
        raise RuntimeError('Nonfinite observations, rewards or dones')


def host_scalar_means(torch, values):
    """Retain float32 mean semantics; transfer each device group together."""
    groups = {}
    result = [None]*len(values)
    for i, value in enumerate(values):
        scalar = torch.as_tensor(value).detach().float().mean()
        groups.setdefault(str(scalar.device), []).append((i, scalar))
    for group in groups.values():
        host = torch.stack([value for _, value in group]).cpu().tolist()
        for (i, _), value in zip(group, host):
            result[i] = value
    return result


'''
change('def main():\n', helpers+'def main():\n')
change("save_interval=50, max_iterations=cfg_doc['max_updates'], check_for_nan=True)",
       "save_interval=50, max_iterations=cfg_doc['max_updates'], check_for_nan=False)\n        # Our step wrapper checks every observation, reward and done before PPO consumes them.\n        # This replaces the duplicate upstream check; action and NaN-reset checks remain.")
change("""            nan_count=int(raw.termination_manager.get_term('nan_state').sum().item())
            nan_since[0]+=nan_count
            require(nan_count==0,'Nonfinite simulator state detected; preserve, no automatic retry')
            observations,rewards,dones,_=result
            require(all(bool(torch.isfinite(v).all()) for v in observations.values()) and bool(torch.isfinite(rewards).all()), 'Nonfinite observations or rewards')""",
"""            observations,rewards,dones,_=result
            require_step_outputs(torch, observations, rewards, dones, raw.termination_manager.get_term('nan_state'))
            # A successful check implies zero nan_state resets; the failure path exits immediately.""")
change("            require(all(bool(torch.isfinite(p).all()) for m in (runner.alg.actor,runner.alg.critic) for p in m.parameters()), 'Nonfinite learned weights')",
       "            require_finite_group(torch, (p for m in (runner.alg.actor,runner.alg.critic) for p in m.parameters()), 'Nonfinite learned weights')")
change("""            episodes={}
            for item in runner.logger.ep_extras:
                for key,v in item.items():
                    if any(s in key for s in ('native_','top_ball_center','double_balance','Episode_Termination')):
                        f=float(torch.as_tensor(v).float().mean().item())
                        if math.isfinite(f):episodes.setdefault(key,[]).append(f)
            original_log(**kw)""",
"""            episodes={}; metric_keys=[]; metric_values=[]
            for item in runner.logger.ep_extras:
                for key,v in item.items():
                    if any(s in key for s in ('native_','top_ball_center','double_balance','Episode_Termination')):
                        metric_keys.append(key); metric_values.append(v)
            for key,f in zip(metric_keys,host_scalar_means(torch,metric_values)):
                if math.isfinite(f):episodes.setdefault(key,[]).append(f)
            original_log(**kw)""")
change("            batches.append(summary)\n", """            batches.append(summary)
            current_fractions=host_scalar_means(torch,[
                (mdp._bb_state(raw).hold<=1e-6).float(),
                lower_unassisted(raw).float(), geometric_joint_stable(raw).float()])
""")
change("""                unassisted_env_fraction=float((mdp._bb_state(raw).hold<=1e-6).float().mean()),
                lower_unassisted_current_fraction=float(lower_unassisted(raw).float().mean()),
                joint_geometric_current_fraction=float(geometric_joint_stable(raw).float().mean()),""",
"""                unassisted_env_fraction=current_fractions[0],
                lower_unassisted_current_fraction=current_fractions[1],
                joint_geometric_current_fraction=current_fractions[2],
                collection_seconds=kw['collect_time'], learning_seconds=kw['learn_time'],""")

(ROOT/'candidate').mkdir(exist_ok=True)
(ROOT/'candidate/train.py').write_text(candidate)
(ROOT/'train-performance-candidate.patch').write_text(''.join(difflib.unified_diff(
    original.splitlines(keepends=True), candidate.splitlines(keepends=True),
    fromfile='a/package/train.py', tofile='b/package/train.py')))
compile(candidate, 'candidate/train.py', 'exec')

remote = (ROOT/'perf_read.py').read_text() + "\nimport sys\nprint('MD12_PERF_STATE='+json.dumps(inspect_slot(sys.argv[1]),ensure_ascii=False,allow_nan=False))\n"
ssh = (OLD/'shared_ssh.py').read_bytes()
controller = (ROOT/'controller.py').read_text()
for key, value in dict(SSH_B85=base64.b85encode(zlib.compress(ssh,9)).decode(),
                       SSH_SHA=hashlib.sha256(ssh).hexdigest(),
                       REMOTE_B85=base64.b85encode(zlib.compress(remote.encode(),9)).decode(),
                       REMOTE_SHA=hashlib.sha256(remote.encode()).hexdigest()).items():
    controller = controller.replace('@@'+key+'@@',repr(value))
compile(controller,'perf-controller.py','exec')
output = WORKSPACE/'inspect_training_performance.py'
output.write_text(controller)
output.chmod(0o755)
receipt = dict(status='OFFLINE_CANDIDATE_NOT_DEPLOYED',
    original_train_sha256=hashlib.sha256(original.encode()).hexdigest(),
    candidate_train_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
    controller_sha256=hashlib.sha256(controller.encode()).hexdigest(),
    current_training_package_identity='dd6d1e1e697bcaa348dbdd92dd8c7b30ea31ae9c32599338106859ae7657523a',
    original_controller_sha256=hashlib.sha256((WORKSPACE/'train.py').read_bytes()).hexdigest(),
    GPU_tested=False, speedup_measured=False, cloud_deployed=False,
    added_simulation_steps=0, added_PPO_updates=0)
(ROOT/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
