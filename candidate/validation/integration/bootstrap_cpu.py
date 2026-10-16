"""Actual native_child/bootstrap under spawn; only native session is a CPU fixture."""
from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace
import argparse
import ast
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0,str(ROOT))


def import_runtime(repo, out, torch, legacy=False):
    """Execute the production factory's exact pre-CUDA import block, no session."""
    here = ROOT/'baseline/g11'
    sys.path[:0] = [str(Path(repo)/'src'),str(here),str(here/'accepted'),str(here/'host-candidate')]
    source = ROOT/'integration/native_factory.py'
    tree = ast.parse(source.read_text())
    owner = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'owner_session')
    block = next(n for n in owner.body if isinstance(n,ast.Try))
    imports = []
    for n in block.body:
        if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id == 'require':
            break
        if isinstance(n,(ast.Import,ast.ImportFrom)):
            imports.append(n)
    assert len(imports) >= 25
    # Import bindings are real; no MjModel, environment, CUDA device, policy,
    # optimizer or checkpoint construction is permitted in this check.
    import mujoco
    def deny(*args,**kwargs):
        raise RuntimeError('Import regression forbids physics/model construction')
    for name in ('mj_forward','mj_step','mj_kinematics'):
        setattr(mujoco,name,deny)
    import warp as wp
    namespace = {}
    exec(compile(ast.Module(body=imports,type_ignores=[]),str(source),'exec'),namespace)
    cfg = namespace['make_cfg']()
    namespace['preflight'](cfg,out,namespace['UniformNoiseCfg'])
    if legacy:
        wp.config.kernel_cache_dir = str(out/'kernel-cache')
    else:
        from integration.runtime_paths import for_owner
        wp.config.kernel_cache_dir = for_owner(out)['warp_cache']
    wp.init()  # Actual native Warp runtime; CPU registration covers the v2 defect.
    assert not wp.is_cuda_available(), 'Developer validation is CPU-only'
    cpu = wp.get_device('cpu')
    assert cpu.is_cpu and cpu.name == 'CPU'
    return dict(production_import_statements=len(imports),real_modules=True,config_preflight=True,
                real_Warp_runtime=True,Warp_CPU_device_label=cpu.name,
                native_integrations=0,CPU_forward=0,GPU_forward=0,environment_constructions=0,
                CUDA_initialized=torch.cuda.is_initialized())


@contextmanager
def fixture_session(repo,out,plan,authorization):
    from validation.integration.real_cpu import backend
    from validation.integration.cpu_cases import fake_env
    from integration.transitions import TerminalBridge
    from integration.durable import write_json
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'
    torch,td = backend()
    out=Path(out);out.mkdir()
    import socket,subprocess
    blocked=[]
    for name,operation in (
            ('unreviewed-subprocess',lambda:subprocess.run(['/usr/bin/git','version'],check=True)),
            ('direct-uname-subprocess',lambda:subprocess.run(['uname','-p'],check=True)),
            ('network-resolution',lambda:socket.getaddrinfo('example.com',443)),
            ('write-outside-owner',lambda:(out.parent/'must-not-exist.txt').write_text('forbidden')),
            ('write-outside-private-tmp',lambda:Path('/tmp/stage12-unowned-write-must-not-exist.txt').write_text('forbidden'))):
        try:operation()
        except RuntimeError:blocked.append(name)
        else:raise AssertionError('Boundary did not block '+name)
    assert not (out.parent/'must-not-exist.txt').exists()
    row=import_runtime(repo,out,torch)
    row['boundary_denials']=blocked
    from validation.integration.cleanup_checks import verify
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    row['partial_cleanup']=verify(ManagerBasedRlEnv)
    from integration.manager_contracts import preflight
    row['manager_interfaces']=preflight(out)
    from validation.integration.owner_chain import verify as verify_chain
    import warp as wp
    row['owner_chain']=verify_chain(out,torch,wp)
    write_json(out/'import-regression.json',row)
    env=fake_env(torch,'TensorDict',td,False,plan['logical_world'] == 1)
    bridge=TerminalBridge(env,torch,td,plan['max_resets'])
    journal=SimpleNamespace(formal_steps=0,checkpoint=lambda *a:None)
    def inspect():
        return dict(public_steps=journal.formal_steps,macros=journal.formal_steps*10,
                    resets=bridge.resets,counts=dict(integrations_completed=journal.formal_steps*1600,
                    integrations_attempted=journal.formal_steps*1600),rng_sha256='fixture-'+str(bridge.resets),
                    observation_sha256=str(bridge.get_observations()['actor'][0,0].item()),
                    episode_length=[0],memory=dict(scope='CPU fixture'),fixture=True)
    yield dict(torch=torch,bridge=bridge,journal=journal,inspect=inspect)
    write_json(out/'native-report.json',dict(status='STAGE12_NATIVE_CANDIDATE_FINISHED_REVIEW_REQUIRED',
               native_ledger=dict(exact=True),fixture=True,actual_native_integrations=0))


def bootstrap_child(connection,owner,repo,out,plan,authorization):
    # Keep the production entry, guarded imports, IPC and closeout. Replace only
    # the native session, which cannot run on this developer's CPU environment.
    from integration import native_factory
    from integration.process_owners import native_child
    native_factory.owner_session=fixture_session
    native_child(connection,owner,repo,out,plan,authorization)


def run(output,repo):
    from unittest.mock import patch
    from validation.integration.real_cpu import backend
    from integration import process_owners,local_campaign
    from integration.planning import proposal
    from integration.durable import write_json
    for key in list(os.environ):
        if key.startswith('GIT_'):os.environ.pop(key)
    torch,td=backend()
    output.mkdir()
    rows=[]
    with patch.object(process_owners,'native_child',bootstrap_child),patch.object(local_campaign,'REPO',str(repo)):
        for point in proposal()['points']:
            rows.append(local_campaign.measure_point(point,output/point['name'],torch,td,120))
    boot=[]
    for p in sorted(output.rglob('bootstrap-report.json')):
        r=json.loads(p.read_bytes())
        assert r['status']=='GUARDED_BOOTSTRAP_CLOSED'
        assert r['git']['completed_queries']==2 and len(r['font']['optional_fontconfig_denials']) == 1
        assert len(r['processor']['optional_processor_denials']) == 1
        assert r['original_cwd'] != r['cwd']
        boot.append(dict(path=str(p),git_queries=2,optional_fontconfig_fallbacks=1))
    assert len(boot)==3 and [r['owners'] for r in rows]==[1,2]
    mappings=[json.loads(p.read_bytes()) for p in sorted(output.rglob('runtime-paths.json'))]
    assert len(mappings)==3 and len({r['short_root'] for r in mappings})==3
    assert all(len(os.fsencode(r['temp']))<=48 and Path(r['short_root']).stat().st_mode & 0o777 == 0o700 for r in mappings)
    report=dict(status='PASS',production_native_child_called=True,production_bootstrap=True,
                actual_spawn_owners=3,real_import_stack=True,bootstrap=boot,
                actual_native_integrations=0,fixture_counters_excluded_from_canonical=True,
                PPO=0,cloud_training=0,training_ready=False)
    write_json(output/'bootstrap-cpu-report.json',report)
    print(json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--repo',type=Path,required=True)
    a=p.parse_args();run(a.output,a.repo)
