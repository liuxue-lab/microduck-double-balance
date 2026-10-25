"""Owner-state and batch regressions using CPU containers."""
from pathlib import Path
import ast, contextlib, hashlib, importlib, json, struct, sys, types, unittest
R=Path(__file__).resolve().parent;P=R/'payload'
REPO=next(p for p in R.parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(REPO/'src'))
from mjlab_microduck.release_layout import verify_source_file
ORIGINAL='experiments/stage11/g11-formal-v3/payload/'
sys.path[:0]=[str(P),str(P/'accepted'),str(P/'host-candidate')]
import formal_owner_keys as f
from runtime_accounting import Journal, make_classes, review_ledger
from macro_state_bridge import MACRO_LIMITS
from ground_pair_driver import macro_cost

class DummyTensor: pass
class DummyArray: pass
T=types.SimpleNamespace(Tensor=DummyTensor,no_grad=contextlib.nullcontext)
W=types.SimpleNamespace(array=DummyArray,synchronize=lambda:None)

class Checks(unittest.TestCase):
    def test_original_tuple_key_dict_binding_and_value_restore(self):
        key=(132,'stage11_head_material_frame');d={key:17};env=types.SimpleNamespace(cache=d)
        snap=f.FormalOwnerSnapshot([('env',env,('cache',))],T,W)
        d[key]=-1;d[(132,'other')]=4;env.cache={key:19}
        snap.restore();self.assertIs(env.cache,d);self.assertIs(next(iter(d)),key);self.assertEqual(d,{key:17})

    def test_nested_aliases_and_typed_key_rebinding(self):
        key=((4,'x'),'y');lst=[1,2];d={key:lst,'same':lst};env=types.SimpleNamespace(a=d,b=d)
        s=f.FormalOwnerSnapshot([('e',env,('a','b'))],T,W)
        lst[:]=[8];d['same']=[4];env.a={};s.restore()
        self.assertIs(env.a,env.b);self.assertIs(d[key],d['same']);self.assertEqual(lst,[1,2])
        # A newly-created equal tuple must be restored to the original key.
        d.clear();d[tuple(list(key))]=lst;d['same']=lst
        with self.assertRaises(RuntimeError):s.assert_restored()
        s.restore();self.assertIs(next(iter(d)),key)

    def test_scalar_exact_bits_preserved(self):
        d={(1,'x'):float('nan'),'negzero':-0.0};e=types.SimpleNamespace(d=d)
        s=f.FormalOwnerSnapshot([('e',e,('d',))],T,W);s.assert_restored()
        d['negzero']=0.0
        with self.assertRaises(RuntimeError):s.assert_restored()
        s.restore();self.assertEqual(struct.pack('d',d['negzero']),struct.pack('d',-0.0))

    def test_unsupported_keys_rejected_without_mutation(self):
        for key in (True,1.5,frozenset((1,)),(1,False),(((((('x',),),),),),)):
            d={key:5};e=types.SimpleNamespace(d=d)
            with self.assertRaises(RuntimeError):f.FormalOwnerSnapshot([('e',e,('d',))],T,W)
            self.assertEqual(d,{key:5})
        self.assertFalse(f.supported_key(tuple(range(17))))

    def test_lossless_export_without_key_string_collisions(self):
        d={1:10,'1':11,(1,):12,'(1,)':13,'x.y':14,'x':{'y':15},(1,'a.b'):16}
        raw,scalars,topology=f.owner_state([('e',types.SimpleNamespace(d=d),('d',))],T,W)
        self.assertFalse(raw);self.assertEqual(sorted(scalars.values()),list(range(10,17)))
        self.assertEqual(len(set(scalars)),7)
        self.assertEqual(topology['e.d']['typed_keys'],[f.key_record(k) for k in d])
        self.assertEqual(len(set(f.key_segment(k) for k in d)),len(d))

    def test_native_source_site_lookup_and_roundtrip_preflight(self):
        path=P/'host-candidate/pinned-source/mdp.py'
        node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_double_balance_tray_site_id')
        node.decorator_list=[]
        module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
        ns={};exec(compile(ast.fix_missing_locations(module),str(path),'exec'),ns)
        robot=types.SimpleNamespace(find_sites=lambda name:([17],[name]))
        env=types.SimpleNamespace(scene={'robot':robot})
        result=f.preflight_cache(env,T,W,types.SimpleNamespace(**ns))
        self.assertEqual(result['site_id'],17);self.assertTrue(result['original_dict_and_key_restored'])
        self.assertEqual(env._double_balance_tray_site_ids,{(id(robot),'stage11_head_material_frame'):17})

    def test_all_accepted_import_bindings_installed_and_restored(self):
        class Gate:
            def __init__(self):self.saved=[]
            def patch(self,obj,key,value):self.saved.append((obj,key,getattr(obj,key)));setattr(obj,key,value)
        gate=Gate();f.install(gate)
        for obj,key,old in gate.saved:
            self.assertIs(getattr(obj,key),f.FormalOwnerSnapshot if key=='StrictOwnerSnapshot' else f.owner_state)
        self.assertEqual({m.__name__ for m,k,v in gate.saved if k=='StrictOwnerSnapshot'},
            {'native_state_bridge','macro_state_bridge','sensor_macro_bridge','pilot_state','portable_state'})
        for obj,key,old in reversed(gate.saved):setattr(obj,key,old)

    def test_new_bounds_cover_original_macro_cost(self):
        p=json.loads((P/'plan.json').read_bytes());cost=macro_cost()
        self.assertEqual((p['formal_steps'],p['max_macros']),(2,20))
        self.assertEqual(p['caps']['integrations'],20*352)
        self.assertEqual(p['caps']['GPU_forward'],20*1413+80)
        for k,v in cost.items():self.assertGreaterEqual(p['caps'][k],20*v,k)
        self.assertEqual((p['caps']['BAM_compute'],p['caps']['Observation_noise']),(23,16))
        self.assertEqual(p['graph_limits']['launch'],p['caps']['GPU_forward'])

    def test_only_remaining_cases_preflight_before_loop(self):
        src=(P/'physics_worker.py').read_text();tree=ast.parse(src)
        loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and [v.id for v in n.target.elts]==['index','case'])
        self.assertEqual(ast.literal_eval(loop.iter.args[0]),('auto-timeout','manual-failure'))
        preflight=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='preflight_cache')
        self.assertLess(preflight.lineno,loop.lineno)
        self.assertIn('data_guard.assert_restored();owner_guard.assert_restored()',src)
        self.assertNotIn('for invalid in',src)
        self.assertNotIn("case == 'normal'",src)

    def test_current_ledger_and_failed_attempt_retained(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            counts=dict(integrations_attempted=0,integrations_completed=0)
            caps=dict(integrations=1,GPU_forward=4,CPU_forward=0)
            j=Journal(d,counts,caps);j.progress(True)
            r=json.loads((Path(d)/'progress.json').read_bytes())
            self.assertEqual((r['cumulative_completed'],r['cumulative_attempted']),(670872,670873))
            _,B=make_classes();b=B({**MACRO_LIMITS,'integrations':1},counts,j,lambda:None)
            b.run('integrations',lambda:None);j.close()
            self.assertTrue(review_ledger(Path(d)/'native-ledger.jsonl')['exact'])
            r=json.loads((Path(d)/'progress.json').read_bytes())
            self.assertEqual((r['cumulative_completed'],r['cumulative_attempted']),(670873,670874))

    def test_relocated_sources_match_original_and_public_identities(self):
        baseline=json.loads((R/'fixtures/nominal-rollout-source-hashes.json').read_bytes())
        for dirname,expected in [('accepted',92),('host-candidate',13)]:
            paths=[p for p in (P/dirname).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
            self.assertEqual(len(paths),expected)
            originals={n:h for n,h in baseline.items() if n.startswith(dirname+'/')}
            self.assertEqual(len(originals),expected)
            verified={verify_source_file(REPO,ORIGINAL+n,h) for n,h in originals.items()}
            self.assertEqual(verified,{p.resolve() for p in paths})
        for n in ('formal_task.py','formal_geometry.py','formal_config.py','formal_command.py','checkpoint_review.py','return_transport.py'):
            self.assertEqual(verify_source_file(REPO,ORIGINAL+n,baseline[n]),(P/n).resolve())

    def test_compile_all_and_no_local_gpu_claim(self):
        for p in R.rglob('*.py'):compile(p.read_text(),str(p),'exec')
        d=json.loads((P/'host-candidate/regression-review.json').read_bytes())
        self.assertEqual((d['checks_passed'],d['checks_total']),(29,29))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,help='Optional path for a new check report; recorded evidence is not overwritten')
    args=parser.parse_args()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(status='PASS' if result.wasSuccessful() else 'FAIL',checks=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),old29_reexecuted=False,
        v2_physical_reexecuted=False,native_integrations=0,CUDA_verified=False,PPO=0,cloud_training=0)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
