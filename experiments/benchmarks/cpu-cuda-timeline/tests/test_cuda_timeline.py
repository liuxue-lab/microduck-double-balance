"""Trace interval accounting, partial-capture refusal and actual CPU profiler lifecycle."""
from contextlib import nullcontext
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'package'))
from cuda_timeline import CudaTimeline,summarize,union_length,FIRST,LAST,complete_timeline
import timeline_contract as contract
import timeline_ops

LABELS=('physics.total','physics.original_sim_step','audit.before_read_and_check',
        'audit.after_read_and_check','audit.before_python','audit.after_python','scene.write_total')


def event(cat,name,ts,dur):return dict(ph='X',cat=cat,name=name,ts=ts,dur=dur,pid=1,tid=1)


def fixture():
    events=[]
    for step in range(FIRST,LAST+1):
        start=(step-FIRST)*200.
        events.append(event('user_annotation','md12.env.%d'%step,start,100.))
        # Overlap: 10..60 and 40..80 => kernel sum90, union70; memcpy80..90 adds10.
        events.extend((event('kernel','a',start+10,50),event('kernel','b',start+40,40),
                       event('gpu_memcpy','DtoH',start+80,10)))
        # Between-step work must be excluded despite being present in the raw trace.
        events.append(event('kernel','policy-excluded',start+120,50))
        for label in LABELS:
            events.extend(event('user_annotation','md12.'+label,start+i*10,3) for i in range(10))
    return dict(traceEvents=events)


def cpu_profiler_factory(torch, inject_gpu=False):
    """Use the real PyTorch CPU exporter. Synthetic GPU rows are CPU-test-only."""
    factory=torch.profiler.profile
    def create(**kwargs):
        kwargs['activities']=[torch.profiler.ProfilerActivity.CPU]
        result=factory(**kwargs)
        export=result.export_chrome_trace
        if inject_gpu:
            def export_with_synthetic_gpu(path):
                export(path)
                data=json.loads(Path(path).read_bytes())
                for e in list(data['traceEvents']):
                    if e.get('ph')=='X' and e.get('cat')=='user_annotation' and e.get('name','').startswith('md12.env.'):
                        data['traceEvents'].append(event('kernel','CPU_TEST_ONLY_SYNTHETIC_GPU',e['ts']+e['dur']*.25,e['dur']*.5))
                Path(path).write_text(json.dumps(data))
            result.export_chrome_trace=export_with_synthetic_gpu
        return result
    return create


class SummaryTests(unittest.TestCase):
    def test_overlap_not_double_counted_and_between_step_work_excluded(self):
        r=summarize(fixture())
        self.assertEqual(r['status'],'CUDA_TIMELINE_CAPTURED')
        self.assertAlmostEqual(r['cuda_activity_union_ms'],.64)
        self.assertAlmostEqual(r['kernel_union_ms'],.56)
        self.assertAlmostEqual(r['no_recorded_cuda_activity_in_environment_windows_ms'],.16)
        self.assertEqual(r['cuda_kernel_events'],16)
        self.assertNotIn('policy-excluded',str(r['ranked_events']))
        self.assertEqual(r['scopes']['md12.physics.total']['count'],80)

    def test_clip_gpu_events_to_each_window(self):
        d=fixture();d['traceEvents'].append(event('gpu_memset','cross-boundary',90,120))
        r=summarize(d)
        self.assertAlmostEqual(r['cuda_activity_union_ms'],.66)
        self.assertEqual(r['ranked_events']['gpu_memset'][0]['count'],1)
        self.assertAlmostEqual(r['ranked_events']['gpu_memset'][0]['clipped_total_ms'],.02)

    def test_missing_cuda_activity_never_passes(self):
        d=fixture();d['traceEvents']=[e for e in d['traceEvents'] if e['cat']!='kernel']
        self.assertEqual(summarize(d)['status'],'CUDA_TIMELINE_INCOMPLETE_PRESERVE')

    def test_incomplete_cuda_step_and_scope_counts(self):
        d=fixture();d['traceEvents']=[e for e in d['traceEvents'] if not(e['cat']=='kernel' and e['ts']<100)]
        self.assertFalse(summarize(d)['every_step_has_cuda_kernel'])
        d=fixture();d['traceEvents'].pop()
        self.assertFalse(summarize(d)['required_scope_counts_complete'])

    def test_missing_duplicate_and_overlapping_markers_fail(self):
        d=fixture();d['traceEvents'].pop(0)
        with self.assertRaises(RuntimeError):summarize(d)
        d=fixture();d['traceEvents'].append(copy.deepcopy(d['traceEvents'][0]))
        with self.assertRaises(RuntimeError):summarize(d)
        d=fixture();d['traceEvents'][0]['dur']=300
        with self.assertRaises(RuntimeError):summarize(d)

    def test_malformed_time_and_empty_trace_fail(self):
        for value in (-1,float('nan'),float('inf'),'5',True):
            d=fixture();d['traceEvents'][1]['dur']=value
            with self.assertRaises(RuntimeError):summarize(d)
        with self.assertRaises(RuntimeError):summarize({})

    def test_union_empty_nested_and_disjoint(self):
        self.assertEqual(union_length([]),0.)
        self.assertEqual(union_length([(1,8),(3,4),(8,10),(20,22)]),11)

    def test_measured_profiles_never_selected_as_speed_improvement(self):
        for passed in (False,True):
            r=timeline_ops.recommendation(dict(full_environment_step_ms={'mean':25}),
                dict(cuda_timeline={'summary':{'status':'CUDA_TIMELINE_CAPTURED'}}),{'passed':passed})
            self.assertEqual(r['choice'],'KEEP_INFERENCE_ONLY');self.assertTrue(r['diagnostic_only'])
            self.assertNotIn('mean_reduction_fraction',r)
            self.assertFalse(r['physical_robot_in_scope'])

    def test_latest_reviewed_report_exact_pin(self):
        with patch.object(contract.runtime,'reviewed') as reviewed:
            r=contract.reviewed(ROOT/'reference/scalar-audit-measured.json')
            self.assertEqual(r['status'],'PC_AUDITFAST_MEASURED');reviewed.assert_called_once()


class LifecycleTests(unittest.TestCase):
    def test_real_cpu_profiler_exports_markers_but_cannot_claim_cuda(self):
        import torch
        with tempfile.TemporaryDirectory() as d:
            recorder=CudaTimeline(torch,Path(d)/'trace.json')
            obj=NS(run=lambda:torch.ones(3).sum().item());original=obj.run
            recorder.attach(obj,'run','cpu.test')
            with patch.object(torch.profiler,'profile',side_effect=cpu_profiler_factory(torch)), \
                 patch.object(torch.profiler,'supported_activities',return_value={torch.profiler.ProfilerActivity.CUDA}):
                for i in range(1,LAST+1):
                    with recorder.window(i):obj.run()
                recorder.close();recorder.close()
            r=recorder.report()
            self.assertIs(obj.run,original)
            self.assertEqual(r['completed_steps'],list(range(FIRST,LAST+1)))
            self.assertEqual(r['summary']['status'],'CUDA_TIMELINE_INCOMPLETE_PRESERVE')
            self.assertEqual(r['summary']['scopes']['md12.cpu.test']['count'],8)
            self.assertFalse(complete_timeline(r))

    def test_missing_cuda_rejected_before_start_and_restored(self):
        api=NS(ProfilerActivity=NS(CUDA='CUDA'),supported_activities=lambda:set())
        with tempfile.TemporaryDirectory() as d:
            r=CudaTimeline(NS(profiler=api),Path(d)/'trace.json')
            obj=NS(run=lambda:42);original=obj.run;r.attach(obj,'run','sample')
            with self.assertRaisesRegex(RuntimeError,'CUDA profiler activity unavailable'):
                with r.window(FIRST):pass
            r.close();self.assertIs(obj.run,original);self.assertFalse(r.report()['started'])

    def test_failure_inside_window_preserves_primary_error_and_restores(self):
        import torch
        with tempfile.TemporaryDirectory() as d:
            recorder=CudaTimeline(torch,Path(d)/'trace.json')
            obj=NS(run=lambda:None);original=obj.run;recorder.attach(obj,'run','sample')
            with patch.object(torch.profiler,'profile',side_effect=cpu_profiler_factory(torch)), \
                 patch.object(torch.profiler,'supported_activities',return_value={torch.profiler.ProfilerActivity.CUDA}):
                with self.assertRaisesRegex(ValueError,'original fault'):
                    with recorder.window(FIRST):raise ValueError('original fault')
                with self.assertRaisesRegex(RuntimeError,'Incomplete'):
                    recorder.close()
            self.assertIs(obj.run,original);self.assertTrue(recorder.closed)
            self.assertIsNotNone(recorder.report()['error']);self.assertFalse(complete_timeline(recorder.report()))

    def test_profiler_stop_failure_restores_hooks_without_retry(self):
        calls=[]
        def stop():calls.append('stop');raise RuntimeError('stop failure')
        with tempfile.TemporaryDirectory() as d:
            r=CudaTimeline(NS(),Path(d)/'trace.json')
            obj=NS(run=lambda:42);original=obj.run;r.attach(obj,'run','sample')
            r.started=r.active=True;r.profiler=NS(stop=stop)
            with self.assertRaisesRegex(RuntimeError,'stop failure'):r.close()
            r.close();self.assertEqual(calls,['stop']);self.assertIs(obj.run,original)
            self.assertIsNotNone(r.report()['error'])

    def test_final_window_failure_not_masked_by_profiler_stop_failure(self):
        calls=[]
        def stop():calls.append('stop');raise RuntimeError('stop failure')
        api=NS(record_function=lambda _:nullcontext())
        with tempfile.TemporaryDirectory() as d:
            r=CudaTimeline(NS(profiler=api),Path(d)/'trace.json')
            r.started=r.active=True;r.profiler=NS(stop=stop)
            with self.assertRaisesRegex(ValueError,'original final-window fault'):
                with r.window(LAST):raise ValueError('original final-window fault')
            r.close();self.assertEqual(calls,['stop']);self.assertTrue(r.closed)
            self.assertIsNotNone(r.report()['error'])


if __name__=='__main__':unittest.main()
