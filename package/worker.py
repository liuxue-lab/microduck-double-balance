"""Bounded local geometry experiment. No simulator or training imports/calls."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / 'probe-plan.json').read_text())
ZERO = dict(physics_integrations=0, simulator_forwards=0, PPO=0, cloud_training=0,
            production_integration_authorized=False, cloud_gpu_allowed=False,
            numerical_tolerances_changed=False, installed_sources_modified=False)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


class Probe:
    def __init__(self, output):
        self.output = output
        self.started = time.monotonic()
        self.counts = dict(original_queries_reserved=0, original_queries_completed=0,
                           candidate_queries_reserved=0, candidate_queries_completed=0,
                           explicit_query_kernel_launches_reserved=0)
        self.state = dict(status='RUNNING', phase='preflight', **ZERO)
        self.report = dict(scope=PLAN['scope'], **ZERO)

    def progress(self, phase, **extra):
        self.state.update(phase=phase, elapsed_seconds=time.monotonic() - self.started,
                          **self.counts, **extra)
        write(self.output / 'progress.json', self.state)

    def reserve(self, kind, count):
        require(time.monotonic() - self.started < PLAN['max_seconds'], 'Probe time cap reached')
        self.counts[kind + '_queries_reserved'] += count
        self.counts['explicit_query_kernel_launches_reserved'] += 12 if kind == 'candidate' else 1
        for name in ('original', 'candidate'):
            require(self.counts[name + '_queries_reserved'] <= PLAN['expected_' + name + '_queries'],
                    'Geometry query cap exceeded')
        require(self.counts['explicit_query_kernel_launches_reserved'] <=
                PLAN['expected_explicit_query_kernel_launches'], 'Kernel launch cap exceeded')

    def source_pins(self):
        dist = importlib.metadata.distribution('warp-lang')
        require(dist.version == PLAN['warp_version'], 'Existing Warp version differs; no install attempted')
        pins = {name: sha(dist.locate_file(name)) for name in PLAN['warp_source_pins']}
        require(pins == PLAN['warp_source_pins'], 'Existing Warp source pins differ; preserve and report')
        return pins

    def parity(self, actual, expected, indices, label):
        np = self.np
        a = np.ascontiguousarray(actual, dtype=np.float64)
        b = np.ascontiguousarray(expected, dtype=np.float64)
        require(a.shape == b.shape and a.shape[-1] == 16, 'Geometry result shape differs')
        bad = np.argwhere(a.view(np.uint64) != b.view(np.uint64))
        if len(bad) or not np.isfinite(a).all() or not np.all(a[:, 10] == 1):
            np.savez_compressed(self.output / 'parity-failure.npz', actual=a, expected=b,
                                indices=np.asarray(indices, dtype=np.int32), mismatches=bad)
            detail = dict(label=label, bitwise_equal=False, mismatch_fields=int(len(bad)),
                          first_case=int(indices[int(bad[0, 0])]) if len(bad) else None)
            write(self.output / 'parity-failure.json', detail)
            raise RuntimeError('Geometry bitwise parity failed: ' + json.dumps(detail))

    def fixture_reference(self, values):
        np = self.np
        for result, reference in zip(values, self.cases['fixture_reference'], strict=True):
            require(np.isfinite(result).all() and result[10] == 1, 'Invalid fixture geometry')
            require((result[3] < 0) == (reference[3] < 0) and result[13] == reference[13]
                    and np.max(abs(result[2:4] - reference[2:4])) < 1e-13
                    and np.max(abs(result[4:10] - reference[4:10])) < 1e-10,
                    'Frozen independent fixture comparison failed')
            if reference[0] >= 0:
                require(result[0] == reference[0] and result[1] >= 0, 'Vertex fast path changed')

    def one(self, kind, index):
        wp, np = self.wp, self.np
        self.reserve(kind, 1)
        wp.synchronize_device(self.device)
        t0 = time.perf_counter()
        self.single_p.assign(self.cases['positions'][index:index+1])
        self.single_r.assign(self.cases['rotations'][index:index+1])
        wp.synchronize_device(self.device)
        t1 = time.perf_counter()
        if kind == 'original':
            wp.launch(self.original.convex_geometry, dim=1,
                      inputs=[self.vertices, self.facets, self.single_p, self.single_r,
                              0, PLAN['vertices'], PLAN['facets'], PLAN['radius'], self.single_out],
                      device=self.device)
        else:
            self.single_parallel.launch(self.single_p, self.single_r, PLAN['radius'], self.single_out, 1)
        value = self.single_out.numpy().copy()
        t2 = time.perf_counter()
        self.counts[kind + '_queries_completed'] += 1
        self.parity(value, self.cases['expected_gpu'][index:index+1], [index], kind)
        return value[0], dict(total_seconds=t2-t0, query_and_readback_seconds=t2-t1)

    def run(self):
        require(Path.home() == Path('/home/lx') and os.geteuid() != 0,
                'Run only on lx local workstation; cloud/root execution is refused')
        require(Path(sys.prefix) == Path(PLAN['runtime_python']).parent.parent,
                'Use the pinned existing local virtual environment')
        require(self.output == ROOT.parent and (self.output / 'started.json').is_file(),
                'Start through the standalone launcher')
        self.progress('preflight')
        pins = self.source_pins()
        write(self.output / 'warp-pins-before.json', pins)
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid', '--format=csv,noheader'],
                             capture_output=True, text=True, check=True, timeout=10)
        names = [line.strip() for line in gpu.stdout.splitlines() if line.strip()]
        require(len(names) == 1 and '5060' in names[0] and '5090' not in names[0],
                'Exactly one local RTX 5060 is required; paid cloud GPU is refused')
        self.report['nvidia_smi_identity'] = names
        self.progress('import-warp')
        import numpy as np
        import warp as wp
        self.np, self.wp = np, wp
        cache = Path(os.environ['MD12_GEOMETRY_PRIVATE_CACHE'])
        require(cache.is_dir() and cache.parent == Path('/tmp') and cache.name.startswith('md12-geometry-'),
                'Private short compiler cache missing')
        wp.config.kernel_cache_dir = str(cache / 'warp')
        wp.init()
        devices = wp.get_cuda_devices()
        require(len(devices) == 1 and '5060' in devices[0].name and '5090' not in devices[0].name,
                'Warp selected a different GPU')
        self.device = devices[0]
        self.report.update(device_name=self.device.name, device_alias=str(self.device),
                           warp_version=wp.__version__, private_cache=str(cache))
        with np.load(ROOT / 'cases.npz', allow_pickle=False) as source:
            self.cases = {name: source[name].copy() for name in source.files}
        c, cap = self.cases, PLAN['batch_capacity']
        require(c['positions'].shape == (PLAN['case_count'], 2, 3)
                and c['rotations'].shape == (PLAN['case_count'], 2, 3, 3)
                and c['expected_gpu'].shape == (PLAN['case_count'], 16)
                and c['vertices'].shape == (PLAN['vertices'], 3)
                and c['facets'].shape == (PLAN['facets'], 3), 'Fixed input shape differs')
        require(c['vertices'].dtype == np.float32 and c['facets'].dtype == np.int32,
                'Fixed mesh dtype differs')
        require(all(np.isfinite(v).all() for v in c.values()), 'Nonfinite fixed inputs')
        import convex_pose_kernel as original
        import parallel_hull
        self.original = original
        self.progress('compile-original')
        wp.load_module(original, device=self.device)
        self.progress('compile-parallel')
        wp.load_module(parallel_hull, device=self.device)
        self.vertices = wp.array(c['vertices'], dtype=wp.vec3, device=self.device)
        self.facets = wp.array(c['facets'], dtype=wp.vec3i, device=self.device)
        self.single_p = wp.zeros((1, 2), dtype=wp.vec3d, device=self.device)
        self.single_r = wp.zeros((1, 2), dtype=wp.mat33d, device=self.device)
        self.single_out = wp.zeros((1, 16), dtype=wp.float64, device=self.device)
        self.single_parallel = parallel_hull.ParallelHull(self.vertices, self.facets, 1, self.device)
        batch_p = wp.zeros((cap, 2), dtype=wp.vec3d, device=self.device)
        batch_r = wp.zeros((cap, 2), dtype=wp.mat33d, device=self.device)
        batch_out = wp.zeros((cap, 16), dtype=wp.float64, device=self.device)
        batch_parallel = parallel_hull.ParallelHull(self.vertices, self.facets, cap, self.device)
        benchmark_ids = np.linspace(PLAN['fixtures'], PLAN['case_count']-1,
                                    PLAN['benchmark_cases'], dtype=np.int32)
        original_ids = np.concatenate((np.arange(PLAN['fixtures'], dtype=np.int32), benchmark_ids))
        baseline = []
        self.progress('original-parity', done=0, total=len(original_ids))
        for j, index in enumerate(original_ids):
            value, _ = self.one('original', int(index))
            baseline.append(value)
            if j % 8 == 7 or j + 1 == len(original_ids):
                self.progress('original-parity', done=j+1, total=len(original_ids))
        baseline = np.asarray(baseline)
        self.fixture_reference(baseline[:PLAN['fixtures']])
        np.savez_compressed(self.output / 'original-values.npz', values=baseline, indices=original_ids)
        padded_p = np.zeros((cap, 2, 3), dtype=np.float64)
        padded_r = np.zeros((cap, 2, 3, 3), dtype=np.float64)
        for repeat in range(PLAN['repeated_candidate_passes']):
            values = np.empty_like(c['expected_gpu'])
            phase = 'parallel-parity-' + str(repeat+1)
            self.progress(phase, done=0, total=PLAN['case_count'])
            for start in range(0, PLAN['case_count'], cap):
                end = min(start+cap, PLAN['case_count'])
                count = end-start
                padded_p[:count], padded_r[:count] = c['positions'][start:end], c['rotations'][start:end]
                self.reserve('candidate', count)
                batch_p.assign(padded_p)
                batch_r.assign(padded_r)
                batch_parallel.launch(batch_p, batch_r, PLAN['radius'], batch_out, count)
                actual = batch_out.numpy()[:count].copy()
                self.counts['candidate_queries_completed'] += count
                self.parity(actual, c['expected_gpu'][start:end], range(start, end), phase)
                values[start:end] = actual
                if start % (cap*8) == 0 or end == PLAN['case_count']:
                    self.progress(phase, done=end, total=PLAN['case_count'])
            self.fixture_reference(values[:PLAN['fixtures']])
            np.savez_compressed(self.output / ('parallel-values-' + str(repeat+1) + '.npz'), values=values)
        timings = []
        total = PLAN['benchmark_cases'] * PLAN['benchmark_rounds'] * 2
        self.progress('single-query-benchmark', done=0, total=total)
        for repeat in range(PLAN['benchmark_rounds']):
            for j, index in enumerate(benchmark_ids):
                # Alternate execution order; compare one pose at a time on both implementations.
                order = ('original', 'candidate') if (repeat+j) % 2 == 0 else ('candidate', 'original')
                for kind in order:
                    _, elapsed = self.one(kind, int(index))
                    timings.append(dict(kind=kind, repeat=repeat, case=int(index), **elapsed))
                self.progress('single-query-benchmark', done=len(timings), total=total)
        write(self.output / 'benchmark-timings.json', timings)
        summary = {}
        for metric in ('total_seconds', 'query_and_readback_seconds'):
            medians = {kind: float(np.median([row[metric] for row in timings if row['kind'] == kind]))
                       for kind in ('original', 'candidate')}
            summary[metric] = dict(median=medians, speedup=medians['original']/medians['candidate'])
        for kind in ('original', 'candidate'):
            require(self.counts[kind + '_queries_completed'] == PLAN['expected_' + kind + '_queries'],
                    'Final geometry query accounting differs')
        require(self.counts['explicit_query_kernel_launches_reserved'] ==
                PLAN['expected_explicit_query_kernel_launches'], 'Final kernel launch accounting differs')
        after = self.source_pins()
        write(self.output / 'warp-pins-after.json', after)
        require(after == pins, 'Installed source changed during probe')
        banned = [name for name in sys.modules if name.split('.')[0] in
                  ('torch', 'mujoco', 'mujoco_warp', 'mjlab', 'rsl_rl')]
        require(not banned, 'Simulator/training module unexpectedly imported: ' + repr(banned))
        speedup = summary['query_and_readback_seconds']['speedup']
        self.report.update(status='GEOMETRY_PARITY_PASS_BENCHMARK_COMPLETE', bitwise_equal=True,
                           fixed_cases=PLAN['case_count'], repeated_passes=2,
                           independent_fixtures=PLAN['fixtures'], benchmark=summary,
                           timing_scope='One pose per query; host launch + synchronous result copy; excludes WAL and full simulation',
                           candidate_for_integration_review=speedup >= PLAN['minimum_geometry_speedup_for_further_review'],
                           full_simulation_speedup_claimed=False,
                           unimported_modules=['torch', 'mujoco', 'mujoco_warp', 'mjlab', 'rsl_rl'])
        self.progress('finished', status=self.report['status'])

    def finish(self):
        self.report.update(elapsed_seconds=time.monotonic()-self.started, **self.counts)
        write(self.output / 'result.json', self.report)
        print('GEOMETRY_RESULT=' + json.dumps(self.report, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    probe = Probe(args.output.resolve())
    try:
        probe.run()
    except BaseException as error:
        probe.report.update(status='GEOMETRY_PROBE_FAILED_PRESERVED', error=str(error),
                            bitwise_equal=False)
        probe.progress('failed', status=probe.report['status'])
        traceback.print_exc()
        probe.finish()
        return 1
    probe.finish()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
