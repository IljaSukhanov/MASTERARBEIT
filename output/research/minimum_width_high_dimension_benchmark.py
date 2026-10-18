"""Bounded feasibility measurements; does not modify thesis or AR analysis code."""
from pathlib import Path
import csv
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).with_suffix('.json')
sys.path.insert(0, str(ROOT))


def worker(spec):
    import numpy as np
    from scipy.spatial import ConvexHull, HalfspaceIntersection
    from Code.Code_AR_processes.ar_tools.functions import (
        jones_ar_coefficients, simulate_uniform_ar, sliding_window_point_cloud,
    )

    started = time.perf_counter()
    if spec['source'] == 'saved_ar3':
        path = ROOT / ('Code/Code_AR_processes/outputs/csv/'
                       'ar3_order_selection_warm_starts/processes/'
                       f"process_{spec['replication']:03d}.csv")
        series = np.array([float(r['X_t']) for r in csv.DictReader(path.open())])
        points = sliding_window_point_cloud(series, 10)[:, :spec['k']]
    else:
        order = spec['order']
        rep0 = spec['replication'] - 1
        seed = 20260923
        phi = jones_ar_coefficients(order, rng=np.random.default_rng(
            np.random.SeedSequence([seed, order, rep0, 0])))
        series = simulate_uniform_ar(phi, sample_size=2000, burnin=1000,
            half_width=1.0, rng=np.random.default_rng(
                np.random.SeedSequence([seed, order, rep0, 1])))
        points = sliding_window_point_cloud(series, spec['k'])
        estimates = ROOT / ('Code/Code_AR_processes/outputs/csv/'
            'ar3_minimum_width_k4/four_method_comparison/per_process_estimates.csv')
        saved = next(r for r in csv.DictReader(estimates.open())
                     if int(r['order']) == order
                     and int(r['replication']) == spec['replication'])
        expected = np.array([float(saved[f'true_phi_{i+1}']) for i in range(order)])
        assert np.allclose(phi, expected, atol=1e-14, rtol=1e-14)

    np_start = time.perf_counter()
    hull = ConvexHull(points)
    vertices = points[hull.vertices]
    record = dict(spec, sample_size=len(series), cloud_points=len(points),
        hull_vertices=len(vertices), original_simplicial_facets=len(hull.simplices),
        first_hull_seconds=time.perf_counter()-np_start,
        difference_rows=len(vertices)**2,
        difference_array_mib=len(vertices)**2*spec['k']*8/2**20)
    print(json.dumps(dict(event='original_hull', **record)), flush=True)
    difference_start = time.perf_counter()
    if spec.get('method') == 'normalized_slab':
        ones = np.ones(len(vertices))
        halfspaces = np.vstack((
            np.column_stack((-vertices, ones, np.zeros(len(vertices)))),
            np.column_stack((vertices, -ones, -ones)),
        ))
        interior = np.r_[np.zeros(spec['k']), -0.5]
        polytope = HalfspaceIntersection(halfspaces, interior)
        candidates = polytope.intersections[:, :spec['k']]
        norms = np.linalg.norm(candidates, axis=1)
        best = int(np.argmax(norms))
        direction = candidates[best] / norms[best]
        width = float(np.ptp(points @ direction))
        error = abs(width - 1.0 / norms[best])
        assert error < 1e-8 * max(1.0, width)
        record.update(halfspace_count=len(halfspaces),
            normalized_polytope_vertices=len(candidates),
            normalized_polytope_seconds=time.perf_counter()-difference_start,
            geometry_seconds=time.perf_counter()-np_start,
            elapsed_seconds=time.perf_counter()-started,
            minimum_width=width, support_crosscheck_error=error)
        print(json.dumps(dict(event='result', **record)), flush=True)
        return
    differences = (vertices[:, None, :] - vertices[None, :, :]).reshape(-1, spec['k'])
    print(json.dumps({'event': 'differences_built'}), flush=True)
    difference_hull = ConvexHull(differences)
    normals = difference_hull.equations[:, :-1]
    distances = -difference_hull.equations[:, -1] / np.linalg.norm(normals, axis=1)
    best = int(np.argmin(distances))
    direction = normals[best] / np.linalg.norm(normals[best])
    width = float(np.ptp(points @ direction))
    error = abs(width - float(distances[best]))
    assert error < 1e-8 * max(1.0, width)
    record.update(difference_simplicial_facets=len(difference_hull.simplices),
        difference_hull_seconds=time.perf_counter()-difference_start,
        geometry_seconds=time.perf_counter()-np_start,
        elapsed_seconds=time.perf_counter()-started,
        minimum_width=width, support_crosscheck_error=error)
    print(json.dumps(dict(event='result', **record)), flush=True)


def bounded_run(spec, seconds=40.0, memory_mib=512):
    import psutil
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
        '--worker', json.dumps(spec)], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, env=env)
    monitored = psutil.Process(process.pid)
    started, peak, reason = time.perf_counter(), 0, None
    while process.poll() is None:
        try:
            peak = max(peak, monitored.memory_info().rss)
        except psutil.NoSuchProcess:
            break
        if peak > memory_mib*2**20:
            reason = 'memory_limit'
        elif time.perf_counter()-started > seconds:
            reason = 'time_limit'
        if reason:
            process.kill()
            break
        time.sleep(0.05)
    stdout, stderr = process.communicate()
    events = [json.loads(line) for line in stdout.splitlines() if line.startswith('{')]
    details = next((r for r in reversed(events) if r['event'] in ('result','original_hull')), {})
    details.pop('event', None)
    result = dict(spec, **{k:v for k,v in details.items() if k not in spec},
        status=reason or ('success' if process.returncode == 0 else 'error'),
        wall_seconds=time.perf_counter()-started, peak_rss_mib=peak/2**20)
    if stderr:
        result['stderr'] = stderr[-3000:]
    print(json.dumps(result), flush=True)
    return result


def main(slab=False):
    cases = [dict(source='saved_ar3', order=3, replication=50, k=k) for k in (4,5,6)]
    cases += [dict(source='regenerated_comparison', order=p, replication=1, k=p+1)
              for p in (4,5)]
    cases += [dict(source='saved_ar3', order=3, replication=50, k=k) for k in (7,8,9,10)]
    destination = REPORT
    if slab:
        cases = [dict(case, method='normalized_slab') for case in cases]
        destination = REPORT.with_name('minimum_width_high_dimension_slab_benchmark.json')
    report = dict(date='2026-10-18', time_limit_seconds_per_case=40,
        memory_limit_mib_per_case=512, arithmetic='floating point',
        method=('ConvexHull(P); enumerate normalized slab polytope vertices; maximum norm'
                if slab else 'ConvexHull(P); ConvexHull(V-V); minimum facet distance'),
        note='Facet counts include the simplicial triangulation returned by Qhull.',
        cases=[])
    for case in cases:
        result = bounded_run(case)
        report['cases'].append(result)
        destination.write_text(json.dumps(report, indent=2)+'\n')
        if case['k'] >= 7 and result['status'] != 'success':
            report['higher_dimensions_not_attempted'] = list(range(case['k']+1,11))
            destination.write_text(json.dumps(report, indent=2)+'\n')
            break
    print(str(destination), flush=True)


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':
        worker(json.loads(sys.argv[2]))
    else:
        main(slab='--slab' in sys.argv)
