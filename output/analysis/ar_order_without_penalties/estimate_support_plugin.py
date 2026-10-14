"""Exploratory plug-in support estimates on the two existing simulation sets.

All estimates use only observed data. The known h=1 is a reporting benchmark.
The previously evaluated validation set is reused here, not fresh validation.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from Code.Code_AR_processes.ar_tools.functions import jones_ar_coefficients, simulate_uniform_ar

OUT = Path(__file__).resolve().parent
INPUT = ROOT / 'Code/Code_AR_processes/outputs/csv/ar_width_comparison'
SETTINGS = json.loads((INPUT / 'configuration.json').read_text())
RADII = pd.read_csv(OUT / 'known_bound_radii.csv')
KNOWN_ORIGINAL = pd.read_csv(OUT / 'selections.csv')
KNOWN_ORIGINAL = KNOWN_ORIGINAL[KNOWN_ORIGINAL.method == 'known_bound_minimum_feasible']
KNOWN_VALIDATION = pd.read_csv(OUT / 'validation_selections.csv')
MAXIMUM_ORDER = 8


def minimum_feasible_order(response, lags, estimated_radius):
    for q in range(MAXIMUM_ORDER + 1):
        if q == 0:
            feasible = np.max(np.abs(response)) <= estimated_radius
        else:
            design = lags[:, :q]
            fit = linprog(
                np.zeros(q), A_ub=np.vstack([-design, design]),
                b_ub=np.r_[estimated_radius-response, estimated_radius+response],
                bounds=[(None, None)]*q, method='highs',
                options={'primal_feasibility_tolerance': 1e-9})
            if fit.status not in (0, 2):
                raise RuntimeError(fit.message)
            feasible = fit.success
            if feasible and np.max(np.abs(response-design@fit.x)) > estimated_radius+1e-8:
                raise RuntimeError('Numerical feasibility violation')
        if feasible:
            return q
    return None


rows = []
with threadpool_limits(limits=1):
    for dataset, count in [('original', 50), ('reused_validation', 100)]:
        for p in (1, 2, 3):
            for replication in range(1, count+1):
                if dataset == 'original':
                    seed = [SETTINGS['seeds'][str(p)], replication-1, 0]
                    known = KNOWN_ORIGINAL
                else:
                    seed = [20260918+p, replication-1, 123]
                    known = KNOWN_VALIDATION
                rng = np.random.default_rng(np.random.SeedSequence(seed))
                phi = jones_ar_coefficients(p, rng=rng)
                series = simulate_uniform_ar(phi, 2000, 1000, 1., rng=rng)
                response = series[MAXIMUM_ORDER:]
                lags = np.column_stack([
                    series[MAXIMUM_ORDER-j:len(series)-j]
                    for j in range(1, MAXIMUM_ORDER+1)])
                coefficients = np.linalg.lstsq(lags, response, rcond=None)[0]
                residuals = response-lags@coefficients
                estimates = {
                    'pilot_ols_maximum_absolute_residual': np.max(np.abs(residuals)),
                    'pilot_ols_uniform_moment': np.sqrt(3*np.mean(residuals**2)),
                }
                known_order = int(known[(known.true_order == p) &
                                       (known.replication == replication)].selected_order.iloc[0])
                rows.append(dict(dataset=dataset, true_order=p, replication=replication,
                                 method='known_radius', estimated_radius=1.,
                                 selected_order=known_order, feasible=True))
                for method, radius in estimates.items():
                    selected = minimum_feasible_order(response, lags, float(radius))
                    rows.append(dict(dataset=dataset, true_order=p, replication=replication,
                                     method=method, estimated_radius=float(radius),
                                     selected_order=selected, feasible=selected is not None))
                if dataset == 'original':
                    group = RADII[(RADII.true_order == p) &
                                  (RADII.replication == replication)].sort_values('candidate_order')
                    radius = float(group[group.candidate_order == MAXIMUM_ORDER].radius.iloc[0])
                    # Tolerance is only for equality of separately solved numerical LP values.
                    eligible = group[group.radius <= radius+1e-8]
                    selected = int(eligible.candidate_order.min())
                    rows.append(dict(dataset=dataset, true_order=p, replication=replication,
                                     method='pilot_minimax_radius', estimated_radius=radius,
                                     selected_order=selected, feasible=True))
            print(f'Completed {dataset} AR({p})', flush=True)

results = pd.DataFrame(rows)
results.to_csv(OUT / 'estimated_support_selections.csv', index=False)
summary = []
for (dataset, method, p), group in results.groupby(['dataset', 'method', 'true_order']):
    summary.append(dict(dataset=dataset, method=method, true_order=p,
                        correct=int((group.selected_order == p).sum()),
                        under=int((group.selected_order < p).sum()),
                        over=int((group.selected_order > p).sum()),
                        infeasible=int((~group.feasible).sum()), total=len(group),
                        estimated_radius_below_true=int((group.estimated_radius < 1).sum()),
                        median_estimated_radius=float(group.estimated_radius.median())))
summary = pd.DataFrame(summary)
summary.to_csv(OUT / 'estimated_support_summary.csv', index=False)
print(summary.to_string(index=False), flush=True)
