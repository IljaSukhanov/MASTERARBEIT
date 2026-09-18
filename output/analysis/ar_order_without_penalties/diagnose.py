"""Exploratory order diagnostics; no tuned penalty or threshold sequences."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from Code.Code_AR_processes.functions import jones_ar_coefficients, simulate_uniform_ar
from Code.Code_AR_processes.descent import batched_geometric_width_curves

OUT = Path(__file__).resolve().parent
INPUT = ROOT / 'Code/Code_AR_processes/outputs/csv/ar_shared_breakpoint_l_methods'
settings = json.loads((INPUT / 'configuration.json').read_text())
curves = pd.read_csv(INPUT / 'width_curves.csv')
rows, candidate_rows, radius_rows = [], [], []
for (p, replication), curve in curves.groupby(['true_order', 'replication']):
    y = curve.sort_values('k').minimum_width.to_numpy()
    drops = -np.diff(y)
    for method in ('drop_to_tail_energy', 'tail_sse_ratio', 'tail_variance_ratio'):
        scores = []
        for q in range(1, len(y)-1):
            if method == 'drop_to_tail_energy':
                top, bottom = drops[q-1]**2, np.sum(drops[q:]**2)
            else:
                previous_tail, tail = y[q-1:], y[q:]
                top = np.sum((previous_tail-previous_tail.mean())**2)
                bottom = np.sum((tail-tail.mean())**2)
                if method == 'tail_variance_ratio':
                    top /= len(previous_tail)
                    bottom /= len(tail)
            score = top/bottom if bottom > 0 else (np.inf if top > 0 else 0.)
            scores.append(score)
            candidate_rows.append(dict(true_order=p,replication=replication,method=method,
                                       candidate_order=q,numerator=top,denominator=bottom,score=score))
        selected = 1 + int(np.argmax(scores))
        rows.append(dict(true_order=p,replication=replication,method=method,selected_order=selected))

# Known [-1,1] bound is additional information; this diagnostic uses the series,
# not only the estimated spherical minimum-width curve.
with threadpool_limits(limits=1):
    for count, ((p, replication), curve) in enumerate(curves.groupby(['true_order','replication']),1):
        seed = settings['seeds'][str(p)]
        rng = np.random.default_rng(np.random.SeedSequence([seed,replication-1,0]))
        phi = jones_ar_coefficients(int(p),rng=rng)
        np.testing.assert_allclose(phi,json.loads(curve.coefficients.iloc[0]),rtol=0,atol=0)
        series = simulate_uniform_ar(phi,settings['sample_size'],settings['burnin'],1.,rng=rng)
        maximum_order = 8
        response = series[maximum_order:]
        lags = np.column_stack([series[maximum_order-lag:len(series)-lag]
                                for lag in range(1,maximum_order+1)])
        selected = None
        for q in range(maximum_order+1):
            if q == 0:
                radius = float(np.max(np.abs(response)))
                feasible = bool(radius <= 1.)
            else:
                design = lags[:,:q]
                constraints = np.vstack([np.c_[-design,-np.ones(len(response))],
                                         np.c_[design,-np.ones(len(response))]])
                fit = linprog(np.r_[np.zeros(q),1.],A_ub=constraints,
                              b_ub=np.r_[-response,response],
                              bounds=[(None,None)]*q+[(0,None)],method='highs')
                if not fit.success:
                    raise RuntimeError(fit.message)
                radius = float(fit.x[-1])
                # This is a numerical feasibility check at the fixed physical bound.
                feasibility = linprog(np.zeros(q),A_ub=np.vstack([-design,design]),
                                      b_ub=np.r_[1.-response,1.+response],
                                      bounds=[(None,None)]*q,method='highs',
                                      options={'primal_feasibility_tolerance':1e-9})
                if feasibility.status not in (0,2):
                    raise RuntimeError(feasibility.message)
                feasible = bool(feasibility.success)
                if feasible and np.max(np.abs(response-design@feasibility.x)) > 1.+1e-8:
                    raise RuntimeError('Feasible result violates the bound beyond numerical tolerance.')
            radius_rows.append(dict(true_order=p,replication=replication,candidate_order=q,
                                    radius=radius,feasible=feasible))
            if selected is None and feasible:
                selected = q
        if selected is None:
            raise RuntimeError('True order must be feasible with the known innovation bound.')
        rows.append(dict(true_order=p,replication=replication,method='known_bound_minimum_feasible',
                         selected_order=selected))
        if count % 25 == 0:
            print(f'Completed {count}/150 feasibility comparisons',flush=True)

results = pd.DataFrame(rows)
results.to_csv(OUT/'selections.csv',index=False)
pd.DataFrame(candidate_rows).to_csv(OUT/'ratio_candidate_scores.csv',index=False)
pd.DataFrame(radius_rows).to_csv(OUT/'known_bound_radii.csv',index=False)
summary=[]
for (method,p),g in results.groupby(['method','true_order']):
    summary.append(dict(method=method,true_order=p,
                        correct=int((g.selected_order==p).sum()),
                        under=int((g.selected_order<p).sum()),
                        over=int((g.selected_order>p).sum()),total=len(g)))
summary=pd.DataFrame(summary)
summary.to_csv(OUT/'summary.csv',index=False)
print(summary.to_string(index=False),flush=True)

# Demonstrate a population fitting error on an actual AR(3) parameter vector.
# For k=1, exact width is 2 sum |psi_j|; for k=2,3 use fine angular/search
# diagnostics elsewhere rather than claiming unverified global optima here.
# Optimization audit: first ten AR(3) curves, chosen before seeing new fits.
audit=[]
with threadpool_limits(limits=1):
    for replication in range(1,11):
        rng=np.random.default_rng(np.random.SeedSequence([settings['seeds']['3'],replication-1,0]))
        phi=jones_ar_coefficients(3,rng=rng)
        series=simulate_uniform_ar(phi,2000,1000,1.,rng=rng)
        _, revised, _=batched_geometric_width_curves(series,np.arange(1,11),
            seed=settings['seeds']['3'],replication=replication-1,starts=3,iterations=2000,
            initial_step=.2,geometric_decay=.99)
        old=curves[(curves.true_order==3)&(curves.replication==replication)].sort_values('k').minimum_width.to_numpy()
        # Canonical padded recurrence directions are feasible empirical competitors.
        oracle=[]
        for k in range(4,11):
            direction=np.r_[1.,-phi,np.zeros(k-4)]/np.sqrt(1.+phi@phi)
            from Code.Code_AR_processes.functions import sliding_window_point_cloud
            projection=sliding_window_point_cloud(series,k)@direction
            oracle.append(float(np.ptp(projection)))
        for k in range(1,11):
            audit.append(dict(true_order=3,replication=replication,k=k,width_500=old[k-1],
                              width_2000=revised[k-1],
                              recurrence_direction_width=oracle[k-4] if k>=4 else np.nan))
    pd.DataFrame(audit).to_csv(OUT/'optimizer_audit.csv',index=False)
print('Completed optimization audit on the first ten AR(3) curves.',flush=True)
