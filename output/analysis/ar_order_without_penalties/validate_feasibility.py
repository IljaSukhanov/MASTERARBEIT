"""Independent-seed validation of known-bound minimum feasible order."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from Code.Code_AR_processes.ar_tools.functions import jones_ar_coefficients,simulate_uniform_ar
OUT=Path(__file__).resolve().parent
rows=[]
with threadpool_limits(limits=1):
 for p in (1,2,3):
  for r in range(100):
   # These seed families were fixed after designing the rule from the model,
   # before looking at any validation outcomes.
   rng=np.random.default_rng(np.random.SeedSequence([20260918+p,r,123]))
   phi=jones_ar_coefficients(p,rng=rng)
   y=simulate_uniform_ar(phi,2000,1000,1.,rng=rng)
   response=y[8:]
   lags=np.column_stack([y[8-j:len(y)-j] for j in range(1,9)])
   selected=None
   for q in range(9):
    if q==0:
     feasible=np.max(np.abs(response))<=1.
    else:
     x=lags[:,:q]
     fit=linprog(np.zeros(q),A_ub=np.r_[-x,x],b_ub=np.r_[1.-response,1.+response],
                 bounds=[(None,None)]*q,method='highs',
                 options={'primal_feasibility_tolerance':1e-9})
     if fit.status not in (0,2):raise RuntimeError(fit.message)
     feasible=fit.success
     if feasible and np.max(np.abs(response-x@fit.x))>1.+1e-8:
      raise RuntimeError('Numerical bound violation')
    if feasible:
     selected=q;break
   if selected is None or selected>p:raise RuntimeError('The true model is feasible.')
   rows.append(dict(true_order=p,replication=r+1,seed=20260918+p,
                    selected_order=selected,final_coefficient=phi[-1]))
  print(f'Completed 100 independent AR({p}) validation series',flush=True)
f=pd.DataFrame(rows)
f.to_csv(OUT/'validation_selections.csv',index=False)
summary=f.assign(correct=f.selected_order==f.true_order).groupby('true_order').correct.agg(['sum','count'])
summary.to_csv(OUT/'validation_summary.csv')
print(summary.to_string(),flush=True)
