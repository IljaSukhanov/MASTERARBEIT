"""Data-only LP and nested warm starts for a ten-process numerical audit."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from Code.Code_AR_processes.ar_tools.functions import jones_ar_coefficients,simulate_uniform_ar,sliding_window_point_cloud
from Code.Code_AR_processes.ar_tools.L_method import shared_breakpoint_l_method_selector
OUT=Path(__file__).resolve().parent
INPUT=ROOT/'Code/Code_AR_processes/outputs/csv/ar_shared_breakpoint_l_methods'
old=pd.read_csv(INPUT/'width_curves.csv')


def data_initialized_width_curve(series,seed,replication,iterations=1000):
    widths, best_directions=[],[]
    for k in range(1,11):
        cloud=sliding_window_point_cloud(series,k)
        if k==1:
            widths.append(float(np.ptp(cloud[:,0])))
            best_directions.append(np.ones(1))
            continue
        x,y=cloud[:,1:],cloud[:,0]
        fit=linprog(np.r_[np.zeros(k-1),1.],
            A_ub=np.vstack([np.c_[-x,-np.ones(len(y))],np.c_[x,-np.ones(len(y))]]),
            b_ub=np.r_[-y,y],bounds=[(None,None)]*(k-1)+[(0,None)],method='highs')
        if not fit.success:raise RuntimeError(fit.message)
        lp_direction=np.r_[1.,-fit.x[:-1]]
        _,evec=np.linalg.eigh(np.cov(cloud,rowvar=False))
        rng=np.random.default_rng(np.random.SeedSequence([seed,replication,k,20]))
        dirs=np.vstack([lp_direction,np.r_[best_directions[-1],0.],evec[:,0],rng.normal(size=(2,k))])
        dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)
        best=np.full(len(dirs),np.inf)
        kept=dirs.copy()
        indices=np.arange(len(dirs))
        for iteration in range(iterations):
            projections=cloud@dirs.T
            maxi=np.argmax(projections,axis=0);mini=np.argmin(projections,axis=0)
            current=projections[maxi,indices]-projections[mini,indices]
            improved=current<best
            best[improved]=current[improved];kept[improved]=dirs[improved]
            subgradient=cloud[maxi]-cloud[mini]
            tangent=subgradient-np.sum(dirs*subgradient,axis=1,keepdims=True)*dirs
            dirs-=.2*.99**iteration*tangent
            dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)
        winner=int(np.argmin(best))
        widths.append(float(best[winner]));best_directions.append(kept[winner])
    return np.array(widths)


def ratio_pick(y):
    d=-np.diff(y)
    ratios=[]
    for q in range(1,len(y)-1):
        denominator=np.sum(d[q:]**2)
        ratios.append(d[q-1]**2/denominator if denominator>0 else (np.inf if d[q-1]!=0 else 0))
    return 1+int(np.argmax(ratios))


if __name__=='__main__':
 rows,selections=[],[]
 with threadpool_limits(limits=1):
  for replication in range(1,11):
   seed=20260816
   rng=np.random.default_rng(np.random.SeedSequence([seed,replication-1,0]))
   phi=jones_ar_coefficients(3,rng=rng)
   y=simulate_uniform_ar(phi,2000,1000,1.,rng=rng)
   improved=data_initialized_width_curve(y,seed,replication-1)
   baseline=old[(old.true_order==3)&(old.replication==replication)].sort_values('k').minimum_width.to_numpy()
   for k,(w0,w1) in enumerate(zip(baseline,improved),1):
    rows.append(dict(replication=replication,k=k,width_original=w0,width_data_initialized=w1))
   for kind,w in [('original',baseline),('data_initialized',improved)]:
    selections.append(dict(replication=replication,kind=kind,method='drop_to_tail_energy',order=ratio_pick(w)))
    for tail in ('affine','constant'):
     order=shared_breakpoint_l_method_selector(np.arange(1,11),w,tail_model=tail)['selected_order']
     selections.append(dict(replication=replication,kind=kind,method=tail,order=order))
   print(f'Completed data-only initialization audit {replication}/10',flush=True)
 pd.DataFrame(rows).to_csv(OUT/'initialization_audit_widths.csv',index=False)
 f=pd.DataFrame(selections);f.to_csv(OUT/'initialization_audit_selections.csv',index=False)
 print(f.assign(correct=f.order==3).groupby(['kind','method']).correct.sum().to_string(),flush=True)
