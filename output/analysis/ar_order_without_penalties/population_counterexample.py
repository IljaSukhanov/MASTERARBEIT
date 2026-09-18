"""Enumerate truncated-zonotope facet normals for an actual AR(3) example."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from scipy.signal import residuez
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from Code.Code_AR_processes.functions import ar_impulse_response
from Code.Code_AR_processes.L_method import shared_breakpoint_l_method_selector
OUT=Path(__file__).resolve().parent
curves=pd.read_csv(ROOT/'Code/Code_AR_processes/outputs/csv/ar_shared_breakpoint_l_methods/width_curves.csv')
records=[]
with threadpool_limits(limits=1):
 for replication,g in curves[curves.true_order==3].groupby('replication'):
  phi=np.array(json.loads(g.coefficients.iloc[0]))
  residues,poles,direct=residuez([1.],np.r_[1.,-phi])
  rho=float(np.max(np.abs(poles)))
  if rho>.9 or np.min(np.abs(poles[:,None]-poles[None,:]+np.eye(3)*10))<1e-6:continue
  C=float(np.sum(np.abs(residues)))
  N=int(np.ceil(np.log(1e-7*(1-rho)/(2*np.sqrt(3)*C))/np.log(rho)))+2
  if N>300:continue
  psi=ar_impulse_response(phi,maximum_lag=N-1)
  # Check the partial-fraction representation used to bound the omitted tail.
  psi_formula=np.real(np.sum(residues[None,:]*poles[None,:]**np.arange(N)[:,None],axis=1))
  np.testing.assert_allclose(psi,psi_formula,rtol=1e-7,atol=1e-10)
  widths=[]
  for k in (1,2,3):
   G=np.column_stack([np.r_[np.zeros(lag),psi[:N-lag]] for lag in range(k)])
   if k==1:
    width=2*np.sum(np.abs(G[:,0]))
   elif k==2:
    normals=np.c_[-G[:,1],G[:,0]]
    norm=np.linalg.norm(normals,axis=1)
    normals=normals[norm>1e-250]/norm[norm>1e-250,None]
    width=float(np.min(2*np.sum(np.abs(G@normals.T),axis=0)))
   else:
    i,j=np.triu_indices(N,1)
    unit_generators=G/np.linalg.norm(G,axis=1,keepdims=True)
    normals=np.cross(unit_generators[i],unit_generators[j])
    norm=np.linalg.norm(normals,axis=1)
    keep=norm>0
    normals=normals[keep]/norm[keep,None]
    width=np.inf
    for start in range(0,len(normals),1024):
     width=min(width,float(np.min(2*np.sum(np.abs(G@normals[start:start+1024].T),axis=0))))
   widths.append(float(width))
  plateau=float(2/np.sqrt(1+phi@phi))
  width_curve=np.r_[widths,np.repeat(plateau,7)]
  choices={}
  for tail in ('affine','constant'):
   fit=shared_breakpoint_l_method_selector(np.arange(1,11),width_curve,tail_model=tail)
   choices[tail]=fit['selected_order']
  records.append(dict(replication=int(replication),phi=json.dumps(phi.tolist()),rho=rho,N=N,
                      m1=widths[0],m2=widths[1],m3=widths[2],plateau=plateau,
                      width_error_upper_bound=2*np.sqrt(3)*C*rho**(N-2)/(1-rho),
                      affine_selected_order=choices['affine'],constant_selected_order=choices['constant']))
  print('Population example',replication,'widths',width_curve[:4],'orders',choices,flush=True)
  if choices['affine']!=3 and choices['constant']!=3:
   info=records[-1]
   (OUT/'population_counterexample.json').write_text(json.dumps(info,indent=2)+'\n')
   rows=[]
   for tail in ('affine','constant'):
    scores=shared_breakpoint_l_method_selector(np.arange(1,11),width_curve,tail_model=tail)['candidate_scores']
    scores['tail_model']=tail
    rows.append(scores)
   pd.concat(rows).to_csv(OUT/'population_counterexample_scores.csv',index=False)
   break
pd.DataFrame(records).to_csv(OUT/'population_search.csv',index=False)
