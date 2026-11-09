"""Record bounded-Brent function evaluations for samples with >0.1% error."""
import csv
from pathlib import Path
import sys

from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kde_tools.experiment_inputs import paired_gaussian_sample_bank
from kde_tools.qhat import GaussianQhatObjective

OUT = ROOT/'outputs/brent_spectral_comparison_m769_reference'
with (OUT/'paired_results.csv').open(newline='') as f:
    rows=list(csv.DictReader(f))
refs={int(r['sample_id']):float(r['selected_h']) for r in rows if r['method']=='brent_linear_769'}
bad=[int(r['sample_id']) for r in rows if r['method']=='brent_unrefined' and
     abs(float(r['selected_h'])-refs[int(r['sample_id'])])/refs[int(r['sample_id'])]>.001]
samples=paired_gaussian_sample_bank(100,200)
traces=[]
for i in bad:
    objective=GaussianQhatObjective(samples[i])
    def observed(h):
        value=objective.energy(h)
        traces.append(dict(sample_id=i, evaluation=1+sum(r['sample_id']==i for r in traces),
                           h=float(h),energy=float(value)))
        return value
    result=minimize_scalar(observed,bounds=(.01,2.),method='bounded',
                           options={'xatol':2e-12,'maxiter':1000})
    if not result.success:
        raise ArithmeticError(result.message)
with (OUT/'unrefined_mismatch_traces.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(traces[0]));writer.writeheader();writer.writerows(traces)
print('Recorded traces for zero-based sample IDs:',bad)
