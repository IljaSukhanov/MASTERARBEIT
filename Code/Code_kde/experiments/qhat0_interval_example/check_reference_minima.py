"""Check curvature and numerical stationarity at the M=769 references."""
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kde_tools.experiment_inputs import paired_gaussian_sample_bank
from kde_tools.qhat import GaussianQhatObjective

OUT = ROOT/'outputs/brent_spectral_comparison_m769_reference'
with (OUT/'paired_results.csv').open(newline='') as f:
    refs=[r for r in csv.DictReader(f) if r['method']=='brent_linear_769']
samples=paired_gaussian_sample_bank(100,200)
results=[]
for r in refs:
    i=int(r['sample_id']); h=float(r['selected_h'])
    value,slope,curvature=GaussianQhatObjective(samples[i]).derivatives(h)
    results.append(dict(sample_id=i,reference_h=h,energy=float(value),
                        derivative=float(slope),second_derivative=float(curvature),
                        newton_correction_abs=abs(float(slope/curvature)),
                        endpoint_selected=r['endpoint_selected']))
if len(results)!=100:
    raise ValueError('Expected 100 references.')
with (OUT/'reference_minimum_checks.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
print('Interior and positive second derivative:',sum(r['endpoint_selected']=='False' and r['second_derivative']>0 for r in results))
print('Largest absolute derivative:',max(abs(r['derivative']) for r in results))
