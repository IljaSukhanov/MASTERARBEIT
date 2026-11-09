"""Find an independent N(0,1) sample with three detected Q_0 minima."""
import csv
import json
from pathlib import Path
import sys

import numpy as np

KDE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KDE_ROOT))
from kde_tools.qhat import grid_brent_qhat_minimum, GaussianQhatObjective

OUTPUT = KDE_ROOT / 'outputs/brent_three_minima_example'
SEED = 31415926
rng = np.random.default_rng(SEED)
for i in range(3000):
    sample = rng.normal(size=200)
    coarse = grid_brent_qhat_minimum(sample, grid_points=513, grid_spacing='linear')
    if coarse['valleys_detected'] >= 3:
        fine = grid_brent_qhat_minimum(sample, grid_points=8193, grid_spacing='linear')
        if fine['valleys_detected'] >= 3:
            obj = GaussianQhatObjective(sample)
            points = []
            for candidate in fine['valley_candidates']:
                h = candidate['h']
                e, slope, curvature = obj.derivatives(h)
                points.append({'h': h, 'energy': float(e), 'slope': float(slope),
                               'second_derivative': float(curvature)})
            OUTPUT.mkdir(parents=True, exist_ok=True)
            with (OUTPUT / 'sample.csv').open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['observation'])
                writer.writerows((float(x),) for x in sample)
            (OUTPUT / 'result.json').write_text(json.dumps({
                'seed': SEED, 'sample_index_zero_based': i, 'n': 200,
                'distribution': 'standard normal', 'interval': [.01, 2.],
                'r': 0, 'scan_spacing': 'linear', 'coarse_M': 513, 'fine_M': 8193,
                'minima': points, 'certified_global': False,
            }, indent=2) + '\n')
            print(json.dumps({'sample_index': i, 'minima': points}), flush=True)
            break
    if (i+1) % 100 == 0:
        print(f'Checked {i+1} samples', flush=True)
else:
    print('No example found in 3000 samples.', flush=True)
