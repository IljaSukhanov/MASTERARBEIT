"""Compare direct bounded Brent minimization with the numerical reference.

The minimizer works directly in bandwidth h. The independent reference
search analyzes stationary points and checks its scan resolution; neither
procedure is a general certificate of global optimality.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

KDE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KDE_ROOT))

from kde_tools.experiment_inputs import load_reference_sample
from kde_tools.qhat import bounded_qhat_minimum, minimize_qhat


def main():
    samples, metadata = load_reference_sample()
    with threadpool_limits(limits=1):
        reference = minimize_qhat(samples, 0.01, 2.0, r=0.0)
        bounded_qhat_minimum(samples)  # Warm up before the timed runs.
        runs = []
        for _ in range(9):
            start = perf_counter()
            candidate = bounded_qhat_minimum(samples)
            runs.append(dict(seconds=perf_counter()-start, result=candidate))
    target = reference["global_minimum"]
    result = dict(sample=metadata, h_interval=[0.01, 2.0], r=0.0,
                  reference=reference, runs=runs,
                  median_time_ms=1000*float(np.median([run["seconds"] for run in runs])),
                  relative_bandwidth_error=abs(runs[-1]["result"]["h"]-target["h"])/target["h"])
    output = KDE_ROOT / "outputs" / "brent_h_interval_comparison"
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(reference_h=target["h"], result=runs[-1]["result"],
                          relative_bandwidth_error=result["relative_bandwidth_error"],
                          median_time_ms=result["median_time_ms"]), indent=2))


if __name__ == "__main__":
    main()
