"""Compatibility adapters for the one shared Q_r minimization engine."""
import numpy as np
from kde_tools.qhat import GaussianQhatObjective, minimize_qhat
from kde_tools.energy_minimization import search_config, DEFAULT_ENERGY_SEARCH

H_MIN, H_MAX, HALF_WIDTH = .01, 2., DEFAULT_ENERGY_SEARCH.half_width


def same_minima(first, second, tolerance=1e-7):
    return len(first) == len(second) and all(
        abs(np.log(float(a["h"])/float(b["h"]))) < tolerance for a, b in zip(first, second))


class GridScanner:
    def __init__(self, points=None, spatial_points=None):
        self.config = search_config(scan_points=points, spatial_points=spatial_points, half_width=HALF_WIDTH)


def sample_minima(samples, methods, scanner, refined_scanner):
    objective = GaussianQhatObjective(samples)
    results, diagnostics = {}, {}
    for method, r in methods.items():
        report = minimize_qhat(objective, H_MIN, H_MAX, r=r, config=scanner.config)
        if report["diagnostics"]["resolution_stable"] is False:
            report = minimize_qhat(objective, H_MIN, H_MAX, r=r, config=refined_scanner.config)
        results[method] = report["local_minima"]
        diagnostics[method] = dict(curvature_fallback=report["diagnostics"]["refinements"] > 1,
            coarse_fine_signature_agrees=bool(report["diagnostics"]["resolution_stable"]),
            interval_minimum=report["global_minimum"], energy_search=report["diagnostics"], analysis=report)
    return results, diagnostics


def exact_dense_minima(samples, methods, points=1025):
    """Exact-scan validation mode of the same engine; no copied algorithm."""
    objective = GaussianQhatObjective(samples)
    return {method: minimize_qhat(objective, H_MIN, H_MAX, r=r,
        config=search_config(scan_backend="exact", scan_points=points))["local_minima"]
        for method, r in methods.items()}
