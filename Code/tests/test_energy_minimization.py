"""Independent examples for interval/global/local bandwidth search behavior."""

# Permit running individual tests and unittest discovery from the project root.
import sys as _sys
from pathlib import Path as _Path
_code_root = str(_Path(__file__).resolve().parents[1])
if _code_root not in _sys.path:
    _sys.path.insert(0, _code_root)

import unittest
import ast
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad

from kde_tools.energy_minimization import (
    EnergySearchConfig, energy_search_id, minimize_bandwidth_energy,
)
from kde_tools.qhat import minimize_qhat, expected_gaussian_qhat_log_energy


class UnifiedEnergySearchTests(unittest.TestCase):
    def test_multiple_polynomial_minima_and_global_ties(self):
        def energy(z):
            return np.array([(z*z-1)**2, 4*z*(z*z-1), 12*z*z-4])
        result = minimize_bandwidth_energy(energy, np.exp(-2), np.exp(2), scan_points=33)
        np.testing.assert_allclose([p["h"] for p in result["local_minima"]], np.exp([-1, 1]), atol=1e-10)
        self.assertEqual(len(result["local_maxima"]), 1)
        self.assertEqual(len(result["global_minimizers"]), 2)
        self.assertTrue(all(p["is_global_minimum"] for p in result["local_minima"]))

    def test_endpoint_wins_without_discarding_interior_minimum(self):
        # f(z)=z^3/3-z has a local minimum at 1, but f(-3)=-6 < f(1).
        def energy(z):
            return np.array([z**3/3-z, z*z-1, 2*z])
        result = minimize_bandwidth_energy(energy, np.exp(-3), np.exp(2), scan_points=31)
        self.assertEqual(result["global_minimum"]["boundary"], -1)
        self.assertEqual(len(result["local_minima"]), 1)
        self.assertFalse(result["local_minima"][0]["is_global_minimum"])

    def test_degenerate_minimum_on_scan_node_is_not_lost(self):
        result = minimize_bandwidth_energy(
            lambda z: np.array([z**4, 4*z**3, 12*z*z]), np.exp(-1), np.exp(1), scan_points=17)
        self.assertEqual(len(result["local_minima"]), 1)
        self.assertAlmostEqual(result["local_minima"][0]["h"], 1)
        self.assertEqual(result["local_minima"][0]["log_curvature"], 0)

    def test_monotone_and_flat_functions_have_no_strict_interior_minima(self):
        for energy in (lambda z: np.array([np.exp(-z), -np.exp(-z), np.exp(-z)]),
                       lambda z: np.array([1., 0., 0.])):
            result = minimize_bandwidth_energy(energy, .01, 2., scan_points=17)
            self.assertEqual(result["local_minima"], [])
            self.assertIsNone(result["best_local_minimum"])

    def test_close_minimum_maximum_pair_is_not_merged_by_resolution_matching(self):
        center, epsilon = .12345, 1e-8
        def energy(z):
            x = z-center
            return np.array([x**3/3-epsilon**2*x, x*x-epsilon**2, 2*x])
        report = minimize_bandwidth_energy(energy, np.exp(-1), np.exp(1),
            config=EnergySearchConfig(scan_points=17, root_xtol=1e-14,
                                      root_match_tolerance=1e-7))
        self.assertEqual(len(report["local_minima"]), 1)
        self.assertEqual(len(report["local_maxima"]), 1)
        self.assertAlmostEqual(np.log(report["local_minima"][0]["h"]), center+epsilon, delta=1e-12)

    def test_multiscale_gaussian_sample_has_two_local_minima_even_when_boundary_wins(self):
        a = np.array([-3., -1., 0., 1., 3.])
        sample = (a[:, None]+.01*a[None, :]).ravel()
        result = minimize_qhat(sample, .005, 100., r=13/14,
                               config=EnergySearchConfig(scan_backend="exact", scan_points=129))
        np.testing.assert_allclose([p["h"] for p in result["local_minima"]],
                                   [.0159405444385612, 1.5938089305185468], rtol=1e-9)
        self.assertEqual(result["global_minimum"]["boundary"], 1)
        self.assertFalse(any(p["is_global_minimum"] for p in result["local_minima"]))

    def test_fourier_and_exact_search_find_same_local_and_interval_minima(self):
        sample = np.array([-3., -1., 0., 1., 3.])
        exact = minimize_qhat(sample, .01, 4., r=13/14, scan_backend="exact", scan_points=65)
        approx = minimize_qhat(sample, .01, 4., r=13/14, scan_points=65)
        np.testing.assert_allclose([p["h"] for p in exact["local_minima"]],
                                   [p["h"] for p in approx["local_minima"]], rtol=1e-8)
        self.assertAlmostEqual(exact["global_minimum"]["h"], approx["global_minimum"]["h"])

    def test_expected_gaussian_threshold_and_shallow_minimum(self):
        for n, count in ((91, 0), (92, 1), (200, 1)):
            result = minimize_bandwidth_energy(
                lambda z: expected_gaussian_qhat_log_energy(z, n, 13/14), .01, 2., scan_points=65)
            self.assertEqual(len(result["local_minima"]), count)
            self.assertEqual(len(result["local_maxima"]), count)
        self.assertAlmostEqual(result["local_minima"][0]["h"], .44241318610233593, places=9)

    def test_expected_energy_normalization_matches_independent_pair_expectation(self):
        n, r, sigma, h = 17, .7, 1.3, .41
        def integrand(distance):
            u = distance**2/(4*h*h)
            density = np.exp(-distance**2/(4*sigma*sigma))/(2*np.sqrt(np.pi)*sigma)
            return np.exp(-u)*(u*u-3*u+.75)*density
        off_diagonal = quad(integrand, -np.inf, np.inf, epsabs=1e-12)[0]
        expected = h**(-1-2*r)/(4*np.sqrt(np.pi))*(.75/n+(1-1/n)*off_diagonal)
        value = expected_gaussian_qhat_log_energy(np.log(h), n, r, sigma*sigma)[0]
        self.assertAlmostEqual(value, expected, places=13)

    def test_migrated_notebooks_use_shared_search_and_control_notebook_exports_pdf(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("03_fixed_sample_experiments", "06_additional_experiments"):
            notebook = json.loads((root/"notebooks"/(name+".ipynb")).read_text())
            calls = []
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    source = "\n".join(line for line in "".join(cell["source"]).splitlines()
                                       if not line.lstrip().startswith(("%", "!")))
                    calls.extend(node.func.id for node in ast.walk(ast.parse(source))
                                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name))
            self.assertIn("minimize_qhat", calls)
            if name == "06_additional_experiments":
                self.assertNotIn("minimize_scalar", calls)
        path = root/"notebooks/06_additional_experiments.ipynb"
        notebook = json.loads(path.read_text())
        code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
        self.assertIn('setup_notebook("06_additional_experiments")', code)
        self.assertIn('NOTEBOOK_NAME = "continuous_ise_qhat_distributions_vs_n"', code)

    def test_configuration_changes_search_identity_and_is_validated(self):
        self.assertNotEqual(energy_search_id(), energy_search_id(scan_points=513))
        for settings in ({"scan_points": 2}, {"root_xtol": 0}, {"scan_backend": "other"},
                         {"spatial_points": 17}, {"comparison_rtol": -1}):
            with self.assertRaises(ValueError):
                EnergySearchConfig(**settings)
        with self.assertRaises(ValueError):
            minimize_qhat([0], 2., 1.)
        with self.assertRaises(ValueError):
            minimize_qhat([0], r=np.nan)


if __name__ == "__main__":
    unittest.main()
