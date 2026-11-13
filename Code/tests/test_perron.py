"""Checks against dense reference solves and known limiting Perron vectors."""

from __future__ import annotations

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
from unittest.mock import patch

import numpy as np
from scipy.linalg import LinAlgError, eigh

from kde_tools.perron import (
    PerronConvergenceError,
    PerronDegeneracyError,
    doob_mass_from_perron,
    perron_eigenpair,
    perron_eigenpair_from_divergence,
    perron_peak_index,
    perron_vector,
    perron_vector_from_divergence,
)
from kde_tools.similarities import (
    beta2_divergence_matrix,
    doob_stationary_mass,
    select_spectral_bandwidth,
)


class PerronTests(unittest.TestCase):
    def setUp(self) -> None:
        self.samples = np.random.default_rng(20260823).normal(size=25)
        self.grid = np.geomspace(0.05, 1.0, 31)
        self.divergence = beta2_divergence_matrix(self.samples, self.grid)

    def test_matches_original_similarity_dense_eigenvector(self) -> None:
        for lambda_value in (1e-6, 1.0, 1e3):
            with self.subTest(lambda_value=lambda_value):
                similarity = np.exp(-lambda_value * self.divergence)
                _, vectors = eigh(similarity)
                reference = vectors[:, -1]
                reference /= reference.sum()
                actual = perron_eigenpair_from_divergence(self.divergence, lambda_value)
                np.testing.assert_allclose(
                    actual.vector, reference, rtol=1e-10, atol=2e-12
                )
                self.assertLess(actual.relative_residual, 1e-11)
                self.assertGreater(actual.relative_gap, 0.0)
                self.assertEqual(actual.matvecs, 0)

    def test_extreme_lambda_recovers_pair_after_s_becomes_identity(self) -> None:
        divergence = np.array([[0.0, 1.0, 4.0], [1.0, 0.0, 3.0], [4.0, 3.0, 0.0]])
        np.testing.assert_array_equal(np.exp(-1000 * divergence), np.eye(3))
        actual = perron_vector_from_divergence(divergence, 1000.0)
        np.testing.assert_allclose(actual, [0.5, 0.5, 0.0], atol=1e-14)
        with self.assertRaises(PerronDegeneracyError):
            perron_vector(np.eye(3))

    def test_nearly_bipartite_matrix_selects_positive_eigenvalue(self) -> None:
        matrix = np.array([[0.0, 1.0, 1e-12], [1.0, 0.0, 2e-12], [1e-12, 2e-12, 0.0]])
        actual = perron_eigenpair(matrix)
        self.assertGreater(actual.eigenvalue, 0.0)
        self.assertTrue(np.all(actual.vector >= 0.0))
        np.testing.assert_allclose(actual.vector[:2], [0.5, 0.5], atol=1e-10)

    def test_general_affinity_with_unequal_diagonal(self) -> None:
        densities = np.array(
            [
                [0.2, 0.7, 0.8, 0.3],
                [0.5, 0.8, 0.4, 0.2],
                [0.3, 0.4, 0.5, 0.9],
                [0.6, 0.3, 0.2, 0.4],
            ]
        )
        affinity = densities @ densities.T
        eigenvalues, eigenvectors = eigh(affinity)
        reference = eigenvectors[:, -1] / eigenvectors[:, -1].sum()
        actual = perron_eigenpair(affinity)
        np.testing.assert_allclose(actual.vector, reference, atol=1e-13)
        self.assertAlmostEqual(actual.eigenvalue, eigenvalues[-1], places=12)
        self.assertAlmostEqual(
            actual.spectral_gap, eigenvalues[-1] - eigenvalues[-2], places=12
        )

    def test_shift_and_scale_preserve_vector(self) -> None:
        matrix = np.exp(-self.divergence)
        original = perron_eigenpair(matrix)
        shifted = perron_eigenpair(7 * matrix + 19 * np.eye(len(matrix)))
        np.testing.assert_allclose(shifted.vector, original.vector, atol=1e-13)
        self.assertAlmostEqual(
            shifted.eigenvalue, 7 * original.eigenvalue + 19, places=11
        )

    def test_inputs_are_not_mutated(self) -> None:
        divergence = self.divergence.copy()
        perron_vector_from_divergence(divergence, 1e8)
        np.testing.assert_array_equal(divergence, self.divergence)
        similarity = np.exp(-divergence)
        saved = similarity.copy()
        perron_vector(similarity)
        np.testing.assert_array_equal(similarity, saved)

    def test_zero_lambda_returns_exactly_uniform_vector(self) -> None:
        actual = perron_vector_from_divergence(self.divergence, 0.0)
        np.testing.assert_array_equal(actual, np.full(len(actual), 1 / len(actual)))

    def test_small_matrix_cases(self) -> None:
        np.testing.assert_array_equal(
            perron_vector_from_divergence([[0.0]], 1e9), [1.0]
        )
        actual = perron_eigenpair_from_divergence([[0.0, 3.0], [3.0, 0.0]], 1e9)
        np.testing.assert_allclose(actual.vector, [0.5, 0.5], atol=1e-15)
        self.assertEqual(actual.matvecs, 0)
        self.assertAlmostEqual(actual.spectral_gap, 2.0)

    def test_doob_mass_uses_single_shared_vector(self) -> None:
        similarity = np.exp(-10 * self.divergence)
        vector = perron_vector(similarity)
        expected = vector**2 / np.sum(vector**2)
        np.testing.assert_allclose(
            doob_stationary_mass(similarity), expected, atol=1e-14
        )
        np.testing.assert_allclose(doob_mass_from_perron(vector), expected, atol=1e-14)

    def test_repeated_largest_eigenvalue_is_reported(self) -> None:
        two_pairs = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        with self.assertRaises(PerronDegeneracyError):
            perron_vector(two_pairs)

    def test_nonconvergence_is_not_silently_accepted(self) -> None:
        error = LinAlgError("deliberate nonconvergence")
        with patch("kde_tools.perron.eigh", side_effect=error):
            with self.assertRaises(PerronConvergenceError) as caught:
                perron_vector_from_divergence(self.divergence)
        self.assertIs(caught.exception.__cause__, error)

    def test_failed_residual_is_not_silently_accepted(self) -> None:
        wrong_vectors = np.eye(31)[:, :2]
        with patch(
            "kde_tools.perron.eigh", return_value=(np.array([0.2, 1.0]), wrong_vectors)
        ):
            with self.assertRaises(PerronConvergenceError):
                perron_vector_from_divergence(self.divergence)

    def test_invalid_inputs(self) -> None:
        invalid = [
            [],
            np.ones((2, 3)),
            [[np.nan]],
            [[np.inf]],
            [[-1.0]],
            [[1.0, 0.2], [0.3, 1.0]],
            np.eye(2, dtype=complex),
        ]
        for matrix in invalid:
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                perron_vector(matrix)
        for lambda_value in (-1.0, np.nan, np.inf):
            with self.subTest(lambda_value=lambda_value), self.assertRaises(ValueError):
                perron_vector_from_divergence(self.divergence, lambda_value)
        with self.assertRaises(ValueError):
            perron_vector_from_divergence(np.eye(2), 1.0)

    def test_selector_handles_large_lambda_without_losing_divergence(self) -> None:
        actual = select_spectral_bandwidth(self.samples, self.grid, lambda_value=1e9)
        expected = perron_vector_from_divergence(self.divergence, 1e9)
        np.testing.assert_allclose(actual["perron_vector"], expected, atol=1e-13)
        self.assertEqual(actual["selected_index"], perron_peak_index(expected))

    def test_closest_pair_peak_is_consistent_across_repeated_solves(self) -> None:
        divergence = self.divergence.copy()
        np.fill_diagonal(divergence, np.inf)
        pair = np.unravel_index(np.argmin(divergence), divergence.shape)
        for _ in range(20):
            result = perron_eigenpair_from_divergence(self.divergence, 1e9)
            self.assertEqual(result.selected_index, min(pair))

    def test_peak_tie_policy_preserves_distinguishable_maxima(self) -> None:
        self.assertEqual(perron_peak_index([0.5, 0.5 + 1e-15, 0.0]), 0)
        self.assertEqual(perron_peak_index([0.5, 0.5 + 1e-12, 0.0]), 1)
        self.assertEqual(perron_peak_index([0.0, 0.0]), 0)
        for invalid in ([], [[1.0]], [np.nan], [np.inf], [1j]):
            with self.subTest(vector=invalid), self.assertRaises(ValueError):
                perron_peak_index(invalid)



class NotebookRoutingTests(unittest.TestCase):
    def test_experiment_scripts_cannot_reintroduce_local_eigensolvers(self) -> None:
        root = Path(__file__).resolve().parents[1]
        allowed = root / "kde_tools" / "perron.py"
        # Existing validation scripts compare the stabilized matrix with an
        # independent eigh solve of the original, unshifted similarity.
        dense_reference_scripts = {
            root / "experiments/spectral_lambda_jump_study/numerical_artefact_checks.py",
        }
        sparse_solvers = {
            root / "experiments/outlook_four_lambdas.py",
            root / "experiments/repeated_alpha_evolution/run_experiment.py",
        }
        for path in root.rglob("*.py"):
            if path == allowed or "tests" in path.relative_to(root).parts:
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Call):
                    name = (
                        node.func.id
                        if isinstance(node.func, ast.Name)
                        else getattr(node.func, "attr", "")
                    )
                    with self.subTest(script=path.relative_to(root)):
                        if name == "eigh" and path in dense_reference_scripts:
                            continue
                        if name == "eigsh" and path in sparse_solvers:
                            continue
                        self.assertNotIn(
                            name, {"eigh", "eigsh", "eigvalsh", "eigs", "eig"}
                        )

    def test_notebooks_cannot_reintroduce_local_eigensolvers(self) -> None:
        root = Path(__file__).resolve().parents[1]
        forbidden = {
            "eigh",
            "eigsh",
            "eigvalsh",
            "symmetric_eigh",
            "stable_perron_vector_from_divergence",
            "stable_power_perron_vector_from_divergence",
            "power_perron_vector",
        }
        notebooks = list(root.glob("notebooks/*.ipynb"))
        self.assertTrue(notebooks, "No categorized KDE notebooks were found.")
        for path in notebooks:
            notebook = json.loads(path.read_text())
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue
                source = "\n".join(
                    line for line in "".join(cell["source"]).splitlines()
                    if not line.lstrip().startswith(("%", "!"))
                )
                for node in ast.walk(ast.parse(source)):
                    if isinstance(node, ast.Call):
                        name = (
                            node.func.id
                            if isinstance(node.func, ast.Name)
                            else getattr(node.func, "attr", "")
                        )
                        with self.subTest(notebook=path.name, cell=index):
                            eigenvalues_only = name == "eigh" and any(
                                keyword.arg == "eigvals_only"
                                and isinstance(keyword.value, ast.Constant)
                                and keyword.value.value is True
                                for keyword in node.keywords
                            )
                            if not eigenvalues_only:
                                self.assertNotIn(name, forbidden)

    def test_every_notebook_retains_automatic_pdf_setup(self) -> None:
        root = Path(__file__).resolve().parents[1]
        notebooks = list(root.glob("notebooks/*.ipynb"))
        self.assertTrue(notebooks, "No categorized KDE notebooks were found.")
        for path in notebooks:
            notebook = json.loads(path.read_text())
            setup = next(
                "".join(cell["source"]) for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            )
            with self.subTest(notebook=path.name):
                self.assertTrue("enable_automatic_pdf_saving(" in setup or "setup_notebook(" in setup)


if __name__ == "__main__":
    unittest.main()
