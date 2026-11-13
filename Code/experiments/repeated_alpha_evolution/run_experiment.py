"""Paired repeated-sample comparison of four spectral concentration regimes.

Run with BLAS limited to one thread, for example:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    PYTHONPATH=Code python Code/experiments/repeated_alpha_evolution/run_experiment.py

Selections and solver diagnostics are checkpointed. Figures are produced by
the companion plotting script and are always saved as PDFs. The thesis
comparison uses N=400 by default; REPEATED_ALPHA_GRID_SIZE selects another N.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import os
import sys
import time

import numpy as np
from scipy.sparse.linalg import eigsh

CODE = Path(__file__).resolve().parents[2]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from kde_tools.fourier.similarities import DEFAULT_FREQUENCY_CUTOFF, DEFAULT_FREQUENCY_INTERVALS
from kde_tools.perron import perron_peak_index


M, N, SAMPLE_SIZE, SEED = 1000, int(os.environ.get('REPEATED_ALPHA_GRID_SIZE', '400')), 200, 20260828
if N < 3:
    raise ValueError('REPEATED_ALPHA_GRID_SIZE must be at least 3.')
DATA = CODE / 'outputs/data/repeated_alpha_evolution'
if N != 1000:
    DATA = DATA / f'n{N}'
GRID = np.geomspace(0.01, 2.0, N)
EXPONENTS = np.array([0.0, 0.5, 1.0, 1.5])
LAMBDAS = N ** EXPONENTS
OMEGA = np.linspace(0.0, DEFAULT_FREQUENCY_CUTOFF, DEFAULT_FREQUENCY_INTERVALS + 1)
WEIGHTS = np.full(len(OMEGA), (OMEGA[1] - OMEGA[0]) / np.pi)
WEIGHTS[[0, -1]] *= 0.5
SQRT_WEIGHTS = np.sqrt(WEIGHTS)
BASIS = np.exp(-0.5 * (GRID[:, None] * OMEGA[None, :]) ** 2)
SAMPLES = np.random.default_rng(SEED).normal(size=(M, SAMPLE_SIZE))
SAMPLE_HASH = hashlib.sha256(SAMPLES.tobytes()).hexdigest()
EXPECTED_SAMPLE_HASH = '6a33f4c51d028c4145179ca536db8f5069055aba68f38820d56faf9355bf3299'
assert SAMPLE_HASH == EXPECTED_SAMPLE_HASH


def one_sample(repetition):
    sample = SAMPLES[repetition]
    characteristic = np.abs(np.exp(-1j * OMEGA[:, None] * sample).mean(axis=1))
    features = BASIS * (SQRT_WEIGHTS * characteristic)[None, :]
    gram = features @ features.T
    norms = np.diag(gram).copy()
    divergence = 0.5 * (norms[:, None] + norms[None, :] - 2.0 * gram)
    np.maximum(divergence, 0.0, out=divergence)
    np.fill_diagonal(divergence, 0.0)
    selections = np.empty(6, dtype=int)
    residuals = np.empty(4, dtype=float)
    gaps = np.empty(4, dtype=float)
    for column, penalty in enumerate(LAMBDAS):
        matrix = np.exp(-penalty * divergence)
        np.fill_diagonal(matrix, 0.0)
        values, vectors = eigsh(matrix, k=2, which='LA', tol=1e-10, v0=np.ones(N))
        vector = vectors[:, -1]
        if vector.sum() < 0:
            vector = -vector
        if np.min(vector) < -1e-10 * np.max(np.abs(vector)):
            raise ArithmeticError(f'Negative Perron vector at sample {repetition}, lambda={penalty}')
        np.maximum(vector, 0.0, out=vector)
        residuals[column] = np.linalg.norm(matrix @ vector - values[-1] * vector) / (abs(values[-1]) * np.linalg.norm(vector))
        gaps[column] = (values[-1] - values[-2]) / abs(values[-1])
        if residuals[column] > 1e-9 or gaps[column] <= 1e-12:
            raise ArithmeticError(f'Perron diagnostics failed for sample {repetition}, lambda={penalty}')
        selections[column] = perron_peak_index(vector)
    cross = np.exp(-sample[None, :] ** 2 / (2.0 * (1.0 + GRID[:, None] ** 2))).mean(axis=1)
    cross /= np.sqrt(2.0 * np.pi * (1.0 + GRID**2))
    selections[4] = int(np.argmin(norms - 2.0 * cross))
    energies = 0.5 * GRID**4 * np.einsum('ij,j,ij->i', features, OMEGA**4, features, optimize=True)
    selections[5] = int(np.argmin(energies))
    return repetition, selections, residuals, gaps


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    checkpoint = DATA / 'paired_selections.npz'
    if checkpoint.exists():
        with np.load(checkpoint) as saved:
            if str(saved['sample_hash'].item()) != SAMPLE_HASH or not np.array_equal(saved['grid'], GRID):
                raise ValueError('Existing checkpoint has different samples or grid.')
            indices = saved['selected_indices'].copy()
            done = saved['completed'].copy()
            residuals = saved['residuals'].copy()
            gaps = saved['gaps'].copy()
    else:
        indices = np.full((M, 6), -1, dtype=int)
        done = np.zeros(M, dtype=bool)
        residuals = np.full((M, 4), np.nan)
        gaps = np.full((M, 4), np.nan)

    def save():
        tmp = checkpoint.with_suffix('.npz.tmp')
        with tmp.open('wb') as f:
            np.savez_compressed(f, grid=GRID, exponents=EXPONENTS, lambdas=LAMBDAS,
                                sample_hash=SAMPLE_HASH, selected_indices=indices,
                                completed=done, residuals=residuals, gaps=gaps)
        tmp.replace(checkpoint)

    start = time.monotonic()
    print(f'Paired sample comparison: {done.sum()}/{M} cached', flush=True)
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(one_sample, int(i)) for i in np.flatnonzero(~done)]
            for future in as_completed(futures):
                i, selected, relative_residual, relative_gap = future.result()
                indices[i], residuals[i], gaps[i], done[i] = selected, relative_residual, relative_gap, True
                total = int(done.sum())
                if total % 25 == 0:
                    save()
                if total % 100 == 0 or total == M:
                    print(f'Completed {total}/{M} samples in {time.monotonic()-start:.1f}s', flush=True)
    finally:
        save()
    assert np.all(done) and np.all((indices >= 0) & (indices < N))
    metadata = dict(n=SAMPLE_SIZE, N=N, M=M, seed=SEED, sample_sha256=SAMPLE_HASH,
                    lambda_values=LAMBDAS.tolist(), exponents=EXPONENTS.tolist(),
                    frequency_cutoff=DEFAULT_FREQUENCY_CUTOFF, frequency_intervals=DEFAULT_FREQUENCY_INTERVALS,
                    solver='scipy.sparse.linalg.eigsh, two largest algebraic eigenpairs',
                    control_method='frequency quadrature (ISE cross-term evaluated analytically)',
                    maximum_relative_residual=float(residuals.max()), minimum_relative_gap=float(gaps.min()))
    (DATA / 'settings_and_diagnostics.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
