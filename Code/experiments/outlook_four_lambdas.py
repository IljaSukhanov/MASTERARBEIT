"""Recompute the Outlook's four sample-size panels using Figure 2.6 settings.

Run from the repository root with PYTHONPATH=Code. The same seeded
sample bank and prefixes as the earlier Outlook sweep are used. Per-size
checkpoints preserve all selections and eigensolver diagnostics.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigsh
from threadpoolctl import threadpool_limits

from kde_tools.bandwidth_histograms import plot_thesis_bandwidth_distributions
from kde_tools.fourier.similarities import beta2_divergence_matrix_fourier
from kde_tools.perron import perron_peak_index


CODE = Path(__file__).resolve().parents[1]
DATA = CODE / 'outputs/data/outlook_four_lambdas_n400'
OUT = CODE / 'outputs/images/thesis'
SIZES = (300, 400, 500, 1000)
N, M, SEED, BINNING = 400, 1000, 20260830, 2
LAMBDAS = np.array([1., 20., 400., 8000.])
GRID = np.geomspace(.01, 2., N)
SAMPLES = np.random.default_rng(SEED).normal(size=(M, max(SIZES)))
SAMPLE_HASH = hashlib.sha256(SAMPLES.tobytes()).hexdigest()


def one_sample(n, i):
    divergence = beta2_divergence_matrix_fourier(SAMPLES[i, :n], GRID)
    selections = np.empty(4, dtype=int)
    residuals = np.empty(4)
    gaps = np.empty(4)
    for j, value in enumerate(LAMBDAS):
        matrix = np.exp(-value * divergence)
        np.fill_diagonal(matrix, 0.)
        eigenvalues, vectors = eigsh(matrix, k=2, which='LA', tol=1e-10, v0=np.ones(N))
        vector = vectors[:, -1]
        if vector.sum() < 0:
            vector = -vector
        if np.min(vector) < -1e-10 * np.max(np.abs(vector)):
            raise ArithmeticError(f'Negative Perron vector: n={n}, repetition={i}, lambda={value}')
        np.maximum(vector, 0., out=vector)
        residuals[j] = np.linalg.norm(matrix @ vector - eigenvalues[-1] * vector) / (abs(eigenvalues[-1]) * np.linalg.norm(vector))
        gaps[j] = (eigenvalues[-1] - eigenvalues[-2]) / abs(eigenvalues[-1])
        if residuals[j] > 1e-9 or gaps[j] <= 1e-12:
            raise ArithmeticError(f'Eigenpair diagnostics failed: n={n}, repetition={i}, lambda={value}')
        selections[j] = perron_peak_index(vector)
    return i, selections, residuals, gaps


def run_size(n):
    checkpoint = DATA / f'n{n}.npz'
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            if (str(saved['sample_hash']) != SAMPLE_HASH or not np.array_equal(saved['grid'], GRID)
                    or not np.array_equal(saved['lambdas'], LAMBDAS)):
                raise ValueError(f'Checkpoint configuration mismatch: {checkpoint}')
            selected = saved['selected'].copy()
            completed = saved['completed'].copy()
            residuals = saved['residuals'].copy()
            gaps = saved['gaps'].copy()
    else:
        selected = np.full((M, 4), -1, dtype=int)
        completed = np.zeros(M, dtype=bool)
        residuals = np.full((M, 4), np.nan)
        gaps = np.full((M, 4), np.nan)

    def save():
        temporary = checkpoint.with_suffix('.npz.tmp')
        with temporary.open('wb') as f:
            np.savez_compressed(f, grid=GRID, lambdas=LAMBDAS, sample_hash=SAMPLE_HASH,
                                selected=selected, completed=completed, residuals=residuals, gaps=gaps)
        temporary.replace(checkpoint)

    print(f'n={n}: {completed.sum()}/{M} cached', flush=True)
    start = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=int(os.environ.get('OUTLOOK_WORKERS', '4'))) as pool:
            futures = [pool.submit(one_sample, n, int(i)) for i in np.flatnonzero(~completed)]
            for future in as_completed(futures):
                i, selections, errors, spectral_gaps = future.result()
                selected[i], residuals[i], gaps[i], completed[i] = selections, errors, spectral_gaps, True
                total = int(completed.sum())
                if total % 25 == 0:
                    save()
                if total % 100 == 0:
                    print(f'n={n}: {total}/{M}, {time.monotonic()-start:.0f}s', flush=True)
    finally:
        save()
    if not np.all(completed) or not np.all((selected >= 0) & (selected < N)):
        raise ArithmeticError('Incomplete or invalid selections')
    return selected, residuals, gaps


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        for n in SIZES:
            selected, residuals, gaps = run_size(n)
            counts = np.stack([np.bincount(selected[:, j], minlength=N) for j in range(4)])
            fig, ax, (_, _, grouped) = plot_thesis_bandwidth_distributions(
                GRID, counts, sample_size=n, repetitions=M, positions_per_bin=BINNING)
            assert len(ax.get_legend().get_texts()) == 4 and np.all(grouped.sum(axis=1) == M)
            fig.savefig(OUT / f'distribution_of_chosen_bandwidths_n{n}.pdf', bbox_inches='tight')
            plt.close(fig)
            print(f'n={n}: PDF saved; max residual={residuals.max():.2g}; min gap={gaps.min():.2g}', flush=True)


if __name__ == '__main__':
    main()
