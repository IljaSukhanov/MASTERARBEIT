"""Gaussian KDE L2 inner products by direct frequency quadrature."""

from __future__ import annotations

import numpy as np

from ..similarities import squared_l2_distance_from_gram

FOURIER_L2_METHOD_VERSION = "frequency_trapezoid_v1"

# Shared production grid: K intervals require K+1 trapezoidal nodes.
DEFAULT_FREQUENCY_CUTOFF = 450.0
DEFAULT_FREQUENCY_INTERVALS = 1536
DEFAULT_FREQUENCY_POINTS = DEFAULT_FREQUENCY_INTERVALS + 1
DEFAULT_FOURIER_GRID_TAG = (
    f"omega{DEFAULT_FREQUENCY_CUTOFF:g}_k{DEFAULT_FREQUENCY_INTERVALS}"
)

# Compatibility controls for callers that still accept (half_width, spatial_points).
# Their conversion below gives exactly the shared cutoff and interval count.
DEFAULT_FOURIER_HALF_WIDTH = np.pi * DEFAULT_FREQUENCY_INTERVALS / DEFAULT_FREQUENCY_CUTOFF
DEFAULT_FOURIER_SPATIAL_POINTS = 2 * DEFAULT_FREQUENCY_INTERVALS


def gaussian_kde_l2_gram_fourier(
    samples: np.ndarray,
    h_grid: np.ndarray,
    *,
    frequency_cutoff: float = DEFAULT_FREQUENCY_CUTOFF,
    frequency_points: int = DEFAULT_FREQUENCY_POINTS,
) -> np.ndarray:
    r"""Integrate the Gaussian KDE inner products directly in frequency space.

    With P_n(w) = mean(exp(-1j * samples * w)), approximate

        G_rs = (1/pi) integral_0^infinity |P_n(w)|^2
                       exp(-(h_r^2 + h_s^2) * w^2 / 2) dw.

    Composite trapezoidal weights on [0, frequency_cutoff] are factored
    into real features B_rk = sqrt(w_k) |P_n(omega_k)| exp(-h_r^2 omega_k^2/2),
    then G = B @ B.T. No inverse FFT, density clipping or renormalization
    is used. This is the numerical approximation in the thesis's
    5_bandwidth_selection.tex, after eq:beta2_exact_formula02.

    The cutoff and node spacing must resolve the integral; accuracy is not
    certified by this routine. Parseval's identity in the thesis convention
    follows from https://dlmf.nist.gov/1.14.E7_5, and the integration rule is
    documented at https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.trapezoid.html.
    """
    samples = np.asarray(samples, dtype=float)
    h_grid = np.asarray(h_grid, dtype=float)
    frequency_cutoff = float(frequency_cutoff)
    if samples.ndim != 1 or not samples.size or not np.all(np.isfinite(samples)):
        raise ValueError("samples must be finite, nonempty and one-dimensional.")
    if (h_grid.ndim != 1 or not h_grid.size or
            not np.all(np.isfinite(h_grid)) or np.any(h_grid <= 0.0)):
        raise ValueError("h_grid must contain finite, strictly positive bandwidths.")
    if not np.isfinite(frequency_cutoff) or frequency_cutoff <= 0.0:
        raise ValueError("frequency_cutoff must be finite and strictly positive.")
    if (isinstance(frequency_points, (bool, np.bool_)) or
            not isinstance(frequency_points, (int, np.integer)) or frequency_points < 2):
        raise ValueError("frequency_points must be an integer of at least two.")

    omega = np.linspace(0.0, frequency_cutoff, int(frequency_points))
    empirical_cf = np.exp(-1j * omega[:, None] * samples[None, :]).mean(axis=1)
    weights = np.full(len(omega), (omega[1] - omega[0]) / np.pi)
    weights[[0, -1]] *= 0.5
    features = np.exp(-0.5 * (h_grid[:, None] * omega[None, :])**2)
    features *= (np.sqrt(weights) * np.abs(empirical_cf))[None, :]
    return features @ features.T


def beta2_divergence_matrix_fourier(
    samples: np.ndarray,
    h_grid: np.ndarray,
    half_width: float = DEFAULT_FOURIER_HALF_WIDTH,
    spatial_points: int = DEFAULT_FOURIER_SPATIAL_POINTS,
) -> np.ndarray:
    r"""Return half squared L2 distances using direct frequency quadrature.

    The grid controls are retained for existing experiments: half_width=H
    and spatial_points=m define delta_omega=pi/H, cutoff=pi*m/(2H), and
    m/2+1 frequency nodes. Defaults give Omega=450 and K=1536 intervals.
    No spatial KDE grid is reconstructed. For an
    independently chosen cutoff and node count use gaussian_kde_l2_gram_fourier.
    """
    half_width = float(half_width)
    if not np.isfinite(half_width) or half_width <= 0.0:
        raise ValueError("half_width must be finite and strictly positive.")
    if (isinstance(spatial_points, (bool, np.bool_)) or
            not isinstance(spatial_points, (int, np.integer)) or
            spatial_points < 2 or spatial_points % 2):
        raise ValueError("spatial_points must be an even integer of at least two.")
    gram = gaussian_kde_l2_gram_fourier(
        samples, h_grid,
        frequency_cutoff=np.pi * int(spatial_points) / (2.0 * half_width),
        frequency_points=int(spatial_points) // 2 + 1,
    )
    return 0.5 * squared_l2_distance_from_gram(gram)


def beta2_divergence_matrix_fft(
    samples: np.ndarray,
    h_grid: np.ndarray,
    half_width: float = DEFAULT_FOURIER_HALF_WIDTH,
    spatial_points: int = DEFAULT_FOURIER_SPATIAL_POINTS,
) -> np.ndarray:
    """Compatibility name for beta2_divergence_matrix_fourier; no FFT is used."""
    return beta2_divergence_matrix_fourier(samples, h_grid, half_width, spatial_points)


def beta2_similarity_matrix_fourier(
    samples: np.ndarray,
    h_grid: np.ndarray,
    lambda_value: float = 1.0,
    half_width: float = DEFAULT_FOURIER_HALF_WIDTH,
    spatial_points: int = DEFAULT_FOURIER_SPATIAL_POINTS,
) -> np.ndarray:
    """Return ``exp(-lambda * D_2)`` using direct frequency quadrature."""
    if not np.isfinite(lambda_value) or lambda_value <= 0.0:
        raise ValueError("lambda_value must be finite and positive.")
    divergence = beta2_divergence_matrix_fourier(
        samples=samples,
        h_grid=h_grid,
        half_width=half_width,
        spatial_points=spatial_points,
    )
    similarity = np.exp(-float(lambda_value) * divergence)
    np.fill_diagonal(similarity, 1.0)
    return similarity


def beta2_similarity_matrix_fft(
    samples: np.ndarray,
    h_grid: np.ndarray,
    lambda_value: float = 1.0,
    half_width: float = DEFAULT_FOURIER_HALF_WIDTH,
    spatial_points: int = DEFAULT_FOURIER_SPATIAL_POINTS,
) -> np.ndarray:
    """Compatibility name for beta2_similarity_matrix_fourier; no FFT is used."""
    return beta2_similarity_matrix_fourier(
        samples, h_grid, lambda_value, half_width, spatial_points
    )
