"""Optional Fourier-based KDE helpers.

The main :mod:`kde_tools` package uses analytic Gaussian formulas or ordinary
spatial quadrature.  Import this subpackage explicitly only for experiments
that intentionally use periodic Fourier reconstruction.
"""

from .reconstruction import kde_family_fft
from .similarities import (
    beta2_divergence_matrix_fft,
    beta2_similarity_matrix_fft,
)
from ..kullback_leibler import directional_kl_similarity_fft

__all__ = [
    "beta2_divergence_matrix_fft",
    "beta2_similarity_matrix_fft",
    "directional_kl_similarity_fft",
    "kde_family_fft",
]
