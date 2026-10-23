"""Direct Fourier quadrature for L2, plus optional periodic KDE reconstruction."""

from .reconstruction import kde_family_fft
from .similarities import (
    FOURIER_L2_METHOD_VERSION,
    gaussian_kde_l2_gram_fourier,
    beta2_divergence_matrix_fourier,
    beta2_similarity_matrix_fourier,
    beta2_divergence_matrix_fft,
    beta2_similarity_matrix_fft,
)
from ..kullback_leibler import directional_kl_similarity_fft

__all__ = [
    "FOURIER_L2_METHOD_VERSION",
    "gaussian_kde_l2_gram_fourier",
    "beta2_divergence_matrix_fourier",
    "beta2_similarity_matrix_fourier",
    "beta2_divergence_matrix_fft",
    "beta2_similarity_matrix_fft",
    "directional_kl_similarity_fft",
    "kde_family_fft",
]
