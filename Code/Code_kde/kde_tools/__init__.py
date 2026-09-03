"""Reusable building blocks for the spectral KDE experiments.

The public imports below are intentionally small.  Import specialized helpers from
the individual modules when a notebook needs more control.
"""

from .bandwidths import (
    bandwidth_grid_linear_in_psi_beta,
    direct_plugin_bandwidth,
    gaussian_mise_oracle_bandwidth,
    least_squares_cv_bandwidth_index,
    nearest_bandwidth_grid_index,
    robust_scale_for_bandwidth,
    silverman_rule_of_thumb_bandwidth,
)
from .density import (
    exact_gaussian_kde_l2_gram,
    exact_oracle_ise_values,
    gaussian_kde_values,
)
from .gaussian_mixtures import (
    exact_mixture_ise_values,
    exact_mixture_mise_values,
    gaussian_density_with_variance,
    gaussian_mixture_mean_and_variance,
    mixture_density,
    sample_gaussian_mixture,
)
from .kullback_leibler import (
    directional_kl_divergence_matrix,
    directional_kl_similarity,
)
from .quadrature import (
    compute_quadrature_similarity_matrices,
    quadrature_beta2_divergence_matrix,
    quadrature_grid_for_kdes,
    quadrature_similarity_matrices,
    select_quadrature_similarity_bandwidths,
    trapezoidal_integral,
)
from .pdf_export import (
    disable_automatic_pdf_saving,
    enable_automatic_pdf_saving,
)
from .similarities import (
    beta2_divergence_matrix,
    beta2_similarity_matrix,
    first_order_perron_vector,
    perron_vector,
    population_gaussian_beta2_divergence_matrix,
    select_spectral_bandwidth,
    spectral_choice_from_similarity,
)

__all__ = [
    "bandwidth_grid_linear_in_psi_beta",
    "beta2_divergence_matrix",
    "beta2_similarity_matrix",
    "compute_quadrature_similarity_matrices",
    "direct_plugin_bandwidth",
    "disable_automatic_pdf_saving",
    "directional_kl_divergence_matrix",
    "directional_kl_similarity",
    "first_order_perron_vector",
    "exact_gaussian_kde_l2_gram",
    "exact_mixture_ise_values",
    "exact_mixture_mise_values",
    "exact_oracle_ise_values",
    "enable_automatic_pdf_saving",
    "gaussian_density_with_variance",
    "gaussian_kde_values",
    "gaussian_mise_oracle_bandwidth",
    "gaussian_mixture_mean_and_variance",
    "least_squares_cv_bandwidth_index",
    "mixture_density",
    "nearest_bandwidth_grid_index",
    "perron_vector",
    "population_gaussian_beta2_divergence_matrix",
    "quadrature_beta2_divergence_matrix",
    "quadrature_grid_for_kdes",
    "quadrature_similarity_matrices",
    "robust_scale_for_bandwidth",
    "sample_gaussian_mixture",
    "select_quadrature_similarity_bandwidths",
    "select_spectral_bandwidth",
    "silverman_rule_of_thumb_bandwidth",
    "spectral_choice_from_similarity",
    "trapezoidal_integral",
]
