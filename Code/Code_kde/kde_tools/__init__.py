"""Reusable building blocks for the spectral KDE experiments.

The public imports below are intentionally small.  Import specialized helpers from
the individual modules when a notebook needs more control.
"""

from .bandwidths import (
    bandwidth_grid_for_gaussian_experiment,
    bandwidth_grid_for_gaussian_mixture_experiment,
    direct_plugin_bandwidth,
    gaussian_mise_oracle_bandwidth,
    least_squares_cv_bandwidth_index,
    nearest_bandwidth_grid_index,
    robust_scale_for_bandwidth,
    silverman_rule_of_thumb_bandwidth,
)
from .density import (
    exact_gaussian_kde_l2_gram,
    exact_mixture_ise_values,
    exact_oracle_ise_values,
    gaussian_density_with_variance,
    gaussian_kde_values,
    gaussian_mixture_mean_and_variance,
    mixture_density,
    quadrature_grid_for_kdes,
    sample_gaussian_mixture,
    trapezoidal_integral,
)
from .similarities import (
    compute_quadrature_similarity_matrices,
    directional_kl_similarity,
    perron_vector,
    quadrature_similarity_matrices,
    select_quadrature_similarity_bandwidths,
    select_spectral_bandwidth,
    spectral_choice_from_similarity,
    symmetrized_renyi_similarity,
    toeplitz_projection,
)

__all__ = [
    "bandwidth_grid_for_gaussian_experiment",
    "bandwidth_grid_for_gaussian_mixture_experiment",
    "compute_quadrature_similarity_matrices",
    "direct_plugin_bandwidth",
    "directional_kl_similarity",
    "exact_gaussian_kde_l2_gram",
    "exact_mixture_ise_values",
    "exact_oracle_ise_values",
    "gaussian_density_with_variance",
    "gaussian_kde_values",
    "gaussian_mise_oracle_bandwidth",
    "gaussian_mixture_mean_and_variance",
    "least_squares_cv_bandwidth_index",
    "mixture_density",
    "nearest_bandwidth_grid_index",
    "perron_vector",
    "quadrature_grid_for_kdes",
    "quadrature_similarity_matrices",
    "robust_scale_for_bandwidth",
    "sample_gaussian_mixture",
    "select_quadrature_similarity_bandwidths",
    "select_spectral_bandwidth",
    "silverman_rule_of_thumb_bandwidth",
    "spectral_choice_from_similarity",
    "symmetrized_renyi_similarity",
    "toeplitz_projection",
    "trapezoidal_integral",
]
