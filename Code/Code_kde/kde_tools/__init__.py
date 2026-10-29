"""Reusable building blocks for the spectral KDE experiments.

The public imports below are intentionally small.  Import specialized helpers from
the individual modules when a notebook needs more control.
"""
from .bandwidths import bandwidth_grid_linear_in_psi_beta, direct_plugin_bandwidth, gaussian_mise_oracle_bandwidth, least_squares_cv_bandwidth_index, nearest_bandwidth_grid_index, robust_scale_for_bandwidth, silverman_rule_of_thumb_bandwidth
from .density import exact_gaussian_kde_l2_gram, exact_oracle_ise_values, gaussian_kde_values
from .energy_minimization import EnergySearchConfig, energy_search_settings, minimize_bandwidth_energy
from .qhat import minimize_qhat, expected_gaussian_qhat_log_energy
from .gaussian_mixtures import exact_mixture_ise_values, exact_mixture_mise_values, gaussian_density_with_variance, gaussian_mixture_mean_and_variance, mixture_density, sample_gaussian_mixture
from .quadrature import trapezoidal_integral
from .pdf_export import disable_automatic_pdf_saving, enable_automatic_pdf_saving
from .perron import PERRON_SOLVER_VERSION, PERRON_TOLERANCE, PerronConvergenceError, PerronDegeneracyError, PerronResult, doob_mass_from_perron, perron_eigenpair, perron_eigenpair_from_divergence, perron_peak_index, perron_vector_from_divergence
from .similarities import beta2_divergence_matrix, beta2_similarity_matrix, first_order_perron_vector, perron_vector, population_gaussian_beta2_divergence_matrix, select_spectral_bandwidth, spectral_choice_from_divergence, spectral_choice_from_similarity
__all__ = ['EnergySearchConfig', 'energy_search_settings', 'minimize_bandwidth_energy', 'minimize_qhat', 'expected_gaussian_qhat_log_energy', 'PERRON_SOLVER_VERSION', 'PERRON_TOLERANCE', 'PerronConvergenceError', 'PerronDegeneracyError', 'PerronResult', 'doob_mass_from_perron', 'perron_eigenpair', 'perron_eigenpair_from_divergence', 'perron_peak_index', 'perron_vector_from_divergence', 'bandwidth_grid_linear_in_psi_beta', 'beta2_divergence_matrix', 'beta2_similarity_matrix', 'direct_plugin_bandwidth', 'disable_automatic_pdf_saving', 'first_order_perron_vector', 'exact_gaussian_kde_l2_gram', 'exact_mixture_ise_values', 'exact_mixture_mise_values', 'exact_oracle_ise_values', 'enable_automatic_pdf_saving', 'gaussian_density_with_variance', 'gaussian_kde_values', 'gaussian_mise_oracle_bandwidth', 'gaussian_mixture_mean_and_variance', 'least_squares_cv_bandwidth_index', 'mixture_density', 'nearest_bandwidth_grid_index', 'perron_vector', 'population_gaussian_beta2_divergence_matrix', 'robust_scale_for_bandwidth', 'sample_gaussian_mixture', 'select_spectral_bandwidth', 'silverman_rule_of_thumb_bandwidth', 'spectral_choice_from_similarity', 'spectral_choice_from_divergence', 'trapezoidal_integral']
