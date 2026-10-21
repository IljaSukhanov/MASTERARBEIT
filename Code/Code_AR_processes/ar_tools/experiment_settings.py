"""Default paths and settings shared by AR notebooks and tools."""

from pathlib import Path

from .coefficient_errors import COEFFICIENT_ERROR_METRIC

AR_DIRECTORY = Path(__file__).resolve().parents[1]
PROJECT_DIRECTORY = AR_DIRECTORY.parents[1]
WARM_START_VARIANTS = ("Original saved", "Independent rerun", "Padded warm start")
ORIGINAL_ORDER_DIRECTORY = AR_DIRECTORY / "outputs/csv/ar_order_minimum_width"
WARM_START_DATA_DIRECTORY = AR_DIRECTORY / "outputs/csv/ar3_order_selection_warm_starts"
WARM_START_IMAGE_DIRECTORY = AR_DIRECTORY / "outputs/images/ar3_order_selection_warm_starts"
WARM_START_PDF_DIRECTORY = AR_DIRECTORY / "outputs/pdfs/ar3_order_selection_warm_starts"
LP_DESCENT_DATA_DIRECTORY = AR_DIRECTORY / "outputs/csv/ar3_descent_chain_lp"
LP_DESCENT_IMAGE_DIRECTORY = AR_DIRECTORY / "outputs/images/ar3_descent_chain_lp"
LP_DESCENT_PDF_DIRECTORY = AR_DIRECTORY / "outputs/pdfs/ar3_descent_chain_lp"
POPULATION_DATA_DIRECTORY = AR_DIRECTORY / "outputs/csv/ar3_population_breakpoint_selectors"
POPULATION_IMAGE_DIRECTORY = AR_DIRECTORY / "outputs/images/ar3_population_breakpoint_selectors"
POPULATION_PDF_DIRECTORY = AR_DIRECTORY / "outputs/pdfs/ar3_population_breakpoint_selectors"
GRADIENT_DATA_DIRECTORY = AR_DIRECTORY / "outputs/csv/ar3_gradient_figure_2_4_examples"
GRADIENT_IMAGE_DIRECTORY = AR_DIRECTORY / "outputs/images/ar3_gradient_figure_2_4_examples"
GRADIENT_PDF_PATH = PROJECT_DIRECTORY / "output/pdf/ar3_gradient_figure_2_4_ten_examples.pdf"
AR3_MAE_OUTPUT_DIRECTORY = AR_DIRECTORY / "outputs/csv/ar3_minimum_width_k4/mae_vs_T_100_processes/absolute_steps"
AR3_MAE_CONFIGURATION = {'order': 3,
 'coefficient_error_metric': COEFFICIENT_ERROR_METRIC,
 'embedding_dimension': 4,
 'replications': 100,
 'sample_sizes': [100, 500, 1000, 1500, 2000],
 'burnin': 1000,
 'innovation_distribution': 'uniform[-1,1]',
 'experiment_seed': 20260928,
 'optimizer_seed': 20260929,
 'starts': 12,
 'iterations': 5000,
 'initial_step': 0.2,
 'geometric_decay': 0.997,
 'hull_vertex_reduction': True,
 'mae_definition': 'mean of three absolute coefficient errors per fit; then mean over 100 '
                   'processes'}
