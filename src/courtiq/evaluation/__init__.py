"""Model evaluation, metrics, and grouped cross-validation."""

from courtiq.evaluation.metrics import (
    calculate_expected_calibration_error,
    compute_binary_classification_metrics,
)
from courtiq.evaluation.cv import compute_t_distribution_ci, run_game_grouped_experiment

__all__ = [
    "calculate_expected_calibration_error",
    "compute_binary_classification_metrics",
    "compute_t_distribution_ci",
    "run_game_grouped_experiment",
]
