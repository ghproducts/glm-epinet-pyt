"""Analysis of uncertainty-quantification predictions.

Turns the per-example prediction CSVs written by inference into the tables and
figures reported in the manuscript.
"""

from .metrics import (
    accuracy_by_uncertainty,
    compute_auroc,
    compute_aurc,
    compute_auco,
    compute_brier,
    compute_ece,
    compute_nll,
    confidence_oracle_curve,
    ece_bin_sensitivity,
    reliability_bins,
)
from .repair import read_predictions
from .results import ResultsIndex, RunKey
from .tasks import TaskPair, TaskRegistry, load_registry

__all__ = [
    "ResultsIndex", "RunKey",
    "TaskPair", "TaskRegistry", "load_registry",
    "compute_ece", "compute_nll", "compute_brier", "compute_auroc",
    "compute_aurc", "compute_auco", "confidence_oracle_curve",
    "ece_bin_sensitivity", "reliability_bins", "accuracy_by_uncertainty",
    "read_predictions",
]
