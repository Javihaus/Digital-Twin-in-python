"""Evaluation harness for honest forecasting assessment."""

from PKG.evaluation.protocol import evaluate
from PKG.evaluation.report import EvalReport
from PKG.evaluation.splitters import temporal_holdout, rolling_origin, random_split
from PKG.evaluation.baselines import persistence, drift, mean_forecast, seasonal_naive
from PKG.evaluation.metrics import (
    rmse, mae, nrmse, mase, theil_u,
    crps, picp, mpiw, skill_score,
)

__all__ = [
    "evaluate",
    "EvalReport",
    "temporal_holdout",
    "rolling_origin",
    "random_split",
    "persistence",
    "drift",
    "mean_forecast",
    "seasonal_naive",
    "rmse",
    "mae",
    "nrmse",
    "mase",
    "theil_u",
    "crps",
    "picp",
    "mpiw",
    "skill_score",
]
