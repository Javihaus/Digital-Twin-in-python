"""High-level evaluation protocol (rigorous by default)."""

import hashlib
from typing import Any

import numpy as np
import numpy.typing as npt

from PKG.evaluation.baselines import get_best_baseline
from PKG.evaluation.metrics import crps, mae, mase, mpiw, nrmse, picp, rmse, theil_u
from PKG.evaluation.report import EvalReport
from PKG.evaluation.splitters import rolling_origin, temporal_holdout


def evaluate(
    model: Any,
    data: npt.NDArray[np.floating],
    protocol: str = "temporal_holdout",
    test_frac: float = 0.2,
    n_folds: int = 5,
    horizon: int = 10,
    return_uncertainty: bool = False,
    seasonal_period: int | None = None,
    seed: int = 42,
) -> EvalReport:
    """
    Evaluate forecasting model rigorously (baselines + temporal split).

    This is the ONE ENTRY POINT for evaluation. It enforces:
    - Temporal split (default) or rolling-origin
    - Mandatory baseline comparison
    - Skill scores as headline metric

    Args:
        model: Model with .predict(X) method (and optional .predict_quantiles(X) for UQ)
        data: Time series data (n_samples, n_features)
        protocol: 'temporal_holdout' or 'rolling_origin'
        test_frac: Fraction for test (temporal_holdout)
        n_folds: Number of folds (rolling_origin)
        horizon: Forecast horizon (rolling_origin)
        return_uncertainty: Whether to compute probabilistic metrics
        seasonal_period: Period for seasonal baselines (optional)
        seed: Random seed

    Returns:
        EvalReport with skill scores, baselines, and all metrics

    Example:
        >>> # Assume model has .predict(X) method
        >>> data = np.arange(100).reshape(-1, 1)
        >>> report = evaluate(model, data, protocol='temporal_holdout')
        >>> print(report)
        EvalReport (temporal_holdout, 1 fold)
        ...
    """
    if protocol == "temporal_holdout":
        train, test = temporal_holdout(data, test_frac=test_frac)
        folds = [(train, test)]
        n_folds_actual = 1

    elif protocol == "rolling_origin":
        min_train = max(50, len(data) // 4)  # At least 50 or 25% of data
        folds = list(
            rolling_origin(data, n_folds=n_folds, min_train=min_train, horizon=horizon)
        )
        n_folds_actual = len(folds)

    else:
        raise ValueError(
            f"Unknown protocol: {protocol}. Use 'temporal_holdout' or 'rolling_origin'."
        )

    # Accumulate metrics across folds
    model_rmse_vals = []
    model_mae_vals = []
    baseline_rmse_vals = []
    baseline_mae_vals = []
    baseline_names = []

    mase_vals = []
    theil_u_vals = []
    crps_vals = []
    picp_vals = []
    mpiw_vals = []

    for train, test in folds:
        # Train model (if it has a .fit method)
        if hasattr(model, "fit"):
            model.fit(train)

        # Model predictions
        y_pred = model.predict(test)

        # Ensure shapes match
        if y_pred.shape != test.shape:
            y_pred = y_pred.reshape(test.shape)

        # Point metrics
        model_rmse_vals.append(rmse(test, y_pred))
        model_mae_vals.append(mae(test, y_pred))

        # Baseline
        baseline_name, baseline_pred, baseline_rmse = get_best_baseline(
            train, test, period=seasonal_period
        )
        baseline_names.append(baseline_name)
        baseline_rmse_vals.append(baseline_rmse)
        baseline_mae_vals.append(mae(test, baseline_pred))

        # Scale-free metrics
        mase_vals.append(
            mase(test, y_pred, train, seasonal_period=seasonal_period or 1)
        )
        theil_u_vals.append(theil_u(test, y_pred, train))

        # Probabilistic metrics (if requested)
        if return_uncertainty:
            if hasattr(model, "predict_quantiles"):
                # Get 5th and 95th percentiles (90% interval)
                lower = model.predict_quantiles(test, q=0.05)
                upper = model.predict_quantiles(test, q=0.95)

                picp_vals.append(picp(test.flatten(), lower.flatten(), upper.flatten()))
                mpiw_vals.append(mpiw(lower.flatten(), upper.flatten()))

            if hasattr(model, "predict_ensemble"):
                # Get ensemble forecasts for CRPS
                ensemble = model.predict_ensemble(test)  # (n_samples, n_members)
                crps_vals.append(crps(test.flatten(), ensemble))

    # Aggregate across folds
    model_rmse_mean = float(np.mean(model_rmse_vals))
    model_mae_mean = float(np.mean(model_mae_vals))
    baseline_rmse_mean = float(np.mean(baseline_rmse_vals))
    baseline_mae_mean = float(np.mean(baseline_mae_vals))

    # Most common baseline
    baseline_name = max(set(baseline_names), key=baseline_names.count)

    # Create report
    report = EvalReport(
        split_protocol=protocol,
        n_folds=n_folds_actual,
        baseline_name=baseline_name,
        baseline_metrics={
            "rmse": baseline_rmse_mean,
            "mae": baseline_mae_mean,
        },
    )

    # Add model metrics
    report.add_point_metrics(
        rmse=model_rmse_mean,
        mae=model_mae_mean,
        nrmse=(
            float(np.mean([nrmse(t, model.predict(t)) for _, t in folds]))
            if hasattr(model, "predict")
            else None
        ),
        mase=float(np.mean(mase_vals)) if mase_vals else None,
        theil_u=float(np.mean(theil_u_vals)) if theil_u_vals else None,
    )

    # Add probabilistic metrics if available
    if crps_vals or picp_vals or mpiw_vals:
        report.add_probabilistic_metrics(
            crps=float(np.mean(crps_vals)) if crps_vals else None,
            picp=float(np.mean(picp_vals)) if picp_vals else None,
            mpiw=float(np.mean(mpiw_vals)) if mpiw_vals else None,
            nominal_level=0.90,
        )

    # Metadata
    report.seed = seed
    report.data_hash = hashlib.sha256(data.tobytes()).hexdigest()[:16]
    report.model_name = (
        model.__class__.__name__ if hasattr(model, "__class__") else "Unknown"
    )

    return report
