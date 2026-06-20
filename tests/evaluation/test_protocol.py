"""Tests for the evaluation protocol (the core differentiator)."""

import numpy as np
import pytest

from otwin.evaluation import evaluate
from otwin.evaluation.report import EvalReport


class _MockModel:
    """Minimal model: near-perfect point predictions + simple UQ hooks."""

    def __init__(self, noise: float = 0.0) -> None:
        self.noise = noise
        self.fitted = False

    def fit(self, train: np.ndarray) -> None:
        self.fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(0)
        return X + rng.normal(0, self.noise, size=X.shape)

    def predict_quantiles(self, X: np.ndarray, q: float = 0.5) -> np.ndarray:
        # Symmetric band around the value.
        shift = -1.0 if q < 0.5 else 1.0
        return X + shift

    def predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(1)
        n = X.size
        return X.flatten()[:, None] + rng.normal(0, 0.5, size=(n, 20))


def _series(n: int = 120) -> np.ndarray:
    t = np.linspace(0, 12, n)
    return (np.sin(t) + 0.05 * t).reshape(-1, 1)


def test_temporal_holdout_protocol_runs_and_reports() -> None:
    report = evaluate(_MockModel(noise=0.01), _series(), protocol="temporal_holdout")
    assert isinstance(report, EvalReport)
    assert report.split_protocol == "temporal_holdout"
    assert report.n_folds == 1
    assert "rmse" in report.point_metrics
    assert report.baseline_name  # a baseline was chosen
    assert report.data_hash and report.seed == 42


def test_rolling_origin_protocol_multiple_folds() -> None:
    report = evaluate(
        _MockModel(noise=0.01),
        _series(200),
        protocol="rolling_origin",
        n_folds=3,
        horizon=10,
    )
    assert report.split_protocol == "rolling_origin"
    assert report.n_folds == 3


def test_unknown_protocol_raises() -> None:
    with pytest.raises(ValueError):
        evaluate(_MockModel(), _series(), protocol="kfold_shuffle")


def test_probabilistic_metrics_are_computed_when_requested() -> None:
    report = evaluate(
        _MockModel(noise=0.01),
        _series(),
        protocol="temporal_holdout",
        return_uncertainty=True,
    )
    pm = report.probabilistic_metrics
    assert "picp" in pm and "mpiw" in pm and "crps" in pm
    assert 0.0 <= float(pm["picp"]) <= 1.0


def test_skill_score_positive_for_good_model() -> None:
    # A near-perfect model should beat the naive baseline -> positive skill.
    report = evaluate(_MockModel(noise=0.001), _series(), protocol="temporal_holdout")
    assert report.skill_score("rmse") > 0.0


def test_report_render_and_json_roundtrip(tmp_path) -> None:
    report = evaluate(
        _MockModel(noise=0.01),
        _series(),
        protocol="temporal_holdout",
        return_uncertainty=True,
    )
    text = str(report)
    assert "Skill Score" in text or "Baseline" in text
    md = report.to_markdown()
    assert "Baseline" in md

    path = tmp_path / "report.json"
    report.to_json(str(path))
    loaded = EvalReport.from_json(str(path))
    assert loaded.split_protocol == report.split_protocol
    assert loaded.baseline_name == report.baseline_name
