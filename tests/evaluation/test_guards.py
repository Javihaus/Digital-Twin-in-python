"""Tests for evaluation guards (enforcing leakage-free evaluation)."""

import warnings

import numpy as np
import pytest

from PKG.evaluation import random_split, temporal_holdout
from PKG.evaluation.report import EvalReport


def test_random_split_emits_warning() -> None:
    """Guard: Random split must emit loud warning."""
    data = np.arange(100).reshape(-1, 1)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        train, test = random_split(data, test_frac=0.2, seed=42)

        # Check warning was raised
        assert len(w) == 1
        assert "RANDOM SPLIT USED FOR TIME SERIES" in str(w[0].message)
        assert "INTERPOLATION" in str(w[0].message)
        assert "FORECASTING" in str(w[0].message)


def test_eval_report_requires_split_protocol() -> None:
    """Guard: EvalReport cannot be created without split protocol."""
    # This is enforced by dataclass required fields
    with pytest.raises(TypeError):
        # Missing required arguments
        EvalReport()  # type: ignore


def test_eval_report_requires_baseline() -> None:
    """Guard: EvalReport requires baseline metrics."""
    # This is enforced by dataclass required fields
    with pytest.raises(TypeError):
        EvalReport(split_protocol="temporal_holdout", n_folds=1)  # type: ignore


def test_eval_report_skill_score_first() -> None:
    """Guard: Skill score appears FIRST in string representation."""
    report = EvalReport(
        split_protocol="temporal_holdout",
        n_folds=1,
        baseline_name="persistence",
        baseline_metrics={"rmse": 0.20, "mae": 0.15},
    )
    report.add_point_metrics(rmse=0.10, mae=0.08)

    report_str = str(report)
    lines = report_str.split("\n")

    # Find "Skill Score" line
    skill_line_idx = None
    for i, line in enumerate(lines):
        if "Skill Score" in line:
            skill_line_idx = i
            break

    assert skill_line_idx is not None, "Skill score not found in report"

    # Should be within first 5 lines (after header)
    assert skill_line_idx < 5, f"Skill score not prominent (line {skill_line_idx})"


def test_eval_report_shows_baseline_name() -> None:
    """Guard: Baseline name is clearly shown."""
    report = EvalReport(
        split_protocol="rolling_origin",
        n_folds=5,
        baseline_name="drift",
        baseline_metrics={"rmse": 0.15},
    )

    report_str = str(report)
    assert "drift" in report_str


def test_temporal_holdout_is_temporal() -> None:
    """Verify: temporal_holdout maintains time order."""
    data = np.arange(100).reshape(-1, 1)
    train, test = temporal_holdout(data, test_frac=0.2)

    # Train should come before test chronologically
    assert train[-1, 0] < test[0, 0]


def test_eval_report_json_roundtrip() -> None:
    """Test: Report can be saved and loaded for reproducibility."""
    import tempfile

    report = EvalReport(
        split_protocol="temporal_holdout",
        n_folds=1,
        baseline_name="persistence",
        baseline_metrics={"rmse": 0.20, "mae": 0.15},
        data_hash="abc123",
        seed=42,
    )
    report.add_point_metrics(rmse=0.10, mae=0.08, mase=0.5, theil_u=0.5)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        report.to_json(f.name)
        loaded = EvalReport.from_json(f.name)

    assert loaded.split_protocol == report.split_protocol
    assert loaded.baseline_name == report.baseline_name
    assert loaded.point_metrics["rmse"] == report.point_metrics["rmse"]
    assert loaded.data_hash == report.data_hash
    assert loaded.seed == report.seed


def test_skill_score_interpretation() -> None:
    """Verify: Skill score correctly interprets better/worse than baseline."""
    report = EvalReport(
        split_protocol="temporal_holdout",
        n_folds=1,
        baseline_name="persistence",
        baseline_metrics={"rmse": 0.20},
    )

    # Model better than baseline
    report.add_point_metrics(rmse=0.10, mae=0.05)
    assert report.skill_score("rmse") == 0.5  # 50% improvement

    # Model worse than baseline
    report2 = EvalReport(
        split_protocol="temporal_holdout",
        n_folds=1,
        baseline_name="persistence",
        baseline_metrics={"rmse": 0.10},
    )
    report2.add_point_metrics(rmse=0.20, mae=0.15)
    assert report2.skill_score("rmse") == -1.0  # 100% worse


def test_eval_report_markdown_generation() -> None:
    """Test: Markdown generation for docs."""
    report = EvalReport(
        split_protocol="rolling_origin",
        n_folds=5,
        baseline_name="drift",
        baseline_metrics={"rmse": 0.20, "mae": 0.15},
        model_name="TestModel",
    )
    report.add_point_metrics(rmse=0.15, mae=0.12, mase=0.75)

    md = report.to_markdown()

    assert "TestModel" in md
    assert "rolling_origin" in md
    assert "drift" in md
    assert "RMSE" in md
    assert "0.15" in md  # Model RMSE
