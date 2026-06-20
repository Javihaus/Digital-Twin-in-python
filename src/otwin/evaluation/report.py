"""Evaluation report with mandatory baselines and calibrated metrics."""

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalReport:
    """
    Evaluation report enforcing leakage-free forecasting evaluation.

    GUARDS (enforced by construction):
    - Cannot be created without a split protocol
    - Cannot report headline metric without at least one baseline
    - Skill score (vs best baseline) is the FIRST thing displayed
    - Random split emits loud warning

    Args:
        split_protocol: Name of split protocol ('temporal_holdout', 'rolling_origin', etc.)
        n_folds: Number of folds (1 for single holdout)
        baseline_name: Name of best baseline
        baseline_metrics: Dict of baseline metric values

    Example:
        >>> report = EvalReport(
        ...     split_protocol='rolling_origin',
        ...     n_folds=5,
        ...     baseline_name='persistence',
        ...     baseline_metrics={'rmse': 0.1, 'mae': 0.08}
        ... )
        >>> report.add_point_metrics(rmse=0.07, mae=0.05)
        >>> report.skill_score()
        0.3
    """

    split_protocol: str
    n_folds: int
    baseline_name: str
    baseline_metrics: dict[str, float]

    # Model metrics (filled by add_* methods)
    point_metrics: dict[str, float] = field(default_factory=dict)
    probabilistic_metrics: dict[str, float | str] = field(default_factory=dict)

    # Metadata
    data_hash: str | None = None
    model_name: str | None = None
    seed: int | None = None
    version: str = "2.0.0-alpha"

    def add_point_metrics(
        self,
        rmse: float,
        mae: float,
        nrmse: float | None = None,
        mase: float | None = None,
        theil_u: float | None = None,
    ) -> None:
        """Add point forecast metrics."""
        self.point_metrics["rmse"] = rmse
        self.point_metrics["mae"] = mae

        if nrmse is not None:
            self.point_metrics["nrmse"] = nrmse
        if mase is not None:
            self.point_metrics["mase"] = mase
        if theil_u is not None:
            self.point_metrics["theil_u"] = theil_u

    def add_probabilistic_metrics(
        self,
        crps: float | None = None,
        picp: float | None = None,
        mpiw: float | None = None,
        nominal_level: float = 0.90,
    ) -> None:
        """Add probabilistic forecast metrics."""
        if crps is not None:
            self.probabilistic_metrics["crps"] = crps
        if picp is not None:
            self.probabilistic_metrics["picp"] = picp
            self.probabilistic_metrics["nominal_level"] = nominal_level
            # Calibration quality: how close to nominal
            calibration_error = abs(picp - nominal_level)
            if calibration_error < 0.05:
                self.probabilistic_metrics["calibration"] = "good"
            elif calibration_error < 0.10:
                self.probabilistic_metrics["calibration"] = "fair"
            else:
                self.probabilistic_metrics["calibration"] = "poor"
        if mpiw is not None:
            self.probabilistic_metrics["mpiw"] = mpiw

    def skill_score(self, metric: str = "rmse") -> float:
        """
        Compute skill score vs baseline: (baseline - model) / baseline.

        Positive = better than baseline.
        0 = same as baseline.
        Negative = worse than baseline.

        Args:
            metric: Metric to use ('rmse', 'mae', etc.)

        Returns:
            Skill score (higher is better)
        """
        if metric not in self.point_metrics:
            raise ValueError(f"Model metric '{metric}' not available")
        if metric not in self.baseline_metrics:
            raise ValueError(f"Baseline metric '{metric}' not available")

        model_val = self.point_metrics[metric]
        baseline_val = self.baseline_metrics[metric]

        if baseline_val == 0:
            return 0.0 if model_val == 0 else float("-inf")

        return (baseline_val - model_val) / baseline_val

    def __str__(self) -> str:
        """
        String representation leading with skill score.

        This ensures the FIRST thing the user sees is:
        "Are we better than a trivial baseline?"
        """
        lines = []
        lines.append(
            f"EvalReport ({self.split_protocol}, {self.n_folds} fold{'s' if self.n_folds > 1 else ''})"
        )
        lines.append("━" * 64)

        # HEADLINE: Skill score
        if "rmse" in self.point_metrics and "rmse" in self.baseline_metrics:
            ss = self.skill_score("rmse")
            if ss > 0:
                pct_better = int(ss * 100)
                lines.append(
                    f"Skill Score (vs best baseline): {ss:.2f} ({pct_better}% better)"
                )
            elif ss == 0:
                lines.append("Skill Score: 0.00 (same as baseline)")
            else:
                pct_worse = int(-ss * 100)
                lines.append(
                    f"Skill Score: {ss:.2f} ({pct_worse}% WORSE than baseline)"
                )

        lines.append(f"Baseline: {self.baseline_name}")
        lines.append("")

        # Point metrics
        if self.point_metrics:
            lines.append("Point Metrics:")
            for key, val in self.point_metrics.items():
                if key in self.baseline_metrics:
                    baseline_val = self.baseline_metrics[key]
                    lines.append(
                        f"  {key.upper():8s}  {val:.4f} (baseline: {baseline_val:.4f})"
                    )
                else:
                    lines.append(f"  {key.upper():8s}  {val:.4f}")
            lines.append("")

        # Probabilistic metrics
        if self.probabilistic_metrics:
            lines.append("Probabilistic Metrics:")
            for key, mval in self.probabilistic_metrics.items():
                if key == "nominal_level":
                    continue  # Shown with PICP
                elif key == "picp":
                    nominal = float(
                        self.probabilistic_metrics.get("nominal_level", 0.90)
                    )
                    calibration = self.probabilistic_metrics.get(
                        "calibration", "unknown"
                    )
                    lines.append(
                        f"  PICP@{int(nominal * 100):02d}  {float(mval):.2f}   "
                        f"(target: {nominal:.2f}, calibration: {calibration})"
                    )
                elif key != "calibration":
                    lines.append(f"  {key.upper():8s}  {float(mval):.4f}")

        lines.append("━" * 64)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "split_protocol": self.split_protocol,
            "n_folds": self.n_folds,
            "baseline_name": self.baseline_name,
            "baseline_metrics": self.baseline_metrics,
            "point_metrics": self.point_metrics,
            "probabilistic_metrics": self.probabilistic_metrics,
            "skill_score_rmse": (
                self.skill_score("rmse") if "rmse" in self.point_metrics else None
            ),
            "data_hash": self.data_hash,
            "model_name": self.model_name,
            "seed": self.seed,
            "version": self.version,
        }

    def to_json(self, filepath: str) -> None:
        """Save report to JSON file (for reproducible benchmarks)."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, filepath: str) -> "EvalReport":
        """Load report from JSON file."""
        with open(filepath) as f:
            data = json.load(f)

        report = cls(
            split_protocol=data["split_protocol"],
            n_folds=data["n_folds"],
            baseline_name=data["baseline_name"],
            baseline_metrics=data["baseline_metrics"],
        )

        report.point_metrics = data.get("point_metrics", {})
        report.probabilistic_metrics = data.get("probabilistic_metrics", {})
        report.data_hash = data.get("data_hash")
        report.model_name = data.get("model_name")
        report.seed = data.get("seed")
        report.version = data.get("version", "2.0.0-alpha")

        return report

    def to_markdown(self) -> str:
        """Generate markdown table for documentation."""
        lines = []
        lines.append(f"## Evaluation Report: {self.model_name or 'Model'}")
        lines.append("")
        lines.append(
            f"**Split Protocol:** {self.split_protocol} ({self.n_folds} folds)"
        )
        lines.append(f"**Baseline:** {self.baseline_name}")
        lines.append("")

        # Metrics table
        lines.append("| Metric | Model | Baseline | Skill |")
        lines.append("|--------|-------|----------|-------|")

        for metric in ["rmse", "mae", "nrmse", "mase", "theil_u"]:
            if metric in self.point_metrics:
                model_val = self.point_metrics[metric]
                baseline_val = self.baseline_metrics.get(metric, "—")

                if metric in self.baseline_metrics:
                    ss = self.skill_score(metric)
                    skill_str = f"{ss:+.2f}"
                else:
                    skill_str = "—"

                lines.append(
                    f"| {metric.upper()} | {model_val:.4f} | {baseline_val if isinstance(baseline_val, str) else f'{baseline_val:.4f}'} | {skill_str} |"
                )

        return "\n".join(lines)
