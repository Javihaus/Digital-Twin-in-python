"""Ensemble-based uncertainty quantification.

An ``Ensemble`` aggregates several forecasting members into a predictive
distribution: a mean trajectory, a per-step standard deviation, and quantile
prediction intervals. The spread across members *is* the uncertainty, so the
members must genuinely differ (different fitted weights, perturbed parameters,
bootstrap resamples, MC-dropout samples, ...). An ensemble of identical
deterministic models has zero spread by construction — that is correct, not a bug.

This module is core (numpy only). Members only need a
``forecast(x0, t, u, method=...) -> {"x": array}`` interface, which both
:class:`PKG.twin.DigitalTwin` and learned models satisfy.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt


class Ensemble:
    """A collection of forecasting members with quantile-based intervals.

    Args:
        members: Objects exposing ``forecast(x0, t, u, method=...) -> {"x": ...}``.

    Example:
        >>> import numpy as np
        >>> from PKG.systems import water_tank
        >>> from PKG.twin import DigitalTwin
        >>> from PKG.uq import Ensemble
        >>> members = [DigitalTwin(water_tank(a=a)) for a in (0.09, 0.10, 0.11)]
        >>> ens = Ensemble(members)
        >>> t = np.linspace(0, 5, 50)
        >>> band = ens.forecast_interval(np.array([2.0]), t, np.zeros((50, 1)))
        >>> band["lower"].shape
        (50, 1)
    """

    def __init__(self, members: Sequence[Any]) -> None:
        members = list(members)
        if len(members) < 2:
            raise ValueError("An ensemble needs at least two members")
        self.members = members

    @property
    def n_members(self) -> int:
        return len(self.members)

    def forecast_trajectories(
        self,
        x0: npt.NDArray[np.floating],
        t: npt.NDArray[np.floating],
        u: npt.NDArray[np.floating],
        method: str = "implicit_midpoint",
    ) -> npt.NDArray[np.floating]:
        """Return all member trajectories, shape ``(n_members, n_steps, n_states)``."""
        trajs = []
        for m in self.members:
            out = m.forecast(x0, t, u, method=method)
            trajs.append(np.asarray(out["x"]))
        arr = np.stack(trajs, axis=0)
        return arr

    def forecast_interval(
        self,
        x0: npt.NDArray[np.floating],
        t: npt.NDArray[np.floating],
        u: npt.NDArray[np.floating],
        level: float = 0.9,
        method: str = "implicit_midpoint",
    ) -> dict[str, npt.NDArray[np.floating]]:
        """Mean, std and a central ``level`` prediction interval per step/state.

        Args:
            level: Central coverage level in (0, 1); e.g. 0.9 -> 5th/95th pctl.

        Returns:
            Dict with 'mean', 'std', 'lower', 'upper', each ``(n_steps, n_states)``,
            and 'trajectories' ``(n_members, n_steps, n_states)``.
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}")
        trajs = self.forecast_trajectories(x0, t, u, method=method)
        mean = trajs.mean(axis=0)
        std = trajs.std(axis=0, ddof=1)
        alpha = 1.0 - level
        lower = np.quantile(trajs, alpha / 2.0, axis=0)
        upper = np.quantile(trajs, 1.0 - alpha / 2.0, axis=0)
        return {
            "mean": mean,
            "std": std,
            "lower": lower,
            "upper": upper,
            "trajectories": trajs,
        }

    def ensemble_matrix(
        self,
        x0: npt.NDArray[np.floating],
        t: npt.NDArray[np.floating],
        u: npt.NDArray[np.floating],
        state: int = 0,
        method: str = "implicit_midpoint",
    ) -> npt.NDArray[np.floating]:
        """Member forecasts for one state dim as ``(n_steps, n_members)``.

        This is exactly the shape expected by
        :func:`PKG.evaluation.metrics.crps`.
        """
        trajs = self.forecast_trajectories(x0, t, u, method=method)  # (M, T, S)
        return np.asarray(trajs[:, :, state].T, dtype=float)  # (T, M)
