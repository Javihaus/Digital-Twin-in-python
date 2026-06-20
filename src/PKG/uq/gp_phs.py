"""Gaussian-Process surrogate with calibrated predictive variance (GP-PHS).

This is an *optional-extra* feature, not a placeholder: it has a real
implementation that runs when the ``[gp]`` extra (scikit-learn) is installed, and
raises a clear :class:`ImportError` otherwise.

A GP gives a principled predictive variance, which is the point of using it for
uncertainty quantification. To keep the port-Hamiltonian structure consistent, the GP
is used to learn the *residual* between a structured analytic prior (a PHS vector
field) and observed derivatives, so the mean stays close to a physically
consistent model while the GP supplies calibrated uncertainty on the correction.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt


def _require_sklearn() -> Any:
    try:
        import sklearn  # noqa: F401
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

        return GaussianProcessRegressor, ConstantKernel, RBF, WhiteKernel
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "GP-PHS requires the GP extra. Install with: pip install PKG[gp]"
        ) from exc


class GPPHS:
    """GP residual surrogate over a port-Hamiltonian prior.

    Args:
        prior_dynamics: Optional ``f0(x, u) -> dx`` analytic prior (e.g. a PHS
            vector field). If ``None``, the GP models the derivative directly.
        n_states: State dimension.
        n_inputs: Input dimension.

    Requires the ``[gp]`` extra (scikit-learn).
    """

    def __init__(
        self,
        n_states: int,
        n_inputs: int = 0,
        prior_dynamics: (
            Callable[
                [npt.NDArray[np.floating], npt.NDArray[np.floating]],
                npt.NDArray[np.floating],
            ]
            | None
        ) = None,
    ) -> None:
        GaussianProcessRegressor, ConstantKernel, RBF, WhiteKernel = _require_sklearn()
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.prior_dynamics = prior_dynamics
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(1e-3)
        # One GP per state dimension.
        self._gps = [
            GaussianProcessRegressor(
                kernel=kernel, normalize_y=True, n_restarts_optimizer=2
            )
            for _ in range(n_states)
        ]
        self._fitted = False

    def _features(
        self, X: npt.NDArray[np.floating], U: npt.NDArray[np.floating] | None
    ) -> npt.NDArray[np.floating]:
        X = np.atleast_2d(X)
        if U is None or self.n_inputs == 0:
            return X
        return np.hstack([X, np.atleast_2d(U)])

    def fit(
        self,
        X: npt.NDArray[np.floating],
        dXdt: npt.NDArray[np.floating],
        U: npt.NDArray[np.floating] | None = None,
    ) -> "GPPHS":
        """Fit the GP to (state[, input]) -> derivative residual data."""
        X = np.atleast_2d(X)
        dXdt = np.atleast_2d(dXdt)
        target = dXdt.copy()
        if self.prior_dynamics is not None:
            U_rows = U if U is not None else [None] * len(X)
            prior = np.array(
                [
                    self.prior_dynamics(x, np.zeros(self.n_inputs) if u is None else u)
                    for x, u in zip(X, U_rows)
                ]
            )
            target = dXdt - prior
        feats = self._features(X, U)
        for d in range(self.n_states):
            self._gps[d].fit(feats, target[:, d])
        self._fitted = True
        return self

    def predict(
        self,
        X: npt.NDArray[np.floating],
        U: npt.NDArray[np.floating] | None = None,
        return_std: bool = True,
    ) -> Any:
        """Predict derivatives with optional per-dimension predictive std."""
        if not self._fitted:
            raise RuntimeError("GPPHS must be fitted before predict()")
        X = np.atleast_2d(X)
        feats = self._features(X, U)
        means, stds = [], []
        for d in range(self.n_states):
            m, s = self._gps[d].predict(feats, return_std=True)
            means.append(m)
            stds.append(s)
        mean = np.stack(means, axis=1)
        std = np.stack(stds, axis=1)
        if self.prior_dynamics is not None:
            U_rows = U if U is not None else [None] * len(X)
            prior = np.array(
                [
                    self.prior_dynamics(x, np.zeros(self.n_inputs) if u is None else u)
                    for x, u in zip(X, U_rows)
                ]
            )
            mean = mean + prior
        if return_std:
            return mean, std
        return mean
