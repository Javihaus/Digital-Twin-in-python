"""Digital Twin: Main interface for PHS-based twins."""

from typing import Any

import numpy as np
import numpy.typing as npt

from PKG.integrate import integrate_with_inputs
from PKG.systems.phs import PortHamiltonianSystem

_VALID_UQ = ("none", "ensemble")


class DigitalTwin:
    """
    Digital Twin with port-Hamiltonian structure.

    Supports analytic PHS models with structure-preserving forecasting and,
    optionally, ensemble-based uncertainty quantification.

    Args:
        model: PortHamiltonianSystem, or a model-type string ('phnn', 'gp-phs')
            for learned models (require the corresponding extra).
        uq: Uncertainty quantification method ('none' or 'ensemble').
        ensemble: Optional :class:`PKG.uq.Ensemble` instance. Required when
            ``uq='ensemble'``.

    Example:
        >>> from PKG.systems import water_tank
        >>> twin = DigitalTwin(model=water_tank())
        >>> x0 = np.array([1.0])
        >>> u = np.zeros((100, 1))
        >>> t = np.linspace(0, 10, 100)
        >>> forecast = twin.forecast(x0, t, u)
        >>> forecast["x"].shape
        (100, 1)
    """

    def __init__(
        self,
        model: PortHamiltonianSystem | str = "phnn",
        uq: str = "none",
        ensemble: Any | None = None,
    ) -> None:
        if isinstance(model, PortHamiltonianSystem):
            self.model = model
            self.model_type = "analytic"
        elif isinstance(model, str):
            # Learned models live behind optional extras (Phase 4 modules).
            if model == "phnn":
                from PKG.learn.phnn import PortHamiltonianNN  # noqa: F401

                raise NotImplementedError(
                    "Pass a fitted PortHamiltonianNN (or a PortHamiltonianSystem) "
                    "as `model`; constructing a twin from the string 'phnn' is not "
                    "supported. Install the learning extra with: pip install PKG[torch]"
                )
            elif model == "gp-phs":
                raise NotImplementedError(
                    "Pass a fitted GP-PHS model as `model`. "
                    "Install the GP extra with: pip install PKG[gp]"
                )
            else:
                raise ValueError(f"Unknown model type: {model}")
        else:
            raise TypeError(
                f"Model must be PortHamiltonianSystem or str, got {type(model)}"
            )

        if uq not in _VALID_UQ:
            raise ValueError(f"uq must be one of {_VALID_UQ}, got {uq!r}")
        if uq == "ensemble" and ensemble is None:
            raise ValueError("uq='ensemble' requires an `ensemble` argument")

        self.uq = uq
        self.ensemble = ensemble
        self._is_fitted = False

    def fit(
        self,
        data: npt.NDArray[np.floating],
        state_cols: list[str] | None = None,
        input_cols: list[str] | None = None,
        time_col: str | None = None,
    ) -> dict[str, Any]:
        """
        Fit the twin to data (for learned models).

        For analytic models, this is a no-op.

        Returns:
            Fit metrics.
        """
        if self.model_type == "analytic":
            self._is_fitted = True
            return {"message": "Analytic model requires no fitting"}

        raise NotImplementedError("Learned model fitting lives on the learned model")

    def forecast(
        self,
        x0: npt.NDArray[np.floating],
        t: npt.NDArray[np.floating],
        u: npt.NDArray[np.floating],
        return_uncertainty: bool = False,
        level: float = 0.9,
        method: str = "implicit_midpoint",
    ) -> dict[str, npt.NDArray[np.floating]]:
        """
        Forecast future states given an initial condition and inputs.

        By default uses the structure-preserving implicit-midpoint integrator so
        the PHS energy balance is respected over long horizons.

        Args:
            x0: Initial state (n_states,).
            t: Time points (n_points,).
            u: Input trajectory (n_points, n_inputs).
            return_uncertainty: Whether to return uncertainty bounds. Requires
                ``uq != 'none'`` — otherwise this raises (no fake zero-width bands).
            level: Nominal coverage level for the prediction interval (0, 1).
            method: Integration method ('implicit_midpoint' by default, or any
                method accepted by the integrator, e.g. 'RK45').

        Returns:
            Dict with 'x' (n_points, n_states), 't', 'success', and — when
            ``return_uncertainty`` — 'lower' and 'upper' bounds plus 'std'.

        Raises:
            ValueError: If ``return_uncertainty=True`` while ``uq='none'``.
            RuntimeError: If integration fails.
        """
        if return_uncertainty and self.uq == "none":
            raise ValueError(
                "return_uncertainty=True requires a UQ method. Construct the twin "
                "with uq='ensemble' (and an `ensemble`). Returning zero-width "
                "uncertainty is intentionally disallowed."
            )

        def dynamics(
            t_val: float,
            x: npt.NDArray[np.floating],
            u_val: npt.NDArray[np.floating],
        ) -> npt.NDArray[np.floating]:
            return self.model.dynamics(x, u_val, t_val)

        result = integrate_with_inputs(dynamics, x0, t, u, method=method)

        if not result["success"]:
            raise RuntimeError(f"Integration failed: {result['message']}")

        forecast_result: dict[str, Any] = {
            "x": result["x"],
            "t": result["t"],
            "success": result["success"],
        }

        if return_uncertainty:
            # uq == 'ensemble' is the only valid path here (guarded above).
            assert self.ensemble is not None
            band = self.ensemble.forecast_interval(x0, t, u, level=level, method=method)
            forecast_result["mean"] = band["mean"]
            forecast_result["std"] = band["std"]
            forecast_result["lower"] = band["lower"]
            forecast_result["upper"] = band["upper"]

        return forecast_result

    def predict(
        self,
        X: npt.NDArray[np.floating],
        dt: float = 1.0,
    ) -> npt.NDArray[np.floating]:
        """
        One-step-ahead prediction for the evaluation framework.

        Treats each row of ``X`` as a state and advances it by one explicit-Euler
        step of size ``dt`` with zero input. ``dt`` is explicit (no hidden
        placeholder); pass the sampling interval of your series.

        Args:
            X: States (n_samples, n_states).
            dt: Time step for the one-step advance.

        Returns:
            Next-step states (n_samples, n_states).
        """
        X = np.atleast_2d(X)
        predictions = []
        for x in X:
            u = np.zeros(self.model.n_inputs)
            dx = self.model.dynamics(x, u)
            predictions.append(x + dx * dt)
        return np.array(predictions)

    def assimilate(
        self,
        x_prior: npt.NDArray[np.floating],
        observation: npt.NDArray[np.floating],
        obs_noise: float = 0.1,
        prior_noise: float = 1.0,
        H: npt.NDArray[np.floating] | None = None,
    ) -> dict[str, npt.NDArray[np.floating]]:
        """
        Linear-Gaussian (Kalman) measurement update of a state estimate.

        Implements the standard scalar/diagonal Kalman correction

            K = P Hᵀ (H P Hᵀ + R)⁻¹
            x⁺ = x⁻ + K (y − H x⁻)
            P⁺ = (I − K H) P

        with isotropic prior covariance ``P = prior_noise² I`` and measurement
        covariance ``R = obs_noise² I``. This is a real update — not a fixed
        weighted average.

        Args:
            x_prior: Prior state estimate (n_states,).
            observation: Measurement (n_obs,).
            obs_noise: Measurement noise std (> 0).
            prior_noise: Prior state noise std (> 0).
            H: Observation matrix (n_obs, n_states). Defaults to identity
                (direct observation of the full state).

        Returns:
            Dict with 'x' (posterior mean) and 'gain' (Kalman gain matrix).
        """
        if obs_noise <= 0 or prior_noise <= 0:
            raise ValueError("obs_noise and prior_noise must be positive")

        x_prior = np.asarray(x_prior, dtype=float)
        observation = np.asarray(observation, dtype=float)
        n = x_prior.size

        if H is None:
            H = np.eye(n)
        H = np.atleast_2d(H)

        P = (prior_noise**2) * np.eye(n)
        R = (obs_noise**2) * np.eye(H.shape[0])

        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x_post = x_prior + K @ (observation - H @ x_prior)
        return {"x": x_post, "gain": K}
