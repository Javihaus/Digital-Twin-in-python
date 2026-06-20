"""Time integration solvers for dynamical systems."""

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp


def integrate(
    dynamics: Callable[[float, npt.NDArray[np.floating]], npt.NDArray[np.floating]],
    x0: npt.NDArray[np.floating],
    t_span: tuple[float, float],
    t_eval: npt.NDArray[np.floating] | None = None,
    method: str = "RK45",
    **kwargs: Any,
) -> dict[str, npt.NDArray[np.floating]]:
    """
    Integrate ODE system: dx/dt = dynamics(t, x).

    Wrapper around scipy.integrate.solve_ivp with a clean interface.

    Args:
        dynamics: Function with signature (t, x) -> dx/dt
        x0: Initial state (n_states,)
        t_span: Integration interval (t_start, t_end)
        t_eval: Time points at which to store solution (optional)
        method: Integration method (RK45, RK23, DOP853, etc.)
        **kwargs: Additional arguments passed to solve_ivp

    Returns:
        Dictionary with:
            - 't': Time points (n_points,)
            - 'x': State trajectory (n_points, n_states)
            - 'success': Whether integration succeeded

    Example:
        >>> dynamics = lambda t, x: np.array([-x[0]])  # dx/dt = -x
        >>> x0 = np.array([1.0])
        >>> result = integrate(dynamics, x0, (0, 1), t_eval=np.linspace(0, 1, 10))
        >>> result['success']
        True
    """
    sol = solve_ivp(
        dynamics, t_span, x0, method=method, t_eval=t_eval, dense_output=True, **kwargs
    )

    return {
        "t": sol.t,
        "x": sol.y.T,  # Transpose to (n_points, n_states)
        "success": sol.success,
        "message": sol.message,
    }


def integrate_with_inputs(
    dynamics: Callable[
        [float, npt.NDArray[np.floating], npt.NDArray[np.floating]],
        npt.NDArray[np.floating],
    ],
    x0: npt.NDArray[np.floating],
    t_eval: npt.NDArray[np.floating],
    u: npt.NDArray[np.floating],
    method: str = "RK45",
    **kwargs: Any,
) -> dict[str, npt.NDArray[np.floating]]:
    """
    Integrate ODE with time-varying inputs: dx/dt = dynamics(t, x, u(t)).

    Args:
        dynamics: Function with signature (t, x, u) -> dx/dt
        x0: Initial state (n_states,)
        t_eval: Time points at which to evaluate (n_points,)
        u: Input trajectory (n_points, n_inputs)
        method: Integration method
        **kwargs: Additional arguments for solve_ivp

    Returns:
        Dictionary with 't', 'x', 'success'

    Example:
        >>> dynamics = lambda t, x, u: u  # dx/dt = u
        >>> x0 = np.array([0.0])
        >>> t = np.linspace(0, 1, 10)
        >>> u = np.ones((10, 1))
        >>> result = integrate_with_inputs(dynamics, x0, t, u)
        >>> result['success']
        True
    """
    if u.ndim == 1:
        u = u.reshape(-1, 1)

    # Structure-preserving path: dispatch to the implicit-midpoint integrator,
    # which respects the PHS energy balance (no spurious energy injection).
    if method in ("implicit_midpoint", "implicit-midpoint"):
        from PKG.integrate.structure_preserving import implicit_midpoint

        return implicit_midpoint(dynamics, x0, t_eval, u, **kwargs)

    # Interpolate inputs for arbitrary time points
    from scipy.interpolate import interp1d

    # Create interpolator for each input dimension
    u_interp = interp1d(
        t_eval, u, axis=0, kind="linear", bounds_error=False, fill_value="extrapolate"
    )

    # Wrapper that looks up input at time t
    def dynamics_wrapper(
        t: float, x: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        u_t = u_interp(t)
        return dynamics(t, x, u_t)

    t_span = (t_eval[0], t_eval[-1])
    return integrate(
        dynamics_wrapper, x0, t_span, t_eval=t_eval, method=method, **kwargs
    )
