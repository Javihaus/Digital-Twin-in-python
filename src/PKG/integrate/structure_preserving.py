"""Structure-preserving time integration for port-Hamiltonian systems.

Standard explicit solvers (RK45, etc.) do not respect the energy balance of a
port-Hamiltonian system: over long horizons they can *inject* energy and push the
state into unphysical regions (e.g. negative water height), which shows up as the
Hamiltonian ticking back up near equilibrium.

The implicit-midpoint rule is a symmetric, symplectic-class integrator. For a PHS
with a **quadratic** energy ``H(x) = 1/2 xᵀ Q x`` it satisfies the *discrete*
power balance exactly:

    H(x_{n+1}) − H(x_n) = ∇H(x_mid)·(x_{n+1} − x_n)
                        = Δt · ∇H(x_mid)·[(J − R)∇H(x_mid) + g u_mid]
                        = −Δt · ∇H(x_mid)ᵀ R(x_mid) ∇H(x_mid) + Δt · yᵀ u_mid
                        ≤ Δt · yᵀ u_mid

so with ``u = 0`` energy is non-increasing **to machine precision**, regardless of
how nonlinear ``R(x)`` is, as long as ``R(x_mid) ⪰ 0``. That is the structural
guarantee RK45 cannot give.

For non-quadratic ``H`` the equality above holds only to second order; use a
discrete-gradient method if exact decay is required there.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import fsolve


def implicit_midpoint(
    dynamics: Callable[
        [float, npt.NDArray[np.floating], npt.NDArray[np.floating]],
        npt.NDArray[np.floating],
    ],
    x0: npt.NDArray[np.floating],
    t_eval: npt.NDArray[np.floating],
    u: npt.NDArray[np.floating],
    newton_tol: float = 1e-12,
    max_iter: int = 100,
) -> dict[str, Any]:
    """Integrate ``dx/dt = dynamics(t, x, u(t))`` with the implicit-midpoint rule.

    Each step solves the implicit equation

        x_{n+1} = x_n + Δt · dynamics(t_mid, (x_n + x_{n+1}) / 2, u_mid)

    for ``x_{n+1}`` (``t_mid`` and ``u_mid`` are the step midpoints). Inputs are
    interpolated linearly between the rows of ``u``.

    Args:
        dynamics: Vector field with signature ``(t, x, u) -> dx``.
        x0: Initial state ``(n_states,)``.
        t_eval: Strictly increasing time points ``(n_points,)``.
        u: Input trajectory ``(n_points, n_inputs)`` (1-D is reshaped to a column).
        newton_tol: Residual tolerance for the per-step implicit solve.
        max_iter: Maximum iterations for the per-step solve.

    Returns:
        Dict with ``'t'``, ``'x'`` ``(n_points, n_states)``, ``'success'`` and
        ``'message'``.

    Example:
        >>> import numpy as np
        >>> from PKG.systems import water_tank
        >>> tank = water_tank()
        >>> t = np.linspace(0, 10, 100)
        >>> u = np.zeros((100, 1))
        >>> res = implicit_midpoint(
        ...     lambda tv, x, uv: tank.dynamics(x, uv), np.array([2.0]), t, u
        ... )
        >>> res["success"]
        True
    """
    t_eval = np.asarray(t_eval, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    u = np.asarray(u, dtype=float)
    if u.ndim == 1:
        u = u.reshape(-1, 1)
    if t_eval.ndim != 1 or t_eval.size < 2:
        raise ValueError("t_eval must be a 1-D array with at least two points")
    if np.any(np.diff(t_eval) <= 0):
        raise ValueError("t_eval must be strictly increasing")

    n_points = t_eval.size
    n_states = x0.size
    xs = np.empty((n_points, n_states), dtype=float)
    xs[0] = x0

    success = True
    message = "Integration successful"

    for n in range(n_points - 1):
        dt = float(t_eval[n + 1] - t_eval[n])
        t_mid = 0.5 * (t_eval[n] + t_eval[n + 1])
        u_mid = 0.5 * (u[n] + u[n + 1])
        x_n = xs[n]

        def residual(x_next: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
            x_mid = 0.5 * (x_n + x_next)
            return np.asarray(
                x_next - x_n - dt * np.asarray(dynamics(t_mid, x_mid, u_mid)),
                dtype=float,
            )

        # Explicit-Euler predictor as a warm start, then solve the implicit step.
        x_guess = x_n + dt * np.asarray(dynamics(t_eval[n], x_n, u[n]))
        x_next, info, ier, msg = fsolve(
            residual,
            x_guess,
            full_output=True,
            xtol=newton_tol,
            maxfev=max_iter * (n_states + 1),
        )
        # fsolve may flag non-convergence near non-smooth points (e.g. sqrt(h)
        # at h->0) even when the residual is effectively zero. Accept on a small
        # residual norm; only fail if the step is genuinely unsolved.
        resid_norm = float(np.linalg.norm(residual(x_next)))
        if ier != 1 and resid_norm > 1e-8:
            success = False
            message = (
                f"Implicit solve failed at step {n} (t={t_eval[n]:.4g}, "
                f"residual={resid_norm:.2e}): {msg}"
            )
            xs[n + 1] = x_next
            xs[n + 2 :] = x_next
            break
        xs[n + 1] = x_next

    return {"t": t_eval, "x": xs, "success": success, "message": message}


def integrate_phs(
    phs: Any,
    x0: npt.NDArray[np.floating],
    t_eval: npt.NDArray[np.floating],
    u: npt.NDArray[np.floating] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Structure-preserving integration of a PHS / IPHS object.

    Convenience wrapper around :func:`implicit_midpoint` that uses
    ``phs.dynamics(x, u, t)``.

    Args:
        phs: Object exposing ``dynamics(x, u, t)`` and ``n_inputs``.
        x0: Initial state.
        t_eval: Time points.
        u: Input trajectory ``(n_points, n_inputs)``; defaults to zeros.
        **kwargs: Forwarded to :func:`implicit_midpoint`.

    Returns:
        Same dict as :func:`implicit_midpoint`.
    """
    t_eval = np.asarray(t_eval, dtype=float)
    if u is None:
        u = np.zeros((t_eval.size, getattr(phs, "n_inputs", 1)))

    def dynamics(
        t: float, x: npt.NDArray[np.floating], u_val: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        return np.asarray(phs.dynamics(x, u_val, t), dtype=float)

    return implicit_midpoint(dynamics, x0, t_eval, u, **kwargs)
