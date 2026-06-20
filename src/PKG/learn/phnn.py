"""Port-Hamiltonian Neural Network (structure-constrained learning).

Requires the ``[torch]`` extra: ``pip install PKG[torch]``.

The structure is enforced *by construction*, so passivity holds regardless of the
learned weights:

    ẋ = (J_θ(x) − R_θ(x)) ∇H_θ(x) + g_θ(x) u

    - H_θ(x):  energy, an MLP plus an optional quadratic floor ``½‖x‖²`` so the
               energy is bounded below.
    - J_θ(x) = A_θ(x) − A_θ(x)ᵀ        (skew-symmetric by construction)
    - R_θ(x) = L_θ(x) L_θ(x)ᵀ          (PSD by construction via a Cholesky factor)
    - g_θ(x):  input map, an MLP.

With ``u = 0`` the energy is non-increasing: ``dH/dt = −∇Hᵀ R ∇H ≤ 0``.

This module imports torch lazily so the core package stays numpy/scipy-only.
"""

from typing import Any, Optional

import numpy as np
import numpy.typing as npt


def _require_torch() -> Any:
    try:
        import torch

        return torch
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "PortHamiltonianNN requires PyTorch. Install with: pip install PKG[torch]"
        ) from exc


class PortHamiltonianNN:
    """Neural PHS with skew ``J``, PSD ``R`` and a bounded-below energy.

    Args:
        n_states: State dimension.
        n_inputs: Input dimension (0 for autonomous systems).
        hidden: Hidden width of the internal MLPs.
        quadratic_floor: If True, add ``½‖x‖²`` to the learned energy.
        seed: Optional torch seed for reproducible initialisation.

    Requires the ``[torch]`` extra.
    """

    def __init__(
        self,
        n_states: int,
        n_inputs: int = 0,
        hidden: int = 32,
        quadratic_floor: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        torch = _require_torch()
        self._torch = torch
        if seed is not None:
            torch.manual_seed(seed)

        self.n_states = n_states
        self.n_inputs = n_inputs
        self.quadratic_floor = quadratic_floor

        nn = torch.nn

        self._H_net = nn.Sequential(
            nn.Linear(n_states, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        ).double()
        # A_θ entries -> skew J = A - A^T
        self._A_net = nn.Sequential(
            nn.Linear(n_states, hidden), nn.Tanh(),
            nn.Linear(hidden, n_states * n_states),
        ).double()
        # L_θ entries -> PSD R = L L^T
        self._L_net = nn.Sequential(
            nn.Linear(n_states, hidden), nn.Tanh(),
            nn.Linear(hidden, n_states * n_states),
        ).double()
        if n_inputs > 0:
            self._g_net = nn.Sequential(
                nn.Linear(n_states, hidden), nn.Tanh(),
                nn.Linear(hidden, n_states * n_inputs),
            ).double()
        else:
            self._g_net = None

        self._modules_list = [self._H_net, self._A_net, self._L_net]
        if self._g_net is not None:
            self._modules_list.append(self._g_net)

    # -- structure pieces -------------------------------------------------
    def _energy_tensor(self, x: Any) -> Any:
        h = self._H_net(x).squeeze(-1)
        if self.quadratic_floor:
            h = h + 0.5 * (x**2).sum(dim=-1)
        return h

    def energy(self, x: npt.NDArray[np.floating]) -> float:
        torch = self._torch
        xt = torch.as_tensor(np.atleast_2d(x), dtype=torch.float64)
        with torch.no_grad():
            return float(self._energy_tensor(xt)[0])

    def _grad_H(self, x: Any) -> Any:
        torch = self._torch
        x = x.clone().requires_grad_(True)
        h = self._energy_tensor(x).sum()
        (grad,) = torch.autograd.grad(h, x, create_graph=True)
        return grad

    def _J(self, x: Any) -> Any:
        n = self.n_states
        A = self._A_net(x).reshape(-1, n, n)
        return A - A.transpose(-1, -2)

    def _R(self, x: Any) -> Any:
        n = self.n_states
        L = self._L_net(x).reshape(-1, n, n)
        return L @ L.transpose(-1, -2)

    def _g(self, x: Any) -> Any:
        if self._g_net is None:
            return None
        return self._g_net(x).reshape(-1, self.n_states, self.n_inputs)

    def dynamics_tensor(self, x: Any, u: Any = None) -> Any:
        torch = self._torch
        grad_H = self._grad_H(x)
        J = self._J(x)
        R = self._R(x)
        JR = J - R
        dx = torch.einsum("bij,bj->bi", JR, grad_H)
        if self._g_net is not None and u is not None:
            g = self._g(x)
            dx = dx + torch.einsum("bij,bj->bi", g, u)
        return dx

    def dynamics(
        self,
        x: npt.NDArray[np.floating],
        u: Optional[npt.NDArray[np.floating]] = None,
        t: float = 0.0,
    ) -> npt.NDArray[np.floating]:
        """Numpy-facing vector field, compatible with the integrators."""
        torch = self._torch
        xt = torch.as_tensor(np.atleast_2d(x), dtype=torch.float64)
        ut = None
        if u is not None and self.n_inputs > 0:
            ut = torch.as_tensor(np.atleast_2d(u), dtype=torch.float64)
        # NOTE: do not wrap in torch.no_grad(): computing ∇H needs autograd.
        dx = self.dynamics_tensor(xt, ut)
        return dx.detach().numpy().reshape(np.shape(x))

    def parameters(self) -> list[Any]:
        params: list[Any] = []
        for m in self._modules_list:
            params.extend(list(m.parameters()))
        return params

    def check_structure(
        self, x: npt.NDArray[np.floating], tol: float = 1e-8
    ) -> dict[str, tuple[bool, float]]:
        """Verify skew(J) and PSD(R) at ``x`` for the learned model."""
        torch = self._torch
        xt = torch.as_tensor(np.atleast_2d(x), dtype=torch.float64)
        with torch.no_grad():
            J = self._J(xt)[0].numpy()
            R = self._R(xt)[0].numpy()
        skew_viol = float(np.abs(J + J.T).max())
        eig = float(np.linalg.eigvalsh(0.5 * (R + R.T)).min())
        return {
            "J_skew": (skew_viol <= tol, skew_viol),
            "R_psd": (eig >= -tol, eig),
        }

    def fit(
        self,
        X: npt.NDArray[np.floating],
        dXdt: npt.NDArray[np.floating],
        U: Optional[npt.NDArray[np.floating]] = None,
        epochs: int = 200,
        lr: float = 1e-2,
        energy_penalty: float = 0.0,
    ) -> dict[str, Any]:
        """Fit to derivative data with an optional passivity/energy penalty.

        Args:
            X: States ``(n, n_states)``.
            dXdt: Target derivatives ``(n, n_states)``.
            U: Inputs ``(n, n_inputs)`` (optional).
            epochs, lr: Optimisation settings (Adam).
            energy_penalty: Weight on a soft penalty discouraging energy growth
                with zero input (the structure already guarantees it; this only
                conditions optimisation).

        Returns:
            Dict with the final and per-epoch losses.
        """
        torch = self._torch
        from PKG.learn.losses import derivative_loss, passivity_penalty

        xt = torch.as_tensor(np.atleast_2d(X), dtype=torch.float64)
        yt = torch.as_tensor(np.atleast_2d(dXdt), dtype=torch.float64)
        ut = (
            torch.as_tensor(np.atleast_2d(U), dtype=torch.float64)
            if (U is not None and self.n_inputs > 0)
            else None
        )

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        history = []
        for _ in range(epochs):
            opt.zero_grad()
            pred = self.dynamics_tensor(xt, ut)
            loss = derivative_loss(pred, yt)
            if energy_penalty > 0.0:
                loss = loss + energy_penalty * passivity_penalty(self, xt)
            loss.backward()
            opt.step()
            history.append(float(loss.detach()))
        return {"final_loss": history[-1], "history": history}
