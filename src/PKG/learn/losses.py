"""Training losses for learned port-Hamiltonian models.

Requires the ``[torch]`` extra. These operate on torch tensors / a
:class:`PKG.learn.phnn.PortHamiltonianNN` and import torch lazily.
"""

from typing import Any


def derivative_loss(pred: Any, target: Any) -> Any:
    """Mean-squared error between predicted and target derivatives."""
    return ((pred - target) ** 2).mean()


def passivity_penalty(model: Any, x: Any) -> Any:
    """Soft penalty on energy growth with zero input.

    The PHS structure already guarantees ``∇Hᵀ(J − R)∇H = −∇Hᵀ R ∇H ≤ 0``; this
    term only conditions optimisation by penalising any positive part of the
    autonomous energy rate ``dH/dt = ∇Hᵀ ẋ`` (which should be ≤ 0).

    Args:
        model: A PortHamiltonianNN.
        x: States tensor ``(n, n_states)``.

    Returns:
        Scalar tensor penalty (0 when the structure is respected, as it must be).
    """
    torch = model._torch
    grad_H = model._grad_H(x)
    dx = model.dynamics_tensor(x, None)
    dH_dt = (grad_H * dx).sum(dim=-1)
    return torch.relu(dH_dt).mean()
