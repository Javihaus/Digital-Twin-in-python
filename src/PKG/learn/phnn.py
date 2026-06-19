"""Port-Hamiltonian Neural Network (structure-constrained learning)."""

# Placeholder for Phase 4 full implementation
# Requires torch extra: pip install PKG[torch]

class PortHamiltonianNN:
    """
    Neural network constrained to port-Hamiltonian structure.

    Structure enforced by construction:
    - H_θ(x): Energy function (MLP with optional quadratic floor)
    - J_θ(x) = A_θ - A_θ^T  (skew by construction)
    - R_θ(x) = L_θ @ L_θ^T  (PSD by construction via Cholesky)
    - g_θ(x): Input map (MLP)

    This ensures passivity and structure regardless of learned weights.

    NOTE: Full implementation requires torch. This is a placeholder.
    """

    def __init__(self, n_states: int, n_inputs: int) -> None:
        raise NotImplementedError(
            "PortHamiltonianNN requires torch. "
            "Install with: pip install PKG[torch]\n"
            "Full implementation coming in Phase 4 completion."
        )
