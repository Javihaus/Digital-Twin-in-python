# Water tank — the strong end (analytic port-Hamiltonian)

The simplest **strong-end** example: a draining water tank written as an analytic
port-Hamiltonian system. It shows the property that makes the strong end
worthwhile — **passivity by construction**: with no inflow, the stored energy is
monotonically non-increasing, so the forecast cannot invent energy no matter how
far ahead you integrate.

![Water tank dynamics](figures/water_tank_dynamics.png)

Left: the water height drains over time. Right: the energy `H(x)` decays
monotonically — the structure-preserving integrator respects the energy balance
`dH/dt = −∇Hᵀ R ∇H + yᵀu ≤ yᵀu`.

## What it demonstrates

- **Port-Hamiltonian structure** (`J` skew-symmetric, `R` positive semidefinite).
- **Passivity / no energy-creating drift** — verified numerically (energy is
  monotonically non-increasing with `u = 0`).
- **Power-balance identity** `dH/dt = dissipated + supplied` holds at sample points.
- **Structure-preserving integration** (implicit-midpoint).

A reproducible benchmark for this system (vs a persistence baseline, rolling-origin)
lives in [`../../benchmarks`](../../benchmarks): the analytic PHS model reaches
skill ≈ 0.94 (94% better than persistence).

## Run

```bash
pip install numpy scipy matplotlib seaborn
python water_tank_demo.py
```

Runs in under a second on CPU. The figure is written to `figures/`.
