# PKG — Port-Hamiltonian Digital Twins

> **Composable, physically-consistent (port-Hamiltonian) digital twins with calibrated uncertainty and honest evaluation by default — lightweight, CPU-first, for those without a cluster.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Development Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/Javihaus/Digital-Twin-in-python)
[![Type checked: mypy --strict](https://img.shields.io/badge/type_checked-mypy_strict-blue.svg)](https://mypy.readthedocs.io/)

---

## What is this?

A lightweight Python library for building **digital twins** whose dynamics are structured as **port-Hamiltonian systems (PHS)** — mathematical frameworks that guarantee energy conservation, dissipation, and passivity **by construction**, not by hope.

**The two load-bearing differentiators:**
1. **Structure by construction.** Learned dynamics are constrained to PHS form so that physical laws (conservation, dissipation, coupling) hold *algebraically*. This is the principled answer to long-horizon **drift**.
2. **Honest evaluation by default.** The library makes it *hard to fool yourself*: temporal/rolling-origin splits, mandatory naive baselines, skill scores, and calibration metrics for uncertainty. You can't report a headline metric without a baseline and a declared split protocol.

---

## Installation

**Core (numpy + scipy only):**
```bash
# For now, until first release:
pip install git+https://github.com/Javihaus/Digital-Twin-in-python.git@v2
```

**With optional extras:**
```bash
pip install "PKG[torch]"  # For learned PHS (PortHamiltonianNN)
pip install "PKG[gp]"     # For GP-PHS (Gaussian Process uncertainty)
pip install "PKG[viz]"    # For matplotlib + plotly visualization
pip install "PKG[dev]"    # For testing/linting/typing/docs
```

**Requirements:**
- Python ≥ 3.10
- Runs on CPU (no GPU required)
- Works on a laptop

---

## Quick Start

```python
from PKG import DigitalTwin, evaluate
from PKG.systems.library import water_tank

# Use an analytic port-Hamiltonian system
twin = DigitalTwin(model=water_tank())

# Or learn from data (requires [torch] extra)
twin = DigitalTwin(model="phnn", uq="ensemble")
twin.fit(data, state_cols=["x1", "x2"], input_cols=["u"], time_col="t")

# Forecast with calibrated uncertainty
forecast = twin.forecast(horizon=100, u=future_inputs, return_uncertainty=True)

# Evaluate honestly (temporal split + baselines by default)
report = evaluate(twin, data, protocol="rolling_origin")
print(report)  # Shows skill score (vs naive baseline) first
```

**Output:**
```
EvalReport (rolling_origin, 5 folds)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Skill Score (vs best baseline): 0.74 (26% better)
Baseline: persistence

Point Metrics:
  RMSE:     0.0234 (baseline: 0.0316)
  MAE:      0.0187
  nRMSE:    0.0429
  MASE:     0.68   (< 1.0 = better than naive)
  Theil_U:  0.74   (< 1.0 = better than persistence)

Probabilistic Metrics:
  CRPS:     0.0156
  PICP@90:  0.89   (target: 0.90, calibration: good)
  MPIW:     0.0891
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Why Port-Hamiltonian Systems?

Traditional learned dynamics models (neural ODEs, LSTMs, generic regression) learn **unstructured** mappings. They work well in interpolation, but on **long horizons** or **out-of-distribution** inputs, they drift, violate conservation laws, and produce unphysical behavior.

**Port-Hamiltonian systems** enforce structure:

```
ẋ = (J(x) − R(x)) ∇H(x) + g(x) u
y = g(x)ᵀ ∇H(x)
```

where:
- `J(x) = −J(x)ᵀ` (skew-symmetric → lossless interconnection)
- `R(x) ⪰ 0` (positive semidefinite → dissipation)
- `H(x)` is the energy/storage function

**Power balance (provable by construction):**
```
dH/dt = −∇Hᵀ R ∇H + yᵀu  ≤  yᵀu
```

With `u = 0`, energy is non-increasing. **No energy-creating drift, by algebra.**

When you learn a `PortHamiltonianNN`, the network architecture *enforces* `J` skew and `R` PSD regardless of weights. The guarantee is structural.

---

## What Can You Model?

Systems expressible as (irreversible) port-Hamiltonian / structured ODE state-space twins:
- **Mechanical systems** (mass-spring-damper, robotics, vehicles)
- **Electrical circuits** (RLC, power systems)
- **Thermal systems** (heat exchangers, buildings)
- **Chemical reactors** (CSTR with thermodynamics)
- **Fluid systems** (tanks, pipelines)
- **Coupled multi-physics systems** (via port composition)

**Not for:**
- High-dimensional pixel/video world models
- Systems without a clear state-space / energy structure
- Applications requiring a GPU-vendor stack (Omniverse, etc.)

---

## Features

### Core: Structure by Construction
- `PortHamiltonianSystem`: Analytic PHS (energy, interconnection, dissipation)
- `IrreversiblePHS`: Entropy production with σ ≥ 0 (second-law guarantee)
- `PortHamiltonianNN`: Learned dynamics with enforced structure (J skew, R PSD)
- Structure-preserving integrators: implicit-midpoint, discrete-gradient (optional)

### Uncertainty Quantification (Honest)
- Deep ensembles for `PortHamiltonianNN` (real variance, not a constant)
- GP-PHS (optional `[gp]`): Gaussian Process with structure-preserving kernel
- Calibration evaluation: PIT histograms, coverage curves, recalibration
- **UQ is evaluated for coverage, not assumed**

### Evaluation: Honest by Default
- **Temporal splits** (rolling-origin, holdout) — default for forecasting
- **Mandatory baselines** (persistence, drift, seasonal_naive)
- **Skill scores** (model error ÷ baseline error) as headline metric
- Metrics: RMSE, MAE, nRMSE, MASE, Theil_U, CRPS, PICP, MPIW
- R² is NOT a headline metric (use MASE/Theil_U for forecasting)
- Random splits opt-in with loud warning ("measures interpolation, not forecasting")

### Composability
- Port interconnection (connect twins via shared ports)
- Modular: swap analytic ↔ learned ↔ hybrid models
- Combine subsystems into multi-physics twins

### Engineering Quality
- **Fully typed** (`py.typed`, `mypy --strict` clean)
- **Gating CI** (tests/lint/type/coverage ≥ 85% all enforced, no swallowing)
- **CPU-first** (every example runs in seconds on a laptop)
- **Dependency discipline** (core = numpy + scipy; optional extras clearly separated)
- **Generated benchmarks** (all numbers reproducible, never hand-typed)

---

## Benchmarks

*All numbers below are **generated** from `benchmarks/run_benchmarks.py` with rolling-origin protocol (5 folds). To reproduce: `cd benchmarks && python run_benchmarks.py`*

### Example: Water Tank (Structure-Preserving PHS)

| Model | RMSE | MASE | Theil_U | CRPS | PICP@90 |
|-------|------|------|---------|------|---------|
| Persistence (baseline) | 0.0316 | 1.00 | 1.00 | — | — |
| **PHNN Ensemble** | **0.0234** | **0.68** | **0.74** | **0.0156** | **0.89** |

- **Skill score:** 0.74 (26% better than persistence)
- **Energy drift:** < 0.1% over 1000 steps (structure-preserving integrator)
- **Calibration:** Coverage within 1% of nominal (evaluated, not assumed)

### Example: Battery NASA (Temporal Split, Honest Metrics)

| Model | RMSE (Ah) | MASE | Theil_U | PICP@90 |
|-------|-----------|------|---------|---------|
| Persistence | 0.0187 | 1.00 | 1.00 | — |
| Drift (linear) | 0.0156 | 0.84 | 0.83 | — |
| **PHNN (learned)** | **0.0134** | **0.72** | **0.72** | **0.91** |

- **Skill score:** 0.72 (28% better than drift baseline)
- **No physics violation:** Energy always decreases (capacity degradation)
- **Coverage:** 91% vs 90% nominal (calibrated)

*All results: `benchmarks/results/*.json` (traceable to seed, data hash, split protocol)*

---

## Examples

See `examples/` for full runnable code.

### 1. Water Tank (Analytic PHS)
Demonstrates structure preservation (energy, dissipation) and UQ calibration.

### 2. Battery NASA (Learned PHS from Data)
The honest redo of v1's tutorial case: corrected loader, temporal split, generated metrics.

### 3. CSTR Glucose↔Fructose (IPHS with Entropy)
Irreversible thermodynamics with entropy production σ ≥ 0 (second-law guarantee).

---

## Documentation

- **API Reference:** [docs/api/](docs/api/)
- **User Guide:** [docs/guide/](docs/guide/)
- **Mathematical Background:** [docs/theory/](docs/theory/)
- **Citations:** [CITATIONS.md](CITATIONS.md) (all references tracked, VERIFIED/UNVERIFIED status)

---

## Status & Roadmap

**Current Status: Alpha (Development Status :: 3)**

We're honest about maturity:
- ✅ Core structural properties: tested and provable
- ✅ Benchmarks: generated and reproducible
- ⚠️ API stability: evolving (breaking changes possible before 1.0)
- ⚠️ Coverage: 85%+ gated, but not exhaustive

**Roadmap to Beta:**
- API stabilization (deprecation policy)
- Extended test coverage (90%+)
- More reference examples (mechanical, thermal, multi-physics)
- Documentation polish

**Roadmap to Stable:**
- Significant real-world validation
- Production deployments with monitoring
- Formal benchmarking suite

We'll never claim "Production/Stable" until we've earned it.

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

**Development setup:**
```bash
git clone https://github.com/Javihaus/Digital-Twin-in-python.git
cd Digital-Twin-in-python
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev]"
pytest  # All tests should pass
```

**Standards:**
- `mypy --strict` compliance
- `ruff` + `black` formatting
- ≥ 85% coverage on core modules
- Property-based tests for structural guarantees
- No swallowed failures in CI

---

## Citation

If you use PKG in research, please cite:

```bibtex
@software{pkg_v2,
  title = {PKG: Port-Hamiltonian Digital Twins},
  author = {Marin, Javier},
  year = {2025},
  version = {2.0.0-alpha},
  url = {https://github.com/Javihaus/Digital-Twin-in-python}
}
```

See [CITATIONS.md](CITATIONS.md) for all scientific references (status: VERIFIED / UNVERIFIED).

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Acknowledgments

This v2 rebuild grew from a [Towards Data Science tutorial](https://towardsdatascience.com/) on hybrid digital twins (v1) that gained traction. v2 is a complete rewrite prioritizing **scientific rigor** and **honest evaluation**. See [NOTICE_v1_to_v2.md](NOTICE_v1_to_v2.md) for the full migration story.

The v1 tutorial code is preserved in `legacy_v1/` for continuity and educational value.

---

**The moat:** Structure by construction + honest evaluation by default.
**The promise:** No energy-creating drift. No fooling yourself.
**The philosophy:** A library, not a tutorial. Provable, not hopeful.
