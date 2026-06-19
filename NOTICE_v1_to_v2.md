# Migration from v1 to v2 — The Story

## Background

This repository started as a **tutorial implementation** accompanying a [Towards Data Science article](https://towardsdatascience.com/) on hybrid digital twins for Li-ion battery modeling. The tutorial code (v1) was designed to illustrate concepts and gained significant traction (GitHub stars, SEO value).

## Why v2?

As the project gained attention and real-world interest, it became clear that **a tutorial is not a library**. v1 had several fundamental issues that prevented it from being used as a reliable, rigorous library for research or production use:

### Scientific Rigor Issues
- **No structural guarantees:** The hybrid physics+ML approach in v1 didn't enforce physical structure (conservation laws, passivity, stability)
- **No long-horizon drift prevention:** Without structural constraints, learned models could violate physics and drift unpredictably
- **Placeholder implementations:** Some features (e.g., uncertainty quantification) returned constants rather than real estimates
- **Unreproducible results:** Benchmark numbers were hand-typed and couldn't be regenerated

### Evaluation Issues
- **Wrong metrics for forecasting:** Used R² and random splits for time-series forecasting problems
- **No honest baselines:** Didn't compare against naive forecasts (persistence, drift)
- **Misleading evaluation:** Interpolation masquerading as forecasting capability

### Engineering Issues
- **Unsatisfiable dependencies:** Python 3.8+ with TensorFlow created dependency conflicts
- **Declared but unused dependencies:** Listed mlflow, hydra, pydantic but never imported them
- **Non-gating CI:** Tests "passed" even when they failed (continue-on-error)
- **Broken implementations:** Path bugs, serialization issues, ignored config keys

## What is v2?

**v2 is a complete, rigorous rebuild** that takes the core insight (physics+data-driven modeling) and makes it scientifically sound and honest by default:

### Core Innovation: Structure by Construction
v2 centers on **port-Hamiltonian systems (PHS)** — a mathematical framework that guarantees:
- Energy conservation and dissipation hold **by construction** (not by hope)
- Passivity is **provable** from the structure
- Long-horizon drift is **prevented structurally** (not through regularization)
- Learned dynamics respect physics **algebraically** (skew-symmetric interconnection, positive semidefinite dissipation)

### Irreversible Thermodynamics Extension
The **irreversible PHS (IPHS)** extension adds entropy production with a **second-law guarantee** (σ ≥ 0), providing the principled answer to dissipative system modeling.

### Honest Evaluation by Default
v2 makes self-deception hard:
- **Temporal splits** (not random) for forecasting
- **Mandatory naive baselines** (persistence, drift, seasonal)
- **Skill scores** (model error ÷ baseline error) as headline metrics
- **Calibrated uncertainty** (coverage evaluated, not assumed)
- **Generated benchmarks** (all numbers traceable and reproducible)

### Lightweight, CPU-First
- No TensorFlow dependency hell
- Optional extras for PyTorch (CPU), GP, viz
- Every example runs in **seconds on a laptop**
- No GPU, no cluster, no vendor stack required

## What Happened to v1?

v1 code has been:
1. **Corrected** (fixed data loader bugs, serialization issues, config parsing)
2. **Moved to `legacy_v1/`** with clear labeling as the original tutorial
3. **Kept runnable** for continuity and to honor the tutorial's educational value
4. **Cleaned up** (removed unreproducible benchmarks, fixed misattributions)

The v1 tutorial remains a valuable educational artifact and the origin story of this project. We're not erasing history — we're growing up.

## Migration Path

If you were using v1 code:
- **For learning/tutorials:** `legacy_v1/` is still there and runnable
- **For research/production:** Migrate to v2's port-Hamiltonian API
- **For battery modeling specifically:** See `examples/battery_nasa/` for the honest, structure-preserving version

Deprecation shims are provided for common v1 import paths (they emit warnings pointing to v2 equivalents).

## Status

v2 launches as **Alpha** (Development Status :: 3). We're honest about maturity:
- Core structural properties: **tested and provable**
- API stability: **evolving** (breaking changes possible before 1.0)
- Coverage: **gated at 85%+** in CI
- Benchmarks: **generated and reproducible**

We'll move to Beta when APIs stabilize, and to Stable only after significant real-world validation.

## Acknowledgments

The v1 tutorial succeeded because it resonated with practitioners who needed practical guidance. v2 exists because those same practitioners needed rigor. Thank you to everyone who starred, forked, cited, and provided feedback — you made this rebuild possible and necessary.

---

**The moat:** Structure by construction + honest evaluation by default.

This is the identity. Everything else is detail.
