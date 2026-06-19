# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0-alpha] - TBD

### Added
- **Complete v2 rebuild:** Port-Hamiltonian systems (PHS) framework as the scientific core
- **Structure by construction:** Skew-symmetric J, PSD R, guaranteed power balance and passivity
- **Irreversible PHS (IPHS):** Entropy production with σ ≥ 0 guarantee (second law)
- **Learned PHS:** `PortHamiltonianNN` with structurally constrained neural networks
- **Honest evaluation harness:** Temporal splits, mandatory baselines, skill scores, calibration metrics
- **Uncertainty quantification:** Deep ensembles and GP-PHS (optional) with evaluated coverage
- **Structure-preserving integrators:** Implicit-midpoint and discrete-gradient methods (optional)
- **Composable twins:** Port interconnection for multi-physics systems
- **Generated benchmarks:** All numbers reproducible and traceable
- **Property-based testing:** Hypothesis-driven tests for structural guarantees
- **Gating CI:** Tests, mypy --strict, ruff, coverage ≥ 85% all enforced (no swallowing)
- **Reference examples:**
  - Battery NASA (corrected, temporal split, honest metrics)
  - Water tank PHS (structure preservation + UQ calibration demo)
  - CSTR glucose↔fructose (IPHS with entropy production)
- **CITATIONS.md:** All references tracked with VERIFIED/UNVERIFIED status

### Changed
- **Breaking:** Complete API rewrite (v1 compatibility via deprecation shims only)
- **Dependency discipline:** Core = numpy + scipy; torch/gp/viz are optional extras
- **Python ≥ 3.10 required** (dropped 3.8, 3.9 to avoid unsatisfiable dependency matrix)
- **No TensorFlow:** Replaced with optional CPU PyTorch for learned models
- **Development Status:** Alpha (honest about maturity)
- **Metrics focus:** Skill scores (vs baselines) and MASE/Theil_U replace R² as headline
- **Split protocol:** Temporal/rolling-origin default; random split opt-in with loud warning

### Fixed (from v1)
- **Data loader double-prefix bug:** `load_nasa_dataset` path construction corrected
- **Placeholder uncertainty:** Removed constant return value; now real ensembles or GP
- **Broken model serialization:** Fixed joblib pickling of Keras models
- **Ignored config keys:** physics_model.* YAML keys now actually used
- **Unreproducible benchmarks:** All numbers generated from code, not hand-typed
- **CI false positives:** Removed continue-on-error; gates actually fail on errors
- **Unused dependencies:** Removed mlflow, hydra, pydantic (not imported in v1)
- **Xu et al. misattribution:** Removed formula attribution not present in source

### Removed
- TensorFlow dependency
- Production/Stable status claim (reverting to honest Alpha)
- Hand-typed benchmark tables
- Random-split-as-forecasting examples
- Mock-only tests

### Deprecated
- v1 import paths (emit warnings, redirect to v2 equivalents)

### Migrated to legacy_v1/
- Original tutorial code (corrected and labeled)
- Hybrid physics+ML battery modeling (non-PHS version)
- Link to Towards Data Science article preserved

---

## [1.0.0] - 2024-08 (Legacy)

Original tutorial implementation for hybrid digital twin battery modeling. See `legacy_v1/` for preserved code and `NOTICE_v1_to_v2.md` for full migration story.

### Original Features (v1)
- Hybrid physics-based + ML approach for Li-ion battery degradation
- NASA battery dataset examples
- Neural network correction of physics model
- Basic uncertainty estimation
- Visualization and plotting utilities

### Known Issues (v1, fixed in v2)
- Non-structural learning (no passivity/conservation guarantees)
- Unreproducible benchmark numbers
- Dependency conflicts (Python 3.8+ with TensorFlow)
- Data loader path bugs
- Placeholder implementations
- Non-gating CI
- Misleading forecasting metrics
