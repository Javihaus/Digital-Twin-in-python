# STATUS

Single source of truth for project state. Mirrors the phase markers in
`src/PKG/__init__.py`. If this file and the code disagree, the code wins and this
file is the bug. No "all complete" claims live anywhere else.

_Last updated: 2026-06-19._

## Phases

| Phase | Scope | Status | Notes |
|------|-------|--------|-------|
| 0 | Packaging, seeding, linalg utils | **done** | numpy+scipy core; extras `[torch]`, `[gp]`, `[viz]`, `[dev]` |
| 1 | PHS core (`systems/phs.py`, library) | **done** | skew-J / PSD-R checks, power balance |
| 2 | Honest evaluation harness | **done** | RMSE/MAE/nRMSE/MASE/Theil's U/CRPS/PICP/MPIW/skill; temporal split default; R² not in headline |
| 3 | `DigitalTwin` | **done** | structure-preserving forecast; real Kalman `assimilate`; explicit `dt` in `predict` |
| 4 | Learned PHS + UQ | **done (core)** | `uq/ensemble.py`, `uq/calibration.py` implemented + tested. `learn/phnn.py` (PHNN) requires `[torch]`; `uq/gp_phs.py` (GP-PHS) requires `[gp]` |
| 5 | IPHS (irreversible, entropy production) | **done** | second-law guarantee **enforced** (L validated PSD), not just documented |
| 6 | Composition + GP-PHS | **in-progress** | GP-PHS done (`[gp]`); `twin/compose.py` still a stub |

## Tests

- Default suite: **79 passed, 1 skipped** (`pip install -e . && pytest`).
- Skipped tests are optional-extra tests that skip cleanly when the extra is
  absent:
  - `tests/learn/test_phnn.py` — requires `[torch]`.
  - `tests/uq/test_gp_phs.py` — requires `[gp]` (sklearn); **passes when installed.**

## Known limitations / honest caveats

- **PHNN (`learn/phnn.py`) tests could not be executed in the build sandbox**
  (torch is not installable there). The implementation is complete and reviewed,
  but run `pip install PKG[torch] && pytest tests/learn` locally to confirm the
  structural guarantees (skew J, PSD R, energy decay) on your machine before
  relying on it.
- `twin/compose.py` is a Phase-6 stub (`NotImplementedError`), not a silent fake.
- No benchmark table is published yet; any future numbers must come from a seeded
  script under temporal split (see `docs/claude_code_brief_addendum.md`, RULE 1).

## What is intentionally **not** done

- Composition of subsystems via shared ports (Phase 6).
- A published PyPI release.
