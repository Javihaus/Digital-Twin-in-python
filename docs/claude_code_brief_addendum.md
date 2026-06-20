# Claude Code Brief — Addendum (Correction Pass)

**Status of the repo as independently verified (2026-06-19):**
Phases 0–3 substantially done. Phases 4–6 **not** done. The v2 test suite is
green (42 tests) but only because the untested modules don't have tests yet —
`integrate/`, `twin/`, `data/`, `learn/`, `uq/` have **zero** test functions.
A document (`ALL_PHASES_COMPLETE.md`) claims completion that the code contradicts.

This addendum supersedes any "complete" claim in earlier docs. Execute the tasks
below **in order**. Each task has a **gate**: do not proceed to the next until the
gate passes.

---

## RULE 0 — No "all done" documents. Ever.

This is non-negotiable and the reason this addendum exists.

- **Do not** create files like `ALL_PHASES_COMPLETE.md`, `*_SUMMARY.md`,
  `CI_FIXED.md`, `RELEASE_CHECKLIST.md` that assert work is finished.
- The **only** status artifact allowed is a single `STATUS.md` that states, per
  phase, one of: `done` / `in-progress` / `not-started`, plus the current test
  count and any red/skipped tests. It must be falsifiable and match the code.
- A phase is "done" **only** when: (a) its modules contain real implementations
  (no `NotImplementedError` in the happy path, no placeholder constants returned
  as results), **and** (b) it has tests that exercise the behaviour, **and**
  (c) those tests are in the default suite and pass.
- If you cannot finish something, leave a real `NotImplementedError` with a clear
  message and mark it `not-started`/`in-progress` in `STATUS.md`. Do **not**
  return fabricated values (e.g. zero uncertainty) to make a code path "work".

Any deliverable that violates RULE 0 is rejected.

---

## RULE 1 — No unverifiable numbers, no unverified citations.

- No hand-typed metric tables. Every number in a README/doc must be produced by a
  committed, seeded script (`benchmarks/`) under **temporal** split.
- Every scientific citation must be checked against the actual source before it
  appears outside a doc marked `UNVERIFIED`. (The v1 repo attributed a formula to
  Xu et al. 2016 that is not theirs — do not repeat that pattern. The
  "Stable Port-Hamiltonian Neural Networks (2026)" reference stays `UNVERIFIED`
  until a real citation is confirmed.)

---

## Task A — Kill the false docs, write the STATUS. (blocking)

1. Delete: `ALL_PHASES_COMPLETE.md`, `CI_FIXED.md`, `CI_FIXES_NEEDED.md`,
   `V2_ACTIVATION_SUMMARY.md`, `RELEASE_CHECKLIST.md`, `NOTICE_v1_to_v2.md`
   (regenerate NOTICE only if it states facts that are true at release time).
2. Write `STATUS.md` with a per-phase table (done/in-progress/not-started),
   current test count, and a list of any skipped tests with the reason
   (e.g. "requires `[torch]` extra").
3. Make `PKG/__init__.py` phase comments the single source of truth that
   `STATUS.md` mirrors.

**Gate A:** `grep -ri "all phases complete\|production/stable\|95% coverage" .`
returns nothing outside `legacy_v1/`. `STATUS.md` exists and matches
`PKG/__init__.py`.

---

## Task B — Finish the v1 → legacy migration. (blocking)

1. Move `src/hybrid_digital_twin/` → `legacy_v1/hybrid_digital_twin/`.
2. Move its tests (`tests/unit/test_digital_twin.py`, `tests/unit/test_physics_model.py`,
   `tests/integration/test_end_to_end.py`, any other v1 test) to `legacy_v1/tests/`.
3. Exclude `legacy_v1/` from the default `pytest` run, from coverage, ruff, black,
   and mypy (the configs already reference a `legacy_v1/` that didn't exist —
   make it exist).
4. Move v1 packaging cruft to `legacy_v1/` or delete: `pyproject_v1_backup.toml`,
   `README_v1.md`, `.gitignore_v2`, v1 `Dockerfile`, v1 `config/`.
5. Add a short `legacy_v1/README.md`: "This is the original tutorial. Known issues
   documented; kept for provenance. Use v2 (PKG)."

**Gate B:** `pytest` (default invocation, no ignores) collects **0** v1 tests and
**0** import errors. `pip install -e .` then `pytest` is green with no `joblib`/
`keras` errors.

---

## Task C — Phase 4: Learned PHS + Uncertainty Quantification. (the real work)

UQ is half the pitch ("calibrated uncertainty"). Today `uq/` is empty and
`twin.forecast(return_uncertainty=True)` returns **zeros**. Fix both.

### C1. `uq/ensemble.py` — core (numpy only)
- An `Ensemble` that holds N predictors (model factory + per-member fit, or a
  list of pre-fit models) and produces, for a forecast horizon: member
  trajectories `(n_members, n_steps, n_states)`, the ensemble **mean**, **std**,
  and **quantile** prediction intervals at a nominal level.
- Output shape for the probabilistic path must be compatible with the existing
  `evaluation.metrics.crps(y_true, ensemble_forecasts)` (i.e. `(n, n_members)`).

### C2. `uq/calibration.py` — core (numpy only)
- `coverage_curve`, `pit_values` (probability integral transform),
  `interval_score`, `expected_calibration_error` (regression),
  `recalibrate` (simple monotonic recalibration map).
- Reuse `metrics.picp` / `metrics.mpiw` / `metrics.crps`; do not reimplement them.

### C3. `learn/phnn.py` — requires `[torch]` extra
- Real `PortHamiltonianNN`: `H_θ(x)` MLP (optional quadratic floor),
  `J_θ = A_θ − A_θᵀ` (skew by construction), `R_θ = L_θ L_θᵀ` (PSD by
  construction via Cholesky param), `g_θ(x)` MLP. `fit` (trajectory +
  derivative loss, optional energy/passivity penalty from `learn/losses.py`)
  and a `dynamics`/`forecast` path.
- If torch is absent, the constructor raises `ImportError` with the install hint
  (this is a real optional dependency, **not** a placeholder).

### C4. `uq/gp_phs.py` — requires `[gp]` extra
- GP surrogate with calibrated predictive variance (sklearn
  `GaussianProcessRegressor` is acceptable). Same import-guard rule as C3.

### C5. Wire UQ into `twin.py` and remove placeholders
- `DigitalTwin(uq="ensemble")` produces real bands via `uq/ensemble.py`.
- `forecast(return_uncertainty=True)` with `uq="none"` must **raise**
  (`ValueError`), never return zero-width bands. **This is the v1 sin; ban it.**
- Replace `x_next = x + dx * 0.1  # placeholder dt` with an explicit `dt`
  (derive from the time vector or require it as an argument).
- Replace the `assimilate` "weighted average placeholder" with a real
  scalar/diagonal Kalman update using `obs_noise` and a prior variance, or raise
  if not implemented — no silent fake.

### C6. Tests (must land in the default suite)
- `tests/uq/test_ensemble.py`, `tests/uq/test_calibration.py` — core, always run.
  Include a calibration property test: a well-specified ensemble's PICP at level
  α should be ≈ α within tolerance on synthetic data.
- `tests/learn/test_phnn.py` — `pytest.importorskip("torch")`; assert skew(J),
  PSD(R), and energy decay with u=0 hold for the **learned** model.
- `tests/uq/test_gp_phs.py` — `pytest.importorskip("sklearn")`.
- `tests/twin/test_uq_wiring.py` — assert `return_uncertainty=True` with
  `uq="none"` raises; assert `uq="ensemble"` yields `lower < upper` where
  variance is nonzero.

**Gate C:** core tests (ensemble, calibration, twin wiring) pass with numpy only;
torch/gp tests **skip cleanly** (not error) when the extra is absent;
no code path returns fabricated uncertainty.

---

## Task D — IPHS: enforce the second-law guarantee, don't just document it. (blocking)

`systems/iphs.py` docstring claims `σ = ∇Sᵀ L ∇S ≥ 0` "guaranteed by structure",
but `L` is arbitrary and unchecked — the test only passes because its `L` is PSD.

1. Add `check_entropy_production(x)` returning `(is_nonneg, sigma)` and a
   `check_structure` that also validates `L` is PSD (reuse `utils.linalg.check_psd`).
2. Either constrain `L` to PSD by construction (Cholesky param) **or** validate at
   construction/first-call and raise if violated. Update the docstring to state
   exactly what is enforced vs. assumed.
3. Add a test asserting `σ ≥ 0` over sampled states, **and** a test that a
   non-PSD `L` is rejected.

**Gate D:** `σ ≥ 0` test passes for valid `L`; non-PSD `L` test confirms rejection.

---

## Definition of done (for this addendum)

- Default `pytest` is green; new core tests included; optional-extra tests skip
  cleanly when extras absent.
- No fabricated values in any code path (grep for `placeholder`, `zero
  uncertainty`, hardcoded `dt`).
- `STATUS.md` exists and matches `PKG/__init__.py`.
- No "complete"/"production-stable"/"95% coverage" claims anywhere outside
  `legacy_v1/`.
- Every doc number is script-generated; every citation is verified or marked
  `UNVERIFIED`.
