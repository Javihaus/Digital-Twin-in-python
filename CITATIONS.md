# Citations and References

All external references with verification status. **UNVERIFIED** references must
not be asserted as fact in code comments, docstrings, or documentation.

## Status definitions
- **VERIFIED**: exact title, authors, year, and venue confirmed.
- **UNVERIFIED**: not yet confirmed — do not cite as fact.

---

## Port-Hamiltonian systems theory

### van der Schaft & Jeltsema — PHS introductory overview
**Status:** VERIFIED
**Reference:** van der Schaft, A., & Jeltsema, D. (2014). *Port-Hamiltonian
Systems Theory: An Introductory Overview.* Foundations and Trends in Systems and
Control, 1(2–3), 173–378. Now Publishers. DOI: 10.1561/2600000002.
**Use:** Foundational framework for the PHS core (`systems/phs.py`).

---

## Hamiltonian / port-Hamiltonian neural networks

### Greydanus, Dzamba & Yosinski — Hamiltonian Neural Networks
**Status:** VERIFIED
**Reference:** Greydanus, S., Dzamba, M., & Yosinski, J. (2019). *Hamiltonian
Neural Networks.* Advances in Neural Information Processing Systems (NeurIPS) 32,
15353–15363. arXiv:1906.01563.
**Use:** Motivation for structure-enforcing learned dynamics.

### Desai, Mattheakis, Sondak, Protopapas & Roberts — Port-Hamiltonian NN
**Status:** VERIFIED
**Reference:** Desai, S. A., Mattheakis, M., Sondak, D., Protopapas, P., &
Roberts, S. J. (2021). *Port-Hamiltonian neural networks for learning explicit
time-dependent dynamical systems.* Physical Review E, 104(3), 034312.
arXiv:2107.08024.
**Use:** Basis for the learned `PortHamiltonianNN` ([torch] extra).

---

## Irreversible port-Hamiltonian systems

### Ramírez, Maschke & Sbarbaro — IPHS and the CSTR
**Status:** VERIFIED
**Reference:** Ramírez, H., Maschke, B., & Sbarbaro, D. (2013). *Irreversible
port-Hamiltonian systems: A general formulation of irreversible processes with
application to the CSTR.* Chemical Engineering Science, 89, 223–234.
**Use:** Foundation for `IrreversiblePHS` (entropy production, σ ≥ 0).

---

## Uncertainty quantification & calibration

### Lakshminarayanan, Pritzel & Blundell — Deep Ensembles
**Status:** VERIFIED
**Reference:** Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). *Simple
and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS 30,
6405–6416. arXiv:1612.01474.
**Use:** Ensemble UQ (`uq/ensemble.py`).

### Kuleshov, Fenner & Ermon — Calibrated regression
**Status:** VERIFIED
**Reference:** Kuleshov, V., Fenner, N., & Ermon, S. (2018). *Accurate
Uncertainties for Deep Learning Using Calibrated Regression.* ICML 2018, PMLR 80.
arXiv:1807.00263.
**Use:** Recalibration map in `uq/calibration.py` (`recalibrate`).

### Gneiting & Raftery — Strictly proper scoring rules
**Status:** VERIFIED
**Reference:** Gneiting, T., & Raftery, A. E. (2007). *Strictly Proper Scoring
Rules, Prediction, and Estimation.* Journal of the American Statistical
Association, 102(477), 359–378.
**Use:** CRPS and the interval score (`evaluation/metrics.py`,
`uq/calibration.py`).

---

## Forecasting evaluation

### Hyndman & Koehler — Forecast accuracy measures (MASE)
**Status:** VERIFIED
**Reference:** Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures
of forecast accuracy.* International Journal of Forecasting, 22(4), 679–688.
**Use:** MASE and scale-free error metrics (`evaluation/metrics.py`).

### Theil's U statistic
**Status:** UNVERIFIED
**Reference:** (Theil, H., original reference to be confirmed.)
**Notes:** Forecast accuracy relative to a naive baseline.

---

## Gaussian-process port-Hamiltonian systems (optional [gp])

### Beckers et al. — GP-PHS
**Status:** UNVERIFIED
**Reference:** Beckers, T., et al. (2022). (Exact title/authors to be confirmed.)
**Notes:** GP approach to learning port-Hamiltonian dynamics.

---

## Structure-preserving integration

### Implicit-midpoint / discrete-gradient methods
**Status:** UNVERIFIED
**Reference:** (Standard symplectic / energy-preserving integrators; exact
textbook citations to be added — e.g. Hairer, Lubich & Wanner.)
**Notes:** Used by the structure-preserving integrator.

---

## Data & legacy

### NASA battery aging dataset
**Status:** UNVERIFIED
**Reference:** NASA Prognostics Center of Excellence (PCoE) Battery Data Set.
(Exact attribution to be confirmed.)
**Use:** `examples/battery_soh`. Confirm proper attribution before publication.

### Xu et al. — battery degradation (legacy v1 only)
**Status:** UNVERIFIED
**Reference:** Xu, B., et al. (2016). (Exact title to be confirmed.)
**Notes:** v1 attributed a degradation formula to this source. v2 does **not**
rely on it; the light-end fade prior is treated as empirical, not attributed.

---

**Before asserting any reference as fact:** confirm authors/title/year/venue,
confirm the specific claim appears in the source, set status to VERIFIED, and
never cite UNVERIFIED references as authoritative.
