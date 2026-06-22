# Citations and References


## Port-Hamiltonian systems theory

### Port-Hamiltonian Systems Theory: : An Introductory Overview
**Reference:** van der Schaft, A., & Jeltsema, D. (2014). *Port-Hamiltonian
Systems Theory: An Introductory Overview.* Foundations and Trends in Systems and
Control, 1(2–3), 173–378. Now Publishers. DOI: 10.1561/2600000002.
**Use:** Foundational framework for the PHS core (`systems/phs.py`).

---

## Hamiltonian / port-Hamiltonian neural networks

### Hamiltonian Neural Networks
**Reference:** Greydanus, S., Dzamba, M., & Yosinski, J. (2019). *Hamiltonian
Neural Networks.* Advances in Neural Information Processing Systems (NeurIPS) 32,
15353–15363. arXiv:1906.01563.
**Use:** Motivation for structure-enforcing learned dynamics.

### Port-Hamiltonian neural networks for learning explicit time-dependent dynamical systems
**Reference:** Desai, S. A., Mattheakis, M., Sondak, D., Protopapas, P., &
Roberts, S. J. (2021). *Port-Hamiltonian neural networks for learning explicit
time-dependent dynamical systems.* Physical Review E, 104(3), 034312.
arXiv:2107.08024.
**Use:** Basis for the learned `PortHamiltonianNN` ([torch] extra).

---

## Irreversible port-Hamiltonian systems

### Irreversible port-Hamiltonian systems: A general formulation of irreversible processes with application to the CSTR.
**Reference:** Ramírez, H., Maschke, B., & Sbarbaro, D. (2013). *Irreversible
port-Hamiltonian systems: A general formulation of irreversible processes with
application to the CSTR.* Chemical Engineering Science, 89, 223–234.
**Use:** Foundation for `IrreversiblePHS` (entropy production, σ ≥ 0).

---

## Uncertainty quantification & calibration

### Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles
**Reference:** Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). *Simple
and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS 30,
6405–6416. arXiv:1612.01474.
**Use:** Ensemble UQ (`uq/ensemble.py`).

### Accurate Uncertainties for Deep Learning Using Calibrated Regression.
**Reference:** Kuleshov, V., Fenner, N., & Ermon, S. (2018). *Accurate
Uncertainties for Deep Learning Using Calibrated Regression.* ICML 2018, PMLR 80.
arXiv:1807.00263.
**Use:** Recalibration map in `uq/calibration.py` (`recalibrate`).

### Strictly Proper Scoring Rules, Prediction, and Estimation.
**Reference:** Gneiting, T., & Raftery, A. E. (2007). *Strictly Proper Scoring
Rules, Prediction, and Estimation.* Journal of the American Statistical
Association, 102(477), 359–378.
**Use:** CRPS and the interval score (`evaluation/metrics.py`,
`uq/calibration.py`).

---

## Forecasting evaluation

### Another look at measures of forecast accuracy.
**Reference:** Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures
of forecast accuracy.* International Journal of Forecasting, 22(4), 679–688.
**Use:** MASE and scale-free error metrics (`evaluation/metrics.py`).


---
