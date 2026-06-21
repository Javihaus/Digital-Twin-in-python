# Battery State-of-Health — empirical-law model (grey-box)

![Battery State-of-Health — block diagram (grey-box workflow)](../../assets/battery_block.svg)

Forecasting lithium-ion **State-of-Health (SoH)** and **Remaining-Useful-Life
(RUL)** on the NASA battery-aging dataset. This is the **empirical end** of otwin (grey-box):
there is no energy dynamics to conserve, only a slow degradation trend, so a
transparent **fade-law prior** carries the trend, a **bounded learned residual**
corrects it, and **horizon-aware conformal intervals** quantify uncertainty.

> Not port-Hamiltonian — capacity fade is a monotone degradation curve. Same
> workflow as the first-principles end (prior → residual → calibrated uncertainty
> → leakage-free evaluation), only the physics is lighter.

## The fleet

Eight curated NASA cells: **B0005/6/7/18** (24 °C) and **B0045–48** (4 °C). The
two temperatures are different operating regimes — used here to stress-test the
method across conditions.

![Fleet overview](figures/00_fleet_overview.png)

## Forecast with calibrated uncertainty

Train on the early cycles, forecast the rest. The hybrid forecast (fade-law prior
+ residual) with its conformal band:

![Hero forecast](figures/01_hero_forecast.png)

Resolved by horizon (short / medium / long term):

![Horizon zoom](figures/10_zoom_horizons.png)

## Does the uncertainty mean what it says?

A stated 90% interval should contain the truth ~90% of the time. Calibration
diagnostics (coverage curve and standardized-residual / Q–Q checks):

![Calibration](figures/03_calibration.png)

![Residual diagnostics](figures/11_residual_diag.png)

## Skill vs baselines (no fooling yourself)

Every metric is reported against naive baselines (persistence, drift) under a
temporal split:

![Skill scores](figures/09_skill.png)

## Remaining Useful Life

![RUL](figures/06_rul_hero.png)

## Results (real, from `results.csv`)

On the **24 °C** cells the hybrid beats persistence over the forecast horizon
(Theil's U < 1): B0007 ≈ 0.19, B0005 ≈ 0.35, B0018 ≈ 0.36, B0006 ≈ 0.51
(hybrid SoH RMSE ≈ 0.013–0.041).

**Known limitation (reported as-is):** on the colder **4 °C** cells
(B0045–B0048) the hybrid does *not* yet beat persistence (Theil's U ≈ 1.1–1.8).
The harder regime is shown, not hidden.

Per-figure data is in `figure_data/*.csv`; the full per-cell table is `results.csv`.

## Run

```bash
pip install numpy scipy scikit-learn pandas matplotlib seaborn
python run_battery_soh.py
```

The raw NASA data is not committed (large). Place the Kaggle NASA battery archive
in the project as `archive.zip` (or `data/nasa/`) before running; the loader
aggregates per-cycle SoH from it. Figures regenerate into `figures/`.
