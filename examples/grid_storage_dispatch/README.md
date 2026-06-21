# Grid-scale storage: predictive maintenance **and** real-time optimization

A worked example for the IEEE PES topic *"AI-powered Digital Twins for Grid-Scale
Energy Storage: Enabling Predictive Maintenance and Real-Time Optimization."*

The title names two things. This example shows they are **one pipeline**:

> The calibrated **predictive-maintenance** twin (battery State-of-Health — the
> light-end model from [`../battery_soh`](../battery_soh)) gives the **real-time
> optimizer** what a plain optimizer lacks: the *shrinking, uncertain usable
> capacity*, a *degradation cost per unit of throughput*, and — crucially — a
> **calibrated uncertainty band** on the capacity that becomes a robust (chance)
> constraint. Calibrated uncertainty is what makes the optimization trustworthy.

## The setup

A grid-scale battery (10 MWh nominal, 5 MW) runs a **receding-horizon (MPC)**
dispatch over two real-time objectives:

- **Peak shaving** — cut the peak of net grid demand (priced via a demand charge).
- **Energy arbitrage** — charge when price is low, discharge when high.

The true usable capacity has degraded and is only known through the SoH twin as a
point estimate `ĉap` with a calibrated band `σ`. Three dispatch strategies:

| Strategy | Capacity it assumes | Degradation cost |
|---|---|---|
| **Naive** | nominal (10 MWh) | ignored |
| **Degradation-aware** | point estimate `ĉap` | included |
| **Robust (calibrated UQ)** | lower band `ĉap − z·σ` (90%) | included |

Only the **usable ceiling** carries the uncertainty; the reserve floor is fixed.
Each strategy plans with its assumed capacity, then dispatch is **realised under
the true capacity** — so over-promising shows up as unmet (undeliverable) energy.

## What the results show

Monte Carlo over 150 days (true capacity drawn from the calibrated band):

| Strategy | Arbitrage value ($/day) | Shortfall days | Peak value ($/day) | Shortfall days |
|---|---|---|---|---|
| Naive | 194 | **100%** | 408 | **100%** |
| Degradation-aware | 304 | 49% | 551 | 45% |
| Robust (calibrated UQ) | 291 | **9%** | 546 | **9%** |

- **Naive** assumes nominal capacity → over-promises **every** day.
- **Degradation-aware (point)** is a coin-flip: the true capacity sits below the
  point estimate roughly half the time.
- **Robust (calibrated UQ)** hits the **9% shortfall** target (≈ the 10% the 90%
  band promises) while giving up only ~4% of value. **The calibration does real
  work** — it is the knob that turns "trust" into a feasible, near-optimal plan.

The **calibration sweet spot** (`05_calibration_sweet_spot.png`): sweeping the
robustness margin `k` (cap = `ĉap − k·σ`), the realised value peaks near the
calibrated margin and then falls — too little margin causes shortfalls, too much
leaves value on the table. At the calibrated `k ≈ 1.28`, empirical coverage is
**90%**, matching the nominal band.

## Figures

| File | What |
|---|---|
| `01_arbitrage_trajectories.png` | Price, SoC, and cumulative unmet energy per strategy (naive Σ ≈ 56 MWh vs robust Σ = 0) |
| `02_peak_trajectories.png` | Same, peak-shaving objective |
| `03_arbitrage_montecarlo.png` | Realised value and shortfall rate over 150 days (arbitrage) |
| `04_peak_montecarlo.png` | Same, peak shaving |
| `05_calibration_sweet_spot.png` | Value and shortfall vs robustness margin; calibrated point marked |

Per-figure data is exported to `figure_data/*.csv`.

## Run

```bash
pip install cvxpy pandas matplotlib seaborn   # plus numpy
python run_dispatch.py
```

Runs on a laptop CPU in seconds (parameterised cvxpy problems, CLARABEL solver).

## Scope and honesty notes

- Prices and loads are **synthetic** (clearly diurnal) to make the mechanism
  legible; swap in real ISO price / site-load series to reproduce on your data.
- The degradation cost and capacity band are illustrative parameterisations of
  the light-end SoH model; the point is the **pipeline** (PM → calibrated UQ →
  RTO), not these specific constants.
- This is a single-asset dispatch; multi-asset / network-coupled dispatch is a
  natural extension via port composition.
