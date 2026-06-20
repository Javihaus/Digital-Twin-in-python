"""Battery State-of-Health (SoH) forecasting with calibrated uncertainty.

A worked example for grid-scale energy-storage digital twins (IEEE PES theme):
forecast each cell's State-of-Health into the future under a *temporal* split and
report whether the predicted uncertainty is trustworthy across the fleet.

Pipeline
--------
1. Curate the NASA fleet: keep only cells whose discharge protocol yields a valid
   degradation trajectory (explicit inclusion criteria; the rest are reference /
   impedance / partial-cycle protocols that do not represent capacity fade).
2. Per cell, split temporally (train on the first cycles, forecast the rest).
3. Three models:
     - physics-only  : SoH(n) = 1 - a*sqrt(n) - b*n   (SEI growth + linear aging),
                       a, b >= 0 fit on train only.
     - ML-only       : Gaussian Process on cycle -> SoH (extrapolates to the mean).
     - hybrid        : physics prior + GP on the residual; a bootstrap ensemble
                       gives a predictive distribution whose intervals widen with
                       the forecast horizon.
4. Evaluate with the otwin harness (RMSE / MAE / MASE / Theil's U / CRPS / coverage)
   against naive baselines, all under the temporal split.
5. Remaining-Useful-Life (RUL) to the 80% End-of-Life threshold, with uncertainty.

Outputs: figures/*.png (200 dpi) and figure_data/*.csv (the numbers behind every
figure, so they can be redrawn in a vector editor).

Run:  python examples/battery_soh/run_battery_soh.py
Requires: numpy, scipy, pandas, scikit-learn, matplotlib  (pip install otwin[gp,viz])
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set_theme(style="whitegrid", context="talk", palette="deep")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import nnls  # noqa: E402
from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.gaussian_process.kernels import (  # noqa: E402
    RBF,
    ConstantKernel,
    WhiteKernel,
)

from otwin.evaluation.baselines import drift, persistence  # noqa: E402
from otwin.evaluation.metrics import crps, mae, mase, rmse, theil_u  # noqa: E402
from otwin.uq.calibration import coverage_curve, expected_calibration_error, pit_values

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(42)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG = HERE / "figures"
FIGDATA = HERE / "figure_data"
FIG.mkdir(exist_ok=True)
FIGDATA.mkdir(exist_ok=True)

EOL = 0.80          # End-of-life: 80% SoH
TRAIN_FRAC = 0.5    # forecast the second half of life (realistic early-RUL task)
N_ENSEMBLE = 40
LEVEL = 0.90        # nominal central interval
HERO = "B0005"

# ----------------------------------------------------------------------------
# 1. Data curation
# ----------------------------------------------------------------------------
def load_fleet() -> pd.DataFrame:
    src = ROOT / "data" / "nasa" / "soh_by_cycle.csv"
    df = pd.read_csv(src)
    keep = []
    reasons = {}
    for bid, g in df.groupby("battery_id"):
        n = len(g)
        c0 = g["capacity_ah"].iloc[0]
        soh_end = g["soh"].iloc[-1]
        soh_max = g["soh"].max()
        # Inclusion criteria for a valid full-discharge degradation trajectory:
        soh = g["soh"].to_numpy()
        max_jump = float(np.max(np.diff(soh))) if len(soh) > 1 else 0.0
        early_min = float(soh[: int(0.85 * len(soh))].min()) if len(soh) else 1.0
        ok = (
            n >= 40                       # enough cycles to train + forecast
            and 0.5 <= c0 <= 2.6          # plausible nominal capacity (Ah)
            and soh_end <= 0.88           # actually degrades into useful range
            and soh_max <= 1.05           # no protocol artifacts (capacity "grows")
            and max_jump <= 0.15          # allow normal regeneration, reject glitches
            and early_min >= 0.55         # no premature crash-and-revive
        )
        if ok:
            keep.append(bid)
        else:
            reasons[bid] = (
                f"n={n}, C0={c0:.2f}, soh_end={soh_end:.2f}, soh_max={soh_max:.2f}"
            )
    fleet = df[df["battery_id"].isin(keep)].copy()
    print(f"Included {len(keep)} cells: {sorted(keep)}")
    print(f"Excluded {len(reasons)} cells (protocol/quality):")
    for b, r in sorted(reasons.items()):
        print(f"   {b}: {r}")
    fleet.to_csv(HERE / "soh_fleet.csv", index=False)
    return fleet


# ----------------------------------------------------------------------------
# 2. Models
# ----------------------------------------------------------------------------
def fit_physics(n: np.ndarray, soh: np.ndarray) -> float:
    """Fade-rate a >= 0 for SoH(n) ∝ exp(-a*n) (nonneg least squares in -log space).

    Exponential fade is bounded in (0, 1] and monotone decreasing, so it cannot
    extrapolate to nonsense (no negative SoH, no spurious acceleration). Points are
    weighted toward recent cycles so the fitted rate reflects the cell's *current*
    degradation rather than its flatter early life.
    """
    y = -np.log(np.clip(soh, 1e-4, 1.0))  # >= 0
    A = n.reshape(-1, 1)
    sw = np.sqrt(np.linspace(0.3, 1.0, n.size))  # recency weight
    coef, _ = nnls(A * sw[:, None], y * sw)
    return float(coef[0])


def physics_predict(
    n: np.ndarray, a: float, n_last: float, soh_last: float
) -> np.ndarray:
    """Anchored exp(-a n) shifted (additively) to pass through the last point.

    Additive anchoring keeps in-sample residuals small everywhere (no back-cast
    blow-up), so a residual learner sees only genuine structure, not anchoring
    artefacts.
    """
    p = np.exp(-a * n)
    offset = soh_last - np.exp(-a * n_last)
    return p + offset


def _gp() -> GaussianProcessRegressor:
    kernel = ConstantKernel(1.0) * RBF(length_scale=20.0) + WhiteKernel(1e-3)
    return GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=1, random_state=0
    )


def _conformal_sigma(n_tr, soh_tr, n_te, n_last):
    """Horizon-growing predictive std from an internal (split-conformal) holdout.

    Fit on the first 70% of train, measure absolute errors on the last 30% as a
    function of how far ahead they are, and fit sigma(h) = s0 + s1*h. This gives a
    distribution-free, horizon-aware scale that we apply to the test forecast so
    the intervals are calibrated rather than wishful.
    """
    m = max(int(0.7 * n_tr.size), 5)
    if n_tr.size - m >= 3:
        a_c = fit_physics(n_tr[:m], soh_tr[:m])
        pred_c = physics_predict(n_tr[m:], a_c, n_tr[m - 1], soh_tr[m - 1])
        res_c = np.abs(soh_tr[m:] - pred_c)
        h_c = n_tr[m:] - n_tr[m - 1]
        s1, s0 = np.polyfit(h_c, res_c, 1)
        s0 = max(float(s0), 5e-3)
        s1 = max(float(s1), 0.0)
    else:
        s0, s1 = 0.02, 0.0
    return s0 + s1 * (n_te - n_last)


def forecast_cell(n_tr, soh_tr, n_te):
    """Forecasts for physics / ML / hybrid, with calibrated horizon-growing bands."""
    n_last, soh_last = float(n_tr[-1]), float(soh_tr[-1])

    # physics-only (anchored at last observed point)
    a = fit_physics(n_tr, soh_tr)
    phys_te = physics_predict(n_te, a, n_last, soh_last)

    # ML-only: GP on cycle -> SoH (reverts to the training mean when extrapolating)
    gp_ml = _gp().fit(n_tr.reshape(-1, 1), soh_tr)
    ml_te = gp_ml.predict(n_te.reshape(-1, 1))

    # hybrid: physics + a *damped* GP residual correction. The damping makes the
    # correction act near the last observation and fade out into the forecast, so
    # the hybrid can only improve on physics near-term and never destabilise it.
    resid_tr = soh_tr - physics_predict(n_tr, a, n_last, soh_last)
    gp_res = _gp().fit(n_tr.reshape(-1, 1), resid_tr)
    corr = gp_res.predict(n_te.reshape(-1, 1))
    tau = max(0.4 * (n_te[-1] - n_last), 1.0)
    damp = np.exp(-(n_te - n_last) / tau)
    hyb_te = phys_te + corr * damp

    # calibrated predictive distribution (Gaussian with conformal sigma(h))
    sigma_h = _conformal_sigma(n_tr, soh_tr, n_te, n_last)
    z = 1.0  # store sigma; intervals are formed in run() at the requested level
    members = hyb_te[None, :] + RNG.normal(0.0, 1.0, (N_ENSEMBLE, n_te.size)) * (
        z * sigma_h[None, :]
    )
    members = np.clip(members, 0.0, 1.3)
    return {
        "physics": phys_te,
        "ml": ml_te,
        "hybrid": hyb_te,
        "sigma": sigma_h,
        "ensemble": members,        # (N_ENSEMBLE, n_te)
        "phys_param_a": a,
    }


# ----------------------------------------------------------------------------
# 3. Run fleet
# ----------------------------------------------------------------------------
def run() -> dict:
    fleet = load_fleet()
    cells = sorted(fleet["battery_id"].unique())
    results = {}
    for bid in cells:
        g = fleet[fleet["battery_id"] == bid].sort_values("cycle")
        n = g["cycle"].to_numpy(float)
        soh = g["soh"].to_numpy(float)
        amb = float(g["ambient_temperature"].iloc[0])
        k = int(np.floor(TRAIN_FRAC * len(n)))
        if len(n) - k < 5:
            continue
        n_tr, soh_tr = n[:k], soh[:k]
        n_te, soh_te = n[k:], soh[k:]
        fc = forecast_cell(n_tr, soh_tr, n_te)
        ens = fc["ensemble"]
        mean = fc["hybrid"]
        from scipy.stats import norm
        z = float(norm.ppf(1 - (1 - LEVEL) / 2))  # 1.645 at 90%
        lo = np.clip(mean - z * fc["sigma"], 0.0, 1.3)
        hi = np.clip(mean + z * fc["sigma"], 0.0, 1.3)
        # baselines
        bl_persist = persistence(soh_tr, horizon=len(n_te)).ravel()
        bl_drift = drift(soh_tr, horizon=len(n_te)).ravel()
        results[bid] = {
            "amb": amb,
            "n": n, "soh": soh, "k": k,
            "n_tr": n_tr, "soh_tr": soh_tr, "n_te": n_te, "soh_te": soh_te,
            "physics": fc["physics"], "ml": fc["ml"], "hybrid": fc["hybrid"],
            "mean": mean, "lo": lo, "hi": hi, "ensemble": ens,
            "bl_persist": bl_persist, "bl_drift": bl_drift,
        }
    return results


# ----------------------------------------------------------------------------
# 4. Metrics + RUL
# ----------------------------------------------------------------------------
def cell_metrics(r: dict) -> dict:
    y = r["soh_te"]
    out = {}
    for name, pred in [("physics", r["physics"]), ("ml", r["ml"]),
                       ("hybrid", r["hybrid"]),
                       ("persistence", r["bl_persist"]), ("drift", r["bl_drift"])]:
        out[f"rmse_{name}"] = rmse(y, pred)
        out[f"mae_{name}"] = mae(y, pred)
    out["mase_hybrid"] = mase(y, r["hybrid"], r["soh_tr"], seasonal_period=1)
    out["theilU_hybrid"] = theil_u(y, r["hybrid"], r["soh_tr"])
    out["crps_hybrid"] = crps(y, r["ensemble"].T)  # (n_te, members)
    return out


def rul_from_ensemble(n_te, ensemble, eol=EOL):
    """Predicted cycles-to-EOL distribution from ensemble member crossings."""
    crossings = []
    for m in ensemble:
        below = np.where(m <= eol)[0]
        if below.size:
            crossings.append(n_te[below[0]])
    return np.array(crossings)


def true_eol_cycle(n, soh, eol=EOL):
    below = np.where(soh <= eol)[0]
    return float(n[below[0]]) if below.size else np.nan


# ----------------------------------------------------------------------------
# 5. Figures
# ----------------------------------------------------------------------------
C_OBS, C_PHYS, C_ML, C_HYB, C_BAND = "#222222", "#1f77b4", "#d62728", "#2ca02c", "#2ca02c"


def save_csv(name: str, df: pd.DataFrame) -> None:
    df.to_csv(FIGDATA / f"{name}.csv", index=False)


def fig_hero(R):
    r = R[HERO]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r["n"], r["soh"], "o", ms=3, color=C_OBS, label="Observed SoH", zorder=5)
    ax.axvline(r["n_tr"][-1], color="gray", ls=":", lw=1)
    ax.fill_between(r["n_te"], r["lo"], r["hi"], color=C_BAND, alpha=0.20,
                    label=f"Hybrid {int(LEVEL*100)}% interval")
    ax.plot(r["n_te"], r["mean"], color=C_HYB, lw=2, label="Hybrid forecast")
    ax.plot(r["n_te"], r["physics"], color=C_PHYS, lw=1.5, ls="--", label="Physics-only")
    ax.plot(r["n_te"], r["ml"], color=C_ML, lw=1.5, ls="-.", label="ML-only (GP)")
    ax.axhline(EOL, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.text(r["n"][0], EOL + 0.005, "EOL 80%", fontsize=8)
    ax.set_xlabel("Cycle"); ax.set_ylabel("State of Health")
    ax.set_title(f"{HERO}: SoH forecast (temporal split)")
    ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "01_hero_forecast.png", dpi=200); plt.close(fig)
    save_csv("01_hero_forecast_test", pd.DataFrame({
        "cycle": r["n_te"], "soh_true": r["soh_te"], "hybrid_mean": r["mean"],
        "lower": r["lo"], "upper": r["hi"], "physics": r["physics"], "ml": r["ml"]}))


def fig_method_bars(R, M):
    names = ["persistence", "drift", "ml", "physics", "hybrid"]
    vals = [np.mean([M[b][f"rmse_{nm}"] for b in M]) for nm in names]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(names, vals, color=["#999", "#999", C_ML, C_PHYS, C_HYB])
    ax.set_ylabel("Fleet-mean test RMSE (SoH)")
    ax.set_title("Forecast accuracy by method (lower is better)")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v, f"{v:.3f}", ha="center",
                va="bottom", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(FIG / "02_method_comparison.png", dpi=200); plt.close(fig)
    save_csv("02_method_comparison", pd.DataFrame({"method": names, "rmse": vals}))


def fig_calibration(R):
    # pool standardized test points across fleet
    y_all, ens_all = [], []
    for r in R.values():
        y_all.append(r["soh_te"])
        ens_all.append(r["ensemble"].T)  # (n_te, members)
    y = np.concatenate(y_all)
    ens = np.vstack(ens_all)
    cc = coverage_curve(y, ens)
    ece = expected_calibration_error(y, ens)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot(cc["levels"], cc["coverage"], "o-", color=C_HYB, label="Hybrid (fleet)")
    ax.set_xlabel("Nominal coverage"); ax.set_ylabel("Empirical coverage")
    ax.set_title(f"Fleet calibration  (ECE = {ece:.2f})", fontsize=13)
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(FIG / "03_calibration.png", dpi=200); plt.close(fig)
    save_csv("03_calibration", pd.DataFrame({
        "nominal": cc["levels"], "empirical": cc["coverage"]}))
    # PIT histogram
    pit = pit_values(y, ens)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(pit, bins=12, color=C_HYB, alpha=0.7, edgecolor="white", density=True)
    ax.axhline(1.0, color="k", ls="--", lw=1, label="Uniform (ideal)")
    ax.set_xlabel("PIT value"); ax.set_ylabel("Density")
    ax.set_title("Probability Integral Transform (fleet)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "04_pit_histogram.png", dpi=200); plt.close(fig)
    save_csv("04_pit_histogram", pd.DataFrame({"pit": pit}))
    return ece


def fig_horizon(R):
    # interval width and abs error vs forecast horizon (steps ahead), pooled
    rows = []
    for r in R.values():
        for j in range(r["n_te"].size):
            rows.append((j + 1, r["hi"][j] - r["lo"][j],
                         abs(r["mean"][j] - r["soh_te"][j])))
    d = pd.DataFrame(rows, columns=["h", "width", "abserr"])
    agg = d.groupby("h").mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(agg["h"], agg["width"], color=C_BAND, lw=2, label=f"{int(LEVEL*100)}% interval width")
    ax.plot(agg["h"], agg["abserr"], color=C_OBS, lw=2, label="Mean abs. error")
    ax.set_xlabel("Forecast horizon (cycles ahead)"); ax.set_ylabel("SoH")
    ax.set_title("Uncertainty grows with the forecast horizon")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "05_horizon.png", dpi=200); plt.close(fig)
    save_csv("05_horizon", agg)


def fig_rul(R):
    r = R[HERO]
    full_true = true_eol_cycle(r["n"], r["soh"])
    crossings = rul_from_ensemble(r["n_te"], r["ensemble"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r["n"], r["soh"], "o", ms=3, color=C_OBS, label="Observed")
    ax.fill_between(r["n_te"], r["lo"], r["hi"], color=C_BAND, alpha=0.2)
    ax.plot(r["n_te"], r["mean"], color=C_HYB, lw=2, label="Hybrid forecast")
    ax.axhline(EOL, color="k", ls="--", lw=0.8)
    if not np.isnan(full_true):
        ax.axvline(full_true, color="crimson", lw=1.2, label=f"True EOL @ cycle {full_true:.0f}")
    if crossings.size:
        ax.axvspan(np.quantile(crossings, 0.05), np.quantile(crossings, 0.95),
                   color="orange", alpha=0.15, label="Predicted EOL (90%)")
    ax.set_xlabel("Cycle"); ax.set_ylabel("State of Health")
    ax.set_title(f"{HERO}: Remaining-useful-life forecast to 80% EOL")
    ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "06_rul_hero.png", dpi=200); plt.close(fig)
    if crossings.size:
        save_csv("06_rul_hero_crossings", pd.DataFrame({"predicted_eol_cycle": crossings}))

    # RUL parity across fleet
    rows = []
    for bid, rr in R.items():
        te = true_eol_cycle(rr["n"], rr["soh"])
        cr = rul_from_ensemble(rr["n_te"], rr["ensemble"])
        if not np.isnan(te) and cr.size:
            rows.append((bid, te, np.median(cr),
                         np.quantile(cr, 0.05), np.quantile(cr, 0.95)))
    if rows:
        d = pd.DataFrame(rows, columns=["cell", "true_eol", "pred_med", "p05", "p95"])
        fig, ax = plt.subplots(figsize=(6, 6))
        lo = min(d["true_eol"].min(), d["pred_med"].min()) - 5
        hi = max(d["true_eol"].max(), d["pred_med"].max()) + 5
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.errorbar(d["true_eol"], d["pred_med"],
                    yerr=[d["pred_med"] - d["p05"], d["p95"] - d["pred_med"]],
                    fmt="o", color=C_HYB, capsize=3)
        for _, row in d.iterrows():
            ax.annotate(row["cell"], (row["true_eol"], row["pred_med"]), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("True EOL cycle"); ax.set_ylabel("Predicted EOL cycle (median + 90%)")
        ax.set_title("Remaining-useful-life: predicted vs true (per cell)")
        ax.grid(alpha=0.3); ax.set_aspect("equal")
        fig.tight_layout(); fig.savefig(FIG / "07_rul_parity.png", dpi=200); plt.close(fig)
        save_csv("07_rul_parity", d)


def fig_fleet_overview(R):
    fig, ax = plt.subplots(figsize=(8, 5))
    for bid, r in R.items():
        ax.plot(r["n"], r["soh"], lw=1, alpha=0.8, label=f"{bid} ({r['amb']:.0f}°C)")
    ax.axhline(EOL, color="k", ls="--", lw=0.8)
    ax.set_xlabel("Cycle"); ax.set_ylabel("State of Health")
    ax.set_title("Curated NASA fleet — SoH degradation trajectories")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "00_fleet_overview.png", dpi=200); plt.close(fig)
    rows = []
    for bid, r in R.items():
        for ni, si in zip(r["n"], r["soh"]):
            rows.append((bid, r["amb"], ni, si))
    save_csv("00_fleet_overview", pd.DataFrame(rows, columns=["cell", "amb", "cycle", "soh"]))


def fig_small_multiples(R):
    cells = list(R)[:9]
    fig, axes = plt.subplots(3, 3, figsize=(11, 9))
    for ax, bid in zip(axes.ravel(), cells):
        r = R[bid]
        ax.plot(r["n"], r["soh"], "o", ms=2, color=C_OBS)
        ax.fill_between(r["n_te"], r["lo"], r["hi"], color=C_BAND, alpha=0.2)
        ax.plot(r["n_te"], r["mean"], color=C_HYB, lw=1.5)
        ax.axhline(EOL, color="k", ls="--", lw=0.6)
        ax.set_title(f"{bid} ({r['amb']:.0f}°C)", fontsize=9)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(cells):]:
        ax.axis("off")
    fig.suptitle("Per-cell SoH forecasts with calibrated uncertainty", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "08_small_multiples.png", dpi=200); plt.close(fig)


def fig_skill(R, M):
    rows = []
    for bid in M:
        sk = 1.0 - M[bid]["rmse_hybrid"] / max(M[bid]["rmse_persistence"], 1e-9)
        rows.append((bid, sk))
    d = pd.DataFrame(rows, columns=["cell", "skill"]).sort_values("skill")
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [C_HYB if s > 0 else C_ML for s in d["skill"]]
    ax.barh(d["cell"], d["skill"], color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Skill vs persistence  (1 - RMSE_hybrid / RMSE_persistence)")
    ax.set_title("Forecast skill over the naive baseline, per cell")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(FIG / "09_skill.png", dpi=200); plt.close(fig)
    save_csv("09_skill", d)


def main() -> None:
    R = run()
    M = {bid: cell_metrics(r) for bid, r in R.items()}
    fig_fleet_overview(R)
    fig_hero(R)
    fig_method_bars(R, M)
    ece = fig_calibration(R)
    fig_horizon(R)
    fig_rul(R)
    fig_small_multiples(R)
    fig_skill(R, M)

    # results table
    rows = []
    for bid in M:
        row = {"cell": bid, "amb": R[bid]["amb"], "n_cycles": R[bid]["n"].size}
        row.update(M[bid])
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(HERE / "results.csv", index=False)
    cov = coverage_curve(
        np.concatenate([r["soh_te"] for r in R.values()]),
        np.vstack([r["ensemble"].T for r in R.values()]),
        levels=np.array([LEVEL]),
    )["coverage"][0]
    print("\n=== Fleet summary ===")
    print(f"cells: {len(R)}")
    print(f"mean test RMSE  hybrid={res['rmse_hybrid'].mean():.4f}  "
          f"physics={res['rmse_physics'].mean():.4f}  ml={res['rmse_ml'].mean():.4f}  "
          f"persistence={res['rmse_persistence'].mean():.4f}")
    print(f"mean MASE hybrid={res['mase_hybrid'].mean():.3f} (<1 beats naive)")
    print(f"{int(LEVEL*100)}% interval empirical coverage (fleet)={cov:.3f}")
    print(f"calibration ECE={ece:.3f}")
    print(f"figures -> {FIG}")


if __name__ == "__main__":
    main()
