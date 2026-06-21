"""Grid-scale battery dispatch: predictive maintenance AND real-time optimization.

Worked example for the otwin thesis behind the IEEE PES topic
"AI-powered Digital Twins for Grid-Scale Energy Storage: Enabling Predictive
Maintenance and Real-Time Optimization".

The two halves of that title are one pipeline:

    The calibrated predictive-maintenance twin (battery State-of-Health, the
    light-end model from examples/battery_soh) feeds a real-time dispatch
    optimizer two things a plain optimizer lacks:
      1. a degradation cost per unit of throughput and the (shrinking, uncertain)
         usable capacity, and
      2. a *calibrated* uncertainty band on that capacity, used as a robust
         (chance) constraint so dispatch stays feasible with stated confidence.

Two real-time objectives (receding-horizon / MPC):
  - peak shaving  (cut the peak of net grid demand)
  - energy arbitrage (charge cheap, discharge expensive)

Three strategies compared:
  - naive                       : assume nominal capacity, ignore degradation
  - degradation-aware (point)   : SoH point estimate + degradation cost
  - robust (calibrated UQ)      : calibrated lower capacity band (chance constraint)

Outputs: figures/*.png and figure_data/*.csv
Requires: numpy, cvxpy, pandas, matplotlib, seaborn
"""

from __future__ import annotations

from pathlib import Path

import cvxpy as cp
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set_theme(style="whitegrid", context="talk")

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
DATA = HERE / "figure_data"
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

BLUE, INK, ACCENT, GREEN, SKY = "#1C4E73", "#1B2430", "#B5651D", "#2E7D32", "#3D80A9"

# ------------------------------------------------------------------ BESS params
E_NOM = 10.0  # MWh nominal usable energy
P_MAX = 5.0  # MW charge/discharge limit
ETA = 0.95  # one-way efficiency (charge and discharge)
SOC_MIN, SOC_MAX = 0.10, 0.95  # fraction of usable capacity
E_FLOOR = SOC_MIN * E_NOM  # absolute reserve floor (MWh) — does NOT scale with the
# uncertain capacity; only the usable *ceiling* (SOC_MAX * cap) carries the SoH band.
DT = 1.0  # hours per step
H = 24  # MPC horizon (hours)
LEVEL = 0.90  # nominal coverage of the capacity band


# ------------------------------------------------ light-end SoH (predictive mt.)
def soh_point(eq_cycles: float, a: float = 6e-4) -> float:
    """Exponential capacity-fade prior (same light-end law as examples/battery_soh).

    SoH(n) = exp(-a n), with n the equivalent full cycles. Returns a fraction.
    """
    return float(np.exp(-a * eq_cycles))


def capacity_belief(eq_cycles: float, rel_sigma: float = 0.05) -> tuple[float, float]:
    """Usable-capacity point estimate and a calibrated 1-sigma (MWh).

    The SoH twin reports usable capacity cap_hat = SoH * E_NOM with a conformal
    std. ``rel_sigma`` is the fractional std of the calibrated band.
    """
    cap_hat = soh_point(eq_cycles) * E_NOM
    sigma = rel_sigma * cap_hat
    return cap_hat, sigma


# marginal degradation cost ($/MWh of throughput): replacement cost amortised
# over lifetime throughput.  cost_repl / (cycle_life * E_NOM * 2)  (charge+discharge)
C_DEG = 220_000.0 / (4000.0 * E_NOM * 2.0)  # ~$/MWh; ≈ 2.75
DEMAND_CHARGE = 200.0  # $/MW over the horizon (peak-shaving objective)
SITE_PEAK = 9.0  # MW — national demand shape is scaled to this site peak


def load_eu_csv() -> dict | None:
    """Load real EU day-ahead price + demand if data/eu_market.csv exists.

    Schema (produced by prepare_opsd.py from Open Power System Data / ENTSO-E):
    columns ``timestamp, price_eur_mwh, load_mw``. Returns daily 24 h windows;
    national demand is normalised per day and rescaled to ``SITE_PEAK`` so a
    site-scale battery is meaningful (the real *shape*, scaled to site size).
    """
    p = HERE / "data" / "eu_market.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df = df.dropna(subset=["price_eur_mwh", "load_mw"])
    n = (len(df) // 24) * 24
    if n < 24:
        return None
    price = df["price_eur_mwh"].to_numpy()[:n].reshape(-1, 24)
    load = df["load_mw"].to_numpy()[:n].reshape(-1, 24)
    load_site = load / load.max(axis=1, keepdims=True) * SITE_PEAK
    return {"price": price, "load": load_site, "n_days": price.shape[0]}


REAL = load_eu_csv()
DATA_SOURCE = "OPSD/ENTSO-E (real)" if REAL is not None else "synthetic"


# ------------------------------------------------------------------- signals
def daily_price(days: int, seed: int) -> np.ndarray:
    """Hourly electricity price: real (OPSD) if available, else synthetic."""
    if REAL is not None:
        idx = np.arange(days) % REAL["n_days"]
        return REAL["price"][idx].ravel()
    rng = np.random.default_rng(seed)
    hours = np.arange(24)
    base = (
        45
        + 25 * np.exp(-((hours - 8) ** 2) / 6)
        + 35 * np.exp(-((hours - 19) ** 2) / 5)
        - 15 * np.exp(-((hours - 3) ** 2) / 8)
    )
    out = []
    for _ in range(days):
        out.append(base * (1 + 0.08 * rng.standard_normal(24)))
    return np.clip(np.concatenate(out), 5, None)


def daily_load(days: int, seed: int) -> np.ndarray:
    """Hourly site load (MW): real demand shape (OPSD, scaled) if available, else synthetic."""
    if REAL is not None:
        idx = np.arange(days) % REAL["n_days"]
        return REAL["load"][idx].ravel()
    rng = np.random.default_rng(seed)
    hours = np.arange(24)
    base = (
        6
        + 3.5 * np.exp(-((hours - 9) ** 2) / 8)
        + 6.5 * np.exp(-((hours - 19) ** 2) / 4)
    )
    out = []
    for _ in range(days):
        out.append(base * (1 + 0.05 * rng.standard_normal(24)))
    return np.clip(np.concatenate(out), 0.5, None)


# -------------------------------------------------- parameterised MPC problems
def _build(objective: str):
    """Build a DPP-parameterised horizon problem once; reuse via parameters."""
    pc = cp.Variable(H, nonneg=True)  # charge power (MW)
    pd = cp.Variable(H, nonneg=True)  # discharge power (MW)
    E = cp.Variable(H + 1)  # energy (MWh)

    E0 = cp.Parameter()  # current energy
    cap = cp.Parameter(nonneg=True)  # assumed usable capacity (upper bound)
    e_floor = cp.Parameter(nonneg=True)  # terminal floor (avoid horizon draining)
    sig = cp.Parameter(H)  # price ($/MWh) or load (MW)
    cdeg = cp.Parameter(nonneg=True)
    dc = cp.Parameter(nonneg=True)  # demand charge ($/MW), peak objective only

    cons = [E[0] == E0]
    for t in range(H):
        cons += [E[t + 1] == E[t] + (ETA * pc[t] - pd[t] / ETA) * DT]
    cons += [pc <= P_MAX, pd <= P_MAX]
    cons += [E >= E_FLOOR, E <= SOC_MAX * cap]  # fixed floor, uncertain ceiling
    cons += [E[H] >= e_floor]

    throughput = cp.sum(pc + pd) * DT
    if objective == "arbitrage":
        revenue = sig @ (pd - pc) * DT
        obj = cp.Maximize(revenue - cdeg * throughput)
    elif objective == "peak":
        peak = cp.Variable()
        cons += [peak >= sig + pc - pd]  # net grid draw at each hour
        cons += [peak >= 0]
        obj = cp.Minimize(dc * peak + cdeg * throughput)  # both terms in $
    else:
        raise ValueError(objective)

    prob = cp.Problem(obj, cons)
    return {
        "prob": prob,
        "pc": pc,
        "pd": pd,
        "E": E,
        "E0": E0,
        "cap": cap,
        "e_floor": e_floor,
        "sig": sig,
        "cdeg": cdeg,
        "dc": dc,
    }


_PROBS = {"arbitrage": _build("arbitrage"), "peak": _build("peak")}


def plan_horizon(
    objective: str, e0: float, cap_assumed: float, signal_H: np.ndarray, c_deg: float
) -> dict:
    """Solve one horizon; return first action and full plan."""
    P = _PROBS[objective]
    s = np.asarray(signal_H, dtype=float)
    if s.size < H:  # pad tail by repeating last value
        s = np.concatenate([s, np.repeat(s[-1], H - s.size)])
    P["E0"].value = float(e0)
    P["cap"].value = float(cap_assumed)
    P["e_floor"].value = float(E_FLOOR + 0.20 * cap_assumed)
    P["sig"].value = s[:H]
    P["cdeg"].value = float(c_deg)
    P["dc"].value = DEMAND_CHARGE if objective == "peak" else 0.0
    P["prob"].solve(solver=cp.CLARABEL)
    if P["pc"].value is None:  # infeasible -> idle
        return {
            "pc0": 0.0,
            "pd0": 0.0,
            "pc": np.zeros(H),
            "pd": np.zeros(H),
            "E": None,
            "ok": False,
        }
    return {
        "pc0": float(P["pc"].value[0]),
        "pd0": float(P["pd"].value[0]),
        "pc": np.clip(np.asarray(P["pc"].value), 0, None),
        "pd": np.clip(np.asarray(P["pd"].value), 0, None),
        "E": np.asarray(P["E"].value),
        "ok": True,
    }


# -------------------------------------------------- realised receding-horizon run
def run_mpc(
    objective: str,
    signal_full: np.ndarray,
    cap_assumed: float,
    cap_true: float,
    c_deg: float,
    e0: float | None = None,
) -> dict:
    """Apply MPC: plan with cap_assumed, realise dynamics under the TRUE capacity.

    Records realised SoC, applied power, degradation throughput and any shortfall
    (energy that could not be delivered/absorbed because true capacity differs).
    """
    T = len(signal_full)
    e = init_energy(cap_true) if e0 is None else e0
    soc, pc_a, pd_a = [], [], []
    shortfall = 0.0
    sf_series = []
    tput = 0.0
    grid = []  # net grid draw (peak) or cashflow signal handled outside
    for k in range(T):
        plan = plan_horizon(objective, e, cap_assumed, signal_full[k : k + H], c_deg)
        pc0, pd0 = plan["pc0"], plan["pd0"]
        # realise under TRUE capacity (clip)
        e_next_ideal = e + (ETA * pc0 - pd0 / ETA) * DT
        e_next = min(max(e_next_ideal, E_FLOOR), SOC_MAX * cap_true)
        sf = abs(e_next_ideal - e_next)  # MWh that could not be realised
        shortfall += sf
        sf_series.append(sf)
        # delivered powers after clipping
        delivered = (e_next - e) / DT
        if delivered >= 0:
            pc_real, pd_real = delivered / ETA, 0.0
        else:
            pc_real, pd_real = 0.0, -delivered * ETA
        tput += (pc_real + pd_real) * DT
        soc.append(e_next / cap_true)
        pc_a.append(pc_real)
        pd_a.append(pd_real)
        grid.append(signal_full[k])
        e = e_next
    pc_a = np.array(pc_a)
    pd_a = np.array(pd_a)
    return {
        "soc": np.array(soc),
        "pc": pc_a,
        "pd": pd_a,
        "shortfall": shortfall,
        "shortfall_series": np.array(sf_series),
        "throughput": tput,
        "signal": np.array(grid),
    }


def realised_value(objective: str, res: dict, signal: np.ndarray) -> float:
    """Economic value: arbitrage net revenue ($) or peak reduction (MW)."""
    if objective == "arbitrage":
        cash = float(np.sum(signal * (res["pd"] - res["pc"]) * DT))
        return cash - C_DEG * res["throughput"]
    # peak shaving: reduction vs no-storage peak
    net = signal + res["pc"] - res["pd"]
    return float(np.max(signal) - np.max(net))


def init_energy(cap_true: float) -> float:
    """Starting energy: 40% of the way up the usable band, above the fixed floor."""
    return E_FLOOR + 0.4 * (SOC_MAX * cap_true - E_FLOOR)


def realise_plan(
    objective: str, plan: dict, signal: np.ndarray, cap_true: float, e0: float
) -> dict:
    """Apply a fixed 24 h plan under the TRUE capacity; return value + shortfall."""
    e = e0
    pc_r, pd_r = [], []
    shortfall = 0.0
    n = min(H, len(signal))
    for t in range(n):
        e_ideal = e + (ETA * plan["pc"][t] - plan["pd"][t] / ETA) * DT
        e_next = min(max(e_ideal, E_FLOOR), SOC_MAX * cap_true)
        shortfall += abs(e_ideal - e_next)
        delivered = (e_next - e) / DT
        if delivered >= 0:
            pc_r.append(delivered / ETA)
            pd_r.append(0.0)
        else:
            pc_r.append(0.0)
            pd_r.append(-delivered * ETA)
        e = e_next
    pc_r, pd_r = np.array(pc_r), np.array(pd_r)
    tput = float(np.sum(pc_r + pd_r) * DT)
    if objective == "arbitrage":
        value = float(np.sum(signal[:n] * (pd_r - pc_r) * DT)) - C_DEG * tput
    else:
        net = signal[:n] + pc_r - pd_r
        value = DEMAND_CHARGE * (float(np.max(signal[:n])) - float(np.max(net)))
        value -= C_DEG * tput
    return {
        "value": value,
        "shortfall": shortfall,
        "throughput": tput,
        "pc": pc_r,
        "pd": pd_r,
    }


def _z() -> float:
    from scipy.stats import norm

    return float(norm.ppf(LEVEL))


def strat_caps(cap_hat: float, sigma: float) -> dict:
    """cap_assumed and degradation-cost flag for each strategy."""
    return {
        "Naive": (E_NOM, 0.0),
        "Degradation-aware": (cap_hat, C_DEG),
        "Robust (calibrated UQ)": (cap_hat - _z() * sigma, C_DEG),
    }


STRAT_COLOR = {
    "Naive": ACCENT,
    "Degradation-aware": SKY,
    "Robust (calibrated UQ)": BLUE,
}
EQ_CYCLES = 800.0  # aged grid cell


# ============================================================ Fig 1-2: trajectories
def fig_trajectories(objective: str, fname: str, ylab: str) -> None:
    cap_hat, sigma = capacity_belief(EQ_CYCLES)
    cap_true = cap_hat - 0.8 * sigma  # a plausible true capacity below the estimate
    sig = (daily_price if objective == "arbitrage" else daily_load)(2, seed=7)
    caps = strat_caps(cap_hat, sigma)
    e0 = init_energy(cap_true)

    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    th = np.arange(len(sig))
    ax[0].plot(th, sig, color=INK, lw=2)
    ax[0].set_ylabel(ylab)
    ax[0].set_title(
        f"{objective.capitalize()} — true capacity {cap_true:.2f} MWh "
        f"(estimate {cap_hat:.2f})"
    )
    rows = []
    for name, (cap_a, cdeg) in caps.items():
        res = run_mpc(objective, sig, cap_a, cap_true, cdeg, e0=e0)
        ax[1].plot(th, 100 * res["soc"], color=STRAT_COLOR[name], lw=2, label=name)
        ax[2].plot(
            th,
            np.cumsum(res["shortfall_series"]),
            color=STRAT_COLOR[name],
            lw=2,
            label=f"{name} (Σ={res['shortfall']:.1f} MWh)",
        )
        for t in th:
            rows.append(
                {
                    "hour": int(t),
                    "strategy": name,
                    "signal": float(sig[t]),
                    "soc_pct": float(100 * res["soc"][t]),
                    "cum_shortfall": float(np.cumsum(res["shortfall_series"])[t]),
                }
            )
    ax[1].axhline(100 * SOC_MAX, ls="--", c="gray", lw=1)
    ax[1].axhline(100 * SOC_MIN, ls="--", c="gray", lw=1)
    ax[1].set_ylabel("SoC (%)")
    ax[1].legend(fontsize=10, loc="best")
    ax[2].set_ylabel("Cumulative\nunmet energy (MWh)")
    ax[2].set_xlabel("Hour")
    ax[2].legend(fontsize=10, loc="best")
    fig.suptitle(
        f"Real-time dispatch under uncertain capacity — {objective}", fontsize=15
    )
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=160)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(DATA / fname.replace(".png", ".csv"), index=False)
    print(f"  wrote {fname}")


# ============================================================ Fig 3-4: Monte Carlo
def fig_montecarlo(objective: str, fname: str, ylab: str, n: int = 150) -> None:
    rng = np.random.default_rng(2024)
    cap_hat, sigma = capacity_belief(EQ_CYCLES)
    sig = (daily_price if objective == "arbitrage" else daily_load)(1, seed=11)
    caps = strat_caps(cap_hat, sigma)
    agg = {k: {"value": [], "sf": [], "viol": 0} for k in caps}
    tol = 0.05
    for _ in range(n):
        cap_true = float(np.clip(rng.normal(cap_hat, sigma), 0.3 * E_NOM, E_NOM))
        e0 = init_energy(cap_true)
        for name, (cap_a, cdeg) in caps.items():
            plan = plan_horizon(objective, e0, cap_a, sig, cdeg)
            r = realise_plan(objective, plan, sig, cap_true, e0)
            agg[name]["value"].append(r["value"])
            agg[name]["sf"].append(r["shortfall"])
            agg[name]["viol"] += int(r["shortfall"] > tol)
    names = list(caps)
    mean_val = [float(np.mean(agg[k]["value"])) for k in names]
    viol_rate = [100 * agg[k]["viol"] / n for k in names]
    colors = [STRAT_COLOR[k] for k in names]

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].bar(names, mean_val, color=colors)
    ax[0].set_ylabel(ylab)
    ax[0].set_title("Realised value (higher = better)")
    ax[0].tick_params(axis="x", labelrotation=20)
    ax[1].bar(names, viol_rate, color=colors)
    ax[1].axhline(
        100 * (1 - LEVEL),
        ls="--",
        c=INK,
        lw=1.5,
        label=f"target ≤ {100*(1-LEVEL):.0f}%",
    )
    ax[1].set_ylabel("Infeasible days (%)")
    ax[1].set_title("Capacity-shortfall rate (lower = better)")
    ax[1].tick_params(axis="x", labelrotation=20)
    ax[1].legend(fontsize=10)
    fig.suptitle(
        f"Monte Carlo over {n} days — {objective} "
        f"(true capacity ~ calibrated band)",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=160)
    plt.close(fig)
    pd.DataFrame(
        {"strategy": names, "mean_value": mean_val, "shortfall_rate_pct": viol_rate}
    ).to_csv(DATA / fname.replace(".png", ".csv"), index=False)
    print(f"  wrote {fname}")


# ============================================================ Fig 5: calibration sweet spot
def fig_calibration(fname: str, n: int = 120) -> None:
    rng = np.random.default_rng(7)
    cap_hat, sigma = capacity_belief(EQ_CYCLES)
    ks = np.array([0.0, 0.4, 0.8, _z(), 1.6, 2.0, 2.5, 3.0])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    for ax, objective, ylab in zip(
        axes, ["arbitrage", "peak"], ["Realised value ($)", "Realised value ($)"]
    ):
        sig = (daily_price if objective == "arbitrage" else daily_load)(1, seed=11)
        draws = np.clip(rng.normal(cap_hat, sigma, n), 0.3 * E_NOM, E_NOM)
        vals, sfr, cov = [], [], []
        for k in ks:
            cap_a = cap_hat - k * sigma
            vv, ss, cc = [], 0, 0
            for cap_true in draws:
                e0 = init_energy(cap_true)
                plan = plan_horizon(objective, e0, cap_a, sig, 0.0 if k == 0 else C_DEG)
                r = realise_plan(objective, plan, sig, float(cap_true), e0)
                vv.append(r["value"])
                ss += int(r["shortfall"] > 0.05)
                cc += int(cap_true >= cap_a)
            vals.append(float(np.mean(vv)))
            sfr.append(100 * ss / n)
            cov.append(100 * cc / n)
        ax.plot(ks, vals, "-o", color=BLUE, lw=2.4, label="realised value")
        ax.set_xlabel("robustness margin  k  (cap = ĉap − k·σ)")
        ax.set_ylabel(ylab, color=BLUE)
        ax.axvline(_z(), ls="--", color=GREEN, lw=2)
        ax.text(
            _z() + 0.05,
            min(vals),
            f"calibrated\n(k={_z():.2f}, {int(LEVEL*100)}%)",
            color=GREEN,
            fontsize=10,
            va="bottom",
        )
        ax2 = ax.twinx()
        ax2.plot(ks, sfr, "-s", color=ACCENT, lw=2, label="shortfall rate")
        ax2.set_ylabel("shortfall rate (%)", color=ACCENT)
        ax.set_title(objective)
        pd.DataFrame(
            {"k": ks, "value": vals, "shortfall_rate_pct": sfr, "coverage_pct": cov}
        ).to_csv(DATA / f"05_calibration_{objective}.csv", index=False)
    fig.suptitle(
        "Calibration is the sweet spot: too little margin → shortfalls, "
        "too much → value left on the table",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=160)
    plt.close(fig)
    print(f"  wrote {fname}")


def main() -> None:
    print(f"Setup OK | data source: {DATA_SOURCE} | C_DEG ≈ {C_DEG:.2f} $/MWh")
    print("Trajectories...")
    fig_trajectories("arbitrage", "01_arbitrage_trajectories.png", "Price ($/MWh)")
    fig_trajectories("peak", "02_peak_trajectories.png", "Load (MW)")
    print("Monte Carlo...")
    fig_montecarlo("arbitrage", "03_arbitrage_montecarlo.png", "Net revenue ($/day)")
    fig_montecarlo("peak", "04_peak_montecarlo.png", "Net peak value ($/day)")
    print("Calibration sweet spot...")
    fig_calibration("05_calibration_sweet_spot.png")
    print("Done.")


if __name__ == "__main__":
    main()
