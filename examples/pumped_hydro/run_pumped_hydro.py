"""Pumped-hydro energy storage — first-principles (white-box) port-Hamiltonian twin.

Pumped hydro is the dominant grid-scale storage technology (~95% of the world's
installed long-duration storage). Here it is a two-reservoir port-Hamiltonian
system: energy is stored as gravitational potential energy, and a reversible
pump-turbine moves water between reservoirs.

This is a *white-box* grid-storage twin — the counterpart to the battery
State-of-Health example, which is grey-box. The storage medium decides the model
class: mechanical/hydraulic storage has an exact first-principles energy, so the
twin can be validated against closed-form answers (energy conservation and
round-trip efficiency); electrochemical aging cannot.

This script does the calculations:
  1. Run a charge -> idle -> discharge cycle (pump water up on cheap power, hold,
     generate on demand).
  2. Validate energy conservation: while idle the stored energy is essentially
     constant (passivity; a perfect store as the penstock valve closes).
  3. Validate the round-trip efficiency against the closed form eta_p * eta_t, and
     check the energy books close (stored energy = eta_p * electrical-in on charge;
     electrical-out = eta_t * stored energy on discharge).
  4. Show the passive case: with the penstock valve open and the pump off, water
     runs downhill and the stored energy is dissipated (dH/dt <= 0).

Outputs: figures/*.png and figure_data/*.csv  (seaborn).
Requires: numpy, scipy, matplotlib, seaborn.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from otwin.systems import pumped_hydro  # noqa: E402

sns.set_theme(style="whitegrid", context="talk")
BLUE, INK, ACCENT, GREEN = "#1C4E73", "#1B2430", "#B5651D", "#2E7D32"

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
DATA = HERE / "figure_data"
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

# ---- plant parameters (a ~0.8 GWh, 300 m head pumped-hydro plant) ----
RHO, G = 1000.0, 9.81
A_U, A_L, Z_U = 5.0e4, 5.0e6, 300.0
ETA_P, ETA_T = 0.90, 0.90  # pump / turbine conversion efficiencies
P_SET = 100.0e6  # charge / discharge power (W) = 100 MW
plant = pumped_hydro(A_u=A_U, A_l=A_L, z_u=Z_U, g=G, rho=RHO)

# ---- initial state: upper reservoir nearly empty, lower full ----
V_U0, V_L0 = 2.0e5, 1.2e7
x0 = np.array([V_U0, V_L0])


def net_head(x: np.ndarray) -> float:
    """Differential head the pump-turbine works against (m)."""
    return float((Z_U + x[0] / A_U) - (x[1] / A_L))


def flow_from_power(x: np.ndarray, p_elec: float) -> float:
    """Convert commanded electrical power (W) to pump-turbine flow (m^3/s).

    Charging (p>0): hydraulic power into the store = eta_p * p_elec.
    Generating (p<0): electrical out = eta_t * hydraulic power extracted.
    """
    head = net_head(x)
    if p_elec > 0:  # pump up
        return ETA_P * p_elec / (RHO * G * head)
    if p_elec < 0:  # generate down
        return p_elec / (ETA_T * RHO * G * head)
    return 0.0


def integrate_phase(x_start: np.ndarray, p_elec: float, t_span, event=None):
    """Integrate the plant over one constant-power phase."""

    def dyn(t: float, x: np.ndarray) -> np.ndarray:
        q = flow_from_power(x, p_elec)
        return plant.dynamics(x, np.array([q]))

    sol = solve_ivp(
        dyn, t_span, x_start, max_step=60.0, rtol=1e-8, atol=1e-3,
        dense_output=True, events=event,
    )
    return sol


# ---- schedule: charge 8 h, idle 3 h, discharge until back to start ----
HR = 3600.0
charge = integrate_phase(x0, +P_SET, (0.0, 8 * HR))
t_c_end = charge.t[-1]
idle = integrate_phase(charge.y[:, -1], 0.0, (t_c_end, t_c_end + 3 * HR))
t_i_end = idle.t[-1]


def back_to_start(t: float, x: np.ndarray) -> float:
    return x[0] - V_U0  # stop when the upper reservoir returns to its start


back_to_start.terminal = True
back_to_start.direction = -1.0
discharge = integrate_phase(
    idle.y[:, -1], -P_SET, (t_i_end, t_i_end + 12 * HR), event=back_to_start
)

# ---- stitch trajectory ----
t = np.concatenate([charge.t, idle.t, discharge.t])
X = np.concatenate([charge.y, idle.y, discharge.y], axis=1)
power = np.concatenate([
    np.full(charge.t.size, P_SET),
    np.zeros(idle.t.size),
    np.full(discharge.t.size, -P_SET),
])
energy = np.array([plant.energy(X[:, i]) for i in range(X.shape[1])])
heads = np.array([net_head(X[:, i]) for i in range(X.shape[1])])

# ---- energy accounting ----
t_d_end = discharge.t[-1]
E_in = P_SET * (t_c_end - 0.0)  # electrical energy in (charge)
E_out = P_SET * (t_d_end - t_i_end)  # electrical energy out (discharge)
H0, H_peak, H_end = energy[0], plant.energy(charge.y[:, -1]), energy[-1]
dH_charge = H_peak - H0
dH_discharge = plant.energy(discharge.y[:, -1]) - plant.energy(idle.y[:, -1])

rt_numeric = E_out / E_in
rt_closed = ETA_P * ETA_T
# bookkeeping: stored = eta_p * E_in (charge);  E_out = eta_t * |stored released|
book_charge = dH_charge / (ETA_P * E_in)            # -> 1.0
book_discharge = E_out / (ETA_T * abs(dH_discharge))  # -> 1.0
idle_drift = (plant.energy(idle.y[:, -1]) - H_peak) / H_peak  # fractional

# ---- passivity: open penstock valve, pump off -> energy runs downhill ----
leaky = pumped_hydro(A_u=A_U, A_l=A_L, z_u=Z_U, R_penstock=5.0e6, g=G, rho=RHO)


def dyn_leak(t: float, x: np.ndarray) -> np.ndarray:
    return leaky.dynamics(x, np.array([0.0]))


leak = solve_ivp(
    dyn_leak, (0.0, 24 * HR), np.array([1.0e6, V_L0]), max_step=120.0,
    rtol=1e-8, atol=1e-3, dense_output=True,
)
leak_E = np.array([leaky.energy(leak.y[:, i]) for i in range(leak.y.shape[1])])
leak_mono = bool(np.all(np.diff(leak_E) <= 1e-6))

print("Pumped-hydro — calculations")
print(f"  usable energy stored this cycle:  {dH_charge / 3.6e9:.1f} MWh")
print(f"  net head (start -> peak):          {heads[0]:.1f} -> {net_head(charge.y[:, -1]):.1f} m")
print(f"  round-trip efficiency (numeric):   {rt_numeric:.4f}")
print(f"  round-trip efficiency (closed form eta_p*eta_t): {rt_closed:.4f}")
print(f"  energy books close: charge {book_charge:.4f}, discharge {book_discharge:.4f} (->1)")
print(f"  idle self-discharge over 3 h:      {idle_drift * 100:.3f}%")
print(f"  open-valve passivity (energy monotonically non-increasing): {leak_mono}")

# ---- figure 1: the cycle (power + state of charge) ----
soc = (X[0] - V_U0) / (plant_max := max(X[0].max() - V_U0, 1.0)) * 100.0
th = t / HR
fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
ax[0].plot(th, power / 1e6, color=BLUE, lw=2.4)
ax[0].axhline(0, color="gray", lw=1)
ax[0].fill_between(th, power / 1e6, 0, where=power > 0, color=GREEN, alpha=0.15, label="charging (pump)")
ax[0].fill_between(th, power / 1e6, 0, where=power < 0, color=ACCENT, alpha=0.15, label="generating (turbine)")
ax[0].set_ylabel("grid power (MW)")
ax[0].set_title("Pumped-hydro storage: charge → hold → generate")
ax[0].legend(fontsize=10, loc="upper right")
ax[1].plot(th, soc, color=INK, lw=2.4)
ax[1].set_ylabel("state of charge (%)")
ax[1].set_xlabel("time (hours)")
fig.tight_layout()
fig.savefig(FIG / "pumped_hydro_cycle.png", dpi=160)
plt.close(fig)

# ---- figure 2: stored energy + round-trip validation ----
fig, axe = plt.subplots(figsize=(10, 5))
axe.plot(th, energy / 3.6e9, color=INK, lw=2.6)
axe.axvspan(0, t_c_end / HR, color=GREEN, alpha=0.07, label="charge")
axe.axvspan(t_c_end / HR, t_i_end / HR, color="gray", alpha=0.07, label="idle (energy held)")
axe.axvspan(t_i_end / HR, t_d_end / HR, color=ACCENT, alpha=0.07, label="discharge")
axe.set_xlabel("time (hours)")
axe.set_ylabel("stored energy H (MWh)")
axe.set_title("Energy is conserved while idle; round-trip matches the closed form")
axe.legend(fontsize=10, loc="upper right")
txt = (
    f"round-trip η = {rt_numeric:.3f}\n"
    f"closed form η_p·η_t = {rt_closed:.3f}\n"
    f"books close to {abs(1 - book_charge):.1e} / {abs(1 - book_discharge):.1e}\n"
    f"idle self-discharge (3 h): {idle_drift * 100:.3f}%"
)
axe.text(
    0.02, 0.04, txt, transform=axe.transAxes, fontsize=11, va="bottom",
    bbox=dict(boxstyle="round", fc="white", ec=BLUE, alpha=0.9),
)
fig.tight_layout()
fig.savefig(FIG / "pumped_hydro_energy.png", dpi=160)
plt.close(fig)

# ---- export figure data ----
pd.DataFrame({
    "t_hours": th, "power_MW": power / 1e6, "V_upper_m3": X[0], "V_lower_m3": X[1],
    "net_head_m": heads, "energy_MWh": energy / 3.6e9,
}).to_csv(DATA / "pumped_hydro_timeseries.csv", index=False)
pd.DataFrame({
    "quantity": ["round_trip_numeric", "round_trip_closed_form", "eta_pump",
                 "eta_turbine", "usable_MWh", "idle_drift_pct_3h"],
    "value": [rt_numeric, rt_closed, ETA_P, ETA_T, dH_charge / 3.6e9, idle_drift * 100],
}).to_csv(DATA / "pumped_hydro_validation.csv", index=False)

print("wrote figures/ and figure_data/")
