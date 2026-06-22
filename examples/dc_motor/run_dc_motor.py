"""DC motor — first-principles (white-box) port-Hamiltonian twin.

Reference: van der Schaft & Jeltsema (2014), "Port-Hamiltonian Systems Theory:
An Introductory Overview", Example 2.5, Eq. (2.30).

State x = [phi, p]: inductor flux-linkage and rotor angular momentum.
Input u = [V]: applied voltage.  Output y = I = phi/L: armature current.
Energy H = phi^2/(2L) + p^2/(2*inertia).

This script does the calculations:
  1. Spin-up under a constant voltage, then cut the voltage and coast down.
  2. Validate the model against the closed-form steady state (the analytic
     omega_ss, I_ss a port-Hamiltonian DC motor must converge to).
  3. Check passivity: with V = 0 the stored energy is non-increasing.

Outputs: figures/*.png and figure_data/*.csv  (seaborn).
Requires: numpy, scipy, matplotlib, seaborn.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from otwin import DigitalTwin  # noqa: E402
from otwin.systems import dc_motor  # noqa: E402

sns.set_theme(style="whitegrid", context="talk")
BLUE, INK, ACCENT, GREEN = "#1C4E73", "#1B2430", "#B5651D", "#2E7D32"

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
DATA = HERE / "figure_data"
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

# ---- parameters (book-style linear DC motor) ----
L, INERTIA, RE, B, K = 0.5, 0.01, 1.0, 0.1, 0.5
V_ON = 10.0  # applied voltage during spin-up
motor = dc_motor(L=L, inertia=INERTIA, Re=RE, b=B, K=K)
twin = DigitalTwin(model=motor)

# ---- time grid: V on for the first half, off for the second ----
T_END, N = 4.0, 800
t = np.linspace(0.0, T_END, N)
t_switch = T_END / 2
V = np.where(t < t_switch, V_ON, 0.0)
u = V.reshape(-1, 1)
x0 = np.array([0.0, 0.0])  # start de-energised, at rest

# ---- forecast (structure-preserving) ----
fc = twin.forecast(x0, t, u)
x = fc["x"]
phi, p = x[:, 0], x[:, 1]
current = phi / L  # I = dH/dphi
omega = p / INERTIA  # angular velocity = dH/dp
energy = np.array([motor.energy(xi) for xi in x])

# ---- 1) closed-form steady state under constant V (validation target) ----
# At steady state:  0 = -Re*I - K*omega + V ,  0 = K*I - b*omega
#   => omega_ss = V*K / (Re*b + K^2) ,  I_ss = V*b / (Re*b + K^2)
denom = RE * B + K**2
omega_ss = V_ON * K / denom
I_ss = V_ON * B / denom

# numeric values reached at the end of the spin-up phase
i_end_on = np.searchsorted(t, t_switch) - 1
omega_num = omega[i_end_on]
I_num = current[i_end_on]
err_omega = abs(omega_num - omega_ss) / abs(omega_ss)
err_I = abs(I_num - I_ss) / abs(I_ss)

# ---- 2) passivity: with V = 0 (coast-down) energy is non-increasing ----
off = t >= t_switch
e_off = energy[off]
mono = bool(np.all(np.diff(e_off) <= 1e-9))
drift_off = (e_off[-1] - e_off[0]) / e_off[0] * 100

# ---- 3) power-balance identity at sample points: dH/dt = -gradH^T R gradH + y u
pb_err = 0.0
for i in range(0, N, max(1, N // 50)):
    pb = motor.power_balance(x[i], u[i])
    pb_err = max(pb_err, abs(pb["dH_dt"] - (pb["dissipated"] + pb["supplied"])))

print("DC motor — calculations")
print(f"  steady state (closed form):  omega_ss = {omega_ss:.4f} rad/s   I_ss = {I_ss:.4f} A")
print(f"  steady state (numeric):      omega    = {omega_num:.4f} rad/s   I    = {I_num:.4f} A")
print(f"  relative error:              omega {err_omega:.2%}   I {err_I:.2%}")
print(f"  passivity (V=0): energy monotonically non-increasing = {mono}  (drift {drift_off:.1f}%)")
print(f"  power-balance identity max error: {pb_err:.2e}")

# ---- figures ----
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
ax[0].plot(t, omega, color=BLUE, lw=2.4, label="ω(t)")
ax[0].axhline(omega_ss, ls="--", color=GREEN, lw=1.8, label=f"analytic ω_ss = {omega_ss:.2f}")
ax[0].axvline(t_switch, ls=":", color="gray")
ax[0].set_xlabel("time (s)")
ax[0].set_ylabel("angular velocity ω (rad/s)")
ax[0].set_title("Spin-up, then coast-down (V off)")
ax[0].legend(fontsize=10, loc="best")
ax[1].plot(t, current, color=ACCENT, lw=2.4, label="I(t)")
ax[1].axhline(I_ss, ls="--", color=GREEN, lw=1.8, label=f"analytic I_ss = {I_ss:.2f}")
ax[1].axvline(t_switch, ls=":", color="gray")
ax[1].set_xlabel("time (s)")
ax[1].set_ylabel("armature current I (A)")
ax[1].set_title("Numeric matches the closed-form steady state")
ax[1].legend(fontsize=10, loc="best")
fig.suptitle("DC motor (port-Hamiltonian): validated against the analytic steady state", fontsize=15)
fig.tight_layout()
fig.savefig(FIG / "dc_motor_response.png", dpi=160)
plt.close(fig)

fig, axe = plt.subplots(figsize=(9, 4.4))
axe.plot(t, energy, color=INK, lw=2.4)
axe.axvspan(0, t_switch, color=BLUE, alpha=0.06, label="V on")
axe.axvspan(t_switch, T_END, color=ACCENT, alpha=0.06, label="V off — coast-down")
axe.set_xlabel("time (s)")
axe.set_ylabel("stored energy H (J)")
axe.set_title("With V off, energy is monotonically non-increasing (passivity)")
axe.legend(fontsize=10, loc="best")
fig.tight_layout()
fig.savefig(FIG / "dc_motor_energy.png", dpi=160)
plt.close(fig)

# ---- export figure data ----
pd.DataFrame(
    {"t": t, "V": V, "current_A": current, "omega_rad_s": omega, "energy_J": energy}
).to_csv(DATA / "dc_motor_timeseries.csv", index=False)
pd.DataFrame(
    {
        "quantity": ["omega_ss", "I_ss"],
        "closed_form": [omega_ss, I_ss],
        "numeric": [omega_num, I_num],
        "rel_error": [err_omega, err_I],
    }
).to_csv(DATA / "dc_motor_steady_state.csv", index=False)

print("wrote figures/ and figure_data/")
