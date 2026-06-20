"""Generate the technical assets for the deck: equation images, a short/medium/
long-horizon zoom, and a residual (predictive-distribution) diagnostic.

Run:  python make_science_assets.py
Outputs to ../figures/  (eq_*.png, 10_zoom_horizons.png, 11_residual_diag.png)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk", palette="deep")

HERE = Path(__file__).resolve().parent
EX = HERE.parent
FIG = EX / "figures"
FIG.mkdir(exist_ok=True)

DEEP, TEAL, MINT, INK = "#065A82", "#1C7293", "#02A37A", "#1B2430"

# ---- load the model module and run the fleet ----
spec = importlib.util.spec_from_file_location("rb", EX / "run_battery_soh.py")
rb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rb)
R = rb.run()
HERO = rb.HERO
LEVEL = rb.LEVEL
Z = 1.6448536269514722  # norm.ppf(0.95)


# ============================================================
# 1. Equation images (matplotlib mathtext, transparent background)
# ============================================================
def eq(name: str, tex: str, fontsize: int = 26, color: str = INK, pad: float = 0.30):
    fig = plt.figure(figsize=(0.1, 0.1))
    t = fig.text(0.5, 0.5, tex, fontsize=fontsize, color=color,
                 ha="center", va="center")
    fig.savefig(FIG / f"{name}.png", dpi=300, transparent=True,
                bbox_inches="tight", pad_inches=pad)
    plt.close(fig)


# Degradation prior (empirical fade law, SEI sqrt + linear aging motivation)
eq("eq_prior",
   r"$Q(n)=Q_0-k_{\mathrm{SEI}}\sqrt{n}-k_{\mathrm{cyc}}\,n"
   r"\quad\Rightarrow\quad f_\theta(n)=\mathrm{SoH}_{n_0}\,e^{-a\,(n-n_0)},\ \ a\geq 0$")

# Hybrid decomposition
eq("eq_hybrid",
   r"$\widehat{\mathrm{SoH}}(n)\;=\;f_\theta(n)\;+\;g_\phi(n)\,e^{-(n-n_0)/\tau}$")

# Port-Hamiltonian structure (the library's core; structure constrains extrapolation)
eq("eq_phs",
   r"$\dot{x}=(J(x)-R(x))\nabla H(x)+g(x)\,u,"
   r"\qquad \dot{H}=-\nabla H^{\top}R\,\nabla H+y^{\top}u\ \leq\ y^{\top}u$",
   fontsize=22)

# Calibrated (conformal) band
eq("eq_band",
   r"$[\,\ell(n),\,u(n)\,]=\widehat{\mathrm{SoH}}(n)\pm z_\alpha\,\sigma(h),"
   r"\quad \sigma(h)=s_0+s_1 h,\quad h=n-n_0$")

# Metrics block
eq("eq_metrics",
   r"$\mathrm{MASE}=\frac{\mathrm{MAE}}{\frac{1}{N-1}\sum_t|y_t-y_{t-1}|}"
   r"\qquad \mathrm{PICP}=\frac{1}{N}\sum_i \mathbf{1}\{\ell_i\leq y_i\leq u_i\}$",
   fontsize=22)

eq("eq_crps",
   r"$\mathrm{CRPS}(F,y)=\int_{-\infty}^{\infty}"
   r"(F(z)-\mathbf{1}\{z\geq y\})^2\,dz$", fontsize=22)

print("equation images written")


# ============================================================
# 2. Short / medium / long-horizon zoom (hero cell)
# ============================================================
r = R[HERO]
n_te, y, mean, lo, hi = r["n_te"], r["soh_te"], r["mean"], r["lo"], r["hi"]
nt = n_te.size
thirds = [(0, nt // 3, "Short term"), (nt // 3, 2 * nt // 3, "Medium term"),
          (2 * nt // 3, nt, "Long term")]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, (i0, i1, label) in zip(axes, thirds):
    sl = slice(i0, i1)
    ax.fill_between(n_te[sl], lo[sl], hi[sl], color=MINT, alpha=0.20,
                    label=f"{int(LEVEL*100)}% interval")
    ax.plot(n_te[sl], mean[sl], color=TEAL, lw=2.2, label="Hybrid forecast")
    ax.plot(n_te[sl], y[sl], "o", ms=5, color=INK, label="Observed")
    h0, h1 = int(n_te[sl][0] - r["n_tr"][-1]), int(n_te[sl][-1] - r["n_tr"][-1])
    ax.set_title(f"{label}  (h = {h0}-{h1} cycles)", fontsize=14)
    ax.set_xlabel("Cycle")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("State of Health")
axes[0].legend(fontsize=9, loc="best")
fig.suptitle(f"{HERO}: forecast resolved by horizon (zoom-in)", fontsize=16)
fig.tight_layout()
fig.savefig(FIG / "10_zoom_horizons.png", dpi=200)
plt.close(fig)
print("zoom figure written")


# ============================================================
# 3. Residual / predictive-distribution diagnostic (fleet)
# ============================================================
z_all = []
for rr in R.values():
    sigma = (rr["hi"] - rr["mean"]) / Z
    sigma = np.where(sigma > 1e-6, sigma, 1e-6)
    z_all.append((rr["soh_te"] - rr["mean"]) / sigma)
z = np.concatenate(z_all)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
# (a) standardized residual histogram vs N(0,1)
ax = axes[0]
ax.hist(z, bins=24, density=True, color=TEAL, alpha=0.65, edgecolor="white")
xs = np.linspace(-4, 4, 200)
ax.plot(xs, np.exp(-xs**2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1.6,
        label=r"$\mathcal{N}(0,1)$")
ax.set_xlabel("Standardized residual  z = (y - mean)/σ")
ax.set_ylabel("Density"); ax.set_title("Are the errors the size we claimed?")
ax.legend(fontsize=10); ax.grid(alpha=0.3)
# (b) Q-Q plot
from scipy.stats import norm

ax = axes[1]
zs = np.sort(z)
p = (np.arange(1, zs.size + 1) - 0.5) / zs.size
ax.plot(norm.ppf(p), zs, "o", ms=3, color=DEEP, alpha=0.6)
lim = [min(zs.min(), -3), max(zs.max(), 3)]
ax.plot(lim, lim, "k--", lw=1.2)
ax.set_xlabel("Theoretical quantiles  N(0,1)")
ax.set_ylabel("Empirical quantiles")
ax.set_title("Q–Q: predictive distribution check")
ax.grid(alpha=0.3)
fig.suptitle("Predictive-distribution diagnostics (fleet, pooled)", fontsize=15)
fig.tight_layout()
fig.savefig(FIG / "11_residual_diag.png", dpi=200)
plt.close(fig)
print("residual diagnostic written")
