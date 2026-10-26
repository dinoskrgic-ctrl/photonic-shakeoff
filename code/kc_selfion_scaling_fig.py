"""
kc_selfion_scaling_fig.py  (reads kc_selfion_scaling.npy from kc_selfion_scaling.py)
(a) loss per leakage event L vs Gamma_free/Omega at fixed detuning d = Omega - w_e   (KC gap 2, kinetic edge)
(b) shake-off ratio L/(1 - Z1) vs Gamma_free/Omega
rows: [s, kappa, d, L, 1-Z1, G_B, G_dir, G_ex, Eb - w_e]
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

plt.style.use(["science"])
RAMP = ["#0b2e59", "#1f5fa8", "#4a90c8", "#9ecae1"]
r = np.load("kc_selfion_scaling.npy")
w_e = 4.0575
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
for col, d in zip(RAMP, [0.01, 0.03, 0.06, 0.10]):
    s = r[:, 2] == d
    x = r[s, 1] * w_e / 2                       # Gamma_free / Omega at the edge
    axs[0].loglog(x, r[s, 3], "o-", color=col, ms=3, lw=1.0, label=fr"$d={d}$")
    axs[1].semilogx(x, r[s, 3] / r[s, 4], "o-", color=col, ms=3, lw=1.0, label=fr"$d={d}$")
xx = np.array([1e-4, 2e-2])
axs[0].loglog(xx, 0.25 * xx / 1e-3 * 0.135, color="#888888", lw=0.6, ls=":")
axs[0].text(4e-4, 5.5e-3, r"$\propto\kappa$", fontsize=7, color="#555555")
axs[0].set_xlabel(r"$\Gamma_{\rm free}/\Omega$")
axs[0].set_ylabel(r"loss per leakage event $L$")
axs[0].legend(fontsize=6, frameon=False, loc="lower right", title=r"$d=\Omega-\omega_e$", title_fontsize=6)
axs[1].axhline(1.0, color="#888888", lw=0.6, ls="--")
axs[1].text(1.3e-4, 0.78, "sudden shake-off", fontsize=6, color="#555555")
axs[1].set_ylim(0, 3.8)
axs[1].set_xlabel(r"$\Gamma_{\rm free}/\Omega$")
axs[1].set_ylabel(r"shake-off ratio $L/(1-Z_1)$")

import string as _s
for _ax, _l in zip(np.ravel(axs), _s.ascii_lowercase):
    _ax.text(-0.12, 1.02, rf"\textbf{{({_l})}}", transform=_ax.transAxes, ha="left", va="bottom", fontsize=10)
fig.tight_layout()
fig.savefig("kc_selfion_scaling.pdf")
print("saved kc_selfion_scaling.pdf")