"""
fig_nl2000_repro.py -> fig_nl2000_repro.pdf  (reads nl2000_reproduce.npz from nl2000_reproduce.py)
Reproduction of Nikolopoulos & Lambropoulos, PRA 61, 053812 (2000), Figs. 4(a), 4(b), 5(a): populations of the ladder
atom |1> (top), |2> (middle), |3> (ground) vs time, with their discretization (N = 150 modes, w_u = 10, C2 = 1.5 C1).
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
plt.style.use(["science"])
C = ["#1f5fa8", "#d9731f", "#2f9e6e"]
d = np.load("nl2000_reproduce.npz")
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.2), sharey=True)
panels = [("Fig4a", r"(a) N\&L Fig.~4(a): $\delta_{12}>0$, $\delta_{23}<0$"),
          ("Fig4b", r"(b) N\&L Fig.~4(b): both in band"),
          ("Fig5a", r"(c) N\&L Fig.~5(a): $\delta_{12}<0$, $\delta_{23}>0$")]
for ax, (key, title) in zip(axs, panels):
    t, p1, p2, p3 = d[key]
    ax.plot(t, p1, color=C[0], lw=1.0, label=r"$|1\rangle$ (top)")
    ax.plot(t, p2, color=C[1], lw=1.0, ls="--", label=r"$|2\rangle$ (middle)")
    ax.plot(t, p3, color=C[2], lw=1.0, ls="-.", label=r"$|3\rangle$ (ground)")
    ax.set_ylim(0, 1.0); ax.set_xlim(0, t.max())
    ax.set_xlabel(r"time [$C_1^{-2/3}$]")#; ax.set_title(title, fontsize=7)
axs[0].set_ylabel("population")
axs[0].legend(fontsize=6, frameon=False, loc="center right")

import string
for ax, lab in zip(axs.ravel(), string.ascii_lowercase):
    ax.text(-0.07, 1.04, rf"\textbf{{({lab})}}", transform=ax.transAxes,
            ha="left", va="bottom",fontweight="bold", fontsize=12)

fig.tight_layout(); fig.savefig("fig_nl2000_repro.pdf"); print("saved fig_nl2000_repro.pdf")
