"""
nl2000_markov_fig.py -> fig_markov.pdf
Markov branching formula vs exact two-excitation dynamics in the Nikolopoulos-Lambropoulos (2000) model.
(a) retention P2/Z of the lower bound state vs two-photon detuning Delta2 (exact: markers; Markov: line)
(b) |exact - Markov| vs Gamma/Delta2, Gamma = total Markov decay rate of the upper level.
Inputs: nl2000_threshold.npy (12-point scan, N=250, w_u=20), nl2000_markov_limit.npy (3 points, N=400-450, w_u=30).
"""
import numpy as np, warnings, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa
from nl2000_reproduce import branching, lower_bound_state
warnings.filterwarnings("ignore")
plt.style.use(["science"])
a = np.load("nl2000_threshold.npy"); b = np.load("nl2000_markov_limit.npy")
d = np.vstack([a[:, :5], b[:, :5]])
yb, Z = lower_bound_state(-1.0)
d12s = np.linspace(0.3, 7.2, 300)
mk = np.array([branching(x, -1.0)[0] / Z for x in d12s])
rows = []
for d12, D2, p1, p2, p3 in d:
    P, _, GB, Gd, Gx, _ = branching(d12, -1.0)
    rows.append([D2, p2 / Z, P / Z, (GB + Gd + Gx) / D2 if D2 > 0 else np.nan])
rows = np.array(rows)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
ax[0].plot(d12s - 1.0, mk, color="#1f5fa8", lw=1.1, label="Markov branching formula")
ax[0].plot(rows[:, 0], rows[:, 1], "o", ms=3.5, color="k", label="exact two-excitation dynamics")
i4 = np.argmin(abs(rows[:, 0] - 0.31))
ax[0].plot(rows[i4, 0], rows[i4, 1], "o", ms=7, mfc="none", mec="#c0392b", mew=1.0)
ax[0].annotate("N\\&L Fig. 4(a)", (rows[i4, 0], rows[i4, 1]), xytext=(12, 8), textcoords="offset points",
               fontsize=6.5, color="#c0392b")
ax[0].axvline(0, color="#888888", lw=0.5, ls="--")
ax[0].set_xlabel(r"two-photon detuning $\Delta^{(2)}$ [$C^{2/3}$]"); ax[0].set_ylabel(r"retention $P_2/Z$")
ax[0].legend(fontsize=6, frameon=False, loc="lower left"); ax[0].set_ylim(0.5, 1.02)
m = rows[:, 0] > 0
b0 = rows[~m, 1].mean()                      # channel-free retention below threshold (non-Markov dressing deficit)
S = (1 - rows[m, 1] / b0) / (1 - rows[m, 2])  # loss suppression factor: exact channel loss / Markov loss
ax[1].semilogx(rows[m, 3], S, "o-", ms=3.5, color="k", lw=0.8)
ax[1].axhline(1, color="#1f5fa8", lw=0.8)
ax[1].axvline(rows[i4, 3], color="#c0392b", lw=0.6, ls=":")
ax[1].text(rows[i4, 3]*1.08, 0.75, "N\\&L 4(a)", fontsize=6.5, color="#c0392b")
ax[1].set_ylim(0, 1.1)
ax[1].set_xlabel(r"$\Gamma/\Delta^{(2)}$ (linewidth / two-photon window)")
ax[1].set_ylabel(r"suppression $S=L_{\rm exact}/L_{\rm Markov}$")

import string as _s
for _ax, _l in zip(np.ravel(ax), _s.ascii_lowercase):
    _ax.text(-0.10, 1.02, rf"\textbf{{({_l})}}", transform=_ax.transAxes, ha="left", va="bottom", fontsize=10)
fig.tight_layout(); fig.savefig("fig_markov.pdf")
np.savetxt("fig_markov_data.txt", rows, header="Delta2  P2/Z_exact  P2/Z_Markov  Gamma/Delta2", fmt="%.4f")
print("b0=%.4f" % b0); print("S:", np.round(S, 3))
print(open("fig_markov_data.txt").read())