"""
fig_schematic.py -> fig_schematic.pdf   (Fig. 1; no data needed)
(a) Level scheme: qubit |1> bound in the gap above the band edge w_e; a leakage event |2> decays either by one band photon
    to the bound state B1 (channel R) or by two band photons to the ground state (channel S), w_k + w_l = 2 Omega - alpha.
(b) Design map for a transmon above a lower gap edge w_e: channel S open for Omega - w_e < alpha/2 (Eq. 6);
    clean filter window alpha/2 < Omega - w_e < alpha; above alpha the leakage line is itself in the gap.
    Overlay: published passive filter of Thorbeck et al. (arXiv:2411.04101): edge 4.5 GHz, alpha/2pi = 325 MHz,
    stated range 4.6-4.8 GHz, optimum 4.745 GHz.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
plt.style.use(["science"])
BAND, GAP, BLUE, RED, GREY = "#dbe7f3", "#ffffff", "#1f5fa8", "#c0392b", "#666666"

def wavy(ax, x0, y0, x1, y1, color, n=7, amp=0.035, lw=1.0):
    t = np.linspace(0, 1, 200); x = x0 + (x1 - x0) * t; y = y0 + (y1 - y0) * t
    dx, dy = x1 - x0, y1 - y0; L = np.hypot(dx, dy); nx, ny = -dy / L, dx / L
    w = amp * np.sin(2 * np.pi * n * t)
    ax.plot(x + nx * w, y + ny * w, color=color, lw=lw)
    ax.annotate("", xy=(x1, y1), xytext=(x1 - 0.06 * dx, y1 - 0.06 * dy),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=7))

fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw=dict(width_ratios=[1.1, 1]))
ax = axs[0]
we, Om, al = 1.0, 1.12, 0.30                 # schematic units
E2 = 2 * Om - al
# photon-frequency axis on the right: band below w_e, gap above
ax.add_patch(plt.Rectangle((2.55, 0.0), 0.35, we, color=BAND, lw=0))
ax.text(2.725, 0.5 * we, "band", rotation=90, ha="center", va="center", fontsize=7, color=BLUE)
ax.text(2.725, we + 0.25, "gap", rotation=90, ha="center", va="center", fontsize=7, color=GREY)
ax.plot([2.55, 2.9], [we, we], color=BLUE, lw=0.8); ax.text(2.93, we, r"$\omega_e$", va="center", fontsize=7)
ax.plot([2.55, 2.9], [Om, Om], color="k", lw=0.6, ls=":"); ax.text(2.93, Om, r"$\Omega$", va="center", fontsize=7)
# ladder
lev = {0: 0.0, 1: Om, 2: E2}
for n, y in lev.items():
    ax.plot([0.15, 0.75], [y, y], color="k", lw=1.2)
ax.text(0.0, 0.0, r"$|0\rangle$", ha="right", va="center", fontsize=8)
ax.text(0.0, Om, r"$|1\rangle\!\to\!|B_1\rangle$", ha="right", va="center", fontsize=8)
ax.text(0.0, E2, r"$|2\rangle$", ha="right", va="center", fontsize=8)
# photon cloud of the bound state
for k in range(5):
    ax.add_patch(matplotlib.patches.Ellipse((0.45, Om), 0.75 + 0.18 * k, 0.07 + 0.02 * k, fill=False,
                                            ec=BLUE, lw=0.4, alpha=0.6 - 0.1 * k))
# channel R
ax.annotate("", xy=(0.95, Om), xytext=(0.95, E2), arrowprops=dict(arrowstyle="-|>", color="k", lw=0.9, mutation_scale=8))
wavy(ax, 0.98, 0.5 * (Om + E2), 1.75, 0.5 * (Om + E2) - 0.05, BLUE)
ax.text(1.02, 0.5 * (Om + E2) + 0.10, r"R: $E_2-E_{b1}$", fontsize=7, color=BLUE)
# channel S
ax.annotate("", xy=(1.95, 0.0), xytext=(1.95, E2), arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.9, mutation_scale=8))
wavy(ax, 1.98, 0.70 * E2, 2.48, 0.62 * E2, RED, n=5)
wavy(ax, 1.98, 0.35 * E2, 2.48, 0.27 * E2, RED, n=5)
ax.text(1.88, 0.42, "S: two band\nphotons,\n" + r"$\omega_k+\omega_l=2\Omega-\alpha$", fontsize=7, color=RED, ha="right")
ax.set_xlim(-0.75, 3.15); ax.set_ylim(-0.25, E2 + 0.15); ax.axis("off")
#ax.set_title("(a)", fontsize=12)

ax = axs[1]
x = np.linspace(0, 450, 400); amax = 420
A = np.linspace(100, amax, 400)
ax.fill_betweenx(A, 0, A / 2, color="#f3d6d3", lw=0)
ax.fill_betweenx(A, A / 2, A, color="#d6e4f3", lw=0)
ax.fill_betweenx(A, A, 450, color="#eeeeee", lw=0)
ax.plot(A / 2, A, color=RED, lw=1.0); ax.plot(A, A, color="k", lw=0.8)
ax.text(40, 380, "channel S\nopen", fontsize=7, color=RED)
ax.text(262, 372, "clean\nfilter", fontsize=7, color=BLUE)
ax.text(355, 160, "leakage line\nin gap", fontsize=7, color=GREY)
# Thorbeck et al. device
ax.plot([100, 300], [325, 325], color="k", lw=2.2, solid_capstyle="butt")
ax.plot([245], [325], marker="*", ms=7, color="#e0a100", mec="k", mew=0.4)
ax.annotate("Thorbeck et al.:\nstated range,\noptimum (star)", xy=(110, 325), xytext=(150, 200), fontsize=6,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=GREY))
ax.set_xlim(0, 450); ax.set_ylim(100, amax)
ax.set_xlabel(r"$(\Omega-\omega_e)/2\pi$ [MHz]"); ax.set_ylabel(r"$\alpha/2\pi$ [MHz]")
#ax.set_title(r"(b) design map: $\Omega<\omega_e+\alpha/2$", fontsize=8)
import string
for ax, lab in zip(axs.ravel(), string.ascii_lowercase):
    ax.text(-0.07, 1.04, rf"\textbf{{({lab})}}", transform=ax.transAxes,
            ha="left", va="bottom",fontweight="bold", fontsize=12)

fig.tight_layout(); fig.savefig("fig_schematic.pdf"); print("saved fig_schematic.pdf")
