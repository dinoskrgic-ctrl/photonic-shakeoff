"""
kc_selfion_fig.py   (reads selfion_KC.npz, selfion_KP.npz)
  (a,b) validation at gap 2 (KC, KP): qubit population P1 after a leakage event vs f = (Omega - w_e)/alpha
        full two-quantum numerics (markers), spectator numerics (KC), theory with / without exchange, dressed picture
  (c,d) shake-off loss  L = 1 - P1/P1_dressed  vs f, gaps 1-4  (theory lines, full numerics markers)
  (e,f) leakage-removal speed: Markov leakage rate / free-line rate of the leakage rung, gaps 1-4
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

plt.style.use(["science"])
KC_RAMP = ["#9ecae1", "#4a90c8", "#1f5fa8", "#0b2e59"]
KP_RAMP = ["#f9c08f", "#ee9248", "#d9731f", "#7f3b08"]
GREY = "#777777"


def load(lat):
    d = np.load(f"selfion_{lat}.npz")
    out = {}
    for n in [1, 2, 3, 4]:
        if f"meta_{n}" not in d:
            continue
        w_e, w_top, alpha, kappa, Cg, Om_c = d[f"meta_{n}"]
        fi, fu = d[f"fine_{n}"], d[f"full_{n}"]
        fi = fi[fi[:, 0] < w_top]                    # qubit must be inside the gap
        fu = fu[fu[:, 0] < w_top]
        out[n] = dict(w_e=w_e, alpha=alpha, kappa=kappa, fine=fi, full=fu,
                      f_fine=(fi[:, 0] - w_e) / alpha, f_full=(fu[:, 0] - w_e) / alpha)
    return out


data = {"KC": load("KC"), "KP": load("KP")}

fig, axs = plt.subplots(3, 2, figsize=(7.0, 7.6))

# ---------------------------------------------------- (a, b)
for ax, lat, ramp, tag in [(axs[0, 0], "KC", KC_RAMP, "(a) KC"), (axs[0, 1], "KP", KP_RAMP, "(b) KP")]:
    g = data[lat].get(2)
    if g is None:
        continue
    fi, fu = g["fine"], g["full"]
    ax.axvline(0.5, color=GREY, lw=0.5, ls=":")
    ax.plot(g["f_fine"], fi[:, 4], color=ramp[2], lw=1.1, label="theory, with exchange")
    ax.plot(g["f_fine"], fi[:, 5], color=ramp[2], lw=0.8, ls="--", label="theory, no exchange")
    ax.plot(g["f_fine"], fi[:, 7], color=GREY, lw=0.8, ls=":", label="dressed rung picture")
    ax.plot(g["f_full"], fu[:, 4], "s", color=ramp[3], ms=3.5, label=r"full numerics $P_1$")
    if lat == "KC" and np.isfinite(fu[:, 11]).any():
        ax.plot(g["f_full"], fu[:, 11], "D", mfc="white", color=ramp[2], ms=3, label=r"spectator numerics $P_1$")
    ax.plot(g["f_full"], fu[:, 3], "o", mfc="white", color=GREY, ms=3, label=r"full numerics $P_2$")
    ax.plot(g["f_fine"], fi[:, 9], color=GREY, lw=0.6, label=r"theory $P_2$")
    ax.set_xlim(0, 1.25)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"$f=(\Omega-\omega_e)/\alpha$")
    ax.set_ylabel(r"population after leakage event")
    #ax.set_title(f"{tag}, gap 2", fontsize=10)
    ax.text(0.51, 0.04, r"$\alpha/2$", fontsize=10, color=GREY)
axs[0, 0].legend(fontsize=5.2, frameon=False, loc="lower right")

# ---------------------------------------------------- (c, d) shake-off loss
for ax, lat, ramp, tag in [(axs[1, 0], "KC", KC_RAMP, "(c) KC"), (axs[1, 1], "KP", KP_RAMP, "(d) KP")]:
    for n, g in data[lat].items():
        fi, fu = g["fine"], g["full"]
        L = 1 - fi[:, 4] / fi[:, 7]
        sel = (g["f_fine"] < 1.0) & (fi[:, 9] < 0.05) & (fi[:, 10] > 0)     # leakage rung radiating (filter regime)
        ax.plot(g["f_fine"][sel], L[sel], color=ramp[n - 1], lw=1.0, label=f"gap {n}")
        Lf = 1 - fu[:, 4] / fu[:, 8]
        s2 = (g["f_full"] < 1.0) & (fu[:, 10] < 0.05) & (fu[:, 3] < 0.05)   # drop runs whose leakage has not cleared
        ax.plot(g["f_full"][s2], Lf[s2], "o", color=ramp[n - 1], ms=2.8)
    ax.axvline(0.5, color=GREY, lw=0.5, ls=":")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.05, 0.65)
    ax.set_xlabel(r"$f=(\Omega-\omega_e)/\alpha$")
    ax.set_ylabel(r"shake-off loss $1-P_1/P_1^{\rm dressed}$")
    #ax.set_title()#f"{tag}", fontsize=10)
    ax.legend(fontsize=6, frameon=False, loc="upper right", title="lines: theory, dots: numerics\n(runs with $P_2>0.05$ at $T$ omitted)", title_fontsize=5.5)

# ---------------------------------------------------- (e, f) leakage removal speed
for ax, lat, ramp, tag in [(axs[2, 0], "KC", KC_RAMP, "(e) KC"), (axs[2, 1], "KP", KP_RAMP, "(f) KP")]:
    for n, g in data[lat].items():
        fi = g["fine"]
        OmR = fi[:, 3]
        Gtot = fi[:, 10] + fi[:, 11] + fi[:, 12]
        purc = np.where(fi[:, 10] > 0, Gtot / (g["kappa"] * OmR * (OmR - g["alpha"])), np.nan)
        sel = g["f_fine"] < 1.0
        ax.plot(g["f_fine"][sel], purc[sel], color=ramp[n - 1], lw=1.0, label=f"gap {n}")
    ax.axhline(1.0, color=GREY, lw=0.5, ls=":")
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 1e2)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(r"$f=(\Omega-\omega_e)/\alpha$")
    ax.set_ylabel(r"leakage rate / free-line rate")
    #ax.set_title()#f"{tag}", fontsize=10)
    ax.legend(fontsize=6, frameon=False, loc="lower left")
# ---------------------------------------------------- panel labels
import string
for ax, lab in zip(axs.ravel(), string.ascii_lowercase):
    ax.text(-0.07, 1.04, rf"\textbf{{({lab})}}", transform=ax.transAxes,
            ha="left", va="bottom",fontweight="bold", fontsize=12)

fig.tight_layout()
fig.savefig("kc_selfion.pdf")
fig.tight_layout()
fig.savefig("kc_selfion.pdf")
print("saved kc_selfion.pdf")
