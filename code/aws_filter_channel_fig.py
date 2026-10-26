#!/usr/bin/env python3
"""Figure for the Appendix "Filter-band heating channel in a pumped cat qubit" -> fig_aws_cat.pdf.
Top: predicted filter weight A_b(w3) at the leftover-photon frequency w3 = 2w_p - ws - 2wb(w_p), no fit.
Bottom: measured storage parity (200-300 us) of Putterman et al., PRX 15, 011070 (2025).
Requires numpy, matplotlib.  Usage: python3 aws_filter_channel_fig.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from aws_filter_channel import A_b, wb_of, solve_wp, pass_band, wp_meas, parity, WS_RANGE

lo, hi, plo, phi = pass_band()
wp = np.linspace(7.08, 7.92, 600)
fig, ax = plt.subplots(2, 1, figsize=(3.4, 3.6), sharex=True)
for ws, ls in [(WS_RANGE[0], ":"), (WS_RANGE[-1], "-")]:
    wb = wb_of(wp)
    ax[0].semilogy(wp, A_b(2 * wp - ws - 2 * wb, wb), ls, color="C0", lw=1,
                   label=rf"$\bar\omega_s/2\pi={ws}$ GHz")
ax[0].set_ylabel(r"$A_b(\omega_3)$ (1/GHz)")
ax[0].legend(fontsize=6, frameon=False)
cut = [solve_wp(hi, ws) for ws in WS_RANGE]
edge = [solve_wp(phi, ws) for ws in WS_RANGE]
for a in ax:
    a.axvspan(min(cut), max(cut), color="0.5", alpha=0.25, lw=0)
    a.axvspan(min(edge), max(edge), color="C1", alpha=0.25, lw=0)
ax[1].plot(wp_meas, parity, "o-", color="k", ms=2.5, lw=0.8)
ax[1].set_ylabel("storage parity")
ax[1].set_xlabel(r"pump frequency $\omega_p/2\pi$ (GHz)")
for a in ax:
    a.tick_params(labelsize=7)
    a.yaxis.label.set_size(8)
ax[1].xaxis.label.set_size(8)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_aws_cat.pdf")
fig.savefig(out)
fig.savefig(out.replace(".pdf", ".png"), dpi=200)
print("wrote", out)
