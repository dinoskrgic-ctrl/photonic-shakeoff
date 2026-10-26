#!/usr/bin/env python3
"""Filter-band heating channel in the pumped cat qubit of Putterman et al., PRX 15, 011070 (2025), arXiv:2409.17556.

Reproduces every number in Sec. IV ("pumped filtered baths") and Appendix "Filter-band heating channel in a pumped
cat qubit" of the shake-off paper. Nothing is fitted.

Process: the two-pump heating 2 w_p -> ws + 3 wb identified by the authors. With the buffer hybridized with its
band-pass filter, the dominant configuration puts two photons on the buffer resonance and one into the filter
continuum at
        w3 = 2 w_p - ws - 2 wb(w_p)
(the same band-sum kinematics as photonic shake-off). The channel is open iff w3 lies in the pass band, and its rate is
        Gamma(w_p) ∝ A_b(w3),  A_b(w) = (kb/2pi) T(w)/T(wb) / ((w - wb)^2 + (kb/2)^2),  T = 10^(S21/10).
Inputs (aws_cat_data/, digitized from the figure files of the arXiv source, see README there):
    filter_S21.txt                     measured filter transmission (uncalibrated outside 2.5-4 GHz)
    parity_late_vs_pump.txt            storage parity averaged over 200-300 us after pump turn-on
    buffer_freq_measured_vs_pump.txt   pumped buffer frequency (S21 minimum) per pump column
    ws = 5.35-5.38 GHz: bare storage frequency, or the value implied by the authors' Floquet model.

Requires numpy only.  Usage: python3 aws_filter_channel.py
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "aws_cat_data")
KB = 10.7e-3                       # buffer linewidth kappa_b/2pi (GHz), quoted in the paper
WS_RANGE = (5.35, 5.36, 5.37, 5.38)
EDGE_DB = 10.0                     # pass band: S21 within 10 dB of the in-band median

s21 = np.loadtxt(os.path.join(D, "filter_S21.txt"))
par = np.loadtxt(os.path.join(D, "parity_late_vs_pump.txt"))
bm = np.loadtxt(os.path.join(D, "buffer_freq_measured_vs_pump.txt"))
fS, dS = s21[:, 0], s21[:, 1]
wp_meas, parity = par[:, 0], par[:, 1]
deficit = 1.0 - parity / parity[wp_meas > 7.45].mean()      # normalized to the clean-region parity


def running_median(y, half=2):
    return np.array([np.median(y[max(0, i - half):i + half + 1]) for i in range(len(y))])


def wb_of(wp, smooth=True):
    y = running_median(bm[:, 1]) if smooth else bm[:, 1]
    return np.interp(wp, bm[:, 0], y)


def S21(w):
    return np.interp(w, fS, dS)


def A_b(w, wb):
    T = 10 ** (S21(w) / 10) / 10 ** (S21(wb) / 10)
    return (KB / (2 * np.pi)) * T / ((w - wb) ** 2 + (KB / 2) ** 2)


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return np.corrcoef(rx, ry)[0, 1]


def solve_wp(w3_target, ws, smooth=True):
    """Pump frequency at which w3 = w3_target, using the measured buffer frequency (fixed-point iteration)."""
    wp = 7.3
    for _ in range(50):
        wp = 0.5 * (w3_target + ws + 2 * wb_of(wp, smooth))
    return wp


def pass_band():
    mid = (fS > 2.7) & (fS < 3.4)
    thr = np.median(dS[mid]) - EDGE_DB
    inb = fS[(dS > thr) & (fS > 2.4) & (fS < 3.8)]
    lo, hi = inb.min(), inb.max()
    m = (fS > 3.3) & (fS < hi)
    peak_hi = fS[m][np.argmax(dS[m])]
    m = (fS > lo) & (fS < 2.8)
    peak_lo = fS[m][np.argmax(dS[m])]
    return lo, hi, peak_lo, peak_hi


if __name__ == "__main__":
    lo, hi, plo, phi = pass_band()
    print(f"Filter pass band ({EDGE_DB:.0f} dB below in-band median): {lo:.3f}-{hi:.3f} GHz;"
          f" edge peaks {plo:.3f}, {phi:.3f} GHz")
    sel = (bm[:, 0] > 7.1) & (bm[:, 0] < 7.45)
    print(f"Measured pumped buffer frequency, 7.10-7.45 GHz pump: {bm[sel,1].min():.3f}-{bm[sel,1].max():.3f} GHz"
          f" (5-column median {running_median(bm[:,1])[sel].min():.3f}-{running_median(bm[:,1])[sel].max():.3f})")

    print("\nKinematic landmarks, pump frequency w_p/2pi (GHz), for ws = 5.35 ... 5.38 GHz:")
    rows = [("discrete resonance  w3 = wb", None), ("upper-edge mode     w3 = %.3f" % phi, phi),
            ("upper cutoff        w3 = %.3f" % hi, hi), ("lower cutoff        w3 = %.3f" % lo, lo)]
    for name, target in rows:
        vals = []
        for ws in WS_RANGE:
            for sm in (True, False):
                if target is None:      # 2 wp = ws + 3 wb
                    wp = 7.1
                    for _ in range(50):
                        wp = 0.5 * (ws + 3 * wb_of(wp, sm))
                else:
                    wp = solve_wp(target, ws, sm)
                vals.append(wp)
        print(f"  {name}: {min(vals):.3f}-{max(vals):.3f}")
    # two continuum photons (one on resonance): w_a + w_c = 2wp - ws - wb, max 2*hi
    wp2 = [0.5 * (2 * hi + ws + wb_of(7.7)) for ws in WS_RANGE]
    print(f"  extent if two photons were in the continuum: up to {min(wp2):.2f}-{max(wp2):.2f}"
          f" (data: clean above 7.43, so this configuration is negligible)")

    print("\nMeasured late-time parity near the predicted cutoff:")
    for wp, p in par:
        if 7.32 < wp < 7.47:
            print(f"  {wp:.3f} GHz  parity {p:+.2f}")

    print("\nRank correlation (Spearman) between A_b(w3) and parity deficit:")
    win = (wp_meas >= 7.09) & (wp_meas <= 7.455)
    for sm in (True, False):
        r_all, r_win = [], []
        for ws in WS_RANGE:
            a = A_b(2 * wp_meas - ws - 2 * wb_of(wp_meas, sm), wb_of(wp_meas, sm))
            r_all.append(spearman(a, deficit)); r_win.append(spearman(a[win], deficit[win]))
        lab = "5-column median wb" if sm else "per-column wb     "
        print(f"  {lab}: 7.10-7.45 GHz {min(r_win):.2f}-{max(r_win):.2f};  full scan {min(r_all):.2f}-{max(r_all):.2f}")
    # inside the band only: w3 between the discrete resonance and the upper cutoff
    for ws in (5.37,):
        w3 = 2 * wp_meas - ws - 2 * wb_of(wp_meas, False)
        inb = (w3 > wb_of(wp_meas, False) + 0.1) & (w3 < hi)
        a = A_b(w3[inb], wb_of(wp_meas[inb], False))
        print(f"  in-band columns only, per-column wb (ws={ws}): {spearman(a, deficit[inb]):.2f} (n={inb.sum()})")

    span = 0.7
    print(f"\nChance that a cutoff anywhere in a {span*1e3:.0f} MHz range lands within +-20 MHz: {0.04/span:.1%}")
    print(f"Predicted degraded band width in pump frequency (half the pass band): {(hi-lo)/2*1e3:.0f} MHz")
