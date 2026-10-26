"""
kc_selfion_run.py  LATTICE
For gaps 1-4 of LATTICE (KC or KP): flux sweep of the qubit just above the lower gap edge w_e, leakage event |2>.
  - analytic (Markov + exchange) on a fine grid of Omega
  - full RWA two-quantum numerics on a coarse grid (validation), T = 220, average over [140, 220]
  - spectator numerics on KC gap 2 only
All RWA runs use the calibrated Omega_R (dressed qubit = exact non-RWA pole, C_g loading included).
Output: selfion_<LATTICE>.npz
"""
import sys
import time
import numpy as np
import kc_selfion as si

LAT = sys.argv[1]
T, T_AVG = 220.0, 80.0
N_RING = {"KC": [250, 250, 160, 140], "KP": [250, 250, 220, 200]}[LAT]
F_FULL = {1: [0.03, 0.08, 0.15, 0.25, 0.35, 0.45, 0.55, 0.7, 0.9, 1.2],
          2: [0.03, 0.08, 0.15, 0.25, 0.35, 0.45, 0.55, 0.7, 0.9, 1.2],
          3: [0.05, 0.15, 0.3, 0.45, 0.6, 0.8, 1.1],
          4: [0.05, 0.15, 0.3, 0.45, 0.6, 0.8, 1.1]}
F_FINE = np.concatenate([np.linspace(0.005, 0.6, 50), np.linspace(0.62, 1.4, 20)])

out = {}
t0 = time.time()
for n in [1, 2, 3, 4]:
    dev = si.device(LAT, n)
    w_e = dev["gap"][0]
    w, F = si.bath(dev, N=N_RING[n - 1])
    wf, Ff = si.bath(dev, N=1000)
    print(f"== {LAT} gap {n}: w_e {w_e:.4f}, alpha {dev['alpha']:.3f}, kappa {dev['kappa']:.5f}, Cg {dev['Cg']:.4f}, "
          f"M {len(w)} (N {N_RING[n-1]})", flush=True)

    fine = []
    for f in F_FINE:
        Om = w_e + f * dev["alpha"]
        if Om >= dev["gap"][1]:
            continue
        c = si.calibrate(Om, dev, w, F)
        if c is None:
            continue
        Eb, Zex, OmR = c
        r = si.markov_exchange(OmR, dev, wf, Ff, 0.004)
        m = si.markov(OmR, dev, w, F, escape_P1=r["P1"])
        m0 = si.markov(OmR, dev, w, F, escape_P1=r["P1_noex"])
        fine.append([Om, Eb, Zex, OmR, m["P1_markov"], m0["P1_markov"], r["Z1"], m["P1_dressed"], m["P1_bare"],
                     m["P2_markov"], r["G_B"], r["G_dir"], r["G_ex"]])
    out[f"fine_{n}"] = np.array(fine)

    full = []
    for f in F_FULL[n]:
        Om = w_e + f * dev["alpha"]
        c = si.calibrate(Om, dev, w, F)
        if c is None:
            continue
        Eb, Zex, OmR = c
        ts = time.time()
        P = si.run_full(OmR, dev, w, F, T, T_AVG)
        r = si.markov_exchange(OmR, dev, wf, Ff, 0.004)
        m = si.markov(OmR, dev, w, F, escape_P1=r["P1"])
        m0 = si.markov(OmR, dev, w, F, escape_P1=r["P1_noex"])
        row = [Om, Eb, OmR, *P, m["P1_markov"], m0["P1_markov"], m["P1_dressed"], m["P1_bare"], m["P2_markov"],
               np.nan, np.nan]
        if LAT == "KC" and n == 2:
            S = si.run_spectator(OmR, dev, w, F, T, T_AVG)
            row[-2], row[-1] = S[1], S[0]
        full.append(row)
        print(f"   Om {Om:.4f} (f {f:.2f}): FULL P2 {P[0]:.3f} P1 {P[1]:.3f} P0 {P[2]:.3f} | theory P1 {row[6]:.3f} "
              f"(no-ex {row[7]:.3f}, dressed {m['P1_dressed']:.3f}, bare {m['P1_bare']:.3f}), P2 {m['P2_markov']:.3f}"
              + (f" | SPECT P1 {row[-2]:.3f}" if not np.isnan(row[-2]) else "")
              + f"  [{time.time()-ts:.0f}s, total {time.time()-t0:.0f}s]", flush=True)
    out[f"full_{n}"] = np.array(full)
    out[f"meta_{n}"] = np.array([w_e, dev["gap"][1], dev["alpha"], dev["kappa"], dev["Cg"], dev["Om_c"]])
    np.savez(f"selfion_{LAT}.npz", **out)

print("done", flush=True)
