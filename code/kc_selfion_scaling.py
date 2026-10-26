"""
kc_selfion_scaling.py
Is self-ionization a second-order effect?  Scaling of the ionization loss with the coupling kappa.

Order counting (RWA, coupling g ~ sqrt(kappa)):
    retained channel |B1;k>   : amplitude ~ g        -> rate Gamma_B   ~ kappa
    ionized channel |0;k,l>   : amplitude ~ g^2 G1   -> rate Gamma_2g  ~ kappa^2 |G1|^2
    loss L = Gamma_2g / (Gamma_B + Gamma_2g)  ~ kappa |G1|^2 Gamma_ph-ratio   ~  O(kappa) at fixed detuning
The qubit's own dressing  1 - Z1 = sum over band of |<1|j>|^2  is also O(kappa) at fixed detuning, so the natural comparison is
    L / (1 - Z1).
Near a 1D van Hove edge, perturbation theory in kappa fails when the detuning from the edge is ~ binding energy ~ kappa^(2/3).

Device: KC gap 2 (as in kc_selfion_run), kappa scaled by s at fixed C_Sigma  (C_g ~ sqrt(s)).
Detunings held fixed in absolute frequency: Omega - w_e = d.
"""
import numpy as np
import kc_selfion as si

dev0 = si.device("KC", 2)
C_sig = dev0["Cg"]**2 / dev0["kappa"]
w_e = dev0["gap"][0]
S = [1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2]
D = [0.01, 0.03, 0.06, 0.10]          # Omega - w_e (alpha = 0.259 -> threshold at 0.129)

rows = []
print("s        kappa     Gfree/Om |  d=Omega-w_e:  L      1-Z1    L/(1-Z1)   (L = 1 - P1/Z1, theory with exchange)")
for s in S:
    dev = dict(dev0)
    dev["kappa"] = dev0["kappa"] * s
    dev["Cg"] = np.sqrt(dev["kappa"] * C_sig)
    w, F = si.bath(dev, N=250)
    wf, Ff = si.bath(dev, N=2000)
    line = f"{s:<8.4f} {dev['kappa']:.2e}  {dev['kappa']*w_e/2:.1e} |"
    for d in D:
        Om = w_e + d
        c = si.calibrate(Om, dev, w, F)
        if c is None:
            line += f"  d={d}: no bound state  "
            continue
        Eb, Zex, OmR = c
        r = si.markov_exchange(OmR, dev, wf, Ff, 0.002, n_grid=20000)
        L = 1 - r["P1"] / r["Z1"]
        rows.append([s, dev["kappa"], d, L, 1 - r["Z1"], r["G_B"], r["G_dir"], r["G_ex"], Eb - w_e])
        line += f"  d={d}: {L:.2e} {1-r['Z1']:.2e} {L/(1-r['Z1']):5.2f} |"
    print(line, flush=True)

rows = np.array(rows)
np.save("kc_selfion_scaling.npy", rows)
print("\nlog-log slopes dL/dlog kappa (small-kappa end, fixed d):")
for d in D:
    sel = rows[:, 2] == d
    k, L = rows[sel, 1], rows[sel, 3]
    if len(k) >= 3:
        p = np.polyfit(np.log(k[:3]), np.log(L[:3]), 1)[0]
        q = np.polyfit(np.log(k[-3:]), np.log(L[-3:]), 1)[0]
        print(f"  d = {d}: slope (3 smallest kappa) {p:.2f},  slope (3 largest) {q:.2f}")
