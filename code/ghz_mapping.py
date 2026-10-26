"""
ghz_mapping.py -- device parameters of the paper in physical units (Appendix F, Table II).

The numerics use units a = v = c_l = 1 (a: cell length, v: wave speed, c_l: capacitance per length), so
frequencies are in v/a and capacitances in c_l*a.  The device is ONE fixed circuit (transmon + coupling capacitor)
placed in gap n by choosing the cell length a_n (kc_selfion.device):
    kappa = K/Omega_c,  C_g = CG1/Omega_c,  alpha = 0.05*Omega_c,   Omega_c = centre of gap n (units v/a_n).
Physical anchor (assumption, stated in the paper): every gap centre maps to NU_C = 6 GHz, on a coplanar host with
v = 1.2e8 m/s and Z0 = 50 Ohm.  Then a_n = Omega_c * v / (2 pi NU_C), and every frequency w maps to w * NU_C / Omega_c.
"""
import numpy as np
import kc_bath as kb
import kc_selfion as si

NU_C = 6.0e9          # Hz, gap centre
V = 1.2e8             # m/s, phase velocity (CPW on silicon, eps_eff ~ 6.25)
Z0 = 50.0             # Ohm
C_L = 1 / (Z0 * V)    # F/m
L_L = Z0 / V          # H/m
E, H = 1.602176634e-19, 6.62607015e-34
R0 = 0.5              # KC loading strength (kc_bath)

print(f"host: v = {V:.2e} m/s, Z0 = {Z0:.0f} Ohm, c_l = {C_L*1e12:.1f} pF/m, l_l = {L_L*1e9:.0f} nH/m; gap centre -> {NU_C/1e9:.1f} GHz")
rows = []
for lat in ["KC", "KP"]:
    for n in [1, 2, 3, 4]:
        d = si.device(lat, n)
        Oc = d["Om_c"]; lo, hi = d["gap"]
        a = Oc * V / (2 * np.pi * NU_C)
        f = NU_C / Oc                                     # Hz per (v/a) unit
        Cg = si.CG1 * C_L * V / (2 * np.pi * NU_C)        # = (CG1/Oc) * c_l * a, independent of n
        Csig = Cg / 0.4
        EC = E**2 / (2 * Csig) / H
        load = (2 * R0 * C_L * a * 1e12, "pF") if lat == "KC" else (L_L * a / kb.V0_KP * 1e9, "nH")
        gam_e = si.K * lo / (2 * Oc) * lo * f             # Gamma_free/2pi at the gap edge: (K w/(2 Oc)) * w, in Hz
        rows.append((lat, n, a * 1e3, load, lo * f / 1e9, hi * f / 1e9, d["alpha"] * f / 1e6, gam_e / 1e6))
        print(f"{lat} gap {n}: a = {a*1e3:5.2f} mm, shunt {load[1]} = {load[0]:5.2f}, gap {lo*f/1e9:.3f}-{hi*f/1e9:.3f} GHz, "
              f"alpha/2pi = {d['alpha']*f/1e6:.0f} MHz, clean window {lo*f/1e9 + d['alpha']*f/2e9:.3f}-{lo*f/1e9 + d['alpha']*f/1e9:.3f} GHz, "
              f"Gamma_free/2pi(edge) = {gam_e/1e6:.0f} MHz")
print(f"circuit (all gaps): C_g = {Cg*1e15:.1f} fF, C_Sigma = {Csig*1e15:.1f} fF, E_C/h = {EC/1e6:.0f} MHz "
      f"(compare alpha/2pi = {0.05*NU_C/1e6:.0f} MHz)")
np.save("ghz_mapping.npy", np.array([(r[1], r[2], r[3][0], r[4], r[5], r[6], r[7]) for r in rows]))
