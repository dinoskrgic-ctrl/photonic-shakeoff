"""
kc_transmon_freq.py
Part I: frequency response of the kinetic-crystal bath, as seen by a transmon.  Everything here is analytic
(closed-form lattice Green's function); the ring mode sums of kc_giant_atom.py are used only as checks.

Physical bath = transmission line loaded with a shunt capacitor C0 every a  (Pozar, periodic structures):
    (c_l + C0 sum_n delta(y - n a)) Phi_tt = Phi_yy / l_l      ->   W = 1 + 2 r0 sum delta,   2 r0 = C0 / c_l
    dispersion  cos(q a) = z(w) = cos(w a) - r0 w sin(w a)        (Pozar: cos(beta d) = cos(theta) - (b/2) sin(theta),  b = w C0 Z0)
KP control = same line loaded with a shunt INDUCTOR:  z_KP = cos(w a) + (V0 / 2w) sin(w a)

Units: a = 1, v = 1, c_l = 1.

Sections
  1. bath: dispersion, gaps (KC widen, KP shrink)
  2. Green's function between branes, closed form; checks against the ring mode sum
  3. one harmonic rung, capacitive coupling: exact pole equation  Omega^2 - E^2 = kappa E^4 G(E)
       band -> Gamma(w) = Gamma_free(w) * P(w),   P = Purcell factor
       gap  -> bound state, real E
  4. transmon ladder: rung n -> n-1 at w_n = Omega - (n-1) alpha, strength n  ->  each rung sees its own P(w_n)
"""

import numpy as np
from scipy.optimize import brentq


# ============================================================ parameters
r0 = 0.5            # brane loading 2 r0 = C0 / c_l  (in units of a)
ETA = 1e-9          # retarded prescription  w -> w + i ETA


# ============================================================ 1. bath dispersion
def z_kc(w):
    return np.cos(w) - r0 * w * np.sin(w)


def make_z_kp(V0):
    return lambda w: np.cos(w) + V0 / (2 * w) * np.sin(w)


def gaps(zf, w_max, n_grid=400000):
    """list of (lo, hi) intervals with |z| > 1 (stop bands)"""
    w = np.linspace(1e-4, w_max, n_grid)
    stop = np.abs(zf(w)) > 1
    edges = np.flatnonzero(np.diff(stop.astype(int)))
    out = []
    lo = 0.0                                  # a stop band can start at w = 0 (KP: inductive shunts = high-pass)
    for i in edges:
        if not stop[i]:                       # band -> gap
            lo = brentq(lambda x: abs(zf(x)) - 1, w[i], w[i + 1])
        else:                                 # gap -> band
            hi = brentq(lambda x: abs(zf(x)) - 1, w[i], w[i + 1])
            out.append((lo, hi))
    return out


def bragg_gaps(zf, w_max):
    """stop bands attached to the Bragg points n pi (drops the KP low-frequency gap that starts at w = 0)"""
    return [gp for gp in gaps(zf, w_max) if gp[0] > 1e-3]


def match_kp_first_gap(width):
    """V0 such that the first KP Bragg gap has the given width"""
    def f(V0):
        return np.diff(bragg_gaps(make_z_kp(V0), 3 * np.pi)[0])[0] - width
    return brentq(f, 0.5, 20.0)


# ============================================================ 2. Green's function on the branes
def bloch_factor(w, zf):
    """lambda(w): the root of lambda + 1/lambda = 2 z with |lambda| < 1 at w + i ETA (outgoing / decaying)"""
    zc = zf(w + 1j * ETA)
    root = np.sqrt(zc**2 - 1 + 0j)
    lam1 = zc - root
    lam2 = zc + root
    return np.where(np.abs(lam1) < np.abs(lam2), lam1, lam2)


def G_brane(w, zf, dn):
    """
    [-d^2/dy^2 - w^2 W] G = delta(y - m)  evaluated at brane n,  dn = |n - m|.
    Between branes the string is free, so the field on the branes obeys
        G_{n+1} + G_{n-1} - 2 z G_n = -(sin w / w) delta_nm
    whose decaying solution is
        G_nm = (sin w / w) lambda^|n-m| / (1/lambda - lambda).
    """
    lam = bloch_factor(w, zf)
    s = np.sin(w) / w
    return s * lam**dn / (1 / lam - lam)


def calG(w, zf, sites, weights):
    """G seen by an emitter coupled at branes `sites` with `weights`:  sum_ij w_i w_j G(n_i, n_j)"""
    out = 0.0
    for ni, wi in zip(sites, weights):
        for nj, wj in zip(sites, weights):
            out = out + wi * wj * G_brane(w, zf, abs(ni - nj))
    return out


def purcell(w, zf, sites=(0,), weights=(1.0,)):
    """emission rate relative to a small atom on a bare line:  Im G / Im G_free,  Im G_free = 1/(2w)"""
    return 2 * w * np.imag(calG(w, zf, sites, weights))


# ============================================================ 3. one harmonic rung, capacitive coupling
def pole_equation(E, Omega, kappa, zf, sites=(0,), weights=(1.0,)):
    """
    Lagrangian:  C_a X'^2/2 - X^2/(2 L_J) + Cg X' Phi'(brane)  + line.   Frequency domain, exact for a linear atom:
        Omega^2 - E^2 = kappa E^4 G(E),        kappa = Cg^2 / (C_a c_l)
    """
    return Omega**2 - E**2 - kappa * E**4 * np.real(calG(E, zf, sites, weights))


def gamma(w, kappa, zf, sites=(0,), weights=(1.0,)):
    """band:  E = Omega - i Gamma/2  ->  Gamma = kappa w^3 Im G(w)  =  Gamma_free(w) * P(w),  Gamma_free = kappa w^2 / 2"""
    return kappa * w**3 * np.imag(calG(w, zf, sites, weights))


def bound_states(Omega, kappa, zf, gap, sites=(0,), weights=(1.0,), n_grid=4000):
    """real roots of the pole equation inside a gap (G is real there)"""
    lo, hi = gap
    E = np.linspace(lo + 1e-9, hi - 1e-9, n_grid)
    f = pole_equation(E, Omega, kappa, zf, sites, weights)
    roots = []
    for i in np.flatnonzero(np.sign(f[:-1]) != np.sign(f[1:])):
        if np.isfinite(f[i]) and np.isfinite(f[i + 1]) and abs(f[i] - f[i + 1]) < 1e3:
            roots.append(brentq(pole_equation, E[i], E[i + 1], args=(Omega, kappa, zf, sites, weights)))
    return roots


# ============================================================ 4. transmon ladder
def rung_frequencies(Omega, alpha, n_rungs):
    """rung n (|n> -> |n-1>) sits at w_n = Omega - (n-1) alpha"""
    return Omega - alpha * np.arange(n_rungs)


def in_gap(w, gap_list):
    return any(lo < w < hi for lo, hi in gap_list)


def rung_class(Omega, alpha, gap_list):
    """(qubit rung 1->0 protected?, leakage rung 2->1 protected?)"""
    return in_gap(Omega, gap_list), in_gap(Omega - alpha, gap_list)


def window_lengths(gap, alpha_ratio, gap_list, n=4000):
    """
    scan the qubit frequency Omega across one gap (alpha = alpha_ratio * Omega) and measure the length of
        'filter'    : qubit rung protected, leakage rung radiates
        'protected' : both rungs protected
    """
    Om = np.linspace(gap[0], gap[1], n)
    dO = Om[1] - Om[0]
    cls = np.array([rung_class(O, alpha_ratio * O, gap_list) for O in Om])
    return dO * np.sum(cls[:, 0] & ~cls[:, 1]), dO * np.sum(cls[:, 0] & cls[:, 1])


def decay_per_cell(w, zf):
    """-ln|lambda(w)|: inverse localisation length (in cells) of a bound state at energy w"""
    return -np.log(np.abs(bloch_factor(w, zf)))


# ============================================================ checks and report
if __name__ == "__main__":
    import kc_giant_atom as kga

    W_MAX = 6 * np.pi + 0.5
    gaps_kc = gaps(z_kc, W_MAX)
    V0 = match_kp_first_gap(np.diff(gaps_kc[0])[0])
    z_kp = make_z_kp(V0)
    gaps_kp = bragg_gaps(z_kp, W_MAX)
    print(f"KP low-frequency (high-pass) gap: {gaps(z_kp, 3.0)[0]}")

    # ---------------------------------------------------- 1
    print("[1] stop bands (gap widths), KC vs KP with matched first gap  (V0 = %.3f)" % V0)
    for i in range(6):
        a, b = gaps_kc[i]
        c, d = gaps_kp[i]
        print(f"    gap {i+1}:  KC ({a:6.3f}, {b:6.3f})  width {b-a:.3f}    KP ({c:6.3f}, {d:6.3f})  width {d-c:.3f}")

    # ---------------------------------------------------- 2. checks against kc_giant_atom
    print("\n[2] Green's function checks (flux coupling, where the old code applies:  Gamma = 2 g^2 Im G)")
    g = 0.3
    geoms = {"small, brane 0": ([0], [1.0]), "giant (1,-2,1)": ([0, 1, 2], [1.0, -2.0, 1.0])}
    for name, (st, wt) in geoms.items():
        rows = []
        for w in [0.5, 1.0, 1.5, 3.3, 3.8]:
            closed = 2 * g**2 * np.imag(calG(w, z_kc, st, wt))
            bloch = kga.gamma_analytic(w, g, [float(s) for s in st], wt)
            rows.append(f"w={w}: {closed:.5e} / {bloch:.5e}")
        print(f"    {name:15s} closed form / Bloch golden rule:  " + "   ".join(rows))

    print("    gap pole, flux coupling  Omega^2 - nu^2 = c^2 G(nu)   vs  ring normal modes (kc_giant_atom)")
    y, mass, w_k, u_k = kga.bath_modes()
    for name, (st, wt) in geoms.items():
        F_k = kga.form_factor(u_k, [float(s) for s in st], wt)
        for Om in [2.0, 2.4, 2.9]:
            c = g * np.sqrt(2 * Om)
            lam, _ = kga.linear_normal_modes(Om, c, F_k, w_k)
            nu = np.sqrt(np.clip(lam, 0, None))
            nu_gap = nu[(nu > gaps_kc[0][0]) & (nu < gaps_kc[0][1])]
            f = lambda E: Om**2 - E**2 - c**2 * np.real(calG(E, z_kc, st, wt))
            Es = np.linspace(gaps_kc[0][0] + 1e-6, gaps_kc[0][1] - 1e-6, 4000)
            fv = f(Es)
            idx = np.flatnonzero(np.sign(fv[:-1]) != np.sign(fv[1:]))
            roots = [brentq(f, Es[i], Es[i + 1]) for i in idx if abs(fv[i] - fv[i + 1]) < 1e3]
            print(f"      {name:15s} Omega {Om}:  closed form {np.round(roots, 5)}   ring {np.round(nu_gap, 5)}")

    # ---------------------------------------------------- 3. Purcell factor, small capacitive transmon
    print("\n[3] Purcell factor P(w) = Gamma/Gamma_free, small transmon on a brane.  Closed form: P = sin(w)/sin(q)")
    for w in [0.5, 1.0, 1.5, 1.70, 3.2, 3.5, 4.0]:
        lam = bloch_factor(w, z_kc)
        q = np.angle(lam)
        print(f"    w = {w:4.2f}:  P = {purcell(w, z_kc):8.4f}   sin w / sin q = {np.sin(w) / np.sin(q):8.4f}")

    # ---------------------------------------------------- 4. band index as a design knob
    print("\n[4] per band index n:  band below the gap, gap, mid-gap decay per cell, and the transmon windows (alpha = 0.05 Omega)")
    print("    KC band width ~ 2/(r0 n pi), gap -> pi;   KP the other way round")
    edges_kp = gaps(z_kp, W_MAX)                     # includes the KP high-pass gap from w = 0
    for i in range(6):
        for label, gl, gl_all, zf in [("KC", gaps_kc, gaps_kc, z_kc), ("KP", gaps_kp, edges_kp, z_kp)]:
            lo, hi = gl[i]
            k = gl_all.index(gl[i])
            band = lo - (gl_all[k - 1][1] if k > 0 else 0.0)
            xi_inv = decay_per_cell(0.5 * (lo + hi), zf)
            filt, prot = window_lengths(gl[i], 0.05, gl_all)
            print(f"    n={i+1} {label}: band below {band:.3f}  gap {hi-lo:.3f}  mid-gap decay/cell {xi_inv:.2f}"
                  f"  | alpha {0.05*lo:.2f}-{0.05*hi:.2f}:  filter window {filt:.3f}  both-protected window {prot:.3f}")
