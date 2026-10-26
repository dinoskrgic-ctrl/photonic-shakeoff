"""
kc_bath.py
The bath seen by a capacitively coupled transmon on brane 0, INCLUDING the coupling capacitor's loading of brane 0.

Lattices (units a = v = c_l = 1):
    KC : shunt capacitors 2 r0 on every brane      z = cos w - r0 w sin w,          bead mass 2 r0
    KP : shunt inductors  V0 = l/L_s on every brane  z = cos w + (V0 / 2w) sin w,     bead mass 0
Brane 0 additionally carries C_g to the transmon.  Seen from the line (transmon node held fixed) this is one extra
shunt capacitor on brane 0:   z_0 = z - (C_g/2) w sin w,   extra mass C_g on brane 0.

Contents
    lattice(name)                 : dispersion and bead mass
    gaps(zf, w_max)               : stop bands
    G00(w, zf), Gd(w, zf, Cg)     : on-site retarded Green's function, without / with the C_g loading (Dyson)
    gamma_ph(w, Om, kappa, ...)   : single-photon emission function 2 pi sum_k g_k^2 delta(w - w_k) = kappa Om w^2 Im Gd(w)
    D(E, ...)                     : exact linear pole function  Om^2 - E^2 - kappa E^4 Gd(E)
    exact_bound_states(...)       : real roots of D in a gap, weight Z = 2E/|D'(E)|
    ring_modes(...)               : exact even ring modes with the defect (no spatial grid), frequencies and F_k = u_k(0)
"""

import numpy as np
from scipy.optimize import brentq

ETA = 1e-9
R0 = 0.5
V0_KP = 7.851044        # matches the first KP Bragg gap to the first KC gap (kc_transmon_freq.match_kp_first_gap)


# ============================================================ lattices
def z_kc(w):
    return np.cos(w) - R0 * w * np.sin(w)


def z_kp(w):
    return np.cos(w) + V0_KP / (2 * w) * np.sin(w)


LATTICES = {"KC": (z_kc, 2 * R0), "KP": (z_kp, 0.0)}


def lattice(name):
    return LATTICES[name]


def gaps(zf, w_max, n_grid=400000):
    w = np.linspace(1e-4, w_max, n_grid)
    stop = np.abs(zf(w)) > 1
    edges = np.flatnonzero(np.diff(stop.astype(int)))
    out = []
    lo = 0.0
    for i in edges:
        if not stop[i]:
            lo = brentq(lambda x: abs(zf(x)) - 1, w[i], w[i + 1], xtol=1e-14)
        else:
            hi = brentq(lambda x: abs(zf(x)) - 1, w[i], w[i + 1], xtol=1e-14)
            out.append((lo, hi))
    if stop[-1]:
        out.append((lo, w_max))           # unfinished gap at the cutoff (was silently dropped before)
    return out


def bands(zf, w_max):
    edges = [0.0]
    for lo, hi in gaps(zf, w_max):
        edges += [lo, hi]
    edges.append(w_max)
    out = [(edges[2 * i], edges[2 * i + 1]) for i in range(len(edges) // 2)]
    return [b for b in out if b[1] - b[0] > 1e-9]


# ============================================================ Green's functions
def lam(w, zf):
    """decaying / outgoing root of lambda + 1/lambda = 2 z at w + i ETA"""
    zc = zf(w + 1j * ETA)
    root = np.sqrt(zc**2 - 1 + 0j)
    l1 = zc - root
    l2 = zc + root
    return np.where(np.abs(l1) < np.abs(l2), l1, l2)


def G00(w, zf):
    """on-site brane Green's function of [-d^2 - w^2 W]:  (sin w / w) / (1/lambda - lambda)"""
    l = lam(w, zf)
    return (np.sin(w) / w) / (1 / l - l)


def Gd(w, zf, Cg):
    """with the extra C_g on brane 0 (Dyson, one-site defect):  G00 / (1 - C_g w^2 G00)"""
    g = G00(w, zf)
    return g / (1 - Cg * w**2 * g)


def gamma_ph(w, Om, kappa, zf, Cg):
    """2 pi sum_k g_k^2 delta(w - w_k) with g_k = (1/2) sqrt(kappa Om w_k) F_k   ->   kappa Om w^2 Im Gd(w)"""
    return kappa * Om * w**2 * np.imag(Gd(w, zf, Cg))


def D(E, Om, kappa, zf, Cg):
    """exact linear pole function (derivation in the notes, Sec. 4 with the C_g denominator)"""
    return Om**2 - E**2 - kappa * E**4 * Gd(E, zf, Cg)


def exact_bound_states(Om, kappa, zf, Cg, gap, n_grid=6000):
    lo, hi = gap
    E = np.linspace(lo + 1e-10, hi - 1e-10, n_grid)
    f = np.real(D(E, Om, kappa, zf, Cg))
    out = []
    for i in np.flatnonzero(np.sign(f[:-1]) != np.sign(f[1:])):
        if not (np.isfinite(f[i]) and np.isfinite(f[i + 1])):
            continue
        g = lambda x: np.real(D(x, Om, kappa, zf, Cg))
        Eb = brentq(g, E[i], E[i + 1], xtol=1e-14)
        h = 1e-7
        dD = (g(Eb + h) - g(Eb - h)) / (2 * h)
        if abs(dD) > 1e6:                       # a pole of D (sign change through infinity), not a root
            continue
        out.append((Eb, 2 * Eb / abs(dD)))
    return out


# ============================================================ exact ring modes with the brane-0 defect
def _cell_integrals(w, u):
    """sum over cells n of int_0^1 (u_n cos ws + b_n sin ws)^2 ds,  b_n = (u_{n+1} - u_n cos w)/sin w"""
    un = u[:-1]
    b = (u[1:] - un * np.cos(w)) / np.sin(w)
    s2 = np.sin(2 * w) / (4 * w)
    return np.sum(un**2 * (0.5 + s2) + b**2 * (0.5 - s2) + un * b * np.sin(w)**2 / w)


def _omega_of_q(q, zf, band):
    lo, hi = band
    f = lambda x: zf(x) - np.cos(q)
    a, b = max(lo, 1e-7), hi
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        return a if abs(fa) < abs(fb) else b
    return brentq(f, a, b, xtol=1e-14)


def ring_modes(name, Cg, N, w_max, q_per_interval=12):
    """
    Even modes (about brane 0) of a ring of N cells, brane 0 loaded with bead + C_g.  Odd modes vanish on brane 0
    and do not couple.  In the uniform part the brane values are u_n = cos(q (n - N/2)), and the brane-0 condition
        u_1 = z_0 u_0   reads   h(q) = sin(q N/2) sin q + (C_g/2) w(q) sin w(q) cos(q N/2) = 0.
    Modes pushed out of a band by C_g become localized defect modes in the gap: roots of 1 = C_g w^2 G00(w).
    Returns (w_k, F_k) with F_k = u_k(0) / sqrt(int W u^2 + C_g u_0^2).
    """
    zf, bead = lattice(name)
    n = np.arange(N + 1)
    ws, Fs = [], []

    def add_mode(w, u):
        if abs(np.sin(w)) < 1e-9 or w < 1e-5:
            return
        norm = _cell_integrals(w, u) + bead * np.sum(u[:-1]**2) + Cg * u[0]**2
        ws.append(w)
        Fs.append(u[0] / np.sqrt(norm))

    for band in bands(zf, w_max):
        if band[1] >= w_max - 1e-9:
            continue                                         # do not keep a band cut by w_max
        qgrid = np.linspace(0, np.pi, (N // 2) * q_per_interval + 1)
        wq = np.array([_omega_of_q(q, zf, band) for q in qgrid])
        h = np.sin(qgrid * N / 2) * np.sin(qgrid) + 0.5 * Cg * wq * np.sin(wq) * np.cos(qgrid * N / 2)
        roots = []
        for i in np.flatnonzero(np.sign(h[:-1]) * np.sign(h[1:]) < 0):
            def hq(q):
                w = _omega_of_q(q, zf, band)
                return np.sin(q * N / 2) * np.sin(q) + 0.5 * Cg * w * np.sin(w) * np.cos(q * N / 2)
            roots.append(brentq(hq, qgrid[i], qgrid[i + 1], xtol=1e-15))
        for i in np.flatnonzero(h == 0):
            roots.append(qgrid[i])
        for q in roots:
            w = _omega_of_q(q, zf, band)
            add_mode(w, np.cos(q * (n - N / 2)))

    # defect modes in gaps (only if C_g > 0)
    if Cg > 0:
        for gap in gaps(zf, w_max):
            lo, hi = gap
            E = np.linspace(lo + 1e-11, min(hi, w_max) - 1e-11, 20000)
            f = 1 - Cg * E**2 * np.real(G00(E, zf))
            for i in np.flatnonzero(np.sign(f[:-1]) != np.sign(f[1:])):
                if not (np.isfinite(f[i]) and np.isfinite(f[i + 1])) or abs(f[i] - f[i + 1]) > 1e3:
                    continue
                w = brentq(lambda x: 1 - Cg * x**2 * np.real(G00(x, zf)), E[i], E[i + 1], xtol=1e-14)
                l = np.real(lam(w, zf))
                u = l ** np.minimum(n, N - n)
                add_mode(w, u)

    order = np.argsort(ws)
    return np.array(ws)[order], np.array(Fs)[order]


# ============================================================ self-tests
if __name__ == "__main__":
    import kc_transmon_part2 as old
    from kc_transmon_freq import z_kc as z_old

    print("[t1] Cg = 0 reproduces the old Bloch ring (frequencies, |F|):")
    w0, F0 = ring_modes("KC", 0.0, 200, 4 * np.pi + 0.5)
    w1, F1 = old.ring_modes(z_old, 1.0, 200, 4)
    m = min(len(w0), len(w1))
    print(f"    modes new {len(w0)}, old {len(w1)};  max |dw| = {np.max(np.abs(w0[:m]-w1[:m])):.2e}, "
          f"max |d|F|| = {np.max(np.abs(np.abs(F0[:m])-np.abs(F1[:m]))):.2e}")

    Cg = 0.02
    print(f"\n[t2] Cg = {Cg}: completeness on a bead, sum F_k^2 -> 1/(2 r0 + Cg) = {1/(2*R0+Cg):.5f} (all modes)")
    for wm in [4 * np.pi + 0.5, 8 * np.pi + 0.5, 16 * np.pi + 0.5]:
        w, F = ring_modes("KC", Cg, 200, wm)
        print(f"    w_max {wm:6.2f}: {len(w)} modes,  sum F^2 = {np.sum(F**2):.5f}")

    print("\n[t3] mode sum vs Dyson Gd in a gap:  sum F_k^2/(w_k^2 - E^2)  vs  Gd(E)")
    w, F = ring_modes("KC", Cg, 200, 16 * np.pi + 0.5)
    for E in [2.0, 2.6, 3.05, 4.3, 5.5]:
        print(f"    E = {E}:  modes {np.sum(F**2/(w**2-E**2)):+.6f}   Gd {np.real(Gd(E, z_kc, Cg)):+.6f}   (G00 {np.real(G00(E, z_kc)):+.6f})")

    print("\n[t4] defect modes found (pure line + C_g):")
    gl = gaps(z_kc, 13)
    for gp in gl:
        sel = (w > gp[0]) & (w < gp[1])
        print(f"    gap ({gp[0]:.4f}, {gp[1]:.4f}): {np.round(w[sel], 6)}")

    print("\n[t5] exact linear pole (with Cg) vs non-RWA generalized eigenproblem on the ring")
    Csig = 0.05
    kappa = Cg**2 / Csig
    w, F = ring_modes("KC", Cg, 200, 16 * np.pi + 0.5)
    M = len(w)
    for Om in [1.75, 2.4, 4.1, 5.2]:
        T = np.eye(M + 1)
        T[0, 0] = Csig
        T[0, 1:] = -Cg * F
        T[1:, 0] = -Cg * F
        V = np.diag(np.concatenate([[Csig * Om**2], w**2]))
        from scipy.linalg import eigh
        nu2 = eigh(V, T, eigvals_only=True)
        nu = np.sqrt(nu2)
        gp = [g for g in gl if g[0] < Om < g[1]][0]
        ring = nu[(nu > gp[0]) & (nu < gp[1])]
        ex = exact_bound_states(Om, kappa, z_kc, Cg, gp)
        print(f"    Omega {Om}: ring {np.round(ring, 6)}   exact roots {[round(e, 6) for e, _ in ex]}  Z {[round(z, 4) for _, z in ex]}")
