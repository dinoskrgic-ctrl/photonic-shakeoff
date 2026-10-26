"""
kc_transmon_part2.py
Part II: a transmon, capacitively coupled to one brane, swept in frequency through the kinetic-crystal bath.

  1. exact ring modes from Bloch waves (no spatial grid): the bath that the dynamics uses
  2. (a) analytic flux-sweep spectroscopy from the exact pole equation  Omega^2 - E^2 = kappa E^4 G(E)
  3. (b) two-quantum dynamics (RWA), transmon levels |0>,|1>,|2>, start in |2> (a leakage event):
         where does the population end up, as a function of Omega?  compared with the rung picture.

Coupling (derived in the notes):   g_k = (1/2) sqrt(kappa Omega w_k) F_k,   F_k = mode amplitude at the brane.
Units a = v = c_l = 1.
"""

import time
import numpy as np
import scipy.sparse as sp
from scipy.optimize import brentq
from scipy.sparse.linalg import expm_multiply

from kc_transmon_freq import z_kc, make_z_kp, gaps, calG, gamma, bloch_factor, r0


# ============================================================ 1. exact ring modes
def band_intervals(zf, w_max):
    """pass bands below w_max as (lo, hi) intervals"""
    gl = gaps(zf, w_max)
    edges = [0.0]
    for lo, hi in gl:
        edges += [lo, hi]
    edges.append(w_max)
    bands = [(edges[2 * i], edges[2 * i + 1]) for i in range(len(edges) // 2)]
    return [(lo, hi) for lo, hi in bands if hi - lo > 1e-9]


def cell_norm(w, lam, bead):
    """
    N_cell = int_0^1 |cos(ws) + C sin(ws)|^2 ds + bead * |u(0)|^2,   C = (lambda - cos w)/sin w,  u(0) = 1.
    The integral is done in closed form.
    """
    C = (lam - np.cos(w)) / np.sin(w)
    s2 = np.sin(2 * w) / (4 * w)
    integral = (0.5 + s2) + np.abs(C)**2 * (0.5 - s2) + np.real(C) * np.sin(w)**2 / w
    return integral + bead


def ring_modes(zf, bead, N, n_bands):
    """
    Ring of N cells.  Bloch momenta q_m = 2 pi m / N.  The emitter sits on brane 0, so only the standing wave
    even about brane 0 couples:  u = sqrt(2) Re u_q  (m = 1 .. N/2 - 1)  or  u_q itself (m = 0, N/2).
    Returns frequencies w_k and brane amplitudes F_k with  int W u_k^2 = 1:
        F_k^2 = c_m / (N N_cell),   c_m = 2 (standing pair) or 1 (q = 0, pi)
    """
    bands = band_intervals(zf, 4 * np.pi * n_bands)[:n_bands]
    w_list = []
    F_list = []
    for lo, hi in bands:
        for m in range(N // 2 + 1):
            target = np.cos(2 * np.pi * m / N)
            f = lambda x: np.real(zf(x)) - target
            a, b = max(lo, 1e-6), hi
            if f(a) * f(b) > 0:                      # edge mode exactly at an end point
                w = a if abs(f(a)) < abs(f(b)) else b
            else:
                w = brentq(f, a, b, xtol=1e-14)
            if w < 1e-4 or abs(np.sin(w)) < 1e-10:   # zero mode, or Bragg mode with a node on the brane
                continue
            lam = np.exp(1j * 2 * np.pi * m / N)
            c_m = 1.0 if m in (0, N // 2) else 2.0
            w_list.append(w)
            F_list.append(np.sqrt(c_m / (N * cell_norm(w, lam, bead))))
    order = np.argsort(w_list)
    return np.array(w_list)[order], np.array(F_list)[order]


def couplings(w_k, F_k, kappa, Omega):
    """RWA couplings for a capacitively coupled transmon of 0-1 frequency Omega"""
    return 0.5 * np.sqrt(kappa * Omega * w_k) * F_k


# ============================================================ 2. one quantum (exact within RWA + ring)
def one_quantum(Omega, g_k, w_k):
    M = len(w_k)
    H = np.zeros((M + 1, M + 1))
    H[0, 0] = Omega
    H[0, 1:] = g_k
    H[1:, 0] = g_k
    H[1:, 1:] = np.diag(w_k)
    return np.linalg.eigh(H)


def trapped_fraction(Omega, g_k, w_k):
    """long-time average of P1 for |1> at t = 0:  sum_j |<1|j>|^4  (dominated by bound states)"""
    E, V = one_quantum(Omega, g_k, w_k)
    return np.sum(V[0, :]**4)


# ============================================================ 3. two quanta
def two_quantum_H(Omega, alpha, g_k, w_k):
    """
    basis:  0           |2; vac>
            1 .. M      |1; k>
            M+1 ..      |0; k, l>  k <= l   (normalised two-photon states)
    matrix elements (charge operator ~ sqrt(n) between neighbouring levels):
        <1;k| H |2;0>     = sqrt(2) g_k
        <0;kl| H |1;k>    = g_l          (k != l)
        <0;kk| H |1;k>    = sqrt(2) g_k
    """
    M = len(w_k)
    kk, ll = np.triu_indices(M)
    P = len(kk)
    i1 = 1 + np.arange(M)
    ip = 1 + M + np.arange(P)
    dim = 1 + M + P

    diag = np.concatenate([[2 * Omega - alpha], Omega + w_k, w_k[kk] + w_k[ll]])

    rows = [np.zeros(M, dtype=int)]
    cols = [i1]
    vals = [np.sqrt(2) * g_k]

    same = kk == ll
    rows += [i1[kk[same]]]
    cols += [ip[same]]
    vals += [np.sqrt(2) * g_k[kk[same]]]

    diff = ~same
    rows += [i1[kk[diff]], i1[ll[diff]]]
    cols += [ip[diff], ip[diff]]
    vals += [g_k[ll[diff]], g_k[kk[diff]]]

    r = np.concatenate(rows)
    c = np.concatenate(cols)
    v = np.concatenate(vals)
    H = sp.coo_matrix((np.concatenate([v, v]), (np.concatenate([r, c]), np.concatenate([c, r]))), shape=(dim, dim))
    return (H + sp.diags(diag)).tocsr(), i1, ip


def run_from_2(Omega, alpha, g_k, w_k, t_end, t_avg):
    """start in |2; vac>; return time-averaged P2, P1, P0 over [t_end - t_avg, t_end]"""
    H, i1, ip = two_quantum_H(Omega, alpha, g_k, w_k)
    psi = np.zeros(H.shape[0], dtype=complex)
    psi[0] = 1.0
    psi = expm_multiply(-1j * H * (t_end - t_avg), psi)
    ts = np.linspace(0, t_avg, 9)
    out = []
    for i in range(len(ts)):
        if i > 0:
            psi = expm_multiply(-1j * H * (ts[i] - ts[i - 1]), psi)
        p = np.abs(psi)**2
        out.append([p[0], p[i1].sum(), p[ip].sum()])
    return np.mean(out, axis=0)


# ============================================================ run
if __name__ == "__main__":

    N_RING = 200
    N_BANDS = 4
    KAPPA = 0.008
    ALPHA_RATIO = 0.05

    w_k, F_k = ring_modes(z_kc, 2 * r0, N_RING, N_BANDS)
    print(f"[1] exact ring: N = {N_RING} cells, {N_BANDS} bands, {len(w_k)} coupled modes, w up to {w_k.max():.3f}")

    # ---------------------------------------------------- check 1: golden rule from the discrete modes
    print("    check: Gamma from the ring modes (smoothed delta) vs closed form kappa w^3 Im G, Omega = w")
    for w in [1.0, 1.5, 3.4, 3.8, 6.5]:
        g_k = couplings(w_k, F_k, KAPPA, w)
        win = np.abs(w_k - w) < 0.06
        G_ring = 2 * np.pi * np.sum(g_k[win]**2) / 0.12
        print(f"      w = {w}:  ring {G_ring:.5f}   closed {gamma(w, KAPPA, z_kc):.5f}")

    # ---------------------------------------------------- check 2: exact linear pole (flux) with these modes
    print("    check: flux-coupled linear gap pole, ring modes vs closed form (as in Part I)")
    for Om in [2.0, 2.4, 2.9]:
        c = 0.3 * np.sqrt(2 * Om)
        V = np.diag(np.concatenate([[Om**2], w_k**2]))
        V[0, 1:] = c * F_k
        V[1:, 0] = c * F_k
        nu = np.sqrt(np.clip(np.linalg.eigvalsh(V), 0, None))
        nu_gap = nu[(nu > 1.7207) & (nu < np.pi - 1e-3)]
        f = lambda E: Om**2 - E**2 - c**2 * np.real(calG(E, z_kc, (0,), (1.0,)))
        root = brentq(f, 1.7208, np.pi - 1e-4)
        print(f"      Omega {Om}:  ring {np.round(nu_gap, 6)}   closed {root:.6f}")
