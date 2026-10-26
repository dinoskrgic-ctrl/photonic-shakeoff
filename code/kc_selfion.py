"""
kc_selfion.py
Leakage event |2; vac> of a transmon on brane 0, with the C_g loading of brane 0, in three levels of theory:

  FULL      : RWA two-quantum dynamics  {|2>, |1;k>, |0;k,l>}                       (numerically exact within RWA)
  SPECTATOR : |2> decays into |j> x |k>, j = one-quantum eigenstate (qubit dressed by the bath), k = leakage photon,
              energies E_j + w_k, coupling sqrt2 g_k <1|j>.  Neglects only the exchange/reabsorption of photon k.
  MARKOV    : golden-rule branching of the spectator model:
                  rate into channel j   ~  |<1|j>|^2  Gamma_ph(E2 - E_j),      E2 = 2 Omega - alpha
                  P1 = sum_{j bound} |<1|j>|^4 Gamma_ph(E2 - E_j)  /  sum_j |<1|j>|^2 Gamma_ph(E2 - E_j)
              limits: Gamma_ph flat -> P1 = sum |<1|j>|^4 (bare picture);  band channels closed -> P1 = Z1 (dressed).

Calibration (C): the RWA bare frequency Omega_R is chosen so that the RWA dressed qubit equals the exact non-RWA
pole E_b1(Omega) (with the C_g loading):  E_b1 = Omega_R (1 + sigma(E_b1)),  sigma(E) = sum_k kappa w_k F_k^2 / (4 (E - w_k)).

Device for gap n (fixed circuit, cell length a chosen to put the transmon in gap n):
    kappa = K / Omega_c,  C_g = CG1 / Omega_c,  alpha = 0.05 Omega_c   (Omega_c = gap centre; flux tuning within a gap)
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

import kc_bath as kb
from kc_transmon_part2 import two_quantum_H

K = 0.0192            # kappa * Omega_c   -> Gamma_free / Omega = K/2 ~ 1%
CG1 = 0.048           # C_g * Omega_c     -> C_g / C_Sigma = 0.4
ALPHA_RATIO = 0.05


def device(name, n_gap, w_max_gaps=40.0):
    zf, _ = kb.lattice(name)
    gl = [g for g in kb.gaps(zf, w_max_gaps) if g[0] > 1e-3]     # Bragg / kinetic gaps (drop KP high-pass gap)
    gap = gl[n_gap - 1]
    Om_c = 0.5 * (gap[0] + gap[1])
    return dict(name=name, n=n_gap, gap=gap, Om_c=Om_c, kappa=K / Om_c, Cg=CG1 / Om_c,
                alpha=ALPHA_RATIO * Om_c, zf=zf, gaps_all=kb.gaps(zf, w_max_gaps))


def bath(dev, N=200, extra=1.2):
    """ring modes up to the end of the first band above Omega_c + extra"""
    zf = dev["zf"]
    w_cut = None
    for lo, hi in kb.bands(zf, 60.0):
        if hi > dev["gap"][1] + extra:
            w_cut = hi + 1e-3                                # cutoff inside the gap above this band
            break
    return kb.ring_modes(dev["name"], dev["Cg"], N, w_cut)


def in_gap(E, gl):
    return any(lo < E < hi for lo, hi in gl)


def couplings(w_k, F_k, kappa, Om):
    return 0.5 * np.sqrt(kappa * Om * w_k) * F_k


def one_quantum(Om, g_k, w_k):
    M = len(w_k)
    H = np.zeros((M + 1, M + 1))
    H[0, 0] = Om
    H[0, 1:] = g_k
    H[1:, 0] = g_k
    H[1:, 1:] = np.diag(w_k)
    return np.linalg.eigh(H)


def calibrate(Om, dev, w_k, F_k):
    """exact (non-RWA) dressed qubit E_b1 and its weight; RWA bare frequency reproducing it"""
    ex = kb.exact_bound_states(Om, dev["kappa"], dev["zf"], dev["Cg"], dev["gap"])
    if not ex:
        return None
    Eb, Zex = max(ex, key=lambda t: t[1])
    sigma = np.sum(dev["kappa"] * w_k * F_k**2 / (4 * (Eb - w_k)))
    return Eb, Zex, Eb / (1 + sigma)


# ============================================================ the three theories
def markov(OmR, dev, w_k, F_k, escape_P1=None):
    """escape_P1: qubit retention for the escaping part of the leakage rung (e.g. the exchange-corrected value);
    default = spectator-Markov value computed here"""
    g_k = couplings(w_k, F_k, dev["kappa"], OmR)
    E, V = one_quantum(OmR, g_k, w_k)
    w1 = V[0, :]**2
    E2 = 2 * OmR - dev["alpha"]
    ph = E2 - E
    rate = np.where(ph > 1e-6, kb.gamma_ph(np.maximum(ph, 1e-6), OmR, dev["kappa"], dev["zf"], dev["Cg"]), 0.0)
    rate = np.clip(rate, 0, None)
    bound = np.array([in_gap(e, dev["gaps_all"]) for e in E])
    P1 = np.sum(w1[bound]**2 * rate[bound]) / np.sum(w1 * rate)
    jb = np.argmax(np.where(bound & (E > dev["gap"][0]) & (E < dev["gap"][1]), w1, -1))
    Z1 = w1[jb]
    Eb1 = E[jb]
    T1 = np.sum(w1**2)
    frac_band = np.sum(w1[~bound] * rate[~bound]) / np.sum(w1 * rate)

    # leakage bound states (dressed rung picture): rung at 2 Om - alpha - Eb1, coupling sqrt(2 Z1) g
    E2r, V2r = one_quantum(2 * OmR - dev["alpha"] - Eb1, np.sqrt(2 * Z1) * g_k, w_k)
    Z2 = np.array([V2r[0, j]**2 for j in range(len(E2r)) if in_gap(E2r[j], dev["gaps_all"])])
    Z2 = Z2 if len(Z2) else np.zeros(1)
    P2 = np.sum(Z2**2)
    if escape_P1 is not None and np.isfinite(escape_P1):
        P1 = escape_P1
    rung2 = 2 * OmR - dev["alpha"] - Eb1
    if in_gap(rung2, dev["gaps_all"]):
        P1 = Z1                       # leakage rung bound: its escaping part is not at E2 - E_b1; Markov not applicable
    P1_full = np.sum(Z2 * (1 - Z2)) + (1 - np.sum(Z2)) * P1
    return dict(P1_markov=P1_full, P2_markov=P2, Z1=Z1, Eb1=Eb1, T1=T1, frac_band=frac_band,
                P1_dressed=np.sum(Z2 * (1 - Z2)) + (1 - np.sum(Z2)) * Z1,
                P1_bare=np.sum(Z2 * (1 - Z2)) + (1 - np.sum(Z2)) * T1, E=E, w1=w1, rate=rate)


def spectator_H(OmR, dev, w_k, F_k):
    g_k = couplings(w_k, F_k, dev["kappa"], OmR)
    E, V = one_quantum(OmR, g_k, w_k)
    J, M = len(E), len(w_k)
    diag = np.concatenate([[2 * OmR - dev["alpha"]], (E[:, None] + w_k[None, :]).ravel()])
    cpl = (np.sqrt(2) * V[0, :][:, None] * g_k[None, :]).ravel()
    idx = 1 + np.arange(J * M)
    H = sp.coo_matrix((np.concatenate([cpl, cpl]),
                       (np.concatenate([np.zeros(J * M, int), idx]), np.concatenate([idx, np.zeros(J * M, int)]))),
                      shape=(1 + J * M, 1 + J * M))
    return (H + sp.diags(diag)).tocsr(), V[0, :]**2, J, M


def run_spectator(OmR, dev, w_k, F_k, T, T_avg=20.0):
    H, w1, J, M = spectator_H(OmR, dev, w_k, F_k)
    psi = np.zeros(H.shape[0], complex)
    psi[0] = 1
    psi = expm_multiply(-1j * H * (T - T_avg), psi)          # single step to T - T_avg (interval mode overflows)
    states = expm_multiply(-1j * H, psi, start=0.0, stop=T_avg, num=9, endpoint=True)
    out = []
    for s in states:
        p = np.abs(s)**2
        pj = p[1:].reshape(J, M).sum(axis=1)
        out.append([p[0], np.sum(pj * w1), np.sum(pj * (1 - w1))])
    return np.mean(out, axis=0)


def run_full(OmR, dev, w_k, F_k, T, T_avg=20.0):
    g_k = couplings(w_k, F_k, dev["kappa"], OmR)
    H, i1, ip = two_quantum_H(OmR, dev["alpha"], g_k, w_k)
    psi = np.zeros(H.shape[0], complex)
    psi[0] = 1
    psi = expm_multiply(-1j * H * (T - T_avg), psi)
    states = expm_multiply(-1j * H, psi, start=0.0, stop=T_avg, num=9, endpoint=True)
    p = np.abs(states)**2
    return np.mean(np.stack([p[:, 0], p[:, i1].sum(1), p[:, ip].sum(1)], axis=1), axis=0)


# ============================================================ Markov with two-photon exchange (T-matrix, second order)
def markov_exchange(OmR, dev, w_f, F_f, eta, n_grid=6000):
    """
    Decay channels of |2> (golden rule, energies E2 = 2 OmR - alpha):
        B1 channel   : |B1; k>      rate  G_B  = 2 Z1 Gph(E2 - Eb1)
        two photons  : |0; k, l>    amplitude M_kl = sqrt2 g_k g_l [G1(w_l) + G1(w_k)]     (both emission orderings)
                       rate G_2g = (1/pi) int dw Gph(w) Gph(E2-w) { |G1(w)|^2  +  Re[G1(w) G1*(E2-w)] }
                                    = direct (= spectator band rate)          + exchange
    G1(z) = 1/(z - OmR - Sigma(z+i0)) is the RWA one-quantum resolvent; Im Sigma = -Gph/2 exactly, Re Sigma from the
    ring mode sum (fine ring w_f, F_f, broadening eta).  P1 = Z1 G_B / (G_B + G_2g).
    """
    kap, zf, Cg = dev["kappa"], dev["zf"], dev["Cg"]
    E2 = 2 * OmR - dev["alpha"]

    def Sigma(z):
        z = np.atleast_1d(z)
        d = z[:, None] - w_f[None, :]
        re = OmR * np.sum(kap * w_f * F_f**2 / 4 * d / (d**2 + eta**2), axis=1)
        im = -0.5 * np.clip(gamma_ph_safe(z, OmR, kap, zf, Cg), 0, None)
        return re + 1j * im

    def G1(z):
        return 1.0 / (z - OmR - Sigma(z))

    # bound qubit: RWA root of E = OmR + Re Sigma(E) in the gap (Sigma real there; use eigen-decomposition instead)
    g_k = couplings(w_f, F_f, kap, OmR)
    E, V = one_quantum(OmR, g_k, w_f)
    inside = (E > dev["gap"][0]) & (E < dev["gap"][1])
    jb = np.argmax(np.where(inside, V[0, :]**2, -1))
    Z1, Eb1 = V[0, jb]**2, E[jb]

    G_B = 2 * Z1 * gamma_ph_safe(E2 - Eb1, OmR, kap, zf, Cg)
    w = np.linspace(1e-3, E2 - 1e-3, n_grid)
    Ga = gamma_ph_safe(w, OmR, kap, zf, Cg)
    Gb = gamma_ph_safe(E2 - w, OmR, kap, zf, Cg)
    mask = (Ga > 0) & (Gb > 0)
    w, Ga, Gb = w[mask], Ga[mask], Gb[mask]
    if len(w) == 0:
        return dict(P1=float(Z1), P1_noex=float(Z1), Z1=Z1, G_B=float(G_B), G_dir=0.0, G_ex=0.0)
    if G_B <= 0:                      # leakage photon would sit in a gap: leakage rung is bound, not a decay problem
        return dict(P1=np.nan, P1_noex=np.nan, Z1=Z1, G_B=0.0, G_dir=0.0, G_ex=0.0)
    ga = G1(w)
    gb = G1(E2 - w)
    dw = (E2 - 2e-3) / (n_grid - 1)
    G_dir = np.sum(Ga * Gb * np.abs(ga)**2) * dw / np.pi
    G_ex = np.sum(Ga * Gb * np.real(ga * np.conj(gb))) * dw / np.pi
    return dict(P1=float(Z1 * G_B / (G_B + G_dir + G_ex)), P1_noex=float(Z1 * G_B / (G_B + G_dir)),
                Z1=Z1, G_B=float(G_B), G_dir=float(G_dir), G_ex=float(G_ex))


def gamma_ph_safe(w, Om, kappa, zf, Cg):
    """emission function, set to exactly 0 outside pass bands (|z| > 1); the i*ETA regulator leaves ~1e-10 there,
    which multiplied by |G1|^2 near the qubit pole gives garbage"""
    w = np.atleast_1d(np.asarray(w, dtype=float))
    out = np.zeros_like(w)
    ok = (w > 1e-6) & (np.abs(zf(np.maximum(w, 1e-6))) < 1 - 1e-12)
    out[ok] = np.clip(kb.gamma_ph(w[ok], Om, kappa, zf, Cg), 0, None)
    return out if out.size > 1 else out[0]
