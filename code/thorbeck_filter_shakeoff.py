"""
thorbeck_filter_shakeoff.py
Photonic shake-off in the passive leakage filter of Thorbeck et al., arXiv:2411.04101, using THEIR circuit model:
  Eqs. (3)-(5): 7th-order Chebyshev chain (RWA), omega_0 = 3.6 GHz, Delta_omega = 1.8 GHz (cutoff 4.5 GHz),
                0.1 dB ripple, port loss kappa_f on site 7;   Eq. (7): transmon on site 1, g_ge = g, g_fe = sqrt2 g.
Units: GHz (frequency / 2pi), times in ns.

Quantity: L = 1 - (probability that an |f> event returns the transmon to its dressed |e>).
Method (exact for this model): no-jump evolution of |f,vac> in the two-excitation sector (36 states); after the first
photon leaves (jump sqrt(kappa_f) f_7), project onto the long-lived dressed-|e> eigenmode of the one-excitation
non-Hermitian Hamiltonian (right r, left l):
      1 - L = ||r||^2  int_0^inf dt | l^T J exp(-i H2 t) |f,0> |^2       (closed form from the eigen-decomposition)
Checks printed: jump sum rule (= 1) and a full Lindblad integration (agreement ~2e-3 where T1e >> T1f).
NOTE: L is only meaningful where the dressed |e> is long-lived (T1e >> T1f); near 4.5-4.6 GHz with the unshifted
filter this fails (T1e < 1 us) and the numbers there are indicative only.
Calibration: the published parameters put the T1f minimum ~150 MHz above the measured 4.745 GHz; we also report the
filter rigidly shifted down by 100-150 MHz so that the T1f minimum moves to the measured value.
"""
import numpy as np
import scipy.linalg as sl

L = 7
W0, DW = 3.6, 1.8
ETA = np.sqrt(10 ** (0.1 / 10) - 1)
ALPHA = -0.325
BETA = np.arcsinh(1 / ETA) / L
J = [DW / 4 * abs(np.sin(n * np.pi / L + 1j * BETA))
     / np.sqrt(np.sin((2 * n - 1) * np.pi / (2 * L)) * np.sin((2 * n + 1) * np.pi / (2 * L))) for n in range(1, L)]
KF = DW * np.sinh(BETA) / (2 * np.sin(np.pi / (2 * L)))
PAIRS = [(n, m) for n in range(L) for m in range(n, L)]
S = np.zeros((L * L, len(PAIRS)))                    # symmetric two-photon basis -> product space
for i, (n, m) in enumerate(PAIRS):
    if n == m:
        S[n * L + n, i] = 1
    else:
        S[n * L + m, i] = S[m * L + n, i] = 1 / np.sqrt(2)


def a_dag(site):
    """matrix of a_site^dagger from one-photon states |1_n> to the symmetric two-photon basis"""
    A = np.zeros((len(PAIRS), L))
    for n in range(L):
        v = np.zeros(L * L)
        v[site * L + n] += 1 / np.sqrt(2)
        v[n * L + site] += 1 / np.sqrt(2)
        A[:, n] = S.T @ v * (np.sqrt(2) if n == site else 1)
    return A


A0, AL = a_dag(0), a_dag(L - 1)


def filt(shift=0.0):
    Hf = np.diag([W0 + shift] * L).astype(complex)
    for n in range(L - 1):
        Hf[n, n + 1] = Hf[n + 1, n] = J[n]
    Hf[L - 1, L - 1] -= 0.5j * KF
    return Hf


def build(Om, g, shift=0.0):
    Hf = filt(shift)
    I = np.eye(L)
    H1 = np.zeros((L + 1, L + 1), complex)           # |e,0>, |g,1_n>
    H1[0, 0] = Om
    H1[1:, 1:] = Hf
    H1[0, 1] = H1[1, 0] = g
    P = len(PAIRS)
    D = 1 + L + P                                    # |f,0>, |e,1_n>, |g,1_n 1_m>
    H2 = np.zeros((D, D), complex)
    H2[0, 0] = 2 * Om + ALPHA
    H2[0, 1] = H2[1, 0] = np.sqrt(2) * g
    H2[1:1 + L, 1:1 + L] = Om * np.eye(L) + Hf
    H2[1 + L:, 1 + L:] = S.T @ (np.kron(Hf, I) + np.kron(I, Hf)) @ S
    H2[1 + L:, 1:1 + L] += g * A0
    H2[1:1 + L, 1 + L:] += g * A0.T
    Jm = np.zeros((L + 1, D))
    Jm[0, L] = 1                                     # |e,1_7> -> |e,0>
    Jm[1:, 1 + L:] = AL.T                            # |g,..1_7..> -> |g,1_n>
    return H1, H2, np.sqrt(KF) * Jm


def analyse(Om, g, shift=0.0):
    H1, H2, Jm = build(Om, g, shift)
    mu, R = np.linalg.eig(H2)
    Rinv = np.linalg.inv(R)
    lam, r = np.linalg.eig(H1)
    k = np.argmax(lam.imag)
    l, rr = np.linalg.inv(r)[k], r[:, k]
    M = -1j / (mu[:, None] - mu.conj()[None, :])
    a = (l @ Jm @ R) * Rinv[:, 0]
    keep = np.real(a @ M @ a.conj()) * np.vdot(rr, rr).real
    b = Jm @ R * Rinv[:, 0][None, :]
    sumrule = np.real(np.einsum('ik,kl,il->', b, M, b.conj()))
    T1e = -1 / (2 * lam[k].imag) / (2 * np.pi)                     # ns
    kf = np.argmax(np.abs(R[0]) ** 2 / np.sum(np.abs(R) ** 2, 0))
    T1f = -1 / (2 * mu[kf].imag) / (2 * np.pi)
    return dict(L=1 - keep, Z1=abs(rr[0]) ** 2 / np.vdot(rr, rr).real, T1e_us=T1e / 1e3, T1f_ns=T1f, sumrule=sumrule)


def lindblad_check(Om, g, shift, T=1000.0):
    H1, H2, Jm = build(Om, g, shift)
    d1, d2 = H1.shape[0], H2.shape[0]
    D = 1 + d1 + d2
    H = np.zeros((D, D), complex)
    H[1:1 + d1, 1:1 + d1] = H1
    H[1 + d1:, 1 + d1:] = H2
    Jf = np.zeros((D, D), complex)
    Jf[1:1 + d1, 1 + d1:] = Jm
    Jf[0, L] = np.sqrt(KF)                           # |g,1_7> -> vac
    I = np.eye(D)
    Ls = -1j * (np.kron(H, I) - np.kron(I, H.conj())) + np.kron(Jf, Jf.conj())
    rho0 = np.zeros((D, D), complex)
    rho0[1 + d1, 1 + d1] = 1
    rho = (sl.expm(Ls * T * 2 * np.pi) @ rho0.ravel()).reshape(D, D)
    r = analyse(Om, g, shift)                       # undo the slow |e> decay after the |f> transient (t > T1f)
    return np.trace(rho[1:1 + d1, 1:1 + d1]).real * np.exp((T - r["T1f_ns"]) / (r["T1e_us"] * 1e3))


if __name__ == "__main__":
    print(f"kappa_f = {KF:.3f} GHz; filter modes (no loss): {np.sort(np.linalg.eigvalsh(filt().real)).round(3)}")
    for w0_, sh_ in ((4.62, -0.10), (4.60, -0.125)):
        r = analyse(w0_, 0.02, sh_)
        print(f"check at {w0_} GHz, shift {sh_*1e3:+.0f} MHz: jump formula 1-L = {1-r['L']:.4f}, "
              f"Lindblad = {lindblad_check(w0_, 0.02, sh_, T=700.0):.4f},"
              f" sum rule {r['sumrule']:.4f}")
    oms = np.round(np.arange(4.45, 5.0001, 0.005), 3)
    out = {}
    for shift in (0.0, -0.10, -0.15):
        for g in (0.015, 0.02, 0.025):
            rows = np.array([[analyse(w, g, shift)[q] for q in ("L", "Z1", "T1e_us", "T1f_ns")] for w in oms])
            out[f"s{int(round(-shift*1e3))}_g{int(round(g*1e3))}"] = rows
            i = np.argmin(rows[:, 3])
            pick = lambda w: rows[np.argmin(abs(oms - w))]
            print(f"shift {shift*1e3:+5.0f} MHz, g {g*1e3:.0f} MHz: T1f min {rows[i,3]:.0f} ns at {oms[i]:.3f} GHz;  L at "
                  + "  ".join(f"{w:.3f}:{pick(w)[0]:.4f}" for w in (4.60, 4.65, 4.70, 4.745, 4.80, 4.85, 4.90))
                  + f";  T1e(4.6,4.7) = {pick(4.6)[2]:.3g}, {pick(4.7)[2]:.3g} us")
    np.savez("thorbeck_filter_shakeoff.npz", omega_ge=oms, **out)
