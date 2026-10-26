"""
nl2000_reproduce.py
Reproduce Nikolopoulos & Lambropoulos, PRA 61, 053812 (2000): ladder atom |1> -> |2> -> |3>, both transitions coupled
to the same isotropic photonic-band-gap reservoir, and test our branching theory on their Fig. 4(a).

Their model (Secs. II-III):
  photon energies x = w - w_e >= 0 (band above the edge, gap below)
  spectral response R(x) = (C/pi) / sqrt(x)          ->  golden-rule rate 2 pi R = 2C/sqrt(x)
  self-energy           Sigma(y) = -C / sqrt(-y - i0) (y = energy of the excitation measured from w_e)
  discretization        x_j = j^2 dw, j = 1..N, w_u = N^2 dw;  constant coupling G = sqrt(2C/(N pi)) * w_u^(1/4)
                        modes above w_u: perturbative static shift
  units                 C1 = 1, frequencies in C1^(2/3), times in C1^(-2/3);  C2 = 1.5 C1
Basis: |1;0>, |2;j>, |3;j,m> (j<=m).  Energies (minus 2 w_e): d12+d23, d23+x_j, x_j+x_m.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
from scipy.integrate import quad
from scipy.optimize import brentq

C1, C2 = 1.0, 1.5


def shift_above(C, d, xu):
    """static shift from modes above w_u:  int_xu^inf (C/pi)/sqrt(x) / (d - x) dx"""
    return quad(lambda x: (C / np.pi) / np.sqrt(x) / (d - x), xu, np.inf, limit=200)[0]


def build(d12, d23, N=150, wu=10.0):
    dw = wu / N**2
    x = dw * np.arange(1, N + 1)**2
    G1 = np.sqrt(2 * C1 / (N * np.pi)) * wu**0.25
    G2 = np.sqrt(2 * C2 / (N * np.pi)) * wu**0.25
    sa = shift_above(C1, d12, wu)
    sb = shift_above(C2, d23, wu)
    kk, ll = np.triu_indices(N)
    P = len(kk)
    i2 = 1 + np.arange(N)
    i3 = 1 + N + np.arange(P)
    dim = 1 + N + P
    diag = np.concatenate([[d12 + d23 + sa], d23 + x + sb, x[kk] + x[ll]])
    rows = [np.zeros(N, int)]
    cols = [i2]
    vals = [np.full(N, G1)]
    same = kk == ll
    rows += [i2[kk[same]]]
    cols += [i3[same]]
    vals += [np.full(same.sum(), np.sqrt(2) * G2)]
    diff = ~same
    rows += [i2[kk[diff]], i2[ll[diff]]]
    cols += [i3[diff], i3[diff]]
    vals += [np.full(diff.sum(), G2), np.full(diff.sum(), G2)]
    r, c, v = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    H = sp.coo_matrix((np.concatenate([v, v]), (np.concatenate([r, c]), np.concatenate([c, r]))), shape=(dim, dim))
    return (H + sp.diags(diag)).tocsr(), i2, i3


def evolve(d12, d23, T=40.0, n_t=401, N=150, wu=10.0):
    H, i2, i3 = build(d12, d23, N, wu)
    psi = np.zeros(H.shape[0], complex)
    psi[0] = 1
    S = expm_multiply(-1j * H, psi, start=0, stop=T, num=n_t, endpoint=True)
    p = np.abs(S)**2
    return np.linspace(0, T, n_t), p[:, 0], p[:, i2].sum(1), p[:, i3].sum(1)


# ============================================================ our theory in their model
def Sigma(y, C):
    return -C / np.sqrt(-y - 1j * 1e-14 + 0j)


def lower_bound_state(d23):
    """bound state of the lower transition below the edge: y = d23 + Sigma(y), y < 0; weight Z"""
    f = lambda s: (-s) - d23 + C2 / np.sqrt(s)          # s = -y > 0
    s = brentq(f, 1e-9, 100)
    Z = 1 / (1 + C2 / (2 * s**1.5))
    return -s, Z


def branching(d12, d23):
    """P(|2>, t->inf) = Z * G_B / (G_B + G_dir + G_ex)  (Markov, total energy E = d12 + d23 above 2 w_e)"""
    E = d12 + d23
    yb, Z = lower_bound_state(d23)
    G1 = lambda x: 2 * C1 / np.sqrt(x)                   # emission functions (photon energy x above edge)
    G2 = lambda x: 2 * C2 / np.sqrt(x)
    G12 = lambda x: 2 * np.sqrt(C1 * C2) / np.sqrt(x)
    g1 = lambda y: 1.0 / (y - d23 - Sigma(y, C2))        # resolvent of the lower transition
    GB = Z * G1(E - yb) if E - yb > 0 else 0.0
    if E <= 0:
        return Z, yb, GB, 0.0, 0.0, Z
    # x = E sin^2(th) removes both 1/sqrt endpoint singularities
    def integ(th, kind):
        x = E * np.sin(th)**2
        jac = 2 * E * np.sin(th) * np.cos(th)
        if kind == "dir":
            v = G1(x) * G2(E - x) * abs(g1(E - x))**2
        else:
            v = G12(x) * G12(E - x) * np.real(g1(E - x) * np.conj(g1(x)))
        return v * jac / (2 * np.pi)
    Gd = quad(integ, 1e-9, np.pi / 2 - 1e-9, args=("dir",), limit=200)[0]
    Gx = quad(integ, 1e-9, np.pi / 2 - 1e-9, args=("ex",), limit=200)[0]
    return Z * GB / (GB + Gd + Gx), yb, GB, Gd, Gx, Z


CASES = {   # name: (d12, d23)  in units C1^(2/3); C2^(2/3) = 1.5^(2/3)
    "Fig2":  (-1.5**(2/3), 0.0),
    "Fig4a": (1.0 * 1.5**(2/3), -1.0),
    "Fig4b": (2.0 * 1.5**(2/3), 3.0),
    "Fig5a": (-2.0 * 1.5**(2/3), 2.0 * 1.5**(2/3)),
    "Fig6":  (-2.0 * 1.5**(2/3), 4.0 * 1.5**(2/3)),
}

if __name__ == "__main__":
    out = {}
    for name, (d12, d23) in CASES.items():
        T = 15.0 if name == "Fig4b" else 40.0
        t, p1, p2, p3 = evolve(d12, d23, T=T)
        out[name] = np.stack([t, p1, p2, p3])
        late = t > 0.75 * T
        print(f"{name}: d12={d12:+.3f} d23={d23:+.3f}  Delta2={d12+d23:+.3f} | late-time <P1>={p1[late].mean():.3f} "
              f"<P2>={p2[late].mean():.3f} <P3>={p3[late].mean():.3f}", flush=True)

    # convergence of Fig. 4(a) in the discretization
    for N, wu in [(150, 10.0), (250, 20.0), (350, 40.0)]:
        t, p1, p2, p3 = evolve(*CASES["Fig4a"], T=40.0, N=N, wu=wu)
        late = t > 30
        print(f"Fig4a convergence N={N} w_u={wu}: <P2>={p2[late].mean():.4f} <P3>={p3[late].mean():.4f}", flush=True)

    P2th, yb, GB, Gd, Gx, Z = branching(*CASES["Fig4a"])
    print(f"\nTheory for Fig. 4(a): lower bound state y_b={yb:.4f}, Z={Z:.4f} (dressed upper limit for P2)")
    print(f"   G_B={GB:.4f}  G_dir={Gd:.4f}  G_ex={Gx:+.4f}  ->  P2 = {P2th:.4f}   (no exchange: {Z*GB/(GB+Gd):.4f};"
          f" shake-off/bare: {Z**2:.4f})")
    np.savez("nl2000_reproduce.npz", **out)
