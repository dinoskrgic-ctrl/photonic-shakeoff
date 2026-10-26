"""
audit_checks.py  -- numbers quoted in response to the internal audit
 (1) Non-adjacent band pairs: Eq. (G2) evaluated band pair by band pair; loss from pairs other than (top band, top band)
     at several f = (Omega - w_e)/alpha, KC and KP gaps 1-4.
 (2) Ring return times N / v_g,max for the bands carrying the leakage / shake-off photons (v_g = |sin q / z'(w)|).
 (3) Validation errors split by half-window and the dropped-run list (P2 > 0.05), from selfion_KC/KP.npz.
"""
import numpy as np
import kc_selfion as si
import kc_bath as kb


def split(OmR, dev, wf, Ff, eta=0.004, n_grid=20000):
    kap, zf, Cg = dev["kappa"], dev["zf"], dev["Cg"]; E2 = 2*OmR-dev["alpha"]
    g_k = si.couplings(wf, Ff, kap, OmR); E, V = si.one_quantum(OmR, g_k, wf)
    inside=(E>dev["gap"][0])&(E<dev["gap"][1]); jb=np.argmax(np.where(inside,V[0]**2,-1)); Z1,Eb1=V[0,jb]**2,E[jb]
    def G1(z):
        d=z[:,None]-wf[None,:]
        re=OmR*np.sum(kap*wf*Ff**2/4*d/(d**2+eta**2),axis=1)
        im=-0.5*np.clip(si.gamma_ph_safe(z,OmR,kap,zf,Cg),0,None)
        return 1/(z-OmR-re-1j*im)
    GB=2*Z1*si.gamma_ph_safe(E2-Eb1,OmR,kap,zf,Cg)
    w=np.linspace(1e-3,E2-1e-3,n_grid); dw=w[1]-w[0]
    Ga=si.gamma_ph_safe(w,OmR,kap,zf,Cg); Gb=si.gamma_ph_safe(E2-w,OmR,kap,zf,Cg)
    m=(Ga>0)&(Gb>0); w,Ga,Gb=w[m],Ga[m],Gb[m]
    bl=kb.bands(zf,60.0); top=[i for i,(lo,hi) in enumerate(bl) if abs(hi-dev["gap"][0])<1e-2][0]
    bi=lambda x: np.array([next((i for i,(lo,hi) in enumerate(bl) if lo-1e-9<=xx<=hi+1e-9),-1) for xx in x])
    b1,b2=bi(w),bi(E2-w)
    if len(w):
        ga,gb=G1(w),G1(E2-w); integ=Ga*Gb*(np.abs(ga)**2+np.real(ga*np.conj(gb)))*dw/np.pi
    else: integ=np.zeros(0)
    adj=(b1==top)&(b2==top)
    GSa,GSn=integ[adj].sum(),integ[~adj].sum()
    pairs=sorted(set(zip(np.minimum(b1,b2)[~adj],np.maximum(b1,b2)[~adj])))
    return Z1,GB,GSa,GSn,pairs


def nonadjacent():
    print("== (1) non-adjacent band pairs (Markov, Eq. G2)")
    for LAT in ("KC", "KP"):
        for n in (1, 2, 3, 4):
            dev = si.device(LAT, n); we = dev["gap"][0]
            w, F = si.bath(dev, N=250); wf, Ff = si.bath(dev, N=1000)
            print(f"{LAT} gap {n}: w_e={we:.4f} alpha={dev['alpha']:.4f}")
            for f in (0.3, 0.45, 0.55, 0.6, 0.75, 0.9):
                c = si.calibrate(we + f * dev["alpha"], dev, w, F)
                if c is None:
                    continue
                Z1, GB, GSa, GSn, pairs = split(c[2], dev, wf, Ff)
                tot = GB + GSa + GSn
                print(f"   f={f:.2f}  L_total={(GSa+GSn)/tot:.2e}  L_nonadjacent={GSn/tot:.2e}  pairs={[tuple(map(int,p)) for p in pairs]}")


def recurrence():
    print("== (2) ring return times")
    N = {"KC": [250, 250, 160, 140], "KP": [250, 250, 220, 200]}
    for LAT in ("KC", "KP"):
        zf, _ = kb.lattice(LAT); bl = kb.bands(zf, 30.0)
        vg = lambda x: np.abs(np.sqrt(np.clip(1 - zf(x)**2, 0, None)) / ((zf(x + 1e-6) - zf(x - 1e-6)) / 2e-6))
        for n in (1, 2, 3, 4):
            dev = si.device(LAT, n); we = dev["gap"][0]
            lo, hi = [b for b in bl if abs(b[1] - we) < 1e-2][0]
            v = vg(np.linspace(lo + 1e-6, hi - 1e-6, 4000)).max()
            print(f"{LAT} gap {n}: N={N[LAT][n-1]}  v_g,max(band below gap)={v:.3f}  return time={N[LAT][n-1]/v:.0f}")


def validation():
    print("== (3) validation by half-window; dropped runs")
    for LAT in ("KC", "KP"):
        d = np.load(f"selfion_{LAT}.npz")
        for n in (1, 2, 3, 4):
            fu = d[f"full_{n}"]; we, _, al = d[f"meta_{n}"][:3]
            f = (fu[:, 0] - we) / al
            keep = (f < 1) & (fu[:, 10] < 0.05) & (fu[:, 3] < 0.05)
            Lx, Lt = 1 - fu[:, 4] / fu[:, 8], 1 - fu[:, 6] / fu[:, 8]
            for name, sel in (("lower", keep & (f < 0.5)), ("upper", keep & (f >= 0.5))):
                if sel.any():
                    print(f"{LAT}{n} {name}: f={np.round(f[sel],2).tolist()} L_exact={np.round(Lx[sel],3).tolist()} "
                          f"L_theory={np.round(Lt[sel],3).tolist()} max|dP1|={np.abs(fu[sel,4]-fu[sel,6]).max():.3f}")
            drop = (f < 1) & ~keep
            if drop.any():
                print(f"{LAT}{n} dropped: f={np.round(f[drop],2).tolist()} P2_num={np.round(fu[drop,3],3).tolist()} "
                      f"P2_theory={np.round(fu[drop,10],3).tolist()}")


if __name__ == "__main__":
    validation()
    recurrence()
    nonadjacent()
