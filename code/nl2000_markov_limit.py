"""
nl2000_markov_limit.py
Extend the N&L threshold scan deeper into the band (d12 up to 7) to test whether the exact loss converges to the
Markov branching formula as Gamma_up/Delta2 -> 0.  Exact two-excitation numerics (their discretization), lower
transition bound at d23 = -1.  Output: nl2000_markov_limit.npy rows [d12, Delta2, P1, P2, P3, N, wu].
"""
import sys, time, numpy as np
from nl2000_reproduce import evolve
rows = []
# grid chosen so the mode spacing at the photon energy keeps the recurrence time > 90 (averaging window 40-60)
for d12, N, wu in [(3.8, 400, 30.0), (5.0, 450, 30.0), (7.0, 450, 30.0)]:
    t0 = time.time()
    t, p1, p2, p3 = evolve(d12, -1.0, T=60.0, N=N, wu=wu)
    late = t > 40
    rows.append([d12, d12 - 1.0, p1[late].mean(), p2[late].mean(), p3[late].mean(), N, wu])
    print(f"d12={d12}: P1={rows[-1][2]:.4f} P2={rows[-1][3]:.4f} P3={rows[-1][4]:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    np.save("nl2000_markov_limit.npy", np.array(rows))
