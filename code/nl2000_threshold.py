"""
nl2000_threshold.py
Threshold scan in the N&L 2000 model: lower transition bound (d23 = -1), upper transition in the band (d12 > 0).
Two-photon channel open iff Delta2 = d12 + d23 > 0.  Exact two-excitation numerics, late-time <P2> vs Z.
"""
import numpy as np
from nl2000_reproduce import evolve, lower_bound_state
d23 = -1.0
yb, Z = lower_bound_state(d23)
rows = []
for d12 in [0.3, 0.5, 0.7, 0.85, 0.95, 1.05, 1.15, 1.31, 1.5, 1.8, 2.2, 2.8]:
    t, p1, p2, p3 = evolve(d12, d23, T=60.0, N=250, wu=20.0)
    late = t > 40
    rows.append([d12, d12 + d23, p1[late].mean(), p2[late].mean(), p3[late].mean()])
    print(f"d12={d12:.2f}  Delta2={d12+d23:+.2f}:  P1={p1[late].mean():.3f}  P2={p2[late].mean():.3f}  "
          f"P3={p3[late].mean():.3f}   P2/Z={p2[late].mean()/Z:.3f}", flush=True)
print(f"Z (lower bound state) = {Z:.4f}")
np.save("nl2000_threshold.npy", np.array(rows))
