"""Phase of the subdominant eigenvalue of P across the period-3 bifurcation (tau_f/2 = 4.740202), fine scan."""
import json, os, sys, numpy as np, scipy.sparse.linalg as sla
from cavity import Grid, images, transfer, OUT
g = Grid(); out = os.path.join(OUT, 'cav_sweep_fine.json')
res = json.load(open(out)) if os.path.exists(out) else []
done = {r['half'] for r in res}
for half in np.round(np.arange(4.730, 4.7501, 0.001), 4):
    if float(half) in done: continue
    P = transfer(g, *images(g, half, 1))
    w = sla.eigs(P.T.tocsc(), k=10, which='LM', return_eigenvectors=False)
    res.append(dict(half=float(half), P=[[z.real, z.imag] for z in w])); json.dump(res, open(out, 'w'))
    c = w[np.argmin(np.abs(np.angle(w) - 2 * np.pi / 3))]
    print(half, abs(c), np.degrees(np.angle(c)), flush=True)
