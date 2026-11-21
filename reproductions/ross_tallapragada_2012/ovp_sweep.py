"""Overnight sweep: turnstile flux of the oscillating vortex pair over (gamma, eps), against Melnikov.

For each (gamma, eps): the saddle p+ of the period map, W^u(p+) up to its first pip q on the symmetry line
x = 0 (reversibility), the core R1, and the exact one-period flux mu(R1 \\ f(R1)) by advecting R1's boundary
(= the turnstile area, whatever the number of lobes).  First order: the Melnikov function of the exact model
along the unperturbed separatrix, M(t0), and mu_1 = eps * integral over one period of max(M, 0).

python ovp_sweep.py melnikov            # first-order curve (fast)
python ovp_sweep.py grid [i0 i1]         # the (gamma, eps) grid, resumable (one json per point)
python ovp_sweep.py plot
"""
import os, sys, json, time, glob
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.validation import make_valid
import ovp as O

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out', 'sweep'); os.makedirs(OUT, exist_ok=True)
GAMS = np.round(np.r_[np.linspace(0.2, 3.0, 57)], 4)
EPSS = np.round([0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2], 4)


def shp(P):
    g = Polygon(P); return g if g.is_valid else make_valid(g)


def flux_point(gam, eps, ds=0.004):
    o = O.OVP(eps, gam, steps=max(200, int(400 * gam / 0.5) if gam > 0.5 else 400))
    x0 = 1.732 * np.exp(-eps)  # start near the saddle
    pR = o.fixed_point(x0)
    lam = float(np.max(np.abs(np.linalg.eigvals(o._jac(pR)))))
    Wu = o.manifold(pR, 'u', length=8.0, branch=+1, ds=ds)
    if Wu[5, 1] < 0: Wu = o.manifold(pR, 'u', length=8.0, branch=-1, ds=ds)
    cr = np.where((np.sign(Wu[1:, 0]) != np.sign(Wu[:-1, 0])) & (Wu[:-1, 0] != 0))[0]
    i = cr[0]; a, b = Wu[i], Wu[i + 1]; q = a + a[0] / (a[0] - b[0]) * (b - a)
    up = np.vstack([Wu[:i + 1], q[None]]); upper = np.vstack([up, np.c_[-up[:, 0], up[:, 1]][::-1][1:]])
    poly = np.vstack([upper, np.c_[upper[:, 0], -upper[:, 1]][::-1][1:-1]])
    R1 = shp(poly)
    # one period of the boundary, with point insertion
    P = np.vstack([poly, poly[:1]])
    X, Y = o.map(P[:, 0], P[:, 1]); Q = np.c_[X, Y]
    for _ in range(40):
        d = np.hypot(*np.diff(Q, axis=0).T); bad = np.where(d > ds)[0]
        if not len(bad): break
        M = .5 * (P[bad] + P[bad + 1]); Xm, Ym = o.map(M[:, 0], M[:, 1])
        P = np.insert(P, bad + 1, M, axis=0); Q = np.insert(Q, bad + 1, np.c_[Xm, Ym], axis=0)
    F = shp(Q[:-1])
    out = R1.difference(F).area
    return dict(gamma=float(gam), eps=float(eps), pR=pR.tolist(), lam=lam, q=q.tolist(), area_R1=R1.area,
                flux=out, flux_frac=out / R1.area, area_err=abs(F.area - R1.area), npip_axis=int(len(cr)))


def separatrix(n=30000, T=30.0):
    """Unperturbed upper separatrix from +sqrt(3) to -sqrt(3), parametrised by time (t = 0 at x = 0)."""
    from scipy.integrate import solve_ivp
    def f(t, z):
        u, v = O.vel(z[0], z[1], 0.0, 0.0, 1.0); return [u, v]
    # top of the separatrix: psi = 0 on x = 0 -> y with -ln((y-1)^2/(y+1)^2)/2 ... find by integrating from top
    from scipy.optimize import brentq
    psi = lambda y: -0.5 * np.log((y - 1) ** 2 / (y + 1) ** 2) - 0.5 * y
    ytop = brentq(psi, 1.2, 3.0)
    t = np.linspace(0, T, n)
    fw = solve_ivp(f, (0, T), [0, ytop], t_eval=t, rtol=1e-12, atol=1e-12).y.T
    bw = solve_ivp(f, (0, -T), [0, ytop], t_eval=-t, rtol=1e-12, atol=1e-12).y.T
    return np.r_[-t[::-1], t[1:]], np.vstack([bw[::-1], fw[1:]]), ytop


def melnikov(gam, nt0=128, d=1e-6):
    ts, Q, _ = separatrix()
    f0 = np.array([O.vel(x, y, 0.0, 0.0, gam) for x, y in Q])
    t0s = np.linspace(0, 2 * np.pi * gam, nt0, endpoint=False); M = []
    for t0 in t0s:
        g = np.array([np.subtract(O.vel(x, y, t + t0, d, gam), O.vel(x, y, t + t0, 0.0, gam)) / d for (x, y), t in zip(Q, ts)])
        M.append(np.trapezoid(f0[:, 0] * g[:, 1] - f0[:, 1] * g[:, 0], ts))
    M = np.array(M)
    return t0s, M, float(2 * np.trapezoid(np.maximum(M, 0), t0s))  # both halves (upper + mirror lobe), per unit eps


def stage_melnikov():
    res = []
    for g in np.round(np.linspace(0.1, 3.0, 59), 4):
        t0, M, a1 = melnikov(g); res.append(dict(gamma=float(g), area_per_eps=a1, Mmax=float(M.max()), Mmin=float(M.min())))
        print(res[-1], flush=True)
    json.dump(res, open(os.path.join(OUT, 'melnikov.json'), 'w'), indent=1)


def stage_grid(i0=0, i1=None):
    pts = [(g, e) for e in EPSS for g in GAMS]
    for k, (g, e) in enumerate(pts[int(i0):int(i1) if i1 else None]):
        f = os.path.join(OUT, f'pt_g{g:.4f}_e{e:.4f}.json')
        if os.path.exists(f): continue
        t = time.time()
        try:
            r = flux_point(g, e)
        except Exception as ex:
            r = dict(gamma=float(g), eps=float(e), error=repr(ex))
        r['sec'] = time.time() - t; json.dump(r, open(f, 'w'))
        print(g, e, r.get('flux_frac'), r.get('error', ''), f"{r['sec']:.1f}s", flush=True)


def stage_fine(i0=0, i1=None):
    """Fine scan around the transport minimum for each eps (step 0.0025 in gamma)."""
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(OUT, 'pt_*.json'))]
    pts = []
    for e in EPSS:
        rr = sorted([r for r in rows if abs(r['eps'] - e) < 1e-9 and 'flux' in r and r['gamma'] > 0.9], key=lambda r: r['flux'])
        g0 = rr[0]['gamma']
        pts += [(round(g, 4), e) for g in np.arange(g0 - 0.1, g0 + 0.1001, 0.0025)]
    for g, e in pts[int(i0):int(i1) if i1 else None]:
        f = os.path.join(OUT, f'pt_g{g:.4f}_e{e:.4f}.json')
        if os.path.exists(f): continue
        t = time.time()
        try: r = flux_point(g, e)
        except Exception as ex: r = dict(gamma=float(g), eps=float(e), error=repr(ex))
        r['sec'] = time.time() - t; json.dump(r, open(f, 'w'))
        print(g, e, r.get('flux_frac'), r.get('error', ''), f"{r['sec']:.1f}s", flush=True)


def stage_plot():
    mel = {round(r['gamma'], 4): r['area_per_eps'] for r in json.load(open(os.path.join(OUT, 'melnikov.json')))}
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(OUT, 'pt_*.json'))]
    rows = [r for r in rows if 'flux' in r]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    gm = np.array(sorted(mel)); am = np.array([mel[g] for g in gm])
    for j, e in enumerate(EPSS):
        rr = sorted([r for r in rows if abs(r['eps'] - e) < 1e-9], key=lambda r: r['gamma'])
        if not rr: continue
        g = np.array([r['gamma'] for r in rr]); fl = np.array([r['flux'] for r in rr])
        c = plt.cm.viridis(j / (len(EPSS) - 1))
        axs[0].semilogy(g, fl / e, 'o-', ms=2.5, color=c, lw=.8, label=fr'$\epsilon$ = {e:g}')
        axs[1].plot(g, fl / (e * np.interp(g, gm, am)), 'o-', ms=2.5, color=c, lw=.8)
    axs[0].semilogy(gm, am, 'k--', lw=1.2, label='Melnikov (first order)')
    axs[0].set_xlabel(r'$\gamma$'); axs[0].set_ylabel(r'turnstile area / $\epsilon$'); axs[0].legend(fontsize=7, ncol=2)
    axs[1].axhline(1, color='k', ls='--', lw=.8); axs[1].set_yscale('log')
    axs[1].set_xlabel(r'$\gamma$'); axs[1].set_ylabel('true flux / Melnikov prediction')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'out', 'ovp_sweep_flux.png'), dpi=160); plt.close(fig)



def stage_minimum():
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(OUT, 'pt_*.json'))]
    mel = json.load(open(os.path.join(OUT, 'melnikov.json')))
    gm = np.array([r['gamma'] for r in mel]); am = np.array([r['area_per_eps'] for r in mel])
    k = np.argmin(np.where(gm > 1.5, am, np.inf)); g0 = gm[k]  # coarse Melnikov zero; refine
    from scipy.optimize import minimize_scalar
    g0 = minimize_scalar(lambda g: melnikov(g, nt0=128)[2], bounds=(g0 - 0.05, g0 + 0.05), method='bounded', options=dict(xatol=1e-4)).x
    E, G, F, NP = [], [], [], []
    for e in EPSS:
        rr = sorted([r for r in rows if abs(r['eps'] - e) < 1e-9 and 'flux' in r and r['gamma'] > 0.9], key=lambda r: r['gamma'])
        g = np.array([r['gamma'] for r in rr]); f = np.array([r['flux'] for r in rr]); i = np.argmin(f)
        c = np.polyfit(g[i - 3:i + 4] - g[i], f[i - 3:i + 4], 2); gs = g[i] - c[1] / (2 * c[0])
        E.append(e); G.append(gs); F.append(np.polyval(c, gs - g[i])); NP.append(rr[i]['npip_axis'])
    E, G, F = map(np.array, (E, G, F))
    pg = np.polyfit(E[:4], G[:4], 1); pf = np.polyfit(E[:4], F[:4] / E[:4] ** 2, 1)
    res = dict(melnikov_zero=float(g0), eps=E.tolist(), gamma_star=G.tolist(), min_flux=F.tolist(), npips_axis=NP,
               fit_gamma_star=pg.tolist(), fit_flux_over_eps2=pf.tolist())
    json.dump(res, open(os.path.join(HERE, 'out', 'ovp_minimum.json'), 'w'), indent=1); print(json.dumps(res, indent=1))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].plot(E, G, 'o-', label=r'$\gamma^*(\epsilon)$: forcing period of least transport')
    axs[0].axhline(g0, color='k', ls='--', lw=.8, label=fr'Melnikov zero, $\gamma_0$ = {g0:.4f}')
    axs[0].set_xlabel(r'$\epsilon$'); axs[0].set_ylabel(r'$\gamma$'); axs[0].legend(fontsize=8)
    axs[1].plot(E, F / E ** 2, 'o-', label='true flux at $\\gamma^*$')
    fz = [flux_at(g0, e, rows) for e in E]
    axs[1].plot(E, np.array(fz) / E ** 2, 's--', label=r'true flux at the Melnikov zero $\gamma_0$')
    axs[1].set_xlabel(r'$\epsilon$'); axs[1].set_ylabel(r'one-period flux / $\epsilon^2$'); axs[1].legend(fontsize=8)
    axs[1].set_ylim(0, None)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'out', 'ovp_minimum.png'), dpi=160); plt.close(fig)


def flux_at(g, e, rows):
    rr = sorted([r for r in rows if abs(r['eps'] - e) < 1e-9 and 'flux' in r], key=lambda r: r['gamma'])
    return np.interp(g, [r['gamma'] for r in rr], [r['flux'] for r in rr])


if __name__ == '__main__':
    st = sys.argv[1] if len(sys.argv) > 1 else 'plot'
    globals()['stage_' + st](*sys.argv[2:])
