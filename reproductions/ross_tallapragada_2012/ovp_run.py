"""Oscillating vortex pair: Ross & Tallapragada (2012) Fig. 11 (manifolds vs FTLE) and Fig. 9 (species
transport by lobe dynamics), with lobe areas checked three ways (boundary advection, Monte Carlo, Lober).

python ovp_run.py [stage ...]   stages: mani ftle lober species mc anim (default: all)
"""
import sys, os, json, subprocess, time
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
from shapely.geometry import Polygon
from shapely.validation import make_valid
from ovp import OVP, flow_many

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out'); os.makedirs(OUT, exist_ok=True)
EPS, GAM = float(os.environ.get('OVP_EPS', 0.1)), float(os.environ.get('OVP_GAM', 0.5))
TAG = f'eps{EPS:g}_gam{GAM:g}'
o = OVP(EPS, GAM)
LOBER = os.environ.get('LOBER', '')


def refl_x(c):  # reversibility R(x, y) = (-x, y): W^s(pL) = R W^u(pR)
    return np.c_[-c[:, 0], c[:, 1]]


def refl_y(c):  # up-down symmetry of the flow
    return np.c_[c[:, 0], -c[:, 1]]


def manifolds(length=40.0):
    f = os.path.join(OUT, f'mani_{TAG}.npz')
    if os.path.exists(f):
        d = np.load(f); return d['pR'], d['Wu']
    pR = o.fixed_point(1.5)
    Wu = o.manifold(pR, 'u', length=length, branch=+1, ds=0.004)
    if Wu[5, 1] < 0: Wu = o.manifold(pR, 'u', length=length, branch=-1, ds=0.004)
    np.savez(f, pR=pR, Wu=Wu); return pR, Wu


def first_cross_x0(c):
    i = np.where(np.sign(c[1:, 0]) != np.sign(c[:-1, 0]))[0]
    i = i[c[i, 0] != 0][0] if len(i) else None
    a, b = c[i], c[i + 1]; s = a[0] / (a[0] - b[0])
    return i, a + s * (b - a)


def region_R1(pR, Wu):
    """Upper boundary B = W^u[pR, q] U W^s[q, pL]; R1 is the core bounded by B and its mirror."""
    i, q = first_cross_x0(Wu)
    up = np.vstack([Wu[:i + 1], q[None]])
    upper = np.vstack([up, refl_x(up)[::-1][1:]])        # pR -> q -> pL
    poly = np.vstack([upper, refl_y(upper)[::-1][1:-1]])  # closed loop via the lower half
    return q, i, upper, poly


def advect_poly(P, k=1, t0=0.0, T=None, ds=0.01, maxpts=4_000_000):
    """Advect a closed curve by the flow from t0 over T (default k periods), inserting points
    wherever neighbours separate by more than ds (done one period at a time)."""
    P = np.vstack([P, P[:1]])
    T = o.Tp * k if T is None else T
    nper = max(1, int(np.ceil(abs(T) / o.Tp - 1e-9))); h = T / nper; t = t0
    for _ in range(nper):
        steps = max(4, int(o.steps * abs(h) / o.Tp))
        X, Y = flow_many(P[:, 0].copy(), P[:, 1].copy(), t, h, steps, EPS, GAM)
        Q = np.c_[X, Y]
        for _ in range(40):
            d = np.hypot(*np.diff(Q, axis=0).T); bad = np.where(d > ds)[0]
            if not len(bad) or len(Q) > maxpts: break
            M = 0.5 * (P[bad] + P[bad + 1])
            Xm, Ym = flow_many(M[:, 0].copy(), M[:, 1].copy(), t, h, steps, EPS, GAM)
            P = np.insert(P, bad + 1, M, axis=0); Q = np.insert(Q, bad + 1, np.c_[Xm, Ym], axis=0)
        P = Q; t += h
    return P[:-1]


def shp(P):
    g = Polygon(P)
    return g if g.is_valid else make_valid(g)


# ---------------------------------------------------------------- Fig. 11(a): manifolds
def stage_mani():
    pR, Wu = manifolds()
    q, i, upper, poly = region_R1(pR, Wu)
    Ws = refl_x(Wu)
    # pips on x = 0 (reversibility puts every pip of the upper tangle that lies on the symmetry line there)
    cr = np.where(np.sign(Wu[1:, 0]) != np.sign(Wu[:-1, 0]))[0]
    pips = [Wu[j] + Wu[j, 0] / (Wu[j, 0] - Wu[j + 1, 0]) * (Wu[j + 1] - Wu[j]) for j in cr[:8]]
    fq = o.map(q[0], q[1], -1); fq = np.array([fq[0][0], fq[1][0]])
    info = dict(eps=EPS, gamma=GAM, Tp=o.Tp, pR=pR.tolist(), lam=np.abs(np.linalg.eigvals(o._jac(pR))).tolist(),
                q=q.tolist(), finv_q=fq.tolist(), area_R1=shp(poly).area, pips_on_axis=[p.tolist() for p in pips])
    json.dump(info, open(os.path.join(OUT, f'ovp_{TAG}.json'), 'w'), indent=1)
    print(json.dumps(info, indent=1))
    L = int(np.searchsorted(np.cumsum(np.r_[0, np.hypot(*np.diff(Wu, axis=0).T)]), 14.0))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(Wu[:L, 0], Wu[:L, 1], 'r', lw=.9, label=r'$W^u(p_+)$')
    ax.plot(Ws[:L, 0], Ws[:L, 1], 'k', lw=.9, label=r'$W^s(p_-)$')
    ax.plot([pR[0], -pR[0]], [0, 0], 'o', ms=5, mfc='w', mec='k')
    ax.plot(q[0], q[1], 'ko', ms=3); ax.annotate('q', q, (8, 4), textcoords='offset points')
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(0, 2.7); ax.set_aspect('equal')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.legend(loc='upper right', fontsize=8, frameon=False)
    ax.set_title(fr'OVP, $(\gamma,\epsilon)=({GAM:g},{EPS:g})$: manifolds of the period-$2\pi\gamma$ map', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f'fig11a_manifolds_{TAG}.png'), dpi=200); plt.close(fig)


# ---------------------------------------------------------------- Fig. 11(b): FTLE
def ftle(T, nx=1001, ny=501, box=(-2.6, 2.6, 0.0, 2.7), t0=0.0):
    x = np.linspace(box[0], box[1], nx); y = np.linspace(box[2], box[3], ny)
    X, Y = np.meshgrid(x, y)
    steps = int(o.steps * abs(T) / o.Tp)
    Xf, Yf = flow_many(X.ravel().copy(), Y.ravel().copy(), t0, T, steps, EPS, GAM)
    Xf = Xf.reshape(X.shape); Yf = Yf.reshape(Y.shape)
    F12, F11 = np.gradient(Xf, y, x)  # dX/dy, dX/dx
    F22, F21 = np.gradient(Yf, y, x)  # dY/dy, dY/dx
    C11 = F11 ** 2 + F21 ** 2; C12 = F11 * F12 + F21 * F22; C22 = F12 ** 2 + F22 ** 2
    lmax = 0.5 * (C11 + C22) + np.sqrt(0.25 * (C11 - C22) ** 2 + C12 ** 2)
    return x, y, np.log(np.sqrt(np.maximum(lmax, 1e-300))) / abs(T)


def stage_ftle():
    pR, Wu = manifolds(); Ws = refl_x(Wu)
    T = float(os.environ.get('OVP_FTLE_T', 3.0)) * o.Tp
    fz = os.path.join(OUT, f'ftle_{TAG}.npz')
    if os.path.exists(fz) and np.load(fz)['T'] == T:
        d = np.load(fz); xs, ys, Sf, Sb = d['x'], d['y'], d['fwd'], d['bwd']
    else:
        xs, ys, Sf = ftle(+T); _, _, Sb = ftle(-T)
    np.savez_compressed(os.path.join(OUT, f'ftle_{TAG}.npz'), x=xs, y=ys, fwd=Sf, bwd=Sb, T=T)
    L = int(np.searchsorted(np.cumsum(np.r_[0, np.hypot(*np.diff(Wu, axis=0).T)]), 14.0))
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, S, cm, ttl in ((axs[0], Sf, 'Reds', f'forward FTLE, T = +{T / o.Tp:g} periods (repelling LCS)'),
                           (axs[1], Sb, 'Greys', f'backward FTLE, T = -{T / o.Tp:g} periods (attracting LCS)')):
        ax.pcolormesh(xs, ys, S, cmap=cm, shading='auto', vmin=np.percentile(S, 5), vmax=np.percentile(S, 99.7))
        ax.set_title(ttl, fontsize=9)
    # overlay: thresholded ridges in paper colours + manifolds
    ax = axs[2]
    XX, YY = np.meshgrid(xs, ys); far = np.hypot(XX, YY - 1.0) > 0.75  # leave out the vortex core (shear, not hyperbolicity)
    mf = far & (Sf > np.percentile(Sf[far], 90)); mb = far & (Sb > np.percentile(Sb[far], 90))
    ax.pcolormesh(xs, ys, np.ma.masked_where(~mb, Sb), cmap='Greys', shading='auto', alpha=.8)
    ax.pcolormesh(xs, ys, np.ma.masked_where(~mf, Sf), cmap='spring', shading='auto', alpha=.8)
    ax.plot(Wu[:L, 0], Wu[:L, 1], color='C0', lw=.6, ls='--', label=r'$W^u(p_+)$')
    ax.plot(Ws[:L, 0], Ws[:L, 1], color='C2', lw=.6, ls='--', label=r'$W^s(p_-)$')
    ax.legend(fontsize=7, loc='upper right', frameon=False)
    ax.set_title('top 10% FTLE outside the vortex core (magenta fwd, grey bwd), manifolds dashed', fontsize=9)
    for a in axs: a.set_aspect('equal'); a.set_xlim(xs[0], xs[-1]); a.set_ylim(ys[0], ys[-1])
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f'fig11b_ftle_{TAG}.png'), dpi=170); plt.close(fig)


# ---------------------------------------------------------------- Lober
def write_tec(fn, c, title):
    with open(fn, 'w') as f:
        f.write('VARIABLES="x""y"\nZONE T="%s"\n' % title)
        np.savetxt(f, c, fmt='%.15e')


def stage_lober():
    pR, Wu = manifolds()
    q, i, upper, poly = region_R1(pR, Wu)
    # manifolds up to a few lobes past q, as Lober wants (unstable from pR, stable from pL)
    s = np.cumsum(np.r_[0, np.hypot(*np.diff(Wu, axis=0).T)])
    L = s[i] * 2.2
    U = Wu[s <= L]; S = refl_x(U)
    d = os.path.join(OUT, f'lober_{TAG}'); os.makedirs(d, exist_ok=True)
    write_tec(os.path.join(d, 'u.n'), U, 'unstable'); write_tec(os.path.join(d, 's.n'), S, 'stable')
    if not LOBER:
        print('set LOBER=/path/to/lober to run it'); return
    r = subprocess.run([LOBER, 'u.n', 's.n', 'lobe.dat', 'mani.dat', 'pips.dat', '0', '0', '1', 'lobeb.dat'],
                       cwd=d, capture_output=True, text=True)
    open(os.path.join(d, 'lober.log'), 'w').write(r.stdout + r.stderr)
    print(r.stdout[-3000:])


# ---------------------------------------------------------------- Fig. 9: species transport
def stage_species(nmax=None):
    nmax = nmax or int(os.environ.get('OVP_NMAX', 8))
    pR, Wu = manifolds(); Ws = refl_x(Wu)
    q, i, upper, poly = region_R1(pR, Wu)
    R1 = shp(poly); A = R1.area
    rows = []; imgs = {0: poly}
    P = poly.copy(); T12p = 0.0
    for n in range(1, nmax + 1):
        t = time.time()
        P = advect_poly(P, 1, t0=(n - 1) * o.Tp, ds=0.006)
        Pn = shp(P)
        T11 = Pn.intersection(R1).area; T12 = A - T11
        rows.append(dict(n=n, T12=T12, a12=T12 - T12p, image_area=Pn.area, npts=len(P)))
        print(rows[-1], f'{time.time() - t:.1f}s', flush=True)
        T12p = T12; imgs[n] = P
        np.save(os.path.join(OUT, f'species_img{n}_{TAG}.npy'), P.astype(np.float32))
    json.dump(dict(area_R1=A, rows=rows), open(os.path.join(OUT, f'species_{TAG}.json'), 'w'), indent=1)
    # figure in the layout of Fig. 9
    show = [n for n in (0, 1, 2, 3, 5, 7) if n in imgs]
    fig = plt.figure(figsize=(10, 8.5))
    for k, n in enumerate(show):
        ax = fig.add_axes([0.02 + (k % 3) * 0.33, 0.66 - (k // 3) * 0.3, 0.3, 0.28])
        ax.fill(imgs[n][:, 0], imgs[n][:, 1], color='#1f2bd0', lw=0)
        ax.plot(poly[:, 0], poly[:, 1], color='k', lw=.4, alpha=.5)
        ax.set_xlim(-3.2, 2.4); ax.set_ylim(-2.6, 2.6); ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(f'n = {n}', y=-0.08, fontsize=10)
    ax = fig.add_axes([0.1, 0.06, 0.55, 0.28])
    nn = [r['n'] for r in rows]
    ax.bar(nn, [r['T12'] / A for r in rows], color='#e0262f', label=r'$T_{1,2}(n)$: species $S_1$ in $R_2$ after $n$ iterates')
    ax.bar(nn, [r['a12'] / A for r in rows], color='#4459b8', width=.8,
           label=r'$a_{1,2}(n)=T_{1,2}(n)-T_{1,2}(n-1)$: flux on the $n$-th iterate')
    ax.set_xlabel('n = iterate of the map'); ax.set_ylabel(r'fraction of $\mu(R_1)$')
    ax.legend(fontsize=8, frameon=False, loc='upper left'); ax.set_xticks(nn)
    fig.savefig(os.path.join(OUT, f'fig9_species_{TAG}.png'), dpi=170); plt.close(fig)


def stage_mc(N=int(os.environ.get('OVP_MC_N', 400000)), nmax=int(os.environ.get('OVP_MC_NMAX', 40))):
    """Monte Carlo check of T_{1,2}(n), and the long-time escape law, with N tracers seeded in R1."""
    pR, Wu = manifolds(); q, i, upper, poly = region_R1(pR, Wu)
    path = Path(poly); rng = np.random.default_rng(1)
    xmin, ymin = poly.min(0); xmax, ymax = poly.max(0)
    pts = np.empty((0, 2))
    while len(pts) < N:
        c = rng.uniform([xmin, ymin], [xmax, ymax], (N, 2)); pts = np.vstack([pts, c[path.contains_points(c)]])
    pts = pts[:N]; X, Y = pts[:, 0].copy(), pts[:, 1].copy()
    inside0 = np.ones(N, bool); ever_out = np.zeros(N, bool)
    frac_in = [1.0]; frac_never = [1.0]
    for n in range(1, nmax + 1):
        X, Y = flow_many(X, Y, (n - 1) * o.Tp, o.Tp, o.steps, EPS, GAM)
        ins = path.contains_points(np.c_[X, Y]); ever_out |= ~ins
        frac_in.append(ins.mean()); frac_never.append((~ever_out).mean())
    json.dump(dict(N=N, frac_in=frac_in, frac_never_left=frac_never),
              open(os.path.join(OUT, f'mc_{TAG}.json'), 'w'), indent=1)
    print('T12/A (MC):', [round(1 - f, 4) for f in frac_in[:9]])


def stage_anim(nper=int(os.environ.get('OVP_ANIM_PER', 4)), fpp=48):
    """Continuous-time movie: species S1 (the core R1 at t = 0) and the moving manifolds W^u(p+(t)), W^s(p-(t))."""
    import matplotlib.animation as ma
    pR, Wu = manifolds(); q, i, upper, poly = region_R1(pR, Wu)
    s = np.cumsum(np.r_[0, np.hypot(*np.diff(Wu, axis=0).T)]); U = Wu[s <= 12.0]; Sm = refl_x(U)
    Us = U[::2]; Ss = Sm[::2]
    phases = np.arange(fpp) * o.Tp / fpp
    WU, WS = [], []
    for ph in phases:
        n = max(2, int(o.steps * ph / o.Tp))
        X, Y = flow_many(Us[:, 0].copy(), Us[:, 1].copy(), 0.0, ph, n, EPS, GAM); WU.append(np.c_[X, Y])
        n = max(2, int(o.steps * (o.Tp - ph) / o.Tp))
        X, Y = flow_many(Ss[:, 0].copy(), Ss[:, 1].copy(), o.Tp, ph - o.Tp, n, EPS, GAM); WS.append(np.c_[X, Y])
    fig, ax = plt.subplots(figsize=(8, 5)); fig.subplots_adjust(0.02, 0.02, 0.98, 0.93)
    st = dict(P=poly.copy(), t=0.0)
    def frame(k):
        t = k * o.Tp / fpp
        if k > 0:
            st['P'] = advect_poly(st['P'], T=t - st['t'], t0=st['t'], ds=0.01, maxpts=1_500_000); st['t'] = t
        ax.clear(); P = st['P']; j = k % fpp
        ax.fill(P[:, 0], P[:, 1], color='#1f2bd0', lw=0, alpha=.9)
        for W, c in ((WU[j], 'r'), (WS[j], 'k')):
            for M in (W, refl_y(W)):
                ax.plot(M[:, 0], M[:, 1], color=c, lw=.7)
        yv = np.exp(EPS * (np.cos(t / GAM) - 1))
        ax.plot([0, 0], [yv, -yv], 'o', ms=6, color='orange')
        ax.set_xlim(-4.5, 2.6); ax.set_ylim(-2.6, 2.6); ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(fr'oscillating vortex pair, $(\gamma,\epsilon)$ = ({GAM:g}, {EPS:g}):  t = {t / o.Tp:.2f} periods'
                     '\nblue: fluid of the core at t = 0;  red $W^u(p_+)$, black $W^s(p_-)$ (and mirror images)', fontsize=9)
    a = ma.FuncAnimation(fig, frame, frames=nper * fpp + 1)
    a.save(os.path.join(OUT, f'anim_ovp_species_{TAG}.mp4'), fps=24, dpi=110, writer='ffmpeg'); plt.close(fig)


def stage_anim_x(nper=int(os.environ.get('OVP_ANIM_PER', 4)), fpp=60, theme=os.environ.get('OVP_THEME', 'dark')):
    """Tightly cropped, title-free version of stage_anim for social media (1080 x 1080, H.264)."""
    import matplotlib.animation as ma
    th = dict(dark=dict(bg='#0b1016', fluid='#3fa7ff', wu='#ff5a4e', ws='#e8edf2', vort='#ffb020', txt='#8fa0ad'),
              light=dict(bg='#ffffff', fluid='#1f2bd0', wu='#d7261e', ws='#111111', vort='#f39c12', txt='#55646f'))[theme]
    pR, Wu = manifolds(); q, i, upper, poly = region_R1(pR, Wu)
    s_ = np.cumsum(np.r_[0, np.hypot(*np.diff(Wu, axis=0).T)]); U = Wu[s_ <= 12.0]; Sm = refl_x(U)
    Us = U[::2]; Ss = Sm[::2]; phases = np.arange(fpp) * o.Tp / fpp; WU, WS = [], []
    for ph in phases:
        n = max(2, int(o.steps * ph / o.Tp))
        X, Y = flow_many(Us[:, 0].copy(), Us[:, 1].copy(), 0.0, ph, n, EPS, GAM); WU.append(np.c_[X, Y])
        n = max(2, int(o.steps * (o.Tp - ph) / o.Tp))
        X, Y = flow_many(Ss[:, 0].copy(), Ss[:, 1].copy(), o.Tp, ph - o.Tp, n, EPS, GAM); WS.append(np.c_[X, Y])
    fig = plt.figure(figsize=(6, 6), facecolor=th['bg']); ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(th['bg'])
    st = dict(P=poly.copy(), t=0.0)
    def frame(k):
        t = k * o.Tp / fpp
        if k > 0:
            st['P'] = advect_poly(st['P'], T=t - st['t'], t0=st['t'], ds=0.008, maxpts=1_500_000); st['t'] = t
        ax.clear(); ax.set_facecolor(th['bg']); P = st['P']; j = k % fpp
        ax.fill(P[:, 0], P[:, 1], color=th['fluid'], lw=0, alpha=.92)
        for W, c, lw in ((WS[j], th['ws'], .6), (WU[j], th['wu'], .8)):
            for M in (W, refl_y(W)): ax.plot(M[:, 0], M[:, 1], color=c, lw=lw, alpha=.9)
        yv = np.exp(EPS * (np.cos(t / GAM) - 1)); ax.plot([0, 0], [yv, -yv], 'o', ms=5, color=th['vort'])
        ax.set_xlim(-2.95, 2.25); ax.set_ylim(-2.6, 2.6); ax.set_aspect('equal'); ax.axis('off')
        ax.text(0.03, 0.03, f't = {t / o.Tp:.2f} T', transform=ax.transAxes, color=th['txt'], fontsize=9, family='monospace')
    a = ma.FuncAnimation(fig, frame, frames=nper * fpp + 1)
    a.save(os.path.join(OUT, f'x_ovp_species_{theme}.mp4'), fps=30, dpi=180, writer='ffmpeg',
           savefig_kwargs=dict(facecolor=th['bg']), extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18'])
    plt.close(fig)


if __name__ == '__main__':
    stages = sys.argv[1:] or ['mani', 'ftle', 'lober', 'species', 'mc']
    for s in stages:
        t = time.time(); globals()['stage_' + s](); print(f'== {s} {time.time() - t:.1f}s', flush=True)
