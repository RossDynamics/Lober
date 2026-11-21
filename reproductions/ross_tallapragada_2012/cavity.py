"""Lid-driven cavity flow (Stremler & Chen 2007; Tallapragada & Ross 2013 sec. 6.1; Ross & Tallapragada 2012
Figs. 14-15): almost-invariant sets from the transfer operator, covariance (set-oriented) FTLE and coherent sets.

psi = sum_n U_n C_n f_n(y) sin(n pi x / a) on [0, a] x [-b, b], a = 6, b = 1;
f_n(y) = (2 pi y / a) cosh(n pi b / a) sinh(n pi y / a) - (2 pi b / a) sinh(n pi b / a) cosh(n pi y / a);
C_n normalised so that the wall speed is |U_n| sin(n pi x / a).  U1 = -1, U2 = 0.841298 on the first half
period, U1 -> +1 on the second; half period tau_f/2 = 4.848 (period-3 islands are born at 4.740202).

python cavity.py [stage ...]   stages: P ftle sets evolve spectrum sweep
"""
import os, sys, json, time
import numpy as np
from numba import njit, prange
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as sla

A, B = 6.0, 1.0
U2 = 0.841298
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out'); os.makedirs(OUT, exist_ok=True)


def Cn(n):
    k = n * np.pi * B / A
    return A * A / (2 * n * np.pi ** 2 * B) / (A / (2 * n * np.pi * B) * np.sinh(2 * k) + 1)


C1, C2 = Cn(1), Cn(2)


@njit(cache=True, fastmath=True)
def vel(x, y, U1):
    u = 0.0; v = 0.0
    for n in (1, 2):
        Un = U1 if n == 1 else U2
        Cc = C1 if n == 1 else C2
        k = n * np.pi / A; kb = k * B
        ch, sh = np.cosh(kb), np.sinh(kb)
        f = (2 * np.pi * y / A) * ch * np.sinh(k * y) - (2 * np.pi * B / A) * sh * np.cosh(k * y)
        fp = (2 * np.pi / A) * ch * np.sinh(k * y) + (2 * np.pi * y / A) * ch * k * np.cosh(k * y) \
            - (2 * np.pi * B / A) * sh * k * np.sinh(k * y)
        u += Un * Cc * fp * np.sin(k * x)
        v -= Un * Cc * f * k * np.cos(k * x)
    return u, v


@njit(cache=True, fastmath=True)
def steady(x, y, U1, T, n):
    h = T / n
    for _ in range(n):
        a1, b1 = vel(x, y, U1); a2, b2 = vel(x + .5 * h * a1, y + .5 * h * b1, U1)
        a3, b3 = vel(x + .5 * h * a2, y + .5 * h * b2, U1); a4, b4 = vel(x + h * a3, y + h * b3, U1)
        x += h / 6 * (a1 + 2 * a2 + 2 * a3 + a4); y += h / 6 * (b1 + 2 * b2 + 2 * b3 + b4)
    x = min(max(x, 0.0), A); y = min(max(y, -B), B)
    return x, y


@njit(parallel=True, cache=True)
def period_map(X, Y, half, nper, nstep):
    Xo = np.empty_like(X); Yo = np.empty_like(Y)
    for i in prange(X.size):
        x, y = X[i], Y[i]
        for _ in range(nper):
            x, y = steady(x, y, -1.0, half, nstep)
            x, y = steady(x, y, 1.0, half, nstep)
        Xo[i] = x; Yo[i] = y
    return Xo, Yo


@njit(parallel=True, cache=True)
def flow_t(X, Y, half, t, nstep_per_unit):
    """Flow from 0 to t (any t >= 0) through the piecewise-steady protocol."""
    Xo = np.empty_like(X); Yo = np.empty_like(Y)
    for i in prange(X.size):
        x, y = X[i], Y[i]; s = 0.0
        while s < t - 1e-12:
            k = int(np.floor(s / half + 1e-9)); end = min((k + 1) * half, t)
            U1 = -1.0 if k % 2 == 0 else 1.0
            n = max(1, int(np.ceil((end - s) * nstep_per_unit)))
            x, y = steady(x, y, U1, end - s, n); s = end
        Xo[i] = x; Yo[i] = y
    return Xo, Yo


class Grid:
    def __init__(self, nx=120, ny=40, m=10):
        self.nx, self.ny, self.m = nx, ny, m
        self.dx, self.dy = A / nx, 2 * B / ny
        ix, iy = np.meshgrid(np.arange(nx), np.arange(ny))  # box index = iy*nx + ix
        self.cx = (ix.ravel() + .5) * self.dx; self.cy = -B + (iy.ravel() + .5) * self.dy
        s = (np.arange(m) + .5) / m
        sx, sy = np.meshgrid(s, s)
        self.X = (ix.ravel()[:, None] + sx.ravel()[None]) * self.dx
        self.Y = -B + (iy.ravel()[:, None] + sy.ravel()[None]) * self.dy
        self.n = nx * ny

    def box(self, X, Y):
        i = np.clip((X / self.dx).astype(int), 0, self.nx - 1); j = np.clip(((Y + B) / self.dy).astype(int), 0, self.ny - 1)
        return j * self.nx + i

    def img(self, v):
        return np.asarray(v).reshape(self.ny, self.nx)


def images(g, half, nper, nstep=500):
    f = os.path.join(OUT, f'cav_img_h{half:g}_n{nper}_g{g.nx}x{g.ny}x{g.m}.npz')
    if os.path.exists(f):
        d = np.load(f); return d['X'], d['Y']
    X, Y = period_map(g.X.ravel().copy(), g.Y.ravel().copy(), half, nper, nstep)
    X = X.reshape(g.X.shape); Y = Y.reshape(g.Y.shape)
    np.savez(f, X=X, Y=Y); return X, Y


def transfer(g, X, Y):
    rows = np.repeat(np.arange(g.n), g.m * g.m); cols = g.box(X.ravel(), Y.ravel())
    P = sp.csr_matrix((np.full(rows.size, 1.0 / (g.m * g.m)), (rows, cols)), shape=(g.n, g.n))
    P.sum_duplicates(); return P


def cov_ftle(X0, Y0, X1, Y1, T):
    def lmax(X, Y):
        X = X - X.mean(1, keepdims=True); Y = Y - Y.mean(1, keepdims=True)
        a = (X * X).mean(1); b = (X * Y).mean(1); c = (Y * Y).mean(1)
        return .5 * (a + c) + np.sqrt(.25 * (a - c) ** 2 + b * b)
    return np.log(np.sqrt(lmax(X1, Y1) / lmax(X0, Y0))) / T


def classic_ftle(half, nper, nx=600, ny=200, nstep=500):
    x = np.linspace(0, A, nx); y = np.linspace(-B, B, ny); Xg, Yg = np.meshgrid(x, y)
    X, Y = period_map(Xg.ravel().copy(), Yg.ravel().copy(), half, nper, nstep)
    X = X.reshape(Xg.shape); Y = Y.reshape(Yg.shape)
    F12, F11 = np.gradient(X, y, x); F22, F21 = np.gradient(Y, y, x)
    C11 = F11 ** 2 + F21 ** 2; C12 = F11 * F12 + F21 * F22; C22 = F12 ** 2 + F22 ** 2
    l = .5 * (C11 + C22) + np.sqrt(.25 * (C11 - C22) ** 2 + C12 ** 2)
    return x, y, np.log(np.sqrt(l)) / (2 * half * nper)


def pf_cov_ftle(g, P, K, T):
    """Eq. (25): covariance of the image density of each box under P^K (no long integration)."""
    PK = P.copy()
    for _ in range(K - 1): PK = PK @ P
    PK = PK.tocsr()
    cx, cy = g.cx, g.cy
    mx = PK @ cx; my = PK @ cy
    sxx = PK @ (cx * cx) - mx ** 2; syy = PK @ (cy * cy) - my ** 2; sxy = PK @ (cx * cy) - mx * my
    # add the within-box spread of a uniform box (dx^2/12, dy^2/12) to both, as for the initial set
    sxx = sxx + g.dx ** 2 / 12; syy = syy + g.dy ** 2 / 12
    l1 = .5 * (sxx + syy) + np.sqrt(.25 * (sxx - syy) ** 2 + sxy ** 2)
    l0 = max(g.dx, g.dy) ** 2 / 12
    return np.log(np.sqrt(l1 / l0)) / T


def show(ax, g, v, cmap='jet', **kw):
    im = ax.imshow(g.img(v), origin='lower', extent=(0, A, -B, B), cmap=cmap, aspect='equal', interpolation='nearest', **kw)
    ax.set_xticks([0, 2, 4, 6]); ax.set_yticks([-1, 0, 1]); return im


HALF = float(os.environ.get('CAV_HALF', 4.848))


def stage_P():
    g = Grid(); t = time.time(); X, Y = images(g, HALF, 1); print('map', time.time() - t)
    P = transfer(g, X, Y)
    Pr = 0.5 * (P + P.T)  # uniform invariant density: Pbar = P^T
    w, V = sla.eigsh(Pr, k=8, which='LA'); o = np.argsort(-w); w, V = w[o], V[:, o]
    print('Pr eigenvalues', np.round(w, 4))
    v2 = V[:, 1] * np.sign(V[:, 1][np.argmax(np.abs(V[:, 1]))])
    # P itself (not reversible-symmetrised): eigenvalues near the cube roots of unity?
    wp, Vp = sla.eigs(P.T.tocsc(), k=12, which='LM'); o = np.argsort(-np.abs(wp)); wp, Vp = wp[o], Vp[:, o]
    print('P eigenvalues', np.round(wp, 4))
    np.savez(os.path.join(OUT, f'cav_P_h{HALF:g}.npz'), wr=w, Vr=V, wp=wp, Vp=Vp)
    json.dump(dict(half=HALF, Pr_eigs=w.tolist(), P_eigs=[[z.real, z.imag] for z in wp]),
              open(os.path.join(OUT, f'cav_eigs_h{HALF:g}.json'), 'w'), indent=1)
    fig, axs = plt.subplots(1, 2, figsize=(11, 2.4))
    # sign convention: the blobs positive
    if v2[np.argmax(np.abs(v2))] < 0: v2 = -v2
    im = show(axs[0], g, v2); plt.colorbar(im, ax=axs[0], fraction=.02)
    show(axs[1], g, (v2 > 0).astype(float), cmap='gray_r')
    axs[0].set_title(fr'second eigenvector $v^2$ of $P_r$, $\lambda_2$ = {w[1]:.4f}', fontsize=9)
    axs[1].set_title('almost-invariant set: positive part of $v^2$ (black)', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig14_cavity_AIS.png'), dpi=170); plt.close(fig)
    return g, P, v2


def stage_ftle():
    g = Grid(); P = transfer(g, *images(g, HALF, 1)); tau = 2 * HALF
    fig, axs = plt.subplots(3, 3, figsize=(16, 7.2)); res = {}
    for k in (1, 2, 3):
        x, y, F = classic_ftle(HALF, k)
        X, Y = images(g, HALF, k); S = cov_ftle(g.X, g.Y, X, Y, k * tau); res[k] = S
        Spf = pf_cov_ftle(g, P, k, k * tau)
        for j, (ttl, im) in enumerate(((f'line-element FTLE, T = {k}' + r'$\tau_f$', None),
                                       (f'covariance FTLE (direct, 100 pts/box), T = {k}' + r'$\tau_f$', S),
                                       (f'covariance FTLE from $P^{k}$, eq. (25)', Spf))):
            ax = axs[k - 1, j]
            if im is None:
                h = ax.imshow(F, origin='lower', extent=(0, A, -B, B), cmap='jet', aspect='equal', vmin=0, vmax=np.percentile(F, 99.5))
            else:
                h = show(ax, g, im, vmin=0, vmax=np.percentile(im, 99.5))
            plt.colorbar(h, ax=ax, fraction=.02); ax.set_title(ttl, fontsize=9)
        np.save(os.path.join(OUT, f'cav_covftle_{k}.npy'), S)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig12_cavity_ftle.png'), dpi=150); plt.close(fig)


def stage_sets(thr=0.06):
    g = Grid(); S = np.load(os.path.join(OUT, 'cav_covftle_1.npy'))
    coh = S <= thr
    # connected components (4-neighbour)
    from scipy.ndimage import label
    lab, n = label(g.img(coh)); sizes = np.bincount(lab.ravel())[1:]
    print('coherent components', n, 'sizes', sorted(sizes)[::-1][:6])
    np.save(os.path.join(OUT, 'cav_coherent.npy'), lab.ravel())


def render(g, X, Y, w=None, nx=600, ny=200):
    H, _, _ = np.histogram2d(Y.ravel(), X.ravel(), bins=(ny, nx), range=((-B, B), (0, A)), weights=w)
    return H


def stage_evolve():
    """Fig. 16 of Tallapragada & Ross (2013): coherent sets vs AIS advected for 3 periods (625 pts/box)."""
    g = Grid(m=25); lab = np.load(os.path.join(OUT, 'cav_coherent.npy'))
    d = np.load(os.path.join(OUT, f'cav_P_h{HALF:g}.npz')); v2 = d['Vr'][:, 1]
    if v2[np.argmax(np.abs(v2))] < 0: v2 = -v2
    sizes = np.bincount(lab)[1:]; big = 1 + np.argsort(-sizes)[:3]
    sets = {'coherent ($\\sigma_I\\leq0.06$)': np.isin(lab, big), 'AIS ($v^2>0$)': v2 > 0}
    fig, axs = plt.subplots(4, 2, figsize=(12, 8.4))
    for j, (name, mask) in enumerate(sets.items()):
        X0, Y0 = g.X[mask].ravel(), g.Y[mask].ravel()
        for k in range(4):
            X, Y = (X0, Y0) if k == 0 else period_map(X0.copy(), Y0.copy(), HALF, k, 500)
            H = render(g, X, Y); ax = axs[k, j]
            ax.imshow(np.minimum(H / H[H > 0].mean(), 1), origin='lower', extent=(0, A, -B, B), cmap='gray_r', aspect='equal')
            ax.set_title(f'{name}, t = {k}' + r'$\tau_f$', fontsize=9); ax.set_xticks([0, 2, 4, 6]); ax.set_yticks([-1, 0, 1])
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig16_cavity_evolve.png'), dpi=150); plt.close(fig)


def stage_anim(nfr=240, per=3):
    """Animation: the three coherent sets (coloured) and a non-coherent square (grey), continuous time."""
    import matplotlib.animation as ma
    g = Grid(m=25); lab = np.load(os.path.join(OUT, 'cav_coherent.npy'))
    sizes = np.bincount(lab)[1:]; big = 1 + np.argsort(-sizes)[:3]
    order = sorted(big, key=lambda L: g.cx[lab == L].mean())
    sets = [np.isin(lab, [L]) for L in order]
    sq = (np.abs(g.cx - 1.7) < 0.35) & (np.abs(g.cy - 0.1) < 0.35) & ~np.isin(lab, big)
    sets.append(sq)
    cols = [(0.85, 0.1, 0.1), (0.1, 0.45, 0.85), (0.1, 0.65, 0.2), (0.45, 0.45, 0.45)]
    P0 = [(g.X[m].ravel(), g.Y[m].ravel()) for m in sets]
    fig, ax = plt.subplots(figsize=(9, 3.4)); fig.subplots_adjust(0.04, 0.08, 0.99, 0.9)
    T = per * 2 * HALF
    def frame(i):
        t = T * i / (nfr - 1); ax.clear()
        rgb = np.ones((200, 600, 3))
        for (X0, Y0), c in zip(P0, cols):
            X, Y = flow_t(X0.copy(), Y0.copy(), HALF, t, 100.0)
            H = render(g, X, Y) > 0
            for ch in range(3): rgb[..., ch][H] = c[ch]
        ax.imshow(rgb, origin='lower', extent=(0, A, -B, B), aspect='equal', interpolation='nearest')
        top = 'top lid -> right, bottom <-' if (t % (2 * HALF)) < HALF else 'lids reversed (U1 = +1)'
        ax.set_title(f'lid-driven cavity, t = {t / (2 * HALF):.2f} ' + r'$\tau_f$' + f'   [{top}]', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    a = ma.FuncAnimation(fig, frame, frames=nfr)
    a.save(os.path.join(OUT, 'anim_cavity_coherent.mp4'), fps=24, dpi=120, writer='ffmpeg'); plt.close(fig)


def stage_sweep():
    """Spectrum of P (not symmetrised) across the period-3 bifurcation at tau_f/2 = 4.740202."""
    g = Grid(); res = []
    for half in np.round(np.r_[np.linspace(4.60, 4.72, 4), np.linspace(4.73, 4.76, 7), np.linspace(4.78, 5.1, 9)], 4):
        X, Y = images(g, half, 1); P = transfer(g, X, Y)
        wp = sla.eigs(P.T.tocsc(), k=10, which='LM', return_eigenvectors=False)
        Pr = 0.5 * (P + P.T); wr = np.sort(sla.eigsh(Pr, k=4, which='LA', return_eigenvectors=False))[::-1]
        wp = wp[np.argsort(-np.abs(wp))]
        res.append(dict(half=float(half), P=[[z.real, z.imag] for z in wp], Pr=wr.tolist()))
        print(half, np.round(wp[:4], 4), np.round(wr, 4), flush=True)
    json.dump(res, open(os.path.join(OUT, 'cav_sweep.json'), 'w'), indent=1)


if __name__ == '__main__':
    for s in sys.argv[1:] or ['P', 'ftle', 'sets', 'evolve']:
        t = time.time(); globals()['stage_' + s](); print(f'== {s} {time.time() - t:.1f}s', flush=True)
