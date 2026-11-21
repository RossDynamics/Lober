"""Double gyre (Shadden et al. 2005): A = 0.25, omega = 2 pi, eps = 0.25 on [0, 2] x [0, 1].

Tallapragada & Ross (2013) Figs. 2, 17, 18, 19 (FTLE, covariance FTLE direct and from P^10, eigenvectors of
P, coherent sets vs AIS) and Ross & Tallapragada (2012) Fig. 18 (a weakly propelled vehicle reaching a target
region by riding the flow), done here as an exact minimum-time (Zermelo) problem by a level-set method.

python doublegyre.py [stage ...]   stages: ftle cov P sets zermelo
"""
import os, sys, json, time
import numpy as np
from numba import njit, prange
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as sla

AMP, OM, EPS = 0.25, 2 * np.pi, 0.25
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out'); os.makedirs(OUT, exist_ok=True)


@njit(cache=True, fastmath=True)
def vel(x, y, t):
    a = EPS * np.sin(OM * t); b = 1 - 2 * a
    f = a * x * x + b * x; fx = 2 * a * x + b
    return (-np.pi * AMP * np.sin(np.pi * f) * np.cos(np.pi * y),
            np.pi * AMP * np.cos(np.pi * f) * np.sin(np.pi * y) * fx)


@njit(cache=True, fastmath=True)
def flow(x, y, t0, T, n):
    h = T / n; t = t0
    for _ in range(n):
        a1, b1 = vel(x, y, t); a2, b2 = vel(x + .5 * h * a1, y + .5 * h * b1, t + .5 * h)
        a3, b3 = vel(x + .5 * h * a2, y + .5 * h * b2, t + .5 * h); a4, b4 = vel(x + h * a3, y + h * b3, t + h)
        x += h / 6 * (a1 + 2 * a2 + 2 * a3 + a4); y += h / 6 * (b1 + 2 * b2 + 2 * b3 + b4); t += h
    return x, y


@njit(parallel=True, cache=True)
def flow_many(X, Y, t0, T, n):
    Xo = np.empty_like(X); Yo = np.empty_like(Y)
    for i in prange(X.size):
        Xo[i], Yo[i] = flow(X[i], Y[i], t0, T, n)
    return Xo, Yo


def ftle_grid(T, t0=0.0, nx=1000, ny=500):
    x = np.linspace(0, 2, nx); y = np.linspace(0, 1, ny); Xg, Yg = np.meshgrid(x, y)
    X, Y = flow_many(Xg.ravel().copy(), Yg.ravel().copy(), t0, T, int(abs(T) * 100))
    X = X.reshape(Xg.shape); Y = Y.reshape(Yg.shape)
    F12, F11 = np.gradient(X, y, x); F22, F21 = np.gradient(Y, y, x)
    C11 = F11 ** 2 + F21 ** 2; C12 = F11 * F12 + F21 * F22; C22 = F12 ** 2 + F22 ** 2
    l = .5 * (C11 + C22) + np.sqrt(.25 * (C11 - C22) ** 2 + C12 ** 2)
    return x, y, np.log(np.sqrt(l)) / abs(T)


class Boxes:
    def __init__(self, nx, ny, m):
        self.nx, self.ny, self.m, self.n = nx, ny, m, nx * ny
        self.dx, self.dy = 2 / nx, 1 / ny
        ix, iy = np.meshgrid(np.arange(nx), np.arange(ny))
        self.cx = (ix.ravel() + .5) * self.dx; self.cy = (iy.ravel() + .5) * self.dy
        s = (np.arange(m) + .5) / m; sx, sy = np.meshgrid(s, s)
        self.X = (ix.ravel()[:, None] + sx.ravel()[None]) * self.dx
        self.Y = (iy.ravel()[:, None] + sy.ravel()[None]) * self.dy

    def box(self, X, Y):
        i = np.clip((X / self.dx).astype(int), 0, self.nx - 1); j = np.clip((Y / self.dy).astype(int), 0, self.ny - 1)
        return j * self.nx + i

    def img(self, v):
        return np.asarray(v).reshape(self.ny, self.nx)


def cov_lmax(X, Y):
    X = X - X.mean(1, keepdims=True); Y = Y - Y.mean(1, keepdims=True)
    a = (X * X).mean(1); b = (X * Y).mean(1); c = (Y * Y).mean(1)
    return .5 * (a + c) + np.sqrt(.25 * (a - c) ** 2 + b * b)


def stage_ftle():
    x, y, F = ftle_grid(10.0); np.save(os.path.join(OUT, 'dg_ftle10.npy'), F)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    h = ax.imshow(F, origin='lower', extent=(0, 2, 0, 1), cmap='jet', vmin=-0.1, vmax=0.75)
    plt.colorbar(h, ax=ax, fraction=.025); ax.set_title('double gyre: line-element FTLE, T = 10 (1000 x 500 grid)', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'dgfig2_ftle.png'), dpi=170); plt.close(fig)


def stage_cov(T=10.0):
    b = Boxes(400, 200, 10)  # boxes of 0.005 x 0.005, 100 points each
    t = time.time(); X, Y = flow_many(b.X.ravel().copy(), b.Y.ravel().copy(), 0.0, T, int(T * 100)); print('flow', time.time() - t)
    X = X.reshape(b.X.shape); Y = Y.reshape(b.Y.shape)
    S = np.log(np.sqrt(cov_lmax(X, Y) / cov_lmax(b.X, b.Y))) / T
    np.save(os.path.join(OUT, 'dg_covftle_direct.npy'), S)


def stage_P(nx=200, ny=100, m=10):
    """P over one period (dt = 1) on boxes of 0.01; FTLE from P^10 (eq. 25); eigenvectors of P^10."""
    b = Boxes(nx, ny, m)
    X, Y = flow_many(b.X.ravel().copy(), b.Y.ravel().copy(), 0.0, 1.0, 100)
    rows = np.repeat(np.arange(b.n), m * m); cols = b.box(X, Y)
    P = sp.csr_matrix((np.full(rows.size, 1 / m ** 2), (rows, cols)), shape=(b.n, b.n)); P.sum_duplicates()
    PK = P.copy()
    for _ in range(9): PK = (PK @ P).tocsr()
    mx = PK @ b.cx; my = PK @ b.cy
    sxx = PK @ b.cx ** 2 - mx ** 2 + b.dx ** 2 / 12; syy = PK @ b.cy ** 2 - my ** 2 + b.dy ** 2 / 12
    sxy = PK @ (b.cx * b.cy) - mx * my
    l1 = .5 * (sxx + syy) + np.sqrt(.25 * (sxx - syy) ** 2 + sxy ** 2)
    Spf = np.log(np.sqrt(l1 / (b.dx ** 2 / 12))) / 10
    Pr = 0.5 * (PK + PK.T)
    w, V = sla.eigsh(Pr, k=6, which='LA'); o = np.argsort(-w); w, V = w[o], V[:, o]
    print('P10_r eigenvalues', np.round(w, 4))
    np.savez(os.path.join(OUT, 'dg_P.npz'), Spf=Spf, w=w, V=V, nx=nx, ny=ny)
    S = np.load(os.path.join(OUT, 'dg_covftle_direct.npy'))
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))
    for ax, v, ttl in ((axs[0, 0], S.reshape(200, 400), 'covariance FTLE, direct (400 x 200 boxes, 100 pts each), T = 10'),
                       (axs[0, 1], Spf.reshape(ny, nx), r'covariance FTLE from $P^{10}$, eq. (25)'),
                       (axs[1, 0], V[:, 1].reshape(ny, nx), fr'$v^2$ of $P_r$ ($P^{{10}}$), $\lambda_2$ = {w[1]:.4f}'),
                       (axs[1, 1], V[:, 2].reshape(ny, nx), fr'$v^3$, $\lambda_3$ = {w[2]:.4f}')):
        h = ax.imshow(v, origin='lower', extent=(0, 2, 0, 1), cmap='jet'); plt.colorbar(h, ax=ax, fraction=.025)
        ax.set_title(ttl, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'dgfig17_18.png'), dpi=150); plt.close(fig)
    json.dump(dict(eigs=w.tolist()), open(os.path.join(OUT, 'dg_eigs.json'), 'w'))


# ------------------------------------------------------------------ Zermelo: minimum-time navigation
@njit(parallel=True, cache=True)
def ls_step(phi, t, dt, umax, dx, nx, ny):
    """phi_t + v_f . grad phi + umax |grad phi| = 0 (Godunov upwind, first order)."""
    out = np.empty_like(phi)
    for j in prange(ny):
        for i in range(nx):
            x = i * dx; y = j * dx
            u, v = vel(x, y, t)
            im = max(i - 1, 0); ip = min(i + 1, nx - 1); jm = max(j - 1, 0); jp = min(j + 1, ny - 1)
            dxm = (phi[j, i] - phi[j, im]) / dx; dxp = (phi[j, ip] - phi[j, i]) / dx
            dym = (phi[j, i] - phi[jm, i]) / dx; dyp = (phi[jp, i] - phi[j, i]) / dx
            adv = (u * dxm if u > 0 else u * dxp) + (v * dym if v > 0 else v * dyp)
            # Godunov for +|grad phi| (front moving outward)
            gx = max(max(dxm, 0.0) ** 2, min(dxp, 0.0) ** 2); gy = max(max(dym, 0.0) ** 2, min(dyp, 0.0) ** 2)
            out[j, i] = phi[j, i] - dt * (adv + umax * np.sqrt(gx + gy))
    return out


def zermelo(start, target, umax, t0=0.0, nx=401, tmax=30.0, keep_dt=0.02):
    dx = 2 / (nx - 1); ny = (nx - 1) // 2 + 1
    x = np.linspace(0, 2, nx); y = np.linspace(0, 1, ny); Xg, Yg = np.meshgrid(x, y)
    phi = np.hypot(Xg - start[0], Yg - start[1]) - 0.03
    tgt = (np.abs(Xg - target[0]) <= target[2]) & (np.abs(Yg - target[1]) <= target[3])
    vmax = np.pi * AMP * (1 + 2 * EPS) * 1.6 + umax; dt = 0.45 * dx / vmax
    t = t0; snaps = [(t, phi.copy())]; k = 0
    while t < t0 + tmax:
        phi = ls_step(phi, t, dt, umax, dx, nx, ny); t += dt; k += 1
        if k % 50 == 0:  # reinitialise-lite: keep phi a bounded signed function
            phi = np.clip(phi, -0.2, 0.2)
        if t - snaps[-1][0] >= keep_dt: snaps.append((t, phi.astype(np.float32)))
        if (phi[tgt] <= 0).any():
            snaps.append((t, phi.copy())); break
    return x, y, snaps, t - t0


def backtrack(x, y, snaps, target, umax):
    """Optimal path: integrate backwards from the first point of the target reached,
    xdot = v_f + umax grad(phi)/|grad(phi)| (the front's outward normal)."""
    from scipy.interpolate import RegularGridInterpolator as RGI
    tf, phif = snaps[-1]
    Xg, Yg = np.meshgrid(x, y)
    tgt = (np.abs(Xg - target[0]) <= target[2]) & (np.abs(Yg - target[1]) <= target[3])
    j, i = np.unravel_index(np.argmin(np.where(tgt, phif, np.inf)), phif.shape)
    p = np.array([x[i], y[j]]); path = [(tf, *p)]; head = []
    for k in range(len(snaps) - 1, 0, -1):
        t1, ph = snaps[k]; t0, _ = snaps[k - 1]
        gy, gx = np.gradient(ph, y, x)
        Gx = RGI((y, x), gx, bounds_error=False, fill_value=0); Gy = RGI((y, x), gy, bounds_error=False, fill_value=0)
        n = 8; h = (t1 - t0) / n
        for s in range(n):
            tt = t1 - s * h
            g = np.array([Gx((p[1], p[0])), Gy((p[1], p[0]))]).ravel(); g /= (np.linalg.norm(g) + 1e-15)
            u, v = vel(p[0], p[1], tt)
            p = p - h * (np.array([u, v]) + umax * g); p = np.clip(p, [0, 0], [2, 1])
        head.append(g); path.append((t0, *p))
    return np.array(path[::-1])


def stage_zermelo():
    start = (0.15, 0.85); target = (1.0, 0.06, 0.08, 0.06)  # box centre (x, y), half-widths
    res = {}
    for umax in (0.02, 0.05, 0.1, 0.2):
        t = time.time(); x, y, snaps, Tstar = zermelo(start, target, umax)
        path = backtrack(x, y, snaps, target, umax)
        res[umax] = dict(Tstar=Tstar, path=path.tolist())
        straight = np.hypot(target[0] - start[0], target[1] - start[1])
        print(f'umax {umax}: minimum time {Tstar:.3f} (no-flow time {straight / umax:.1f}); {time.time() - t:.0f}s', flush=True)
        np.savez_compressed(os.path.join(OUT, f'dg_zermelo_{umax:g}.npz'), x=x, y=y,
                            t=np.array([s[0] for s in snaps])[::5], phi=np.array([s[1] for s in snaps])[::5].astype(np.float32),
                            path=path)
    json.dump({str(k): dict(Tstar=v['Tstar']) for k, v in res.items()}, open(os.path.join(OUT, 'dg_zermelo.json'), 'w'), indent=1)


def stage_zplot(umax=0.05):
    """The optimal path over the attracting LCS (backward FTLE, T = -5) at four times, the reachable sets, and a movie."""
    import matplotlib.animation as ma
    d = np.load(os.path.join(OUT, f'dg_zermelo_{umax:g}.npz')); x, y, ts, phis, path = d['x'], d['y'], d['t'], d['phi'], d['path']
    start = (0.15, 0.85); tgt = (1.0, 0.06, 0.08, 0.06); Tstar = path[-1, 0]
    def ftle_b(t): return ftle_grid(-5.0, t0=t, nx=500, ny=250)[2]
    fig, axs = plt.subplots(2, 2, figsize=(12, 6.4))
    for ax, frac in zip(axs.ravel(), (0.0, 0.33, 0.66, 1.0)):
        t = frac * Tstar; F = ftle_b(t)
        ax.imshow(F, origin='lower', extent=(0, 2, 0, 1), cmap='gray_r', vmin=0, vmax=np.percentile(F, 99))
        k = np.argmin(np.abs(ts - t)); ax.contour(x, y, phis[k], [0], colors='tab:orange', linewidths=1)
        m = path[:, 0] <= t + 1e-9
        ax.plot(path[:, 1], path[:, 2], color='tab:green', lw=.8, alpha=.35); ax.plot(path[m, 1], path[m, 2], color='tab:green', lw=2)
        ax.plot(*start, 'ko', ms=5); ax.add_patch(plt.Rectangle((tgt[0] - tgt[2], tgt[1] - tgt[3]), 2 * tgt[2], 2 * tgt[3], fc='none', ec='tab:red'))
        ax.set_title(f't = {t:.2f}   (optimal arrival {Tstar:.2f})', fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(fr'minimum-time path of a vehicle with speed $\leq$ {umax:g} (flow speed up to 1.2): green; reachable set: orange; grey: attracting LCS', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f'dg_zermelo_{umax:g}.png'), dpi=150); plt.close(fig)
    # movie
    fig, ax = plt.subplots(figsize=(8, 4.3)); fig.subplots_adjust(0.01, 0.01, 0.99, 0.92)
    fr = np.linspace(0, Tstar, 150)
    def frame(i):
        t = fr[i]; ax.clear(); F = ftle_b(t)
        ax.imshow(F, origin='lower', extent=(0, 2, 0, 1), cmap='gray_r', vmin=0, vmax=np.percentile(F, 99))
        k = np.argmin(np.abs(ts - t)); ax.contour(x, y, phis[k], [0], colors='tab:orange', linewidths=1)
        m = path[:, 0] <= t + 1e-9; ax.plot(path[m, 1], path[m, 2], color='tab:green', lw=2)
        p = np.array([np.interp(t, path[:, 0], path[:, 1]), np.interp(t, path[:, 0], path[:, 2])]); ax.plot(*p, 'o', color='tab:green', ms=6)
        ax.plot(*start, 'ko', ms=5); ax.add_patch(plt.Rectangle((tgt[0] - tgt[2], tgt[1] - tgt[3]), 2 * tgt[2], 2 * tgt[3], fc='none', ec='tab:red'))
        ax.set_xlim(0, 2); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'double gyre, vehicle speed <= {umax:g}:  t = {t:.2f} / {Tstar:.2f}', fontsize=9)
    ma.FuncAnimation(fig, frame, frames=len(fr)).save(os.path.join(OUT, f'anim_dg_zermelo_{umax:g}.mp4'), fps=20, dpi=110, writer='ffmpeg')
    plt.close(fig)


# ------------------------------------------------------------------ Zermelo by front tracking (Huygens)
@njit(parallel=True, cache=True)
def front_vel(P, nrm, t, umax):
    V = np.empty_like(P)
    for i in prange(P.shape[0]):
        u, v = vel(P[i, 0], P[i, 1], t); V[i, 0] = u + umax * nrm[i, 0]; V[i, 1] = v + umax * nrm[i, 1]
    return V


def normals(P):
    """Outward unit normals of a closed counter-clockwise polygon (no repeated end point)."""
    T = np.roll(P, -1, 0) - np.roll(P, 1, 0); n = np.c_[T[:, 1], -T[:, 0]]
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-300)


def clean(P, ds):
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union
    g = Polygon(P).buffer(0).intersection(box(0, 0, 2, 1))
    if g.geom_type != 'Polygon':
        g = max(getattr(g, 'geoms', [g]), key=lambda q: q.area)
    g = g.simplify(ds / 8)
    Q = np.asarray(g.exterior.coords)[:-1]
    if Polygon(Q).exterior.is_ccw is False: Q = Q[::-1]
    # insert points where needed
    d = np.hypot(*np.diff(np.vstack([Q, Q[:1]]), axis=0).T); k = np.maximum(1, np.ceil(d / ds).astype(int))
    if k.max() > 1:
        R = []
        Qn = np.roll(Q, -1, 0)
        for j in range(len(Q)):
            s_ = np.arange(k[j])[:, None] / k[j]; R.append(Q[j] + s_ * (Qn[j] - Q[j]))
        Q = np.vstack(R)
    return Q, g.area


def front_zermelo(start, target, umax, r0=0.03, dt=0.004, ds=0.004, tmax=40.0, keep_dt=0.02):
    from shapely.geometry import Polygon, box
    th = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    P = np.c_[start[0] + r0 * np.cos(th), start[1] + r0 * np.sin(th)]
    tb = box(target[0] - target[2], target[1] - target[3], target[0] + target[2], target[1] + target[3])
    t = 0.0; snaps = [(t, P.copy())]; areas = [(t, Polygon(P).area)]
    while t < tmax:
        n = normals(P); V1 = front_vel(P, n, t, umax); P1 = P + dt * V1
        V2 = front_vel(P1, n, t + dt, umax); P = P + 0.5 * dt * (V1 + V2); t += dt
        P, a = clean(P, ds)
        if t - snaps[-1][0] >= keep_dt - 1e-12:
            snaps.append((t, P.copy())); areas.append((t, a))
        if Polygon(P).intersects(tb):
            snaps.append((t, P.copy())); break
    return snaps, t, areas


def front_backtrack(snaps, target, umax):
    """xdot = v_f + umax n, backwards from the first target point reached; n from the front at each time."""
    from scipy.spatial import cKDTree
    from shapely.geometry import box, Polygon
    tf, Pf = snaps[-1]
    tb = box(target[0] - target[2], target[1] - target[3], target[0] + target[2], target[1] + target[3])
    inside = np.array([tb.contains(__import__('shapely.geometry', fromlist=['Point']).Point(p)) for p in Pf])
    p = Pf[inside][0] if inside.any() else Pf[np.argmin([tb.distance(__import__('shapely.geometry', fromlist=['Point']).Point(q)) for q in Pf])]
    path = [(tf, *p)]
    for k in range(len(snaps) - 1, 0, -1):
        t1, P1 = snaps[k]; t0, _ = snaps[k - 1]; n1 = normals(P1); tree = cKDTree(P1)
        m = 4; h = (t1 - t0) / m
        for s_ in range(m):
            tt = t1 - s_ * h; j = tree.query(p)[1]; u, v = vel(p[0], p[1], tt)
            p = np.clip(p - h * (np.array([u, v]) + umax * n1[j]), [0, 0], [2, 1])
            P1n = P1  # snap the path onto the front (it lies on the boundary of the reachable set)
        jj = cKDTree(snaps[k - 1][1]).query(p)[1]; p = 0.5 * p + 0.5 * snaps[k - 1][1][jj]
        path.append((t0, *p))
    return np.array(path[::-1])


def stage_front():
    start = (0.15, 0.85); target = (1.0, 0.06, 0.08, 0.06); res = {}
    for umax in (0.2, 0.15, 0.1, 0.075, 0.05, 0.03, 0.02, 0.01, 0.005, 0.0025, 0.0):
        t = time.time(); snaps, Tstar, areas = front_zermelo(start, target, umax, dt=0.002, ds=0.002)
        path = front_backtrack(snaps, target, umax)
        res[str(umax)] = dict(Tstar=Tstar, reached=Tstar < 40.0 - 1e-9)
        print(f'front: umax {umax}: minimum time {Tstar:.3f}, {len(snaps)} snapshots, {time.time() - t:.0f}s', flush=True)
        np.savez_compressed(os.path.join(OUT, f'dg_front_{umax:g}.npz'), t=np.array([s[0] for s in snaps]),
                            fronts=np.array([s[1] for s in snaps], dtype=object), path=path, areas=np.array(areas), allow_pickle=True)
        json.dump(res, open(os.path.join(OUT, 'dg_front.json'), 'w'), indent=1)


if __name__ == '__main__':
    for s in sys.argv[1:] or ['ftle', 'cov', 'P']:
        t = time.time(); globals()['stage_' + s](); print(f'== {s} {time.time() - t:.1f}s', flush=True)
