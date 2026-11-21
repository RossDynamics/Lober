"""Oscillating vortex pair (Rom-Kedar, Leonard & Wiggins 1990), exact (not O(eps)) form.

Frame moving with the vortices.  Vortices of strength +-1 at (0, +-yv(t)) in a strain field
of rate (eps/gamma) sin(t/gamma):
    u = -(y-yv)/I- + (y+yv)/I+ - vv(t) + (eps/gamma) x sin(t/gamma)
    v =  x/I- - x/I+                 - (eps/gamma) y sin(t/gamma)
with I+- = x^2 + (y +- yv)^2, yv(t) = exp(eps (cos(t/gamma) - 1)) (the vortices ride the strain),
vv = 1/(2 yv) (the pair's self-induced speed).  To first order in eps this is the system of
Naik, Lekien & Ross (2017), eqs. (56)-(58) (yv = 1, vv = 1/2, saddles at +-sqrt(3) when eps = 0).
Period of the forcing: Tp = 2 pi gamma.
"""
import numpy as np
from numba import njit, prange

@njit(cache=True, fastmath=True)
def vel(x, y, t, eps, gam):
    s = np.sin(t / gam); yv = np.exp(eps * (np.cos(t / gam) - 1.0)); vv = 0.5 / yv
    Im = x * x + (y - yv) ** 2; Ip = x * x + (y + yv) ** 2
    u = -(y - yv) / Im + (y + yv) / Ip - vv + eps / gam * x * s
    v = x / Im - x / Ip - eps / gam * y * s
    return u, v

@njit(cache=True, fastmath=True)
def flow(x, y, t0, T, n, eps, gam):
    h = T / n; t = t0
    for _ in range(n):
        k1u, k1v = vel(x, y, t, eps, gam)
        k2u, k2v = vel(x + .5 * h * k1u, y + .5 * h * k1v, t + .5 * h, eps, gam)
        k3u, k3v = vel(x + .5 * h * k2u, y + .5 * h * k2v, t + .5 * h, eps, gam)
        k4u, k4v = vel(x + h * k3u, y + h * k3v, t + h, eps, gam)
        x += h / 6 * (k1u + 2 * k2u + 2 * k3u + k4u); y += h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        t += h
    return x, y

@njit(parallel=True, cache=True)
def flow_many(X, Y, t0, T, n, eps, gam):
    Xo = np.empty_like(X); Yo = np.empty_like(Y)
    for i in prange(X.size):
        Xo[i], Yo[i] = flow(X[i], Y[i], t0, T, n, eps, gam)
    return Xo, Yo

class OVP:
    def __init__(self, eps=0.1, gam=0.5, steps=400):
        self.eps, self.gam, self.steps = eps, gam, steps
        self.Tp = 2 * np.pi * gam
    def map(self, X, Y, k=1, t0=0.0):
        X = np.atleast_1d(np.asarray(X, float)).copy(); Y = np.atleast_1d(np.asarray(Y, float)).copy()
        T = self.Tp * k
        return flow_many(X, Y, t0, T, int(self.steps * abs(k)), self.eps, self.gam)
    def fixed_point(self, x0, y0=0.0):
        p = np.array([x0, y0], float)
        for _ in range(30):
            fx, fy = self.map(p[0], p[1]); F = np.array([fx[0], fy[0]]) - p
            dp = np.linalg.solve(self._jac(p) - np.eye(2), -F); p = p + dp
            if np.linalg.norm(dp) < 1e-13: break
        return p
    def _jac(self, p, h=1e-7):
        J = np.empty((2, 2))
        for j in range(2):
            a = p.copy(); b = p.copy(); a[j] += h; b[j] -= h
            fa = self.map(*a); fb = self.map(*b)
            J[:, j] = [(fa[0][0] - fb[0][0]) / (2 * h), (fa[1][0] - fb[1][0]) / (2 * h)]
        return J
    def manifold(self, p, which='u', length=12.0, d0=1e-6, ds=0.004, nmax=200000, branch=+1, window=None):
        """Globalize W^u (map forward) or W^s (inverse map) of saddle p by fundamental-segment iteration
        with point insertion. Returns an (N,2) curve ordered from p."""
        J = self._jac(p); w, V = np.linalg.eig(J)
        i = np.argmax(np.abs(w)) if which == 'u' else np.argmin(np.abs(w))
        v = np.real(V[:, i]); v /= np.linalg.norm(v); v *= branch
        lam = np.real(w[i]); k = 1 if which == 'u' else -1
        # fundamental segment from p + d0 v to f(p + d0 v) (linear: lam d0 v)
        s = np.linspace(0, 1, 60)
        a = abs(lam) if which == 'u' else 1 / abs(lam)
        seg = p[None, :] + (d0 * a ** s)[:, None] * v[None, :]
        curve = [p[None, :], seg]; L = 0.0
        prev = seg
        while L < length:
            X, Y = self.map(prev[:, 0], prev[:, 1], k)
            new = np.c_[X, Y]
            # refine where spacing too big
            for _ in range(30):
                d = np.hypot(*np.diff(new, axis=0).T)
                bad = np.where(d > ds)[0]
                if bad.size == 0: break
                mids = 0.5 * (prev[bad] + prev[bad + 1])
                Xm, Ym = self.map(mids[:, 0], mids[:, 1], k)
                prev = np.insert(prev, bad + 1, mids, axis=0)
                new = np.insert(new, bad + 1, np.c_[Xm, Ym], axis=0)
            L += np.hypot(*np.diff(new, axis=0).T).sum()
            curve.append(new[1:]); prev = new
            if sum(len(c) for c in curve) > nmax: break
        return np.vstack(curve)
