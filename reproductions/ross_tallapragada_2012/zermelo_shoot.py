"""Certified minimum-time navigation in the double gyre by direct shooting.

A vehicle of speed <= umax starts at the point (0.15, 0.85) at t = 0; heading theta(t) piecewise constant on N
segments of [0, T]; find the smallest T (scan) for which some heading history ends inside the target box
|x - 1| <= 0.08, |y - 0.06| <= 0.06.  Each T reported is achieved by an explicit control (an upper bound on the
true minimum time, found by multistart local optimisation).
"""
import os, sys, json, time
import numpy as np
from numba import njit
from scipy.optimize import minimize
from doublegyre import vel, OUT

START = np.array([0.15, 0.85]); TGT = (1.0, 0.06, 0.08, 0.06)


@njit(cache=True, fastmath=True)
def fly(th, T, umax, x0, y0, sub):
    N = th.size; h = T / (N * sub); x, y = x0, y0; t = 0.0
    for k in range(N):
        c, s = umax * np.cos(th[k]), umax * np.sin(th[k])
        for _ in range(sub):
            a1, b1 = vel(x, y, t); a2, b2 = vel(x + .5 * h * (a1 + c), y + .5 * h * (b1 + s), t + .5 * h)
            a3, b3 = vel(x + .5 * h * (a2 + c), y + .5 * h * (b2 + s), t + .5 * h)
            a4, b4 = vel(x + h * (a3 + c), y + h * (b3 + s), t + h)
            x += h / 6 * (a1 + 2 * a2 + 2 * a3 + a4 + 6 * c); y += h / 6 * (b1 + 2 * b2 + 2 * b3 + b4 + 6 * s)
            x = min(max(x, 0.0), 2.0); y = min(max(y, 0.0), 1.0); t += h
    return x, y


def cost(th, T, umax):
    x, y = fly(th, T, umax, START[0], START[1], 8)
    dx = max(abs(x - TGT[0]) - TGT[2], 0.0); dy = max(abs(y - TGT[1]) - TGT[3], 0.0)
    return dx * dx + dy * dy


def best_at(T, umax, N=24, starts=24, rng=None):
    rng = rng or np.random.default_rng(0); best = (np.inf, None)
    for k in range(starts):
        th0 = rng.uniform(-np.pi, np.pi, N) if k else np.full(N, -np.pi / 4)
        r = minimize(cost, th0, args=(T, umax), method='L-BFGS-B', options=dict(maxiter=200))
        if r.fun < best[0]: best = (r.fun, r.x)
        if best[0] == 0.0: break
    return best


def passive_time(Tmax=40.0, dt=0.001):
    x, y = START; t = 0.0
    while t < Tmax:
        x, y = fly(np.zeros(1), dt, 0.0, x, y, 1) if False else _step(x, y, t, dt); t += dt
        if abs(x - TGT[0]) <= TGT[2] and abs(y - TGT[1]) <= TGT[3]: return t
    return None


def _step(x, y, t, h):
    a1, b1 = vel(x, y, t); a2, b2 = vel(x + .5 * h * a1, y + .5 * h * b1, t + .5 * h)
    a3, b3 = vel(x + .5 * h * a2, y + .5 * h * b2, t + .5 * h); a4, b4 = vel(x + h * a3, y + h * b3, t + h)
    return x + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4), y + h / 6 * (b1 + 2 * b2 + 2 * b3 + b4)



def traj(th, T, umax, sub=40):
    N = th.size; h = T / (N * sub); x, y = START; t = 0.0; out = [(0.0, x, y)]
    for k in range(N):
        c, s = umax * np.cos(th[k]), umax * np.sin(th[k])
        for _ in range(sub):
            a1, b1 = vel(x, y, t); a2, b2 = vel(x + .5 * h * (a1 + c), y + .5 * h * (b1 + s), t + .5 * h)
            a3, b3 = vel(x + .5 * h * (a2 + c), y + .5 * h * (b2 + s), t + .5 * h); a4, b4 = vel(x + h * (a3 + c), y + h * (b3 + s), t + h)
            x += h / 6 * (a1 + 2 * a2 + 2 * a3 + a4 + 6 * c); y += h / 6 * (b1 + 2 * b2 + 2 * b3 + b4 + 6 * s)
            x = min(max(x, 0.0), 2.0); y = min(max(y, 0.0), 1.0); t += h; out.append((t, x, y))
    return np.array(out)


def plots():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.animation as ma
    from doublegyre import ftle_grid
    res = json.load(open(os.path.join(OUT, 'dg_shoot.json')))
    us = sorted([float(k) for k in res if k != 'passive' and res[k]['Tstar']])
    Tp = res['passive']
    # passive trajectory
    P = [(0.0, *START)]; x, y = START; t = 0.0
    while t < Tp: x, y = _step(x, y, t, 0.002); t += 0.002; P.append((t, x, y))
    P = np.array(P)
    fig, ax = plt.subplots(figsize=(6, 4))
    dist = np.hypot(TGT[0] - START[0], TGT[1] - START[1]) - 0.08
    ax.loglog(us, [res[str(u)]['Tstar'] for u in us], 'o-', label='certified minimum time (shooting)')
    uu = np.logspace(-3, np.log10(0.25), 50); ax.loglog(uu, dist / uu, 'k:', label='still water (distance / speed)')
    ax.axhline(Tp, color='tab:red', ls='--', label=f'passive drift ({Tp:.1f})')
    ax.set_xlabel('vehicle speed  $u_{max}$  (peak flow speed 1.18)'); ax.set_ylabel('arrival time at the target')
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(OUT, 'dg_arrival_vs_speed.png'), dpi=160); plt.close(fig)
    u = 0.02 if '0.02' in res else us[0]; r = res[str(u)]; Tr = traj(np.array(r['theta']), r['Tstar'], u)
    fig, axs = plt.subplots(2, 2, figsize=(12, 6.4))
    for ax, frac in zip(axs.ravel(), (0.0, 0.33, 0.66, 1.0)):
        t = frac * r['Tstar']; F = ftle_grid(-5.0, t0=t, nx=500, ny=250)[2]
        ax.imshow(F, origin='lower', extent=(0, 2, 0, 1), cmap='gray_r', vmin=0, vmax=np.percentile(F, 99))
        m = Tr[:, 0] <= t + 1e-9; mp = P[:, 0] <= t + 1e-9
        ax.plot(P[mp, 1], P[mp, 2], color='tab:red', lw=1.2, alpha=.8)
        ax.plot(Tr[m, 1], Tr[m, 2], color='tab:green', lw=2.2)
        ax.plot(*START, 'ko', ms=5); ax.add_patch(plt.Rectangle((TGT[0] - TGT[2], TGT[1] - TGT[3]), 2 * TGT[2], 2 * TGT[3], fc='none', ec='tab:blue', lw=1.5))
        ax.set_title(f't = {t:.2f}', fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'green: vehicle with speed {u:g} (time-optimal, arrives t = {r["Tstar"]:.1f}); red: passive drift (arrives t = {Tp:.1f}); grey: attracting LCS (backward FTLE)', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f'dg_optimal_path_{u:g}.png'), dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.3)); fig.subplots_adjust(0.01, 0.01, 0.99, 0.92)
    fr = np.linspace(0, r['Tstar'], 120)
    def frame(i):
        t = fr[i]; ax.clear(); F = ftle_grid(-5.0, t0=t, nx=500, ny=250)[2]
        ax.imshow(F, origin='lower', extent=(0, 2, 0, 1), cmap='gray_r', vmin=0, vmax=np.percentile(F, 99))
        m = Tr[:, 0] <= t + 1e-9; mp = P[:, 0] <= t + 1e-9
        ax.plot(P[mp, 1], P[mp, 2], color='tab:red', lw=1.2); ax.plot(P[mp, 1][-1], P[mp, 2][-1], 'o', color='tab:red', ms=5)
        ax.plot(Tr[m, 1], Tr[m, 2], color='tab:green', lw=2); ax.plot(Tr[m, 1][-1], Tr[m, 2][-1], 'o', color='tab:green', ms=6)
        ax.plot(*START, 'ko', ms=4); ax.add_patch(plt.Rectangle((TGT[0] - TGT[2], TGT[1] - TGT[3]), 2 * TGT[2], 2 * TGT[3], fc='none', ec='tab:blue'))
        ax.set_xlim(0, 2); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'double gyre: vehicle speed {u:g} (green) vs passive drift (red), t = {t:.2f}', fontsize=9)
    ma.FuncAnimation(fig, frame, frames=len(fr)).save(os.path.join(OUT, f'anim_dg_vehicle_{u:g}.mp4'), fps=20, dpi=110, writer='ffmpeg')
    plt.close(fig)


if __name__ == '__main__':
    if sys.argv[1:] == ['plots']:
        plots(); sys.exit()
    f0 = os.path.join(OUT, 'dg_shoot.json')
    res = json.load(open(f0)) if os.path.exists(f0) else dict(passive=passive_time())
    print('passive arrival', res['passive'], flush=True)
    Ts = np.round(np.arange(0.5, 12.01, 0.1), 2)
    for umax in [float(u) for u in (sys.argv[1:] or (0.2, 0.1, 0.05, 0.02, 0.01, 0.005))]:
        if str(umax) in res: continue
        t = time.time(); Tstar = None
        for T in Ts:
            N = int(np.clip(round(T * 8), 8, 64))
            f, th = best_at(T, umax, N=N)
            if f == 0.0:
                Tstar = float(T); res[str(umax)] = dict(Tstar=Tstar, theta=th.tolist(), N=N); break
        print(f'umax {umax}: certified arrival T <= {Tstar}  ({time.time() - t:.0f}s)', flush=True)
        json.dump(res, open(os.path.join(OUT, 'dg_shoot.json'), 'w'), indent=1)
