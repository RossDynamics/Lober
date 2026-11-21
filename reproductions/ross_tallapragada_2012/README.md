# Ross & Tallapragada (2012), reproduced and extended

Computations for the figures of S. D. Ross and P. Tallapragada, *Detecting and exploiting chaotic transport in
mechanical systems* (2012 chapter), with the set-oriented FTLE of P. Tallapragada and S. D. Ross, *CNSNS* 18 (2013)
1106-1126. Everything is in `out/` (PNG, MP4, JSON). Python 3 with numpy, scipy, numba, matplotlib, shapely; Lober
for the lobe areas (build notes in `../../Install-issues.txt`).

| Script | What it computes | Chapter / paper figure |
|---|---|---|
| `ovp.py`, `ovp_run.py` | oscillating vortex pair: period-map saddles, manifolds, forward/backward FTLE, Lober lobes, species transport T12(n) by boundary advection and by Monte Carlo, movie | ch. Figs. 9, 11 |
| `ovp_sweep.py` | turnstile flux over (gamma, eps) against the Melnikov function; least-transport forcing period | new |
| `cavity.py`, `cavity_fine.py` | lid-driven cavity: transfer operator, AIS, line-element and covariance FTLE (direct and from P^K), coherent sets, their evolution, movie; spectrum of P across the period-3 bifurcation | ch. Figs. 14-15; paper Figs. 11-16; new |
| `doublegyre.py`, `zermelo_shoot.py` | double gyre: FTLE, covariance FTLE, eigenvectors of P^10; minimum-time navigation of a weak vehicle (front tracking, certified by shooting) | paper Figs. 2, 17-18; ch. Fig. 18 |

```
python ovp_run.py mani ftle lober species mc anim     # LOBER=/path/to/lober for the lober stage
python ovp_sweep.py melnikov; python ovp_sweep.py grid; python ovp_sweep.py fine; python ovp_sweep.py minimum
python cavity.py P ftle sets evolve anim sweep; python cavity_fine.py
python doublegyre.py ftle cov P front; python zermelo_shoot.py; python zermelo_shoot.py plots
```

## Model checks

- **OVP** (exact Rom-Kedar-Leonard-Wiggins form, vortices riding the strain; to O(eps) it is eqs. (56)-(58) of
  Naik, Lekien & Ross 2017). (gamma, eps) = (0.5, 0.1): saddles of the period-pi map at (+-1.451260, 0),
  multipliers 5.0757, 0.1970; first pip on the symmetry line q = (0, 2.06848) (Naik et al.: (0, 2.065)).
  Lober: core area 2 x 4.573268 = 9.146536 (the boundary polygon here: 9.1465362), each turnstile lobe 0.254105.
  Boundary advection: T12(1) = 0.508217 = 2 x 0.254105 (5.56 % of the core); image areas conserved to 1e-6.
  Monte Carlo (400,000 tracers) agrees with boundary advection to <= 5e-4 for n = 1..8; after 40 periods 44.5 % of
  the core has never left (vortex-core islands and sticky orbits).
- **Cavity** (120 x 40 boxes, 100 points each, tau_f/2 = 4.848): eigenvalues of P_r 0.9904, 0.9797, 0.9454,
  0.9436, 0.9375 (paper: 0.9925, 0.9829, 0.9516, 0.9473, 0.9434). P_r's top eigenvalue is 1.0014, not 1: points
  clipped at the walls make P slightly non-doubly-stochastic. Coherent sets at sigma_I <= 0.06: three components of
  390, 385, 209 boxes.
- **Double gyre** (A, omega, eps = 0.25, 2 pi, 0.25): P^10 on 200 x 100 boxes, eigenvalues of P_r 0.9959, 0.9941
  (paper: 0.9997, 0.9995 for its operator; 0.9959^(1/10) = 0.99959).

## New results

1. **The least-transport forcing period is not Melnikov's zero.** The first-order Melnikov function of the OVP
   vanishes at gamma_0 = 1.7831 (separatrix integrated over t in [-30, 30]; a [-12, 12] window put it at 1.792).
   The true one-period flux (boundary advection, 1,064 (gamma, eps) points) is smallest at
   gamma*(eps) ~ 1.780 - 3.27 eps (gamma* = 1.748, 1.617, 1.472, 1.232 at eps = 0.01, 0.05, 0.1, 0.2), where it is
   ~ 12.5 eps^2 (1 - 2 eps), against ~ 55 eps^2 at gamma_0 itself: tuning the forcing period by -3.3 eps cuts the
   residual transport 4.5-fold. Away from the zero, Melnikov overestimates the flux up to 5x on the low-gamma side
   (eps = 0.2) and underestimates it 1.1-7x at gamma = 0.2 (eps = 0.01-0.2). Converged: halving the boundary spacing changes the flux at
   eps = 0.01 by 5e-5 relative.
2. **The non-symmetrised transfer operator sees the period-3 bifurcation.** In the cavity, P (not P_r) has a
   complex pair near exp(+-2 pi i/3): the three almost-cyclic blobs. Its phase locks to 120 deg across
   tau_f/2 = 4.73-4.75: |arg - 120 deg| falls from 20 deg (4.60) through 2 deg (4.730) and 0.04 deg (4.740) to
   1e-4 deg (4.76-4.90), bracketing the known birth of the period-3 islands at 4.740202. P_r's lambda_2 only rises
   smoothly (0.949 -> 0.983) through the same range.
3. **A weak vehicle has a thrust threshold.** Double gyre, start (0.15, 0.85), target box |x - 1| <= 0.08,
   y <= 0.12 (the foot of the jet between the gyres): passive drift arrives at t = 18.28; certified minimum times
   (shooting, an explicit control for each) are 1.9, 2.3, 2.9, 3.0, 3.0 for speeds 0.2, 0.1, 0.05, 0.02, 0.0175
   and 8.0, 8.0, 8.1 for 0.015, 0.0125, 0.01 (none <= 12 for 0.005). At speed 0.02 (1.7 % of the peak flow
   speed) the vehicle is 6x faster than drifting and 19x faster than in still water (58). The fast route stays on
   the near side of the attracting LCS between the gyres and rides it down into the target; the passive particle
   is swept back into the left gyre there. Below u ~ 0.017 the vehicle cannot hold that side in time (seeded
   re-optimisation at T = 3: miss 0.002 at u = 0.0165, 0.028 at 0.015) and must wait for the next pass, t ~ 8.
   Shooting gives upper bounds; the front tracker gives 1.87, 2.14, 2.84, 2.97 for 0.2 .. 0.02. A level-set
   solver (first-order upwind) loses the thin reachable sets of weak vehicles and was dropped.

## Not reproduced

Figs. 10, 12 and 19 (NCEP/NAM data) and the PCR3BP section of Fig. 9: Lober's test manifolds
(`lober-1.8-test/*.n`) are that kind of homoclinic tangle, but the system they came from is not recorded; Fig. 9's
species transport is done here in the OVP instead.
