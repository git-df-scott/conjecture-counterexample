"""Why no local method can find a Sidorenko counterexample.

Perturb the constant kernel: W = p(1 + f) with f symmetric and mean zero
(so t(K_2, W) = p is unchanged).  Expanding the edge product,

    t(H, W) = p^{e(H)} * sum_{S subset E(H)} t(H_S, f),

where H_S is the spanning subgraph with edge set S and t(H_S, f) integrates f
over the edges of S with all vertices free.  If S has a vertex of degree 1 then
the inner integral over that vertex is  int f(x, .) = 0, so t(H_S, f) = 0.
Every nonvanishing S therefore has minimum degree >= 2 on its support, hence
contains a cycle, hence |S| >= girth(H) = 2g.  The unique minimal such S are
the 2g-cycles themselves, and for a cycle

    t(C_{2g}, f) = tr(f^{2g}) = sum_i lambda_i^{2g} >= 0,

strictly positive unless f = 0.  So

    t(H, W) - t(K_2, W)^{e(H)} = p^{e(H)} * [ c_{2g}(H) * tr(f^{2g}) + O(|f|^{2g+1}) ]

with c_{2g}(H) = the number of 2g-cycles of H > 0: the constant kernel is a
strict local minimum of the deficit for *every* graph of even girth, and the
deficit vanishes there to order exactly 2g.

This module checks that prediction per graph -- it fits the exponent of the
deficit along random mean-zero perturbations and compares it to 2g.  A mismatch
would mean either the contraction or the cycle counter is wrong, so this doubles
as a cross-check of `sidorenko.contract` against pure combinatorics.

The probe has to be done in **exact rational arithmetic**, not floats.  The
deficit at the constant kernel is a difference of two quantities that agree to
order 2g, so at eps = 1/32 and girth 8 the true value is ~1e-12 while the float
cancellation error is ~1e-16 -- only four usable digits, and the fitted exponent
comes out near 6.5 instead of 8.  So the perturbation is built with integer
entries and eps = 1/(2^j * max|f|), which makes W rational, and the deficit is
obtained as an exact Fraction from `sidorenko.certify`.  Logs are then taken of
exact numerators and denominators, so there is no cancellation at any eps and
the asymptotic slope is clean.  A useful side effect: this makes the local
analysis a third independent consumer of the exact integer tier.

Consequence for the search design: any counterexample must be far from the
constant kernel in the spectral sense, which is why `sidorenko.optimize` seeds
from 0/1 blowups and wide log-normal kernels rather than from perturbations,
and why `sidorenko.lattice` sweeps the extreme points exhaustively.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from .certify import certify
from .contract import min_fill_order

__all__ = ["perturbation_order", "local_report"]


def _integer_mean_zero(k, rng):
    """Random symmetric integer f with every row sum zero.

    For an integer symmetric A with row sums r_i and total s, the matrix
    f_ij = k^2 A_ij - k(r_i + r_j) + s is symmetric, integral, and has
    sum_j f_ij = k^2 r_i - k(k r_i + s) + k s = 0.  Zero row sums give
    t(K_2, 1 + eps f) = 1 exactly, for any eps, so the edge density is pinned
    and the deficit is t(H, W) - 1.
    """
    A = rng.integers(-3, 4, size=(k, k))
    A = np.triu(A) + np.triu(A, 1).T
    r = A.sum(axis=1)
    s = int(A.sum())
    f = (k * k) * A - k * (r[:, None] + r[None, :]) + s
    return f.astype(object)


def _log_fraction(fr):
    return math.log(fr.numerator) - math.log(fr.denominator)


def perturbation_order(g, k=4, n_dirs=3, seed=0, scales=(3, 4, 5, 6)):
    """Exact estimate of the vanishing order of the deficit at the constant
    kernel, along random integer mean-zero directions.

    For each direction f and each scale j, sets eps = 1/(2^j max|f|) so that
    W = 1 + eps f is a positive rational kernel with common denominator
    D = 2^j max|f| and integer numerators D + f.  The deficit is then an exact
    Fraction, and the slope of log(deficit) against log(eps) is the vanishing
    order.  Returns (median slope, per-direction slopes).
    """
    rng = np.random.default_rng(seed)
    orders = []
    for _ in range(n_dirs):
        f = _integer_mean_zero(k, rng)
        mx = int(max(abs(int(x)) for row in f for x in row))
        if mx == 0:
            continue
        xs, ys = [], []
        for j in scales:
            D = (2**j) * mx
            M = [[int(D + f[i][l]) for l in range(k)] for i in range(k)]
            if any(x <= 0 for row in M for x in row):
                continue
            cert = certify(g, [1] * k, M, D)
            p = Fraction(*cert["edge_density"])
            if p != 1:
                raise AssertionError("mean-zero direction did not preserve the edge density")
            d = Fraction(*cert["deficit"])
            if d <= 0:
                continue  # d == 0 happens only for acyclic H
            xs.append(-math.log(D))  # log eps
            ys.append(_log_fraction(d))
        if len(xs) >= 2:
            orders.append(float(np.polyfit(xs, ys, 1)[0]))
    if not orders:
        return None, orders
    return float(np.median(orders)), orders


def _probe_blocks(g, cap=300000):
    """Largest block count whose exact contraction stays affordable."""
    _order, width = min_fill_order(g.n, g.edges)
    for k in (5, 4, 3, 2):
        if k ** (width + 1) <= cap:
            return k
    return 2


def local_report(g, k=None, seed=0):
    """Girth, shortest-cycle count, and the exactly measured vanishing order."""
    girth = g.girth()
    colour = g.bipartition()
    kk = _probe_blocks(g) if k is None else k
    order, per_dir = perturbation_order(g, k=kk, seed=seed)
    return {
        "graph": g.name,
        "n": g.n,
        "e": g.m,
        "bipartite": colour is not None,
        "girth": girth,
        "shortest_cycle_count": None if girth is None else g.count_cycles_of_length(girth),
        "predicted_vanishing_order": girth,
        "measured_vanishing_order": order,
        "probe_blocks": kk,
        "per_direction": per_dir,
        "matches_prediction": (
            None if (order is None or girth is None) else bool(abs(order - girth) < 0.35)
        ),
    }
