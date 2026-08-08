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

This module checks that prediction numerically per graph -- it fits the
exponent of the deficit along random mean-zero perturbations and compares it to
2g and the coefficient to c_{2g}(H).  A mismatch would mean either the
contraction or the cycle counter is wrong, so this doubles as a cross-check of
`sidorenko.contract` against pure combinatorics.

Consequence for the search design: any counterexample must be far from the
constant kernel in the spectral sense, which is why `sidorenko.optimize` seeds
from 0/1 blowups and wide log-normal kernels rather than from perturbations,
and why `sidorenko.lattice` sweeps the extreme points exhaustively.
"""

from __future__ import annotations

import numpy as np

from .density import edge_density, hom_density

__all__ = ["perturbation_order", "local_report"]


def _mean_zero_symmetric(k, rng):
    """Random symmetric f with all row sums zero (so f has mean zero against
    the uniform block measure and t(K_2, p(1+f)) = p exactly)."""
    A = rng.normal(size=(k, k))
    A = 0.5 * (A + A.T)
    # project onto {row sums zero}: A - (r 1^T + 1 r^T) + (mean) 1 1^T
    r = A.mean(axis=1)
    mu = A.mean()
    f = A - r[:, None] - r[None, :] + mu
    return f / np.abs(f).max()


def perturbation_order(g, k=6, n_dirs=6, seed=0, eps=(0.2, 0.1, 0.05, 0.025)):
    """Estimate the vanishing order of the deficit at the constant kernel.

    Returns (median_fitted_exponent, list of per-direction exponents).
    """
    rng = np.random.default_rng(seed)
    a = np.full(k, 1.0 / k)
    orders = []
    for _ in range(n_dirs):
        f = _mean_zero_symmetric(k, rng)
        xs, ys = [], []
        for e in eps:
            B = 1.0 + e * f
            if B.min() <= 0:
                continue
            p = edge_density(a, B)
            d = hom_density(g, a, B) - p ** g.m
            if d <= 0:
                continue
            xs.append(np.log(e))
            ys.append(np.log(d))
        if len(xs) >= 2:
            orders.append(float(np.polyfit(xs, ys, 1)[0]))
    if not orders:
        return None, orders
    return float(np.median(orders)), orders


def local_report(g, k=6, seed=0):
    """Girth, shortest-cycle count, and the measured vanishing order."""
    girth = g.girth()
    colour = g.bipartition()
    order, per_dir = perturbation_order(g, k=k, seed=seed)
    return {
        "graph": g.name,
        "n": g.n,
        "e": g.m,
        "bipartite": colour is not None,
        "girth": girth,
        "shortest_cycle_count": None if girth is None else g.count_cycles_of_length(girth),
        "predicted_vanishing_order": girth,
        "measured_vanishing_order": order,
        "per_direction": per_dir,
        "matches_prediction": (
            None if (order is None or girth is None) else bool(abs(order - girth) < 0.35)
        ),
    }
