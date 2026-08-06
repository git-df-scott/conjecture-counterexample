"""Float-tier evaluation of the volume product P(K) = vol(K) * vol(K polar).

K = conv({rows of V} U {-rows of V}).  Fast (two qhull calls), NOT certified —
the exact tier in certify.py is the rigorous layer.

Gauge handling: P is GL(d)-invariant, so the parameter space has d^2 flat
directions along which float error grows with the conditioning of V.  Measured
on GL-images of the cube, float P drifts signed-negative (-1.4e-8 at condition
1e8), which would fire false counterexample triggers.  The objective therefore
whitens V by the inverse square root of its second-moment matrix before every
hull call, which bounds conditioning and fixes the scale.
"""
import numpy as np
from scipy.spatial import ConvexHull, QhullError

CONJECTURED_MIN = 32.0 / 3.0  # 4^3/3! = 4^4/4! = 32/3 in both dimensions
TRIGGER_MARGIN = 1e-6
PENALTY_BASE = 50.0
# (floor, ceiling) with slack: Kuperberg lower bound (pi/4)^(d-1) 4^d/d!
# is ~6.58 (d=3) / ~5.17 (d=4); Santalo upper bound vol(B)^2 is ~17.55 / ~24.35.
SANITY_WINDOW = {3: (6.0, 18.5), 4: (4.8, 25.5)}


class EvalStats:
    """Per-start bookkeeping shared with the objective closure."""

    def __init__(self):
        self.nfev = 0
        self.min_P = np.inf
        self.argmin_V = None
        self.n_fault = 0
        self.n_trigger = 0


def whiten(V, cond_limit=1e7):
    """Gauge-normalize V so its second moment is isotropic; None if too degenerate.

    A pure gauge move: the exact volume product is unchanged; conditioning and
    scale are normalized (mean squared row norm becomes 1).  Uses the SVD of V
    directly — forming V'V would square the condition number and lose the small
    singular directions.  Inputs beyond cond_limit are rejected (the caller
    penalizes them): past that point the whitened copy itself is unreliable,
    and since every evaluation whitens a copy, the objective is exactly flat
    along gauge directions — nothing ever pushes the search toward this wall.
    """
    m, d = V.shape
    if m < d or not np.all(np.isfinite(V)):
        return None
    A, s, Bt = np.linalg.svd(V, full_matrices=False)
    if s[-1] <= 0.0 or s[0] / s[-1] > cond_limit:
        return None
    return (A @ Bt) * np.sqrt(m / d)


def volume_product(V):
    """P(K) for K = conv(+-rows).  Returns (P, volK, volKpolar, n_facet_simplices).

    Raises QhullError on degenerate input or if 0 is not strictly interior.
    Facets a.x + b <= 0 (qhull, b<0) rescale to <a/(-b), x> <= 1, so the polar
    body is the convex hull of the points a_j/(-b_j).
    """
    pts = np.vstack([V, -V])
    hull = ConvexHull(pts)
    A = hull.equations[:, :-1]
    b = hull.equations[:, -1]
    if b.max() > -1e-9:
        raise QhullError("origin not strictly interior")
    U = A / (-b)[:, None]
    hull2 = ConvexHull(U)
    return hull.volume * hull2.volume, hull.volume, hull2.volume, len(A)


def make_objective(m, d, stats):
    """log P(K) as a function of x = V.ravel(), with whitening and graded penalties."""
    lo, hi = SANITY_WINDOW[d]

    def f(x):
        stats.nfev += 1
        V = x.reshape(m, d)
        if not np.all(np.isfinite(V)):
            stats.n_fault += 1
            return PENALTY_BASE + 20.0
        Vw = whiten(V)
        if Vw is None:
            stats.n_fault += 1
            return PENALTY_BASE + 10.0
        try:
            P, _, _, _ = volume_product(Vw)
        except QhullError:
            stats.n_fault += 1
            return PENALTY_BASE + 5.0
        if not (lo < P < hi):
            # outside the [Kuperberg, Santalo] window: numerical fault, not geometry
            stats.n_fault += 1
            return PENALTY_BASE + 1.0
        if P < stats.min_P:
            stats.min_P = P
            stats.argmin_V = Vw.copy()
        if P < CONJECTURED_MIN - TRIGGER_MARGIN:
            stats.n_trigger += 1
        return float(np.log(P))

    return f
