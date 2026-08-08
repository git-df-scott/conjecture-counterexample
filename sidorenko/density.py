"""Float search tier: t(H, W) on step kernels, and the Sidorenko objective.

Scale invariance
----------------
The Sidorenko deficit  D(W) = t(H, W) - t(K_2, W)^{e(H)}  is homogeneous of
degree e(H):  replacing W by cW (c > 0) multiplies t(H, W) by c^{e(H)} (one
factor per edge) and t(K_2, W)^{e(H)} by the same c^{e(H)}.  So the *sign* of
the deficit does not depend on the scale of W.  Two consequences, both used
throughout:

* The codomain restriction W <= 1 in the definition of a graphon can be
  dropped: search over nonnegative kernels, and divide by sup W at the end to
  land back in [0, 1] without changing the sign of the deficit.
* The search objective may be taken to be the scale-invariant

      F(a, B) = log t(H, W) - e(H) * log t(K_2, W),

  and evaluated after rescaling B so that t(K_2, W) = 1 exactly.  Then
  F = log t(H, W) with t(H, W) near 1, which removes the catastrophic
  cancellation that  t(H, W) - p^{e(H)}  suffers near the constant kernel
  (both terms there agree to 16 digits).  Sidorenko holds for H iff
  inf F = 0, attained at the constant kernel.

Completeness of the step-kernel search space
--------------------------------------------
For fixed H, both t(H, .) and t(K_2, .) are continuous in the cut metric, and
step kernels are dense.  Hence the infimum of the deficit over all graphons
equals its infimum over step kernels, and a strict counterexample is witnessed
by some step kernel with finitely many blocks.  Restricting to step kernels
therefore loses nothing as the block count grows; it is a *resolution* limit,
not a structural one.

Why the search has to be global
-------------------------------
Write W = p(1 + f) with the perturbation f mean-zero.  Expanding the product
over edges,  t(H, W) = p^{e(H)} * sum over edge subsets S of t(H_S, f), and any
S with a degree-1 vertex integrates to 0 because f has mean zero.  The
lowest-order surviving S are the shortest cycles, of length 2g, contributing
(number of 2g-cycles of H) * tr(f^{2g}) = (count) * sum_i lambda_i^{2g} >= 0.
So the constant kernel is a local minimum of the deficit for every bipartite H
-- no local method started near quasirandomness can ever succeed, and any
counterexample must live far from the constant kernel.  `sidorenko.local`
verifies this numerically per graph.
"""

from __future__ import annotations

import numpy as np

from .contract import contract, contract_open, plan_for

__all__ = [
    "hom_density",
    "edge_density",
    "normalize",
    "evaluate",
    "objective",
    "objective_and_grad",
    "hom_density_bruteforce",
    "support_hom_count",
    "REJECT",
]


def edge_density(weights, mat):
    w = np.asarray(weights, dtype=float)
    return float(w @ np.asarray(mat, dtype=float) @ w)


def hom_density(g, weights, mat):
    """t(H, W) for the step kernel (weights, mat)."""
    plan = plan_for(g.n, g.edges)
    return float(contract(plan, np.asarray(weights, float), np.asarray(mat, float)))


def hom_density_bruteforce(g, weights, mat):
    """Direct k^n sum; reference implementation for tests only."""
    import itertools

    w = np.asarray(weights, dtype=float)
    B = np.asarray(mat, dtype=float)
    k = len(w)
    total = 0.0
    for phi in itertools.product(range(k), repeat=g.n):
        term = 1.0
        for v in range(g.n):
            term *= w[phi[v]]
        for u, v in g.edges:
            term *= B[phi[u], phi[v]]
        total += term
    return total


def normalize(weights, mat):
    """Rescale mat so that t(K_2, W) == 1 (see the scale-invariance note)."""
    p = edge_density(weights, mat)
    if not (p > 0) or not np.isfinite(p):
        return None
    return np.asarray(mat, dtype=float) / p


# Below this, a float64 is subnormal and carries far fewer than 15 significant
# digits; 1e-280 keeps every quantity in the normal range with room to spare.
_NORMAL_FLOOR = 1e-280
REJECT = 1e100


def support_hom_count(g, weights, mat):
    """Exact number of homomorphisms H -> support(mat), over positive-weight
    blocks only (loops allowed).  Integer arithmetic, so exact.

    This is what decides whether t(H, W) = 0 *genuinely* -- i.e. every map hits
    a structural zero of the kernel -- as opposed to having merely underflowed.
    Without the distinction, a degenerate kernel whose true t is 1e-400 reads as
    a hard violation, which is how a proven-positive graph can appear to fail.
    """
    a = np.asarray(weights, float)
    B = np.asarray(mat, float)
    live = a > 0
    if not live.any():
        return 0
    S = ((B > 0) & live[:, None] & live[None, :]).astype(np.int64)
    ones = np.ones(S.shape[0], dtype=np.int64)
    return int(contract(plan_for(g.n, g.edges), ones, S))


def evaluate(g, weights, mat):
    """Guarded evaluation of F = log t(H, W) - e(H) log t(K_2, W).

    Returns (F, status).  `status` is "ok", or a reason string when the point
    cannot be evaluated to float precision, in which case F = REJECT (for the
    optimiser to avoid) -- except for a genuine structural zero, which is a
    real violation and returns -inf.

    F is scale invariant, so the matrix is first divided by its largest entry:
    every entry is then at most 1 and no product over edges can overflow.  What
    max-normalisation does *not* fix is the magnitude of the results themselves:
    with a very wide dynamic range, t and p^e(H) can both land in the subnormal
    range (~1e-320), where they retain about four significant digits and their
    log-difference carries an error of ~1e-3 -- large enough to manufacture a
    violation out of nothing.  So anything that leaves the normal float range is
    rejected rather than believed.
    """
    a = np.asarray(weights, float)
    B = np.asarray(mat, dtype=float)
    if not np.all(np.isfinite(B)) or not np.all(np.isfinite(a)):
        return REJECT, "nonfinite input"
    mx = float(np.max(B))
    if not (mx > 0 and np.isfinite(mx)):
        return REJECT, "degenerate matrix"
    B = B / mx
    p = edge_density(a, B)
    if not (p > 0 and np.isfinite(p)):
        return REJECT, "nonpositive edge density"
    if g.m * np.log(p) < -600.0:
        return REJECT, "p^e(H) below the normal float range"
    t = hom_density(g, a, B)
    if not np.isfinite(t):
        return REJECT, "nonfinite t"
    if t <= 0.0:
        # Either a structural zero (a real violation) or underflow (not).
        if support_hom_count(g, a, B) == 0:
            return -np.inf, "ok"
        return REJECT, "t underflowed"
    if t < _NORMAL_FLOOR:
        return REJECT, "t below the normal float range"
    return float(np.log(t) - g.m * np.log(p)), "ok"


def objective(g, weights, mat):
    """F = log t(H, W) - e(H) log t(K_2, W); Sidorenko for H iff min F >= 0.

    The careful path, used to re-verify any candidate hit from the optimiser.
    Returns REJECT for points that cannot be evaluated reliably.
    """
    return evaluate(g, weights, mat)[0]


def _dt_dmat(g, weights, mat):
    """Q = sum over edges e of P_e, where P_e[i, j] is the weighted count of
    maps with the endpoints of e sent to (i, j) and e's own factor removed.

    Then, parametrising the *symmetric* matrix by its upper triangle S,
        dt/dS_ij = Q[i,j] + Q[j,i]  (i < j),   dt/dS_ii = Q[i,i].
    Euler's identity for the degree-e(H) homogeneous t gives the test
        sum_{i<=j} S_ij * dt/dS_ij = e(H) * t.
    """
    w = np.asarray(weights, float)
    B = np.asarray(mat, float)
    k = len(w)
    Q = np.zeros((k, k))
    for idx, (u, v) in enumerate(g.edges):
        rest = tuple(e for j, e in enumerate(g.edges) if j != idx)
        plan = plan_for(g.n, rest, open_vars=(u, v))
        Q += contract_open(plan, w, B)
    return Q


def _dt_dweights(g, weights, mat):
    """dt/da_i = sum over vertices v of M_v[i], where M_v[i] is the weighted
    count of maps with v sent to i and v's own weight factor removed.
    Euler test: sum_i a_i dt/da_i = n * t."""
    w = np.asarray(weights, float)
    B = np.asarray(mat, float)
    out = np.zeros(len(w))
    for v in range(g.n):
        plan = plan_for(g.n, g.edges, open_vars=(v,), drop_unary=(v,))
        out += contract_open(plan, w, B)
    return out


def _sym_grad(Q):
    D = Q + Q.T
    np.fill_diagonal(D, np.diag(Q))
    return D


def triu_indices(k):
    return np.triu_indices(k)


def pack(theta, psi=None):
    """Flatten a symmetric theta (upper triangle) and optional psi."""
    k = theta.shape[0]
    iu = np.triu_indices(k)
    parts = [np.asarray(theta, float)[iu]]
    if psi is not None:
        parts.append(np.asarray(psi, float))
    return np.concatenate(parts)


def unpack(x, k, with_psi):
    """Inverse of `pack`: returns (symmetric theta, psi or None)."""
    iu = np.triu_indices(k)
    nt = len(iu[0])
    theta = np.zeros((k, k))
    theta[iu] = x[:nt]
    theta = theta + theta.T - np.diag(np.diag(theta))
    psi = np.asarray(x[nt:], float) if with_psi else None
    return theta, psi


def weights_from_psi(psi):
    z = np.asarray(psi, float)
    z = z - z.max()
    a = np.exp(z)
    return a / a.sum()


def gradients(g, weights, mat):
    """(t, p, dF/dS as a symmetric matrix, dF/da) at the given kernel.

    dF/dS_ij is the derivative with respect to the free upper-triangle entry
    S_ij of the symmetric matrix (so it already sums the (i,j) and (j,i)
    positions for i != j).
    """
    a = np.asarray(weights, float)
    B = np.asarray(mat, float)
    e = g.m
    t = float(contract(plan_for(g.n, g.edges), a, B))
    p = float(a @ B @ a)
    dt_dS = _sym_grad(_dt_dmat(g, a, B))
    outer = np.outer(a, a)
    dp_dS = 2.0 * outer
    np.fill_diagonal(dp_dS, np.diag(outer))
    dF_dS = dt_dS / t - e * dp_dS / p
    dt_da = _dt_dweights(g, a, B)
    dp_da = 2.0 * (B @ a)
    dF_da = dt_da / t - e * dp_da / p
    return t, p, dF_dS, dF_da, dt_dS, dt_da


def objective_and_grad(g, x, k, optimize_weights=False, weights=None):
    """F and dF/dx for the flat parameter vector produced by `pack`.

    B = exp(theta) (smooth, and enforces B > 0; exact zeros are reached only in
    the rational rounding stage of `sidorenko.certify`).  Block weights are
    either fixed (`weights`) or a softmax of the psi block of `x`.

    The matrix is divided by its largest entry before anything is contracted.
    F and dF/dtheta are both invariant under B -> cB (F is homogeneous of
    degree 0 and dF/dS is homogeneous of degree -1, so dF/dtheta = dF/dS * S is
    degree 0), so this changes nothing mathematically -- but it is essential
    numerically.  L-BFGS will happily walk to a degenerate boundary point such
    as block weights (1e-86, 1) with matrix entries up to 1e83; there the
    unnormalised product over e edges overflows to inf, and the resulting
    garbage can come back *negative*, i.e. as a spurious counterexample.  With
    every entry at most 1 no product can overflow, and if a quantity underflows
    to exactly 0 the point is rejected rather than believed.
    """
    theta, psi = unpack(np.asarray(x, float), k, optimize_weights)
    B = np.exp(np.clip(theta, -700.0, 700.0))
    a = weights_from_psi(psi) if optimize_weights else np.asarray(weights, float)
    F, status = evaluate(g, a, B)
    if status != "ok" or not np.isfinite(F):
        return REJECT, np.zeros_like(np.asarray(x, float))

    mx = float(np.max(B))
    B = B / mx
    _, _, dF_dS, dF_da, _, _ = gradients(g, a, B)
    grad_theta = dF_dS * B  # chain rule through S = exp(theta)
    if optimize_weights:
        grad_psi = a * (dF_da - float(a @ dF_da))
        return F, pack(grad_theta, grad_psi)
    return F, pack(grad_theta)
