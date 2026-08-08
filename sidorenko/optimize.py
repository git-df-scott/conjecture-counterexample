"""Multi-start global search for a negative Sidorenko deficit.

Objective: F(a, B) = log t(H, W) - e(H) log t(K_2, W), scale invariant and
equal to 0 at the constant kernel.  Sidorenko for H is exactly inf F = 0, so
the search hunts for F < 0 by a large margin over float noise.

Two engines:
  * L-BFGS-B on the analytic gradient (`density.objective_and_grad`), from many
    structured and random starts;
  * CMA-ES on the same objective without gradients, which is what actually
    explores the far-from-constant region that the local analysis says a
    counterexample would have to inhabit.

Starts are deliberately biased away from quasirandomness: near-0/1 blowups,
sparse kernels, two-scale kernels, and log-normal kernels with large spread.
A start seeded near the constant kernel is included only as a sanity check that
the optimiser reproduces F -> 0 there.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

from .density import objective, objective_and_grad, unpack, weights_from_psi

__all__ = ["make_starts", "run_graph", "verify_hit", "TRIGGER"]

# Threshold for declaring a float hit.  The matrix is max-normalised before any
# contraction, so both t and p are O(1) in the flat region and the accumulated
# float error of one contraction is a few 1e-16 relative; -1e-9 leaves six
# orders of magnitude of headroom.  Any candidate below it is re-evaluated by
# `verify_hit` before it is believed, and certified by `sidorenko.certify`
# before it is reported.
TRIGGER = -1e-9


def make_starts(k, rng, n_random, optimize_weights):
    """Yield flat parameter vectors for `k` blocks."""
    iu = np.triu_indices(k)
    starts = []

    def add(theta, psi=None):
        theta = np.asarray(theta, float)
        theta = 0.5 * (theta + theta.T)
        vec = [theta[iu]]
        if optimize_weights:
            vec.append(np.zeros(k) if psi is None else np.asarray(psi, float))
        starts.append(np.concatenate(vec))

    # sanity start: the constant kernel (F must stay ~0)
    add(np.zeros((k, k)))

    # near-0/1 blowups: log of a random 0/1 symmetric matrix, zeros softened
    for _ in range(max(4, n_random // 3)):
        A = rng.integers(0, 2, size=(k, k))
        A = np.triu(A) + np.triu(A, 1).T
        if A.sum() == 0:
            A[0, 0] = 1
        add(np.where(A > 0, 0.0, -rng.uniform(4.0, 12.0)))

    # two-scale kernels: a dense core plus a sparse rim
    for _ in range(max(3, n_random // 6)):
        lo, hi = rng.uniform(-8.0, -1.0), rng.uniform(0.5, 6.0)
        A = rng.random((k, k))
        A = 0.5 * (A + A.T)
        add(np.where(A > rng.uniform(0.3, 0.7), hi, lo))

    # log-normal with a wide spread (the workhorse random start)
    for _ in range(n_random):
        s = rng.uniform(0.5, 6.0)
        add(rng.normal(0.0, s, size=(k, k)))

    # highly skewed weights, if weights are free
    if optimize_weights:
        for _ in range(max(2, n_random // 6)):
            A = rng.normal(0.0, rng.uniform(1.0, 5.0), size=(k, k))
            add(A, rng.normal(0.0, 2.0, size=k))
    return starts


# Parameter box for L-BFGS-B.  Without it the optimiser drifts into degenerate
# corners -- block weights ~1e-67, matrix entries spanning 1e47 -- where the
# objective is not evaluable in float64 at all (see `density.evaluate`).  The
# box costs no coverage: a block of weight below e^-THETA_MAX / k is within
# float rounding of the same kernel with that block deleted, which the search
# covers directly at smaller k, and an entry ratio beyond e^(2 THETA_MAX) is
# within rounding of the same kernel with that entry set to exactly 0, which the
# lattice tier decides exactly.  Since t(H, .) is continuous, a strictly
# negative deficit survives both perturbations.
THETA_MAX = 25.0
PSI_MAX = 12.0


def _bounds(k, optimize_weights):
    nt = k * (k + 1) // 2
    b = [(-THETA_MAX, THETA_MAX)] * nt
    if optimize_weights:
        b += [(-PSI_MAX, PSI_MAX)] * k
    return b


def _polish(g, x0, k, optimize_weights, weights, maxiter=600):
    fun = lambda x: objective_and_grad(g, x, k, optimize_weights, weights)
    bounds = _bounds(k, optimize_weights)
    x0 = np.clip(np.asarray(x0, float), [b[0] for b in bounds], [b[1] for b in bounds])
    res = minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": maxiter, "maxfun": 4 * maxiter, "ftol": 1e-16,
                            "gtol": 1e-12})
    return float(res.fun), np.asarray(res.x, float)


def _cma(g, x0, k, optimize_weights, weights, sigma, budget, seed):
    try:
        import cma
    except ImportError:
        return None
    fun = lambda x: objective_and_grad(g, x, k, optimize_weights, weights)[0]
    bounds = _bounds(k, optimize_weights)
    es = cma.CMAEvolutionStrategy(
        list(map(float, np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds]))), sigma,
        {"verbose": -9, "maxfevals": budget, "seed": int(seed) % (2**31 - 1),
         "tolfun": 1e-14, "tolx": 1e-11,
         "bounds": [[b[0] for b in bounds], [b[1] for b in bounds]]},
    )
    es.optimize(fun)
    return float(es.result.fbest), np.asarray(es.result.xbest, float)


def kernel_of(x, k, optimize_weights):
    theta, psi = unpack(np.asarray(x, float), k, optimize_weights)
    B = np.exp(np.clip(theta, -700.0, 700.0))
    a = weights_from_psi(psi) if optimize_weights else np.full(k, 1.0 / k)
    return a, B


def verify_hit(g, x, k, optimize_weights):
    """Re-evaluate a candidate hit through the careful `density.objective` path.

    The optimiser's objective is fast but is driven by L-BFGS into degenerate
    corners of the parameter space; a reported F < TRIGGER there can be an
    artefact of extreme dynamic range rather than a real violation.  This
    recomputes F from the reconstructed kernel and returns (survives, F_recheck),
    so an artefact is counted and discarded instead of aborting the search.
    Confirmation is still only advisory -- the certified verdict always comes
    from `sidorenko.certify`.
    """
    a, B = kernel_of(x, k, optimize_weights)
    F2 = objective(g, a, B)
    return bool(F2 < TRIGGER), float(F2)


def run_graph(g, block_counts=(2, 3, 4, 5, 6, 8), n_random=12, seed=0,
              cma_budget=4000, time_budget=None, optimize_weights=True,
              use_cma=True):
    """Search one graph.  Returns (record dict, best (a, B) float kernel)."""
    rng = np.random.default_rng([seed, g.n, g.m, abs(hash(g.name)) % 10**6])
    t0 = time.time()
    best = (np.inf, None, None)  # (F, k, x)
    n_starts = 0
    n_lbfgs_fail = 0
    per_k = []
    hit = None
    rejects = []  # candidate hits that did not survive the careful recheck

    for k in block_counts:
        weights = np.full(k, 1.0 / k)
        kbest = np.inf
        for x0 in make_starts(k, rng, n_random, optimize_weights):
            if time_budget and time.time() - t0 > time_budget:
                break
            n_starts += 1
            try:
                F, x = _polish(g, x0, k, optimize_weights, weights)
            except Exception:
                n_lbfgs_fail += 1
                continue
            if not np.isfinite(F) or F >= 1e99:
                continue
            kbest = min(kbest, F)
            if F < best[0]:
                best = (F, k, x)
            if F < TRIGGER:
                survives, F2 = verify_hit(g, x, k, optimize_weights)
                if survives:
                    hit = {"engine": "lbfgs", "k": k, "F": F, "F_recheck": F2}
                    break
                rejects.append({"engine": "lbfgs", "k": k, "F": F, "F_recheck": F2})
        if hit:
            break
        if use_cma and not (time_budget and time.time() - t0 > time_budget):
            for sigma in (0.8, 2.5):
                x0 = best[2] if (best[1] == k and best[2] is not None) else \
                    make_starts(k, rng, 1, optimize_weights)[-1]
                out = _cma(g, x0, k, optimize_weights, weights, sigma,
                           cma_budget, seed + 17 * k + int(sigma * 10))
                n_starts += 1
                if out is None:
                    continue
                F, x = out
                if not np.isfinite(F) or F >= 1e99:
                    continue
                kbest = min(kbest, F)
                if F < best[0]:
                    best = (F, k, x)
                if F < TRIGGER:
                    survives, F2 = verify_hit(g, x, k, optimize_weights)
                    if survives:
                        hit = {"engine": "cma", "k": k, "F": F, "F_recheck": F2}
                        break
                    rejects.append({"engine": "cma", "k": k, "F": F, "F_recheck": F2})
        per_k.append({"k": k, "best_F": None if not np.isfinite(kbest) else kbest})
        if hit:
            break

    F, kbest_k, xbest = best
    a, B = (None, None) if xbest is None else kernel_of(xbest, kbest_k, optimize_weights)
    rec = {
        "graph": g.name,
        "n": g.n,
        "e": g.m,
        "starts": n_starts,
        "lbfgs_failures": n_lbfgs_fail,
        "best_F": None if not np.isfinite(F) else F,
        "best_k": kbest_k,
        "per_k": per_k,
        "float_hit": hit,
        "numerical_rejects": len(rejects),
        "numerical_reject_detail": rejects[:8],
        "seconds": time.time() - t0,
    }
    if B is not None:
        rec["best_F_verified"] = objective(g, a, B)
        Bn = B / max(float(a @ B @ a), 1e-300)
        rec["best_weights"] = [float(v) for v in a]
        rec["best_matrix"] = [[float(v) for v in row] for row in Bn]
        rec["best_matrix_spread"] = float(np.max(Bn) / max(np.min(Bn), 1e-300))
    return rec, (a, B)
