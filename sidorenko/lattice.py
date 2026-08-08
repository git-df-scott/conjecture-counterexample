"""Exhaustive *exact* sweep over small-value integer step kernels.

The continuous search in `sidorenko.optimize` is a local method and can only
ever report "nothing found here".  This module complements it with a genuinely
exhaustive statement over a precisely delimited family: for a block count k and
a value cap m, it enumerates **every** symmetric matrix M in {0..m}^{k x k} up
to simultaneous row/column permutation, with uniform block weights a_i = 1/k,
and decides the sign of the Sidorenko deficit for each, exactly.

Exactness without pure-Python cost
----------------------------------
With uniform weights the deficit sign is the sign of the integer

    Delta = T * k^{2e} - S^e * k^n,
    T = sum_phi prod_{uv in E} M_{phi(u)phi(v)},   S = sum_{i,j} M_ij

(the c_i = 1, C = k case of `sidorenko.certify`).  T is computed by int64
variable elimination, which is exact -- not merely accurate -- provided no
intermediate value reaches 2^63.  Every intermediate factor entry is a sum over
partial maps of products of at most e matrix entries, hence bounded by
k^n * m^e; the sweep computes that bound in Python integers first and switches
to object dtype (unbounded Python ints) when it is not comfortably below 2^62.
The final comparison is done in Python integers regardless.  So every verdict
below is unconditional, and the extremal points are re-verified through the
completely independent pure-stdlib path in `sidorenko.certify`.

Coverage is bounded by combinatorics, not by rigour: k <= 5 with m = 1 and
k <= 4 with m <= 3 are exhaustive; larger (k, m) are randomly sampled, and the
sample size is reported so the claim stays honest.
"""

from __future__ import annotations

import itertools
import random
import time

import numpy as np

from .contract import contract, plan_for

__all__ = ["kernel_classes", "sweep", "TIERS"]

# (k, m, exhaustive?, sample size when not exhaustive)
TIERS = (
    (2, 8, True, 0),
    (3, 4, True, 0),
    (4, 3, True, 0),
    (5, 1, True, 0),
    (5, 2, False, 40000),
    (6, 1, False, 40000),
    (7, 1, False, 40000),
)

_CLASS_CACHE: dict = {}


def _triu_pairs(k):
    return [(i, j) for i in range(k) for j in range(i, k)]


def kernel_classes(k, m, exhaustive=True, sample=0, seed=20240817):
    """Symmetric {0..m}^{k x k} matrices up to simultaneous permutation.

    Returns (classes, n_scanned, exhaustive) where each class is a tuple of
    upper-triangle entries in `_triu_pairs(k)` order.  Matrices that are
    identically zero, or whose support is disconnected from every block in a way
    that makes the edge density zero, are dropped (they have t(K_2, W) = 0 and
    the deficit is then trivially 0 = 0).
    """
    key = (k, m, exhaustive, sample, seed)
    if key in _CLASS_CACHE:
        return _CLASS_CACHE[key]

    pairs = _triu_pairs(k)
    index = {p: i for i, p in enumerate(pairs)}
    perms = list(itertools.permutations(range(k)))
    if len(perms) > 720:
        # Deduplication is a pure efficiency device -- a missed identification
        # only means the same kernel is decided twice, never that a kernel is
        # skipped -- so for k >= 7 we quotient by the dihedral subgroup instead
        # of the full symmetric group to keep the setup cost bounded.
        rots = [tuple((i + s) % k for i in range(k)) for s in range(k)]
        perms = rots + [tuple(reversed(r)) for r in rots]
    perm_maps = []
    for p in perms:
        perm_maps.append(
            tuple(index[(min(p[i], p[j]), max(p[i], p[j]))] for (i, j) in pairs)
        )

    def canon(vec):
        best = None
        for pm in perm_maps:
            cand = tuple(vec[t] for t in pm)
            if best is None or cand < best:
                best = cand
        return best

    if exhaustive:
        # Vectorised: encode each upper triangle as a base-(m+1) integer, take
        # the elementwise minimum of the k! permuted encodings, and keep the
        # distinct minima.  Equivalent to the `canon` loop above but ~100x
        # faster, which is what makes the k=4, m=3 tier (4^10 matrices) usable.
        base = m + 1
        L = len(pairs)
        total = base**L
        powers = base ** np.arange(L - 1, -1, -1, dtype=np.int64)
        digits = (np.arange(total, dtype=np.int64)[:, None] // powers) % base
        keys = None
        for pm in perm_maps:
            cand = digits[:, list(pm)] @ powers
            keys = cand if keys is None else np.minimum(keys, cand)
        keys = np.unique(keys)
        out = []
        for key in keys.tolist():
            vec = tuple(int(key // p) % base for p in powers.tolist())
            if any(vec):
                out.append(vec)
        scanned = total
    else:
        seen = set()
        out = []
        scanned = 0
        rng = random.Random(seed)
        for _ in range(sample):
            vec = tuple(rng.randint(0, m) for _ in pairs)
            scanned += 1
            if not any(vec):
                continue
            c = canon(vec)
            if c in seen:
                continue
            seen.add(c)
            out.append(c)

    res = (out, scanned, bool(exhaustive))
    _CLASS_CACHE[key] = res
    return res


def _matrix_from_vec(vec, k):
    pairs = _triu_pairs(k)
    M = np.zeros((k, k), dtype=object)
    for t, (i, j) in enumerate(pairs):
        M[i, j] = int(vec[t])
        M[j, i] = int(vec[t])
    return M


def _hom_sum_int(g, M, k, m):
    """T over the integers, exactly.  Chooses int64 when provably safe."""
    bound = k**g.n * max(1, m) ** g.m
    ones = np.ones(k, dtype=np.int64)
    if bound < (1 << 62):
        Mi = M.astype(np.int64)
        val = contract(plan_for(g.n, g.edges), ones, Mi)
        return int(val)
    val = contract(plan_for(g.n, g.edges), np.ones(k, dtype=object), M)
    return int(val)


def sweep(g, tiers=TIERS, stop_on_hit=True, progress=None, tier_seconds=90.0,
          total_seconds=None, seed=7):
    """Exact verdicts over the lattice tiers for one graph H.

    Per-kernel cost grows like k^(treewidth+1), so for the widest graphs an
    exhaustive tier can be unaffordable.  Rather than silently truncating, each
    tier is timed on its first kernels and, if the full class list does not fit
    in `tier_seconds`, a deterministic random subsample is taken and the tier is
    reported with exhaustive=False plus the exact `covered` fraction.  A tier is
    only ever labelled exhaustive when every class in it was decided.
    """
    n, e = g.n, g.m
    rng = np.random.default_rng([seed, n, e])
    report = {"graph": g.name, "n": n, "e": e, "tiers": [], "counterexamples": []}
    t_start = time.time()
    for (k, m, exhaustive, sample) in tiers:
        if total_seconds and time.time() - t_start > total_seconds:
            report["tiers"].append({"k": k, "value_cap": m, "skipped": "total budget"})
            continue
        classes, scanned, was_exh = kernel_classes(k, m, exhaustive, sample)
        if not classes:
            continue
        order = rng.permutation(len(classes))
        best = None
        hits = []
        done = 0
        t0 = time.time()
        budget_stop = False
        for pos in order:
            if done >= 8 and (time.time() - t0) > tier_seconds:
                budget_stop = True
                break
            vec = classes[pos]
            M = _matrix_from_vec(vec, k)
            S = int(M.sum())
            done += 1
            if S == 0:
                continue
            T = _hom_sum_int(g, M, k, m)
            delta = T * k ** (2 * e) - S**e * k**n
            if best is None or delta < best[0]:
                best = (delta, vec)
            if delta < 0:
                hits.append({"k": k, "m": m, "matrix_triu": list(vec), "delta": delta})
                if stop_on_hit:
                    break
        complete = (not budget_stop) and not (hits and stop_on_hit)
        tier = {
            "k": k,
            "value_cap": m,
            "exhaustive": bool(was_exh and complete),
            "classes": len(classes),
            "classes_decided": done,
            "covered": done / len(classes),
            "matrices_scanned": scanned,
            "budget_truncated": budget_stop,
            "seconds": time.time() - t0,
            "min_delta_witness_triu": list(best[1]) if best else None,
            # delta magnitudes are astronomically large integers; store the
            # sign and a decimal digit count rather than the full number.
            "min_delta_sign": (
                None if best is None else (0 if best[0] == 0 else (1 if best[0] > 0 else -1))
            ),
            "min_delta_digits": (None if best is None else len(str(abs(best[0])))),
        }
        report["tiers"].append(tier)
        report["counterexamples"].extend(hits)
        if progress:
            progress(g, tier)
        if hits and stop_on_hit:
            break
    report["clean"] = not report["counterexamples"]
    report["fully_exhaustive_tiers"] = sum(
        1 for t in report["tiers"] if t.get("exhaustive")
    )
    report["kernels_decided"] = sum(t.get("classes_decided", 0) for t in report["tiers"])
    report["seconds"] = time.time() - t_start
    return report
