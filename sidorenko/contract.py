"""Tensor-network contraction of homomorphism counts / densities on step kernels.

A *step kernel* is the data (a, B) where a = (a_1..a_k) are nonnegative block
weights and B is a symmetric k x k nonnegative matrix; it represents the
symmetric measurable function

    W(x, y) = B_{ij}   for x in block i, y in block j,

where block i has Lebesgue measure a_i.  For a simple graph H on vertex set
V = {0..n-1} with edge set E, the homomorphism density is the finite sum

    t(H, W) = sum over maps phi: V -> [k] of  prod_v a_{phi(v)} prod_{uv in E} B_{phi(u)phi(v)}.

Summing over k^n maps directly is hopeless for the graphs of interest (the
Desargues graph has n = 20), so this module evaluates the sum as a tensor
network: one variable per vertex of H, one unary factor `a` per vertex, one
binary factor `B` per edge, contracted by variable elimination in a greedy
min-fill order.  The cost is O(n * k^(w+1)) where w is the width of the
elimination order (an upper bound on the treewidth of H).  For every graph in
this repository's corpus w <= 8, so evaluations are milliseconds.

The routine is deliberately dtype-agnostic: it works for float64 (search
tier), for int64 (exact integer sweeps over small-value lattices, where the
caller is responsible for the overflow bound) and for numpy object arrays of
Python ints or Fractions (exact, unbounded).  `sidorenko.certify` does NOT use
this module -- it re-implements the same contraction in pure stdlib Python so
that the certified path is independent of both numpy and of the code exercised
by the search.
"""

from __future__ import annotations

import numpy as np

__all__ = ["min_fill_order", "contract", "ContractionPlan", "plan_for"]


def min_fill_order(n, edges, keep=()):
    """Greedy min-fill elimination order for the vertices of H outside `keep`.

    Returns (order, width).  `width` is max over elimination steps of
    1 + #(live neighbours), i.e. the largest tensor rank materialised, so the
    peak memory is k^width entries.  Vertices in `keep` stay live forever
    (they are the open/uncontracted indices of the result).
    """
    keep = set(keep)
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)
    live = set(range(n))
    order = []
    width = 1 if n else 0
    while live - keep:
        best, best_key = None, None
        for v in sorted(live - keep):
            nb = adj[v] & live
            fill = sum(
                1
                for i, x in enumerate(sorted(nb))
                for y in sorted(nb)[i + 1 :]
                if y not in adj[x]
            )
            key = (fill, len(nb))
            if best_key is None or key < best_key:
                best, best_key = v, key
        nb = sorted(adj[best] & live)
        width = max(width, len(nb) + 1)
        for i, x in enumerate(nb):
            for y in nb[i + 1 :]:
                adj[x].add(y)
                adj[y].add(x)
        for x in nb:
            adj[x].discard(best)
        live.discard(best)
        order.append(best)
    # The residual clique on `keep` must also be materialised.
    if keep:
        width = max(width, len(keep & live))
    return order, width


class ContractionPlan:
    """Cached elimination order for one (H, open_vars, dropped edges) shape."""

    __slots__ = ("n", "edges", "open_vars", "drop_unary", "order", "width")

    def __init__(self, n, edges, open_vars=(), drop_unary=()):
        self.n = n
        self.edges = tuple(tuple(sorted(e)) for e in edges)
        self.open_vars = tuple(open_vars)
        self.drop_unary = frozenset(drop_unary)
        self.order, self.width = min_fill_order(n, self.edges, keep=self.open_vars)


_PLAN_CACHE: dict = {}


def plan_for(n, edges, open_vars=(), drop_unary=()):
    key = (
        n,
        tuple(sorted(tuple(sorted(e)) for e in edges)),
        tuple(open_vars),
        tuple(sorted(drop_unary)),
    )
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        plan = ContractionPlan(n, edges, open_vars, drop_unary)
        _PLAN_CACHE[key] = plan
    return plan


def _expand(vars_from, arr, vars_to):
    """Reshape `arr` (indexed by the sorted tuple vars_from) to broadcast
    against the sorted tuple vars_to."""
    shape = []
    it = iter(vars_from)
    cur = next(it, None)
    pos = 0
    for v in vars_to:
        if cur is not None and v == cur:
            shape.append(arr.shape[pos])
            pos += 1
            cur = next(it, None)
        else:
            shape.append(1)
    return arr.reshape(shape)


def _multiply(f1, f2):
    v1, a1 = f1
    v2, a2 = f2
    if v1 == v2:
        return v1, a1 * a2
    vs = tuple(sorted(set(v1) | set(v2)))
    return vs, _expand(v1, a1, vs) * _expand(v2, a2, vs)


def contract(plan, weights, mat):
    """Evaluate the tensor network for `plan` with block weights and matrix.

    `weights` is a length-k array, `mat` a symmetric k x k array; both must
    share a dtype (float64, int64 or object).  Returns an array indexed by
    *sorted* plan.open_vars (a 0-d array if there are none); use
    `contract_open` to get the axes in the requested order.

    Vertices listed in plan.drop_unary contribute no `weights` factor; this is
    what makes d t / d a_i computable as a single contraction.
    """
    weights = np.asarray(weights)
    mat = np.asarray(mat)
    if weights.ndim != 1 or mat.shape != (weights.shape[0], weights.shape[0]):
        raise ValueError("shape mismatch between weights and mat")

    factors = []  # list of (sorted vars tuple, ndarray)
    for v in range(plan.n):
        if v not in plan.drop_unary:
            factors.append(((v,), weights))
    for u, v in plan.edges:
        if u == v:
            raise ValueError("H must be a simple loopless graph")
        factors.append(((u, v), mat))

    # Bucket factors by their earliest-eliminated variable.
    position = {v: i for i, v in enumerate(plan.order)}
    buckets = [[] for _ in plan.order]
    inert = []
    for f in factors:
        idxs = [position[v] for v in f[0] if v in position]
        if idxs:
            buckets[min(idxs)].append(f)
        else:
            inert.append(f)

    for step, v in enumerate(plan.order):
        bucket = buckets[step]
        if not bucket:
            continue
        acc = bucket[0]
        for f in bucket[1:]:
            acc = _multiply(acc, f)
        vs, arr = acc
        axis = vs.index(v)
        arr = arr.sum(axis=axis)
        vs = vs[:axis] + vs[axis + 1 :]
        if vs:
            idxs = [position[x] for x in vs if x in position]
            (buckets[min(idxs)] if idxs else inert).append((vs, arr))
        else:
            inert.append(((), arr))

    if not inert:
        return np.asarray(weights.dtype.type(1) if weights.dtype != object else 1)
    acc = inert[0]
    for f in inert[1:]:
        acc = _multiply(acc, f)
    # Open vars that appear in no surviving factor (isolated in H once its
    # unary factor and/or an incident edge was dropped) need explicit
    # broadcasting to full size.
    missing = [v for v in sorted(plan.open_vars) if v not in acc[0]]
    for v in missing:
        acc = _multiply(acc, ((v,), np.ones(weights.shape[0], dtype=weights.dtype)))
    return acc[1]


def contract_open(plan, weights, mat):
    """Like `contract` but guarantees the axis order matches plan.open_vars."""
    arr = contract(plan, weights, mat)
    want = tuple(plan.open_vars)
    if len(want) <= 1:
        return arr
    src = tuple(sorted(want))
    if src == want:
        return arr
    return np.moveaxis(arr, [src.index(v) for v in want], range(len(want)))
