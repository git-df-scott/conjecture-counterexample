"""Exact certification of the Sidorenko deficit on rational step kernels.

Pure stdlib: no numpy, no float arithmetic anywhere, and no code shared with
the search tier -- the contraction is re-implemented here over Python integers
so that a certificate cannot inherit a bug from `sidorenko.contract`.

Soundness
---------
Fix a bipartite (or any) simple graph H on n vertices with e edges.  A rational
step kernel is given by integer data

    block weights  a_i = c_i / C     (c_i >= 0 integers, sum c_i = C > 0),
    kernel values  W  = M_ij / D     (M symmetric nonnegative integers, D > 0).

Then, exactly,

    t(H, W)     = T / (C^n  D^e),     T = sum over phi: V(H) -> [k] of
                                          prod_v c_{phi(v)} prod_{uv in E} M_{phi(u)phi(v)}
    t(K_2, W)   = S / (C^2  D),       S = sum_{i,j} c_i c_j M_ij

and therefore

    t(H, W) - t(K_2, W)^e = ( T * C^{2e} - S^e * C^n ) / ( C^{n + 2e} * D^e ).

The denominator is a positive integer, so the *sign of the deficit is the sign
of the integer*

    Delta = T * C^{2e} - S^e * C^n.

T is an integer computed by exact integer variable elimination; S, C, D and the
powers are integers.  Delta < 0 is therefore a complete, unconditional
counterexample certificate, and Delta >= 0 is an unconditional verification
that this kernel is not a counterexample.  There is no error direction and no
tolerance: unlike the geometric certification in `mahler/`, this decision
procedure is exact in both directions.

Graphon-valued form
-------------------
The definition of a graphon also asks for W <= 1.  Since the deficit is
homogeneous of degree e under W -> cW, dividing M by max M leaves the sign of
Delta unchanged, so any certificate here transfers verbatim to a genuine
graphon: report `graphon_matrix` = M / (D * max M) with D chosen so the entries
are the rationals M_ij / max M in [0, 1].
"""

from __future__ import annotations

import itertools
from fractions import Fraction

__all__ = ["exact_hom_sum", "certify", "rational_candidates", "certify_best"]


# --------------------------------------------------------------------------
# exact integer contraction (independent of sidorenko.contract)
# --------------------------------------------------------------------------


def _elimination_order(n, edges):
    """Greedy min-degree order (deliberately a different heuristic from the
    search tier's min-fill, so the two paths agree only if both are right)."""
    adj = {v: set() for v in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    live = set(range(n))
    order = []
    width = 1 if n else 0
    while live:
        v = min(live, key=lambda x: (len(adj[x] & live), x))
        nb = sorted(adj[v] & live)
        width = max(width, len(nb) + 1)
        for i, x in enumerate(nb):
            for y in nb[i + 1 :]:
                adj[x].add(y)
                adj[y].add(x)
        live.discard(v)
        order.append(v)
    return order, width


class _Factor:
    """Dense table over a sorted tuple of variables, flat row-major list."""

    __slots__ = ("vars", "data", "k")

    def __init__(self, vars_, data, k):
        self.vars = tuple(vars_)
        self.data = data
        self.k = k

    @staticmethod
    def unary(v, values, k):
        return _Factor((v,), list(values), k)

    @staticmethod
    def binary(u, v, mat, k):
        assert u < v
        return _Factor((u, v), [mat[i][j] for i in range(k) for j in range(k)], k)

    def _strides(self):
        k, r = self.k, len(self.vars)
        st = [0] * r
        s = 1
        for i in range(r - 1, -1, -1):
            st[i] = s
            s *= k
        return st

    def multiply(self, other):
        k = self.k
        out_vars = tuple(sorted(set(self.vars) | set(other.vars)))
        pos = {v: i for i, v in enumerate(out_vars)}
        st_a = self._strides()
        st_b = other._strides()
        map_a = [(pos[v], st_a[i]) for i, v in enumerate(self.vars)]
        map_b = [(pos[v], st_b[i]) for i, v in enumerate(other.vars)]
        a, b = self.data, other.data
        out = []
        for assign in itertools.product(range(k), repeat=len(out_vars)):
            ia = 0
            for p, s in map_a:
                ia += assign[p] * s
            ib = 0
            for p, s in map_b:
                ib += assign[p] * s
            out.append(a[ia] * b[ib])
        return _Factor(out_vars, out, k)

    def sum_out(self, v):
        k = self.k
        axis = self.vars.index(v)
        out_vars = self.vars[:axis] + self.vars[axis + 1 :]
        st = self._strides()
        s_ax = st[axis]
        keep = [(i, st[i]) for i in range(len(self.vars)) if i != axis]
        data = self.data
        out = []
        for assign in itertools.product(range(k), repeat=len(out_vars)):
            base = 0
            j = 0
            for i, s in keep:
                base += assign[j] * s
                j += 1
            acc = 0
            for x in range(k):
                acc += data[base + x * s_ax]
            out.append(acc)
        return _Factor(out_vars, out, k)


def exact_hom_sum(n, edges, c, M):
    """T = sum over phi of prod_v c_{phi(v)} prod_{uv in E} M_{phi(u)phi(v)}.

    All arguments and the result are Python integers (arbitrary precision).
    """
    k = len(c)
    for i in range(k):
        for j in range(k):
            if M[i][j] != M[j][i]:
                raise ValueError("kernel matrix must be symmetric")
    edges = [(min(u, v), max(u, v)) for u, v in edges]
    order, _ = _elimination_order(n, edges)
    factors = [_Factor.unary(v, c, k) for v in range(n)]
    factors += [_Factor.binary(u, v, M, k) for u, v in edges]
    for v in order:
        touching = [f for f in factors if v in f.vars]
        factors = [f for f in factors if v not in f.vars]
        if not touching:
            continue
        acc = touching[0]
        for f in touching[1:]:
            acc = acc.multiply(f)
        factors.append(acc.sum_out(v))
    total = 1
    for f in factors:
        assert f.vars == ()
        total *= f.data[0]
    return total


def exact_hom_sum_bruteforce(n, edges, c, M):
    """Direct k^n integer sum; used by tests to pin the contraction."""
    k = len(c)
    total = 0
    for phi in itertools.product(range(k), repeat=n):
        term = 1
        for v in range(n):
            term *= c[phi[v]]
        for u, v in edges:
            term *= M[phi[u]][phi[v]]
        total += term
    return total


# --------------------------------------------------------------------------
# certificate
# --------------------------------------------------------------------------


def certify(g, c, M, D=1):
    """Exact verdict for the step kernel (a_i = c_i/C, W = M/D).

    Returns a dict with the integer discriminant `delta`, the exact rational
    values of t(H, W) and t(K_2, W), and `is_counterexample` (delta < 0).
    """
    c = [int(x) for x in c]
    M = [[int(x) for x in row] for row in M]
    D = int(D)
    if D <= 0:
        raise ValueError("D must be positive")
    if any(x < 0 for x in c) or any(x < 0 for row in M for x in row):
        raise ValueError("weights and kernel values must be nonnegative")
    C = sum(c)
    if C <= 0:
        raise ValueError("block weights must sum to a positive integer")
    n, e = g.n, g.m
    T = exact_hom_sum(n, g.edges, c, M)
    S = sum(c[i] * c[j] * M[i][j] for i in range(len(c)) for j in range(len(c)))
    delta = T * C ** (2 * e) - S**e * C**n
    t_H = Fraction(T, C**n * D**e)
    t_K2 = Fraction(S, C**2 * D)
    mx = max((x for row in M for x in row), default=0)
    return {
        "graph": g.name,
        "n": n,
        "e": e,
        "k": len(c),
        "weights_num": c,
        "weights_den": C,
        "matrix": M,
        "matrix_den": D,
        "hom_density": [t_H.numerator, t_H.denominator],
        "edge_density": [t_K2.numerator, t_K2.denominator],
        "deficit": [
            (t_H - t_K2**e).numerator,
            (t_H - t_K2**e).denominator,
        ],
        "delta": delta,
        "delta_sign": (0 if delta == 0 else (1 if delta > 0 else -1)),
        "is_counterexample": delta < 0,
        "graphon_max_entry": mx,
    }


# --------------------------------------------------------------------------
# float -> rational rounding
# --------------------------------------------------------------------------


def rational_candidates(weights, mat, denominators=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128),
                        zero_thresholds=(0.0, 1e-6, 1e-4, 1e-3, 1e-2, 3e-2)):
    """Rational step kernels near a float one, as (c, C, M, D) integer data.

    The float optimum is only ever a combinatorial *hint*: every emitted
    candidate is an exact object in its own right, and `certify` decides it
    without reference to the float input.
    """
    import math

    k = len(weights)
    mx = max(max(row) for row in mat)
    if not (mx > 0) or not math.isfinite(mx):
        return []
    out = []
    seen = set()
    for wq in (0, 6, 12, 24, 60, 120, 360):
        if wq == 0:
            c, C = [1] * k, k
        else:
            c = [max(0, round(w * wq)) for w in weights]
            if sum(c) == 0 or min(c) == 0:
                continue
            C = sum(c)
        for D in denominators:
            for z in zero_thresholds:
                M = [
                    [0 if mat[i][j] / mx <= z else max(0, round(mat[i][j] / mx * D))
                     for j in range(k)]
                    for i in range(k)
                ]
                M = [[(M[i][j] + M[j][i]) // 2 for j in range(k)] for i in range(k)]
                if all(x == 0 for row in M for x in row):
                    continue
                key = (tuple(c), C, tuple(tuple(r) for r in M), D)
                if key in seen:
                    continue
                seen.add(key)
                out.append((c, C, M, D))
    return out


def certify_best(g, weights, mat, max_candidates=400):
    """Round a float kernel to rationals and return the exact verdicts.

    Returns (best_certificate, n_certified) where `best` minimises the exact
    deficit; `is_counterexample` on it is the certified answer for the whole
    rounded family.
    """
    cands = rational_candidates(weights, mat)[:max_candidates]
    best = None
    for c, _C, M, D in cands:
        cert = certify(g, c, M, D)
        key = Fraction(*cert["deficit"])
        if best is None or key < Fraction(*best["deficit"]):
            best = cert
        if cert["is_counterexample"]:
            return cert, len(cands)
    return best, len(cands)
