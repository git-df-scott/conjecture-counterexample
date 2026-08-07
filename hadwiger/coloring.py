"""Chromatic-number machinery.

Soundness invariant: a graph is only ever discarded as "k-colorable" on
the strength of an explicit proper coloring that verify_coloring() has
checked. Heuristic failure alone never decides anything; SAT UNSAT is
the only accepted proof of non-k-colorability.
"""

from __future__ import annotations

from pysat.solvers import Solver

SOLVER_PREFERENCE = ("cd19", "cd15", "cd", "g42", "g4", "m22")


def make_solver(clauses, name=None):
    names = (name,) if name else SOLVER_PREFERENCE
    last = None
    for nm in names:
        try:
            return Solver(name=nm, bootstrap_with=clauses), nm
        except Exception as e:  # solver not built into this pysat
            last = e
    raise RuntimeError(f"no SAT solver available: {last}")


def verify_coloring(n, adj, colors, k) -> bool:
    if colors is None or len(colors) != n:
        return False
    if any(c < 0 or c >= k for c in colors):
        return False
    for v in range(n):
        nb = adj[v]
        while nb:
            wb = nb & -nb
            nb ^= wb
            w = wb.bit_length() - 1
            if w > v and colors[w] == colors[v]:
                return False
    return True


def dsatur(n, adj, k, rng=None, tries=1):
    """Try to find a proper <=k-coloring; returns verified coloring or None."""
    import random as _random

    rng = rng or _random.Random(12345)
    degs = [bin(a).count("1") for a in adj]
    for _ in range(tries):
        colors = [-1] * n
        nbr_colors = [0] * n  # bitmask of colors used by colored neighbors
        noise = [rng.random() for _ in range(n)]
        ok = True
        for _step in range(n):
            best, best_key = -1, None
            for v in range(n):
                if colors[v] >= 0:
                    continue
                key = (bin(nbr_colors[v]).count("1"), degs[v], noise[v])
                if best_key is None or key > best_key:
                    best, best_key = v, key
            c = 0
            while c < k and (nbr_colors[best] >> c) & 1:
                c += 1
            if c >= k:
                ok = False
                break
            colors[best] = c
            nb = adj[best]
            while nb:
                wb = nb & -nb
                nb ^= wb
                nbr_colors[wb.bit_length() - 1] |= 1 << c
        if ok and verify_coloring(n, adj, colors, k):
            return colors
    return None


def greedy_clique(n, adj, rng=None, tries=8):
    import random as _random

    rng = rng or _random.Random(999)
    best = []
    for _ in range(tries):
        order = sorted(range(n), key=lambda v: -(bin(adj[v]).count("1") + rng.random()))
        clique = []
        cmask = 0
        for v in order:
            if (adj[v] & cmask) == cmask:
                clique.append(v)
                cmask |= 1 << v
        if len(clique) > len(best):
            best = clique
    return best


def color_clauses(n, adj, k, precolor=None):
    # var for (v, c): v*k + c + 1
    clauses = []
    for v in range(n):
        clauses.append([v * k + c + 1 for c in range(k)])
    for u in range(n):
        nb = adj[u]
        while nb:
            wb = nb & -nb
            nb ^= wb
            v = wb.bit_length() - 1
            if v > u:
                for c in range(k):
                    clauses.append([-(u * k + c + 1), -(v * k + c + 1)])
    if precolor:
        for c, v in enumerate(precolor):
            clauses.append([v * k + c + 1])
    return clauses


def sat_kcolorable(n, adj, k, conf_budget=None, solver_name=None,
                   clique_shortcut=True, rng=None):
    """Exact k-colorability. Returns (status, coloring, meta):
    (True, verified coloring, meta) / (False, None, meta) / (None, None, meta)
    when a conflict budget was exhausted. meta carries solver name +
    conflict stats for annealing fitness."""
    clique = greedy_clique(n, adj, rng=rng)
    if clique_shortcut and len(clique) > k:
        return False, None, {"proof": f"clique{len(clique)}", "clique": clique}
    precolor = clique[: min(len(clique), k)]
    clauses = color_clauses(n, adj, k, precolor=precolor)
    solver, nm = make_solver(clauses, name=solver_name)
    try:
        if conf_budget:
            solver.conf_budget(conf_budget)
            res = solver.solve_limited(expect_interrupt=False)
        else:
            res = solver.solve()
        stats = solver.accum_stats()
        if res is True:
            model = solver.get_model()
            colors = [-1] * n
            for v in range(n):
                for c in range(k):
                    if model[v * k + c] > 0:  # model is 1-indexed & ordered
                        colors[v] = c
                        break
            assert verify_coloring(n, adj, colors, k), \
                "SAT said k-colorable but extracted coloring invalid"
            return True, colors, {"solver": nm, "stats": stats}
        if res is False:
            return False, None, {"solver": nm, "proof": "unsat", "stats": stats}
        return None, None, {"solver": nm, "stats": stats}
    finally:
        solver.delete()


def chi_at_least_7(n, adj, dsatur_tries=4, rng=None):
    """Fast filter: returns (False, coloring) if a verified 6-coloring was
    found, else (True_pending, None) meaning 'needs exact confirmation'."""
    c = dsatur(n, adj, 6, rng=rng, tries=dsatur_tries)
    if c is not None:
        return False, c
    return True, None
