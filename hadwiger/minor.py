"""K7-minor detection.

Soundness contract (the invariant the whole search rests on):

* "HAS a K7 minor" is only ever concluded from an explicit branch-set
  model that verify_k7_model() has checked. Heuristics propose models;
  they never decide.
* "Has NO K7 minor" is only ever concluded from an exact solver's UNSAT
  answer (direct encoding or CEGAR loop run to completion). Budget
  exhaustion returns 'unknown', which callers must escalate, never
  treat as 'no'.

Two exact deciders with independent encodings (sat_minor_direct,
sat_minor_cegar) plus a branch-and-bound third opinion (bb_minor) for
cross-validation and candidate verification.
"""

from __future__ import annotations

from .graphs import (components_of, connected_component, contract,
                     is_connected_induced)

T = 7  # target clique minor order


def verify_k7_model(n, adj, sets) -> bool:
    """sets: list of 7 vertex bitmasks. Checks disjoint, nonempty,
    induced-connected, pairwise adjacent."""
    if len(sets) != T:
        return False
    seen = 0
    for s in sets:
        if s == 0 or (s & seen):
            return False
        if s >> n:
            return False
        seen |= s
        if not is_connected_induced(adj, s):
            return False
    for i in range(T):
        for j in range(i + 1, T):
            si, sj = sets[i], sets[j]
            hit = False
            m = si
            while m:
                vb = m & -m
                m ^= vb
                if adj[vb.bit_length() - 1] & sj:
                    hit = True
                    break
            if not hit:
                return False
    return True


def find_clique7(n, adj):
    """Return a 7-clique as list of vertices, or None. Bitset BK w/ pivot."""
    result = None

    def bk(r_list, p_mask, x_mask):
        nonlocal result
        if result is not None:
            return
        if len(r_list) == T:
            result = list(r_list)
            return
        if len(r_list) + bin(p_mask).count("1") < T:
            return
        if p_mask == 0:
            return
        # pivot: vertex in P|X with most neighbors in P
        pux = p_mask | x_mask
        best_u, best_cnt = -1, -1
        m = pux
        while m:
            vb = m & -m
            m ^= vb
            v = vb.bit_length() - 1
            c = bin(adj[v] & p_mask).count("1")
            if c > best_cnt:
                best_u, best_cnt = v, c
        cand = p_mask & ~adj[best_u]
        while cand:
            vb = cand & -cand
            cand ^= vb
            v = vb.bit_length() - 1
            bk(r_list + [v], p_mask & adj[v], x_mask & adj[v])
            if result is not None:
                return
            p_mask &= ~vb
            x_mask |= vb

    bk([], (1 << n) - 1, 0)
    return result


def greedy_contraction_minor(n, adj, rng, tries=8):
    """Randomized contraction heuristic. Returns a VERIFIED model or None."""
    base_adj = adj
    for _t in range(tries):
        cn, cadj = n, list(adj)
        masks = [1 << v for v in range(n)]  # original-vertex mask per supernode
        while cn >= T:
            if cn <= 13:
                cl = find_clique7(cn, tuple(cadj))
                if cl is not None:
                    sets = [masks[v] for v in cl]
                    if verify_k7_model(n, base_adj, sets):
                        return sets
                    break  # heuristic proposed junk; try next restart
            if cn == T:
                break
            # sample candidate edges, contract the one whose endpoints share
            # the most neighbors (densifies the quotient)
            pairs = []
            for _ in range(min(24, cn * 2)):
                u = rng.randrange(cn)
                if cadj[u] == 0:
                    continue
                nb = cadj[u]
                choices = []
                m = nb
                while m:
                    vb = m & -m
                    m ^= vb
                    choices.append(vb.bit_length() - 1)
                v = rng.choice(choices)
                score = bin(cadj[u] & cadj[v]).count("1")
                pairs.append((score, u, v))
            if not pairs:
                break
            pairs.sort(reverse=True)
            _, u, v = pairs[0]
            if v < u:
                u, v = v, u
            nn, nadj = contract(cn, tuple(cadj), u, v)
            masks[u] |= masks[v]
            del masks[v]
            cn, cadj = nn, list(nadj)
        else:
            continue
    return None


def minorminer_find(n, adj, tries=10):
    """Optional heuristic via minorminer. Returns VERIFIED model or None."""
    try:
        import minorminer
    except Exception:
        return None
    g_edges = [(u, v) for u in range(n) for v in range(u + 1, n)
               if (adj[u] >> v) & 1]
    k7_edges = [(i, j) for i in range(T) for j in range(i + 1, T)]
    try:
        emb = minorminer.find_embedding(k7_edges, g_edges, tries=tries,
                                        verbose=0)
    except Exception:
        return None
    if not emb or len(emb) != T:
        return None
    sets = []
    for i in range(T):
        m = 0
        for v in emb.get(i, []):
            m |= 1 << v
        sets.append(m)
    if verify_k7_model(n, adj, sets):
        return sets
    return None


def fast_minor_search(n, adj, rng, greedy_tries=8, use_minorminer=False,
                      mm_tries=10):
    """Cheap positive path. Returns verified model or None (=unknown)."""
    cl = find_clique7(n, adj)
    if cl is not None:
        sets = [1 << v for v in cl]
        if verify_k7_model(n, adj, sets):
            return sets
    m = greedy_contraction_minor(n, adj, rng, tries=greedy_tries)
    if m is not None:
        return m
    if use_minorminer:
        m = minorminer_find(n, adj, tries=mm_tries)
        if m is not None:
            return m
    return None


# ---------------------------------------------------------------- exact SAT

def _minor_base_clauses(n, adj, var):
    """Partition vars + at-most-one + nonempty + pairwise set adjacency.
    Shared by both encodings. var(tag,...) allocates/returns ints."""
    clauses = []
    x = [[var("x", v, i) for i in range(T)] for v in range(n)]
    for v in range(n):
        for i in range(T):
            for j in range(i + 1, T):
                clauses.append([-x[v][i], -x[v][j]])
    for i in range(T):
        clauses.append([x[v][i] for v in range(n)])
    edge_list = [(u, v) for u in range(n) for v in range(u + 1, n)
                 if (adj[u] >> v) & 1]
    for i in range(T):
        for j in range(i + 1, T):
            ors = []
            for (u, v) in edge_list:
                for (a, b) in ((u, v), (v, u)):
                    y = var("a", a, b, i, j)
                    clauses.append([-y, x[a][i]])
                    clauses.append([-y, x[b][j]])
                    ors.append(y)
            clauses.append(ors)
    return clauses, x


def sat_minor_direct(n, adj, solver_name=None, conf_budget=None):
    """Complete one-shot encoding: connectivity via canonical roots +
    layered reachability. Returns ('yes', verified sets) / ('no', None)
    / ('unknown', None)."""
    from .coloring import make_solver

    counter = [0]
    vmap = {}

    def var(*key):
        if key not in vmap:
            counter[0] += 1
            vmap[key] = counter[0]
        return vmap[key]

    clauses, x = _minor_base_clauses(n, adj, var)
    layers = max(1, n - T)  # branch set size <= n-6, so paths need <= n-7 hops
    for i in range(T):
        for v in range(n):
            r = var("root", v, i)
            clauses.append([-r, x[v][i]])
            for u in range(v):
                clauses.append([-r, -x[u][i]])
            # x[v][i] & no smaller member -> root[v][i]
            clauses.append([r, -x[v][i]] + [x[u][i] for u in range(v)])
            l0 = var("L", v, i, 0)
            clauses.append([-l0, r])
            clauses.append([-r, l0])
        for t in range(1, layers + 1):
            for v in range(n):
                lt = var("L", v, i, t)
                sup = [-lt, var("L", v, i, t - 1)]
                nb = adj[v]
                while nb:
                    wb = nb & -nb
                    nb ^= wb
                    sup.append(var("L", wb.bit_length() - 1, i, t - 1))
                clauses.append(sup)
                clauses.append([-lt, x[v][i]])
        for v in range(n):
            clauses.append([-x[v][i], var("L", v, i, layers)])
    # symmetry: root of set i precedes root of set i+1
    for i in range(T - 1):
        for v in range(n):
            clauses.append([-var("root", v, i + 1)]
                           + [var("root", u, i) for u in range(v)])

    solver, nm = make_solver(clauses, name=solver_name)
    try:
        if conf_budget:
            solver.conf_budget(conf_budget)
            res = solver.solve_limited(expect_interrupt=False)
        else:
            res = solver.solve()
        if res is True:
            model = set(l for l in solver.get_model() if l > 0)
            sets = []
            for i in range(T):
                m = 0
                for v in range(n):
                    if vmap[("x", v, i)] in model:
                        m |= 1 << v
                sets.append(m)
            assert verify_k7_model(n, adj, sets), \
                "direct encoding SAT but extracted model invalid — encoding bug"
            return "yes", sets
        if res is False:
            return "no", None
        return "unknown", None
    finally:
        solver.delete()


def sat_minor_cegar(n, adj, solver_name=None, max_iters=20000):
    """Independent exact decider: partition+adjacency encoding, lazy
    connectivity cuts. Returns ('yes', verified sets) / ('no', None) /
    ('unknown', None) if iteration cap hit."""
    from .coloring import make_solver

    counter = [0]
    vmap = {}

    def var(*key):
        if key not in vmap:
            counter[0] += 1
            vmap[key] = counter[0]
        return vmap[key]

    clauses, x = _minor_base_clauses(n, adj, var)
    solver, nm = make_solver(clauses, name=solver_name)
    try:
        for _it in range(max_iters):
            if not solver.solve():
                return "no", None
            model = set(l for l in solver.get_model() if l > 0)
            sets = []
            for i in range(T):
                m = 0
                for v in range(n):
                    if vmap[("x", v, i)] in model:
                        m |= 1 << v
                sets.append(m)
            bad = False
            for i, s in enumerate(sets):
                comps = components_of(adj, s)
                if len(comps) <= 1:
                    continue
                bad = True
                # Sound cut, for every pair of components (Ca, Cb) with
                # witnesses u=min(Ca), v=min(Cb): a connected set containing
                # both u and v must contain a vertex of N(Ca)\Ca. (If a
                # future model keeps only one of u,v the clause is satisfied
                # by the negative literal — single components alone are NOT
                # blocked, which is what makes this sound.)
                for a in range(len(comps)):
                    boundary = 0
                    m = comps[a]
                    while m:
                        vb = m & -m
                        m ^= vb
                        boundary |= adj[vb.bit_length() - 1]
                    boundary &= ~comps[a]
                    bnd = []
                    bm = boundary
                    while bm:
                        wb = bm & -bm
                        bm ^= wb
                        bnd.append(vmap[("x", wb.bit_length() - 1, i)])
                    u = (comps[a] & -comps[a]).bit_length() - 1
                    for b in range(len(comps)):
                        if a == b:
                            continue
                        v = (comps[b] & -comps[b]).bit_length() - 1
                        solver.add_clause([-vmap[("x", u, i)],
                                           -vmap[("x", v, i)]] + bnd)
            if not bad:
                assert verify_k7_model(n, adj, sets), \
                    "CEGAR model passed connectivity but failed verification"
                return "yes", sets
        return "unknown", None
    finally:
        solver.delete()


def bb_minor(n, adj, node_budget=2_000_000):
    """Third opinion: complete contraction recursion. G has a K7 minor iff
    some sequence of contractions produces a K7 subgraph. Returns
    'yes'/'no'/'unknown' (budget)."""
    budget = [node_budget]

    def rec(cn, cadj):
        if budget[0] <= 0:
            return None
        budget[0] -= 1
        if cn < T:
            return False
        if find_clique7(cn, cadj) is not None:
            return True
        if cn == T:
            return False
        seen_quotients = set()
        for u in range(cn):
            nb = cadj[u] >> (u + 1)
            while nb:
                wb = nb & -nb
                nb ^= wb
                v = wb.bit_length() - 1 + u + 1
                qn, qadj = contract(cn, cadj, u, v)
                if qadj in seen_quotients:
                    continue
                seen_quotients.add(qadj)
                r = rec(qn, qadj)
                if r is None:
                    return None
                if r:
                    return True
        return False

    r = rec(n, tuple(adj))
    if r is None:
        return "unknown"
    return "yes" if r else "no"


def elimination_width_ub(n, adj, strategy="mindeg"):
    """Greedy elimination order; returns the max bag size - 1 encountered
    (an upper bound on treewidth). The simulation itself is the
    certificate: eliminating v turns its current neighborhood into a
    clique, and the width is the largest such neighborhood."""
    cur = list(adj)
    alive = (1 << n) - 1
    width = 0
    for _ in range(n):
        best_v, best_key = -1, None
        m = alive
        while m:
            vb = m & -m
            m ^= vb
            v = vb.bit_length() - 1
            deg = bin(cur[v] & alive).count("1")
            if strategy == "minfill":
                nbm = cur[v] & alive
                fill = 0
                mm = nbm
                while mm:
                    ub = mm & -mm
                    mm ^= ub
                    u = ub.bit_length() - 1
                    fill += bin(nbm & ~cur[u] & ~ub).count("1")
                key = (fill, deg)
            else:
                key = (deg,)
            if best_key is None or key < best_key:
                best_v, best_key = v, key
        v = best_v
        nbm = cur[v] & alive
        width = max(width, bin(nbm).count("1"))
        mm = nbm
        while mm:
            ub = mm & -mm
            mm ^= ub
            u = ub.bit_length() - 1
            cur[u] |= nbm & ~(1 << u)
        alive &= ~(1 << v)
    return width


def treewidth_le5(n, adj) -> bool:
    """Sound K7-minor-free shortcut: any elimination order of width <= 5
    proves tw <= 5 < tw(K7) = 6, hence no K7 minor."""
    if elimination_width_ub(n, adj, "mindeg") <= 5:
        return True
    if elimination_width_ub(n, adj, "minfill") <= 5:
        return True
    return False


def exact_minor_decision(n, adj, rng, require_agreement=False):
    """Main exact entry. Fast witnessed path first; then direct SAT.
    With require_agreement (candidate protocol) CEGAR must concur.
    Returns (verdict, sets_or_None, detail)."""
    sets = fast_minor_search(n, adj, rng, use_minorminer=True)
    if sets is not None:
        return "yes", sets, {"path": "fast"}
    if not require_agreement and treewidth_le5(n, adj):
        # tw<=5 also implies chi<=6, so no candidate can ever live here;
        # the candidate protocol (require_agreement) still runs both SATs.
        return "no", None, {"path": "tw<=5"}
    v1, s1 = sat_minor_direct(n, adj)
    if v1 == "yes":
        return "yes", s1, {"path": "direct"}
    if v1 == "unknown":
        return "unknown", None, {"path": "direct"}
    if not require_agreement:
        return "no", None, {"path": "direct"}
    v2, s2 = sat_minor_cegar(n, adj)
    if v2 == "no":
        return "no", None, {"path": "direct+cegar-agree"}
    return ("conflict" if v2 == "yes" else "unknown"), s2, \
        {"path": "direct-no/cegar-" + v2}
