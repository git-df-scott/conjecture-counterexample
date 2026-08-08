"""Tests for the Sidorenko search pipeline.

Run: python3 tests/test_sidorenko.py

The tests pin four independent things against each other:
  * the min-fill tensor contraction vs. the direct k^n sum;
  * the analytic gradient vs. central finite differences, and both Euler
    identities implied by homogeneity;
  * the pure-stdlib exact integer certifier vs. the numpy float tier;
  * the predicted local vanishing order (= girth) vs. a numerical log-log fit.
Plus sign controls in both directions: odd cycles must be flagged as
counterexamples, and known-positive bipartite classes must not be.
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sidorenko import density as D
from sidorenko.certify import (
    certify,
    exact_hom_sum,
    exact_hom_sum_bruteforce,
    rational_candidates,
)
from sidorenko.graph import (
    Graph,
    complete_bipartite,
    corpus,
    crown,
    cycle,
    desargues,
    enumerate_bipartite,
    grid,
    heawood,
    hypercube,
    kt_minus_hamilton_cycle,
    mobius_kantor,
    path,
    petersen,
    projective_plane_incidence,
)
from sidorenko.lattice import sweep
from sidorenko.optimize import run_graph
from sidorenko.local import local_report

FAILURES = []


def check(cond, msg):
    if cond:
        print(f"  ok   {msg}")
    else:
        print(f"  FAIL {msg}")
        FAILURES.append(msg)


def approx(a, b, tol):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def test_graph_basics():
    print("graph basics")
    check(cycle(3).bipartition() is None, "C_3 is not bipartite")
    check(cycle(4).bipartition() is not None, "C_4 is bipartite")
    check(petersen().bipartition() is None, "Petersen is not bipartite")
    check(heawood().girth() == 6 and heawood().m == 21, "Heawood: girth 6, 21 edges")
    check(mobius_kantor().girth() == 6 and mobius_kantor().n == 16,
          "Moebius-Kantor: girth 6, 16 vertices")
    check(desargues().girth() == 6 and desargues().m == 30, "Desargues: girth 6, 30 edges")
    k55 = kt_minus_hamilton_cycle(5)
    check(k55.n == 10 and k55.m == 15 and set(k55.degrees()) == {3},
          "K_{5,5} minus C_10 is cubic on 10 vertices")
    check(sorted(crown(4).degrees()) == [3] * 8, "K_{4,4}-PM is cubic")
    check(crown(4).is_isomorphic_to(hypercube(3)), "K_{4,4}-PM is isomorphic to Q_3")
    check(projective_plane_incidence(2).is_isomorphic_to(heawood()),
          "PG(2,2) incidence graph is the Heawood graph")
    check(heawood().count_cycles_of_length(6) == 28, "Heawood has 28 hexagons")
    check(cycle(6).count_cycles_of_length(6) == 1, "C_6 has one hexagon")
    check(complete_bipartite(3, 3).count_cycles_of_length(4) == 9, "K_{3,3} has 9 squares")
    # known-positive labelling
    check("tree" in (path(4).known_positive_reason() or ""), "P_4 labelled a tree")
    check("even cycle" in (cycle(8).known_positive_reason() or ""), "C_8 labelled an even cycle")
    check("complete bipartite" in (complete_bipartite(3, 4).known_positive_reason() or ""),
          "K_{3,4} labelled complete bipartite")
    check("hypercube" in (hypercube(3).known_positive_reason() or ""), "Q_3 labelled a hypercube")
    check(kt_minus_hamilton_cycle(5).known_positive_reason() is None,
          "K_{5,5}-C_10 is in no proven class")
    check(heawood().known_positive_reason() is None, "Heawood is in no proven class")


def test_contraction_vs_bruteforce():
    print("contraction vs. brute force")
    rng = np.random.default_rng(11)
    graphs = [cycle(3), cycle(4), cycle(6), path(4), complete_bipartite(2, 3),
              complete_bipartite(3, 3), hypercube(3), grid(3, 3)]
    worst = 0.0
    for g in graphs:
        for k in (2, 3, 4):
            a = rng.random(k)
            a /= a.sum()
            B = rng.random((k, k))
            B = 0.5 * (B + B.T)
            t1 = D.hom_density(g, a, B)
            t2 = D.hom_density_bruteforce(g, a, B)
            worst = max(worst, abs(t1 - t2) / max(t2, 1e-300))
    check(worst < 1e-11, f"max relative error over 24 cases = {worst:.2e}")

    # a graph with an isolated vertex and one with a pendant path
    g = Graph.make("pendant", 5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 1)])
    a = np.array([0.3, 0.7])
    B = np.array([[0.2, 0.9], [0.9, 0.4]])
    check(approx(D.hom_density(g, a, B), D.hom_density_bruteforce(g, a, B), 1e-12),
          "duplicate-edge input normalises to a simple graph")


def test_gradients():
    print("gradients and Euler identities")
    rng = np.random.default_rng(5)
    for g in [cycle(4), complete_bipartite(2, 3), kt_minus_hamilton_cycle(5), hypercube(3)]:
        k = 4
        a = rng.random(k)
        a /= a.sum()
        B = rng.random((k, k))
        B = 0.5 * (B + B.T)
        t, p, _dF_dS, _dF_da, dt_dS, dt_da = D.gradients(g, a, B)
        iu = np.triu_indices(k)
        check(approx(float((B[iu] * dt_dS[iu]).sum()), g.m * t, 1e-10),
              f"{g.name}: Euler identity in the matrix (degree e = {g.m})")
        check(approx(float(a @ dt_da), g.n * t, 1e-10),
              f"{g.name}: Euler identity in the weights (degree n = {g.n})")

        x = D.pack(np.log(B), np.log(a))
        _F, grad = D.objective_and_grad(g, x, k, True, None)
        h = 1e-6
        num = np.zeros_like(grad)
        for i in range(len(x)):
            xp, xm = x.copy(), x.copy()
            xp[i] += h
            xm[i] -= h
            num[i] = (D.objective_and_grad(g, xp, k, True, None)[0]
                      - D.objective_and_grad(g, xm, k, True, None)[0]) / (2 * h)
        err = float(np.max(np.abs(num - grad)))
        check(err < 1e-6 * max(1.0, float(np.max(np.abs(grad)))),
              f"{g.name}: analytic gradient matches finite differences ({err:.1e})")


def test_scale_invariance_and_zero_at_constant():
    print("scale invariance; F = 0 at the constant kernel")
    rng = np.random.default_rng(3)
    for g in [cycle(6), heawood(), kt_minus_hamilton_cycle(5)]:
        k = 5
        a = np.full(k, 1.0 / k)
        B = rng.random((k, k))
        B = 0.5 * (B + B.T)
        f1 = D.objective(g, a, B)
        f2 = D.objective(g, a, 137.5 * B)
        check(approx(f1, f2, 1e-12), f"{g.name}: F invariant under W -> cW")
        check(abs(D.objective(g, a, np.full((k, k), 0.37))) < 1e-12,
              f"{g.name}: F = 0 at the constant kernel")
        check(f1 > -1e-9, f"{g.name}: random kernel does not violate Sidorenko")


def test_exact_contraction():
    print("exact integer contraction")
    for g in [cycle(3), cycle(4), path(4), complete_bipartite(2, 3), grid(3, 3)]:
        for c, M in (
            ([1, 1], [[0, 1], [1, 0]]),
            ([1, 2, 3], [[0, 2, 1], [2, 1, 3], [1, 3, 0]]),
            ([2, 1, 1], [[3, 0, 1], [0, 2, 2], [1, 2, 1]]),
        ):
            a = exact_hom_sum(g.n, g.edges, c, M)
            if len(c) ** g.n <= 3 ** 9:
                b = exact_hom_sum_bruteforce(g.n, g.edges, c, M)
                check(a == b, f"{g.name}: exact contraction == brute force (k={len(c)})")
            else:
                check(isinstance(a, int), f"{g.name}: exact contraction returns an int")


def test_exact_matches_float():
    print("exact tier agrees with the float tier")
    rng = np.random.default_rng(9)
    for g in [heawood(), mobius_kantor(), crown(5), grid(3, 4)]:
        k = 4
        M = rng.integers(0, 5, size=(k, k))
        M = np.triu(M) + np.triu(M, 1).T
        Ml = [[int(x) for x in row] for row in M]
        T = exact_hom_sum(g.n, g.edges, [1] * k, Ml)
        exact = T / float(k) ** g.n
        flt = D.hom_density(g, np.full(k, 1.0 / k), M.astype(float))
        check(approx(exact, flt, 1e-11), f"{g.name}: exact t = float t ({exact:.10g})")


def test_positive_controls():
    print("positive controls: non-bipartite H must be flagged")
    # the bipartite-blowup kernel kills every odd cycle
    for g in [cycle(3), cycle(5), cycle(7), petersen()]:
        cert = certify(g, [1, 1], [[0, 1], [1, 0]])
        check(cert["is_counterexample"], f"{g.name}: exactly certified violation")
        check(cert["hom_density"][0] == 0, f"{g.name}: t(H, W) = 0 on the blowup kernel")
    # and the float tier must see it too
    a = np.array([0.5, 0.5])
    B = np.array([[0.0, 1.0], [1.0, 0.0]])
    for g in [cycle(3), cycle(5)]:
        check(D.objective(g, a, B) == -np.inf, f"{g.name}: float objective reports -inf")
    r = sweep(cycle(3))
    check(not r["clean"] and r["counterexamples"], "lattice sweep flags C_3")
    r = sweep(petersen())
    check(not r["clean"], "lattice sweep flags the Petersen graph")


def test_negative_controls():
    print("negative controls: proven classes must never be flagged")
    for g in [cycle(4), cycle(6), path(4), complete_bipartite(2, 3),
              complete_bipartite(3, 3), hypercube(3)]:
        r = sweep(g, tiers=((2, 8, True, 0), (3, 4, True, 0), (4, 2, True, 0)),
                  tier_seconds=60.0)
        check(r["clean"], f"{g.name}: exhaustive lattice sweep finds no violation")
    # a directly checkable exact value: C_4 on the two-block bipartite kernel
    cert = certify(cycle(4), [1, 1], [[0, 1], [1, 0]])
    check(cert["hom_density"] == [1, 8] and cert["edge_density"] == [1, 2],
          "C_4 on the blowup kernel: t = 1/8, p = 1/2, so t - p^4 = 1/16 > 0")
    check(not cert["is_counterexample"], "C_4 not flagged")


def test_local_order():
    print("local vanishing order equals the girth")
    for g in [cycle(4), cycle(6), complete_bipartite(3, 3), hypercube(3), heawood(),
              mobius_kantor(), kt_minus_hamilton_cycle(5), grid(3, 3)]:
        r = local_report(g, seed=3)
        check(r["matches_prediction"] is True,
              f"{g.name}: order {r['measured_vanishing_order']:.3f} ~ girth {r['girth']}")


def test_numerical_guards():
    """Regression: degenerate kernels must be rejected, not believed.

    L-BFGS walks to points such as block weights (2.8e-86, 1) with matrix
    entries spanning 1e83.  There the unnormalised edge product overflows, and
    even after max-normalisation t and p^e(H) can both be subnormal (~1e-320)
    where they carry ~4 significant digits -- enough error to manufacture a
    violation of a *proven* case out of nothing.  Both concrete points below
    did exactly that (C_8 reported -6.4e-06, Q_3 reported -0.405).
    """
    print("numerical guards on degenerate kernels")
    bad = [
        (cycle(8), np.array([2.82e-86, 1.0]),
         np.array([[1.0, 2.61e83], [2.61e83, 1.0]])),
        (hypercube(3), np.array([3.15e-67, 1.0, 1.218e-66]),
         np.array([[1.0, 1.1267e-35, 5.117e-47],
                   [1.1267e-35, 2.0752e-27, 2.3488e-47],
                   [5.117e-47, 2.3488e-47, 1.7621e-40]])),
    ]
    for g, a, B in bad:
        F, status = D.evaluate(g, a, B)
        check(F >= 1e99 and status != "ok",
              f"{g.name}: degenerate kernel rejected ({status})")

    # A genuine structural zero is still a violation, decided by an exact
    # integer homomorphism count rather than by float underflow.
    a2 = np.array([0.5, 0.5])
    Bz = np.array([[0.0, 1.0], [1.0, 0.0]])
    check(D.evaluate(cycle(3), a2, Bz) == (-np.inf, "ok"), "C_3 structural zero -> -inf")
    check(D.evaluate(cycle(5), a2, Bz) == (-np.inf, "ok"), "C_5 structural zero -> -inf")
    F4, _ = D.evaluate(cycle(4), a2, Bz)
    check(abs(F4 - np.log(2.0)) < 1e-12, "C_4 on the same kernel -> +log 2, not a zero")
    check(D.support_hom_count(cycle(3), a2, Bz) == 0, "support hom count of C_3 is 0")
    check(D.support_hom_count(cycle(4), a2, Bz) == 2, "support hom count of C_4 is 2")
    check(D.support_hom_count(cycle(4), a2, np.ones((2, 2))) == 16, "C_4 -> full support: 2^4")

    # end-to-end: no proven-positive graph may produce a surviving float hit
    for g in [cycle(6), cycle(8), complete_bipartite(3, 3), hypercube(3)]:
        rec, _ = run_graph(g, block_counts=(2, 3, 4), n_random=6, seed=1, time_budget=25)
        check(rec["float_hit"] is None and rec["best_F"] > -1e-9,
              f"{g.name}: no float hit, floor {rec['best_F']:+.2e}")


def test_rounding_candidates():
    print("float -> rational rounding"),
    k = 3
    a = np.full(k, 1.0 / k)
    B = np.array([[0.01, 1.0, 0.5], [1.0, 0.02, 0.9], [0.5, 0.9, 0.03]])
    cands = rational_candidates([float(x) for x in a], [[float(x) for x in r] for r in B])
    check(len(cands) > 20, f"{len(cands)} rational candidates generated")
    check(all(all(x >= 0 for row in M for x in row) for _c, _C, M, _D in cands),
          "all candidates are nonnegative")
    check(any(any(x == 0 for row in M for x in row) for _c, _C, M, _D in cands),
          "some candidates set small entries to exact zero")
    for c, C, M, Dn in cands[:40]:
        cert = certify(cycle(6), c, M, Dn)
        check_once = cert["delta_sign"] >= 0
        if not check_once:
            check(False, f"C_6 wrongly flagged on {M}")
            return
    check(True, "C_6 survives every rounded candidate exactly")


def test_enumeration():
    print("small exhaustive bipartite enumeration")
    import networkx as nx

    def independent_count(p, q):
        """Pairwise-VF2 dedup with no Weisfeiler-Leman bucketing: a different
        code path from enumerate_bipartite, so agreement is real evidence."""
        slots = [(i, p + j) for i in range(p) for j in range(q)]
        reps = []
        for mask in range(1 << len(slots)):
            edges = [slots[b] for b in range(len(slots)) if mask >> b & 1]
            if len(edges) < max(p, q):
                continue
            G = nx.Graph()
            G.add_nodes_from(range(p + q))
            G.add_edges_from(edges)
            if not nx.is_connected(G):
                continue
            if not any(nx.is_isomorphic(G, H) for H in reps):
                reps.append(G)
        return len(reps)

    for (p, q), expect in (((2, 2), 2), ((2, 3), 4), ((3, 3), 10), ((3, 4), 34)):
        gs = enumerate_bipartite(p, q)
        check(len(gs) == expect == independent_count(p, q),
              f"parts {p}+{q}: {len(gs)} graphs, matches the independent count")
        check(all(g.is_connected() for g in gs), f"parts {p}+{q}: all connected")
    gs = enumerate_bipartite(3, 3)
    check(not any(gs[i].is_isomorphic_to(gs[j])
                  for i in range(len(gs)) for j in range(i + 1, len(gs))),
          "enumeration output is pairwise non-isomorphic")


def test_corpus_sanity():
    print("corpus sanity")
    entries = corpus()
    names = [g.name for g, _ in entries]
    check(len(names) == len(set(names)), f"corpus names unique ({len(names)} entries)")
    check(all(g.is_connected() for g, _ in entries), "every corpus graph is connected")
    nonbip = [g.name for g, _ in entries if g.bipartition() is None]
    check(len(nonbip) >= 4, f"positive controls present: {nonbip}")
    hard = [g.name for g, _ in entries
            if g.bipartition() is not None and g.known_positive_reason() is None]
    check(len(hard) >= 15, f"{len(hard)} bipartite graphs in no proven class")
    check("K_{5,5}-C_{10}" in names, "the cited smallest open case is in the corpus")


def main():
    for fn in (
        test_graph_basics,
        test_contraction_vs_bruteforce,
        test_gradients,
        test_scale_invariance_and_zero_at_constant,
        test_exact_contraction,
        test_exact_matches_float,
        test_positive_controls,
        test_negative_controls,
        test_local_order,
        test_numerical_guards,
        test_rounding_candidates,
        test_enumeration,
        test_corpus_sanity,
    ):
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("all Sidorenko tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
