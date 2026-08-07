#!/usr/bin/env python3
"""Validation gate for the Hadwiger t=7 pipeline.

There is no known-answer small case to checkpoint against (unlike Mahler
dim-3), so this gate substitutes: known-by-theorem controls in both
directions, cross-validation of three independent exact minor deciders,
chromatic-number controls, and negative tests of the witness verifiers
themselves. Any failure => exit 1; the search must not run.

Usage: python3 scripts/hadwiger_validate.py [--quick]
"""

import argparse
import random
import sys
import time

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(
    __import__("os").path.abspath(__file__))))

from hadwiger import family
from hadwiger.coloring import sat_kcolorable, verify_coloring
from hadwiger.graphs import (edge_count, parse_graph6, random_gnp, to_graph6)
from hadwiger.minor import (bb_minor, fast_minor_search, sat_minor_cegar,
                            sat_minor_direct, treewidth_le5, verify_k7_model)
from hadwiger.runner_util import jsonl_append, results_path

FAILURES = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not ok:
        FAILURES.append(name)
    jsonl_append(results_path("validation.jsonl"),
                 {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "check": name, "ok": ok, "detail": detail})


def exact_no(n, adj, cegar_limit=14):
    """Direct decider must say 'no'; CEGAR must concur for n <= cegar_limit
    (it converges too slowly on larger no-instances — measured 31s direct
    vs >260s CEGAR at n=22)."""
    t0 = time.time()
    v1, _ = sat_minor_direct(n, adj)
    d1 = time.time() - t0
    if n <= cegar_limit:
        t0 = time.time()
        v2, _ = sat_minor_cegar(n, adj)
        d2 = time.time() - t0
        return (v1 == "no" and v2 == "no",
                f"direct={v1}({d1:.1f}s) cegar={v2}({d2:.1f}s)")
    return v1 == "no", f"direct={v1}({d1:.1f}s) cegar=skipped(n>{cegar_limit})"


def exact_yes(n, adj):
    v1, s1 = sat_minor_direct(n, adj)
    v2, s2 = sat_minor_cegar(n, adj)
    ok = v1 == "yes" and v2 == "yes"
    ok = ok and verify_k7_model(n, adj, s1) and verify_k7_model(n, adj, s2)
    return ok, f"direct={v1} cegar={v2}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="reduced random cross-validation only")
    args = ap.parse_args()
    rng = random.Random(20260807)
    t_start = time.time()

    # -- A. graph6 roundtrip + geng interop ------------------------------
    for _ in range(200):
        n = rng.randrange(2, 40)
        adj = random_gnp(n, rng.random(), rng)
        n2, adj2 = parse_graph6(to_graph6(n, adj))
        if (n2, adj2) != (n, adj):
            check("graph6-roundtrip", False, f"n={n}")
            break
    else:
        check("graph6-roundtrip", True, "200 random graphs")
    big = family.subdivide_all_edges(*family.complete(13))  # n = 91: long g6 format
    nb, adjb = big
    n2, adj2 = parse_graph6(to_graph6(nb, adjb))
    check("graph6-roundtrip-large-n", (n2, adj2) == (nb, adjb), f"n={nb}")

    import subprocess
    out = subprocess.run(["nauty-geng", "-q", "6"], capture_output=True,
                         text=True).stdout.strip().splitlines()
    parsed = [parse_graph6(l) for l in out]
    check("geng-n6-count", len(parsed) == 156, f"got {len(parsed)} (A000088: 156)")
    out_t = subprocess.run(["nauty-geng", "-q", "-t", "8"], capture_output=True,
                           text=True).stdout.strip().splitlines()
    tri_ok = True
    for l in out_t:
        n, adj = parse_graph6(l)
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1 and adj[u] & adj[v]:
                    tri_ok = False
    check("geng-t-trianglefree", tri_ok, f"{len(out_t)} graphs on 8 vertices")

    # -- B. chromatic controls ------------------------------------------
    chi_known = [
        ("K7", family.complete(7), 7),
        ("C5", family.cycle(5), 3),
        ("Petersen", family.petersen(), 3),
        ("Kneser(7,2)", family.kneser(7, 2), 5),
        ("Kneser(8,2)", family.kneser(8, 2), 6),
        ("Mycielski(C5)=Groetzsch", family.mycielski(5, family.cycle(5)[1]), 4),
        ("Mycielski^2(C5)", family.mycielski(11, family.mycielski(5, family.cycle(5)[1])[1]), 5),
    ]
    for name, (n, adj), chi in chi_known:
        lo, _, _ = sat_kcolorable(n, adj, chi - 1)
        hi, col, _ = sat_kcolorable(n, adj, chi)
        ok = lo is False and hi is True and verify_coloring(n, adj, col, chi)
        check(f"chi-{name}", ok, f"expected chi={chi}")
    if not args.quick:
        n, adj = family.kneser(9, 2)
        t0 = time.time()
        res, _, _ = sat_kcolorable(n, adj, 6, clique_shortcut=False)
        check("chi-Kneser(9,2)-not-6-colorable", res is False,
              f"UNSAT in {time.time()-t0:.1f}s (chi=7 by Lovasz/Kneser)")

    # -- C. minor must-detect -------------------------------------------
    detect = [
        ("K7", family.complete(7)),
        ("K8", family.complete(8)),
        ("K10-PM", family.complete_minus_perfect_matching(10)),
        ("K7-1subdivision", family.subdivide_all_edges(*family.complete(7))),
    ]
    for name, (n, adj) in detect:
        sets = fast_minor_search(n, adj, rng, use_minorminer=True)
        if sets is None and n <= 30:
            v, sets = sat_minor_direct(n, adj)
        check(f"detect-{name}", sets is not None and verify_k7_model(n, adj, sets),
              f"n={n}")
    for nn in (12, 16, 20):
        for s in range(2):
            n, adj = family.mader_supercritical(nn, rng)
            sets = fast_minor_search(n, adj, rng, use_minorminer=True)
            if sets is None:
                v, sets = sat_minor_direct(n, adj)
                sets = sets if v == "yes" else None
            check(f"detect-mader-{nn}-{s}",
                  sets is not None and verify_k7_model(n, adj, sets),
                  "e=5n-14 forces K7 minor (Mader)")
    # exact deciders must also say yes on the small ones
    for name, (n, adj) in detect[:2] + [detect[3]]:
        ok, d = exact_yes(n, adj)
        check(f"exact-yes-{name}", ok, d)
    # verify_candidate's aggregation on a known non-candidate: K7 has
    # chi=7 but obviously has a K7 minor => must NOT be confirmed
    import os
    import subprocess as _sp
    from hadwiger.graphs import to_graph6 as _tg
    p = _sp.run([sys.executable, os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "verify_candidate.py"),
        _tg(*family.complete(7))], capture_output=True, text=True)
    check("verify-candidate-rejects-K7",
          '"VERDICT": "NOT CONFIRMED"' in p.stdout, "chi>=7 but minor present")
    if not args.quick:
        n, adj = family.kneser(9, 2)
        sets = fast_minor_search(n, adj, rng, use_minorminer=True)
        check("detect-Kneser(9,2)", sets is not None and
              verify_k7_model(n, adj, sets), "36 vertices, heuristic path")

    # -- D. minor must-clear --------------------------------------------
    clear = [("K6", family.complete(6))]
    for nn in (10, 13) + ((16,) if not args.quick else ()):
        for s in range(2):
            clear.append((f"5tree-{nn}-{s}", family.random_5tree(nn, rng)))
    clear.append(("planar4x5+2apex", family.planar_grid_2apex(4, 5, rng)))
    # treewidth shortcut: must fire on low-width clears, must NOT fire on
    # graphs that do have K7 minors
    n3, adj3 = family.planar_grid_2apex(3, 7, rng)
    check("tw-shortcut-fires-3x7+2apex", treewidth_le5(n3, adj3), f"n={n3}")
    for nn in (10, 14):
        check(f"tw-shortcut-fires-5tree-{nn}",
              treewidth_le5(*family.random_5tree(nn, rng)), "")
    check("tw-shortcut-refuses-K7", not treewidth_le5(*family.complete(7)), "")
    check("tw-shortcut-refuses-mader16",
          not treewidth_le5(*family.mader_supercritical(16, rng)), "")
    for name, (n, adj) in clear:
        ok, d = exact_no(n, adj)
        check(f"clear-{name}", ok, d + f" n={n} e={edge_count(n, adj)}")
        wit = fast_minor_search(n, adj, rng, use_minorminer=True)
        check(f"clear-noheurwitness-{name}", wit is None,
              "heuristic must not fabricate a witness")

    # -- E. three-way cross-validation on randoms ------------------------
    trials = 60 if args.quick else 400
    mism = 0
    tested = 0
    for _t in range(trials):
        n = rng.randrange(8, 13)
        style = rng.randrange(3)
        if style == 0:
            adj = random_gnp(n, rng.choice([0.2, 0.35, 0.5, 0.65]), rng)
        elif style == 1:
            e = max(0, min(n * (n - 1) // 2, 5 * n - 15 + rng.randrange(-3, 4)))
            from hadwiger.graphs import random_gnm
            adj = random_gnm(n, e, rng)
        else:
            adj = family.random_5tree(max(n, 6), rng)[1]
            n = max(n, 6)
            if rng.random() < 0.5:
                from hadwiger.graphs import add_edge
                u, v = rng.randrange(n), rng.randrange(n)
                if u != v:
                    adj = add_edge(adj, u, v)
        v1, s1 = sat_minor_direct(n, adj)
        v2, s2 = sat_minor_cegar(n, adj)
        v3 = bb_minor(n, adj, node_budget=300_000) if n <= 10 else v1
        tested += 1
        if not (v1 == v2 and (v3 in (v1, "unknown"))):
            mism += 1
            jsonl_append(results_path("validation.jsonl"),
                         {"check": "crossval-mismatch", "g6": to_graph6(n, adj),
                          "direct": v1, "cegar": v2, "bb": v3})
    check("crossval-3way", mism == 0, f"{tested} graphs, {mism} mismatches")

    # -- F. the verifiers themselves must reject corrupted witnesses -----
    n, adj = family.complete(8)
    good = fast_minor_search(n, adj, rng)
    ok = verify_k7_model(n, adj, good)
    bad1 = list(good); bad1[0] = 0                      # empty set
    bad2 = list(good); bad2[1] = bad2[0]                # overlap
    bad3 = list(good); bad3[2] = bad3[2] | (1 << (n - 1)) | (1 << 0)  # overlap/absorb
    nn6, adj6 = family.complete(6)
    check("verifier-rejects-corrupt",
          ok and not verify_k7_model(n, adj, bad1)
          and not verify_k7_model(n, adj, bad2)
          and not verify_k7_model(n, adj, bad3)
          and not verify_k7_model(nn6, adj6, good), "")
    colors_bad = [0] * 7
    check("coloring-verifier-rejects", not verify_coloring(7, family.complete(7)[1],
          colors_bad, 6) and not verify_coloring(3, family.cycle(3)[1], [0, 1], 6), "")

    # disconnected-set witness must be rejected (regression for CEGAR cut bug)
    path4 = (0b0010, 0b0101, 0b1010, 0b0100)  # path 0-1-2-3
    disc = [0b1001]  # {0,3} disconnected in path
    check("verifier-rejects-disconnected",
          not verify_k7_model(4, path4, disc * 7), "")

    dt = time.time() - t_start
    print(f"\n{'='*60}")
    if FAILURES:
        print(f"GATE FAILED ({len(FAILURES)}): {FAILURES}  [{dt:.0f}s]")
        sys.exit(1)
    print(f"GATE PASSED — all checks green in {dt:.0f}s")


if __name__ == "__main__":
    main()
