#!/usr/bin/env python3
"""Controls checkpoint for the Sidorenko pipeline -- run this before the search.

Three things are established here, and the main search is only meaningful if
all three hold:

1. **The trigger fires when it should.**  For non-bipartite H (odd cycles, K_4,
   Petersen) Sidorenko provably fails, and the pipeline must produce an *exact*
   counterexample certificate.  A search that cannot detect a violation it is
   handed is worthless as evidence of absence.

2. **The trigger does not fire when it shouldn't.**  For bipartite H in a proven
   class (trees, even cycles, complete bipartite, hypercubes) the exhaustive
   lattice sweep must come back clean.  Any hit here is a bug in this repository,
   not a disproof of a theorem.

3. **The local structure is what the theory says.**  The deficit must vanish to
   order exactly girth(H) at the constant kernel, with a nonnegative leading
   coefficient.  This is measured, not assumed, and it is the reason the search
   is designed to start far from quasirandomness.

Output: results/sidorenko/controls.json
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sidorenko import density as D
from sidorenko.certify import certify
from sidorenko.graph import (
    complete_bipartite,
    crown,
    cycle,
    desargues,
    grid,
    heawood,
    hypercube,
    kt_minus_hamilton_cycle,
    mobius_kantor,
    path,
    petersen,
)
from sidorenko.lattice import sweep
from sidorenko.local import local_report

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "sidorenko")

POSITIVE = [cycle(3), cycle(5), cycle(7), cycle(9),
            __import__("sidorenko.graph", fromlist=["Graph"]).Graph.make(
                "K_4", 4, [(i, j) for i in range(4) for j in range(i + 1, 4)]),
            petersen()]

NEGATIVE = [path(3), path(4), path(6), cycle(4), cycle(6), cycle(8), cycle(10),
            complete_bipartite(2, 2), complete_bipartite(2, 3),
            complete_bipartite(3, 3), complete_bipartite(3, 4),
            complete_bipartite(4, 4), hypercube(3), hypercube(4)]

LOCAL = [cycle(4), cycle(6), cycle(8), complete_bipartite(3, 3), hypercube(3),
         hypercube(4), grid(3, 3), grid(3, 4), kt_minus_hamilton_cycle(5),
         crown(5), heawood(), mobius_kantor(), desargues()]

# Exhaustive tiers only: a control has to be a complete statement.
CONTROL_TIERS = ((2, 8, True, 0), (3, 4, True, 0), (4, 3, True, 0), (5, 1, True, 0))


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    report = {"positive_controls": [], "negative_controls": [], "local": [],
              "float_tier_cross_checks": []}
    ok = True

    print("== positive controls: non-bipartite H, Sidorenko provably fails ==")
    for g in POSITIVE:
        # The bipartite blowup kernel has t(H, W) = 0 for every H with an odd
        # cycle, while t(K_2, W) = 1/2 > 0.
        cert = certify(g, [1, 1], [[0, 1], [1, 0]])
        lat = sweep(g, tiers=CONTROL_TIERS, tier_seconds=120.0)
        passed = cert["is_counterexample"] and not lat["clean"]
        ok &= passed
        print(f"  {g.name:12s} exact deficit {cert['deficit'][0]}/{cert['deficit'][1]}"
              f"  certified_violation={cert['is_counterexample']}"
              f"  lattice_flagged={not lat['clean']}  -> {'PASS' if passed else 'FAIL'}")
        report["positive_controls"].append(
            {"graph": g.name, "bipartite": g.bipartition() is not None,
             "certificate": cert, "lattice_flagged": not lat["clean"], "passed": bool(passed)}
        )

    print("== negative controls: proven-positive bipartite H, must stay clean ==")
    for g in NEGATIVE:
        lat = sweep(g, tiers=CONTROL_TIERS, tier_seconds=120.0)
        exh = sum(1 for t in lat["tiers"] if t.get("exhaustive"))
        passed = lat["clean"] and exh == len(CONTROL_TIERS)
        ok &= passed
        print(f"  {g.name:12s} {g.known_positive_reason()[:44]:46s}"
              f" kernels={lat['kernels_decided']:>6d} exhaustive_tiers={exh}"
              f" -> {'PASS' if passed else 'FAIL'}")
        report["negative_controls"].append(
            {"graph": g.name, "reason": g.known_positive_reason(),
             "kernels_decided": lat["kernels_decided"], "exhaustive_tiers": exh,
             "clean": lat["clean"], "passed": bool(passed)}
        )

    print("== local structure: deficit vanishes to order girth(H) at the constant kernel ==")
    for g in LOCAL:
        r = local_report(g, k=6, seed=3)
        passed = r["matches_prediction"] is True
        ok &= passed
        print(f"  {g.name:16s} girth={r['girth']} #shortest_cycles={r['shortest_cycle_count']:>4d}"
              f" measured_order={r['measured_vanishing_order']:.3f}"
              f" -> {'PASS' if passed else 'FAIL'}")
        report["local"].append(r)

    print("== float/exact cross-check on integer kernels ==")
    rng = np.random.default_rng(4)
    worst = 0.0
    from sidorenko.certify import exact_hom_sum

    for g in LOCAL:
        k = 4
        M = rng.integers(0, 6, size=(k, k))
        M = np.triu(M) + np.triu(M, 1).T
        T = exact_hom_sum(g.n, g.edges, [1] * k, [[int(x) for x in r] for r in M])
        exact = T / float(k) ** g.n
        flt = D.hom_density(g, np.full(k, 1.0 / k), M.astype(float))
        rel = abs(exact - flt) / max(abs(exact), 1e-300)
        worst = max(worst, rel)
        report["float_tier_cross_checks"].append(
            {"graph": g.name, "exact": exact, "float": flt, "rel_err": rel})
    print(f"  worst relative disagreement over {len(LOCAL)} graphs: {worst:.2e}")
    ok &= worst < 1e-10

    report["all_controls_passed"] = bool(ok)
    report["worst_float_exact_rel_err"] = worst
    report["seconds"] = time.time() - t0
    with open(os.path.join(OUT, "controls.json"), "w") as fh:
        json.dump(report, fh, indent=1, default=str)
    print(f"\n{'ALL CONTROLS PASSED' if ok else 'CONTROLS FAILED'}"
          f"  ({time.time() - t0:.0f}s)  -> results/sidorenko/controls.json")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
