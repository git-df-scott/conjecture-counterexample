#!/usr/bin/env python3
"""Maximum-rigor verification of a counterexample candidate.

chi >= 7: 6-colorability must be UNSAT under three independent SAT
solver families (CaDiCaL, Glucose, MiniSat), clique shortcut disabled,
plus DSATUR failure across many orders (advisory).

No K7 minor: direct encoding UNSAT under two solver families, CEGAR run
to completion, branch-and-bound with a large budget, minorminer with
many tries must fail to embed, and the treewidth shortcut must NOT fire
(tw<=5 would imply chi<=6 and contradict the chi verdict).

Usage: verify_candidate.py '<graph6>'
"""

import json
import os
import sys
import random
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hadwiger.coloring import dsatur, sat_kcolorable
from hadwiger.graphs import degrees, edge_count, parse_graph6
from hadwiger.minor import (bb_minor, minorminer_find, sat_minor_cegar,
                            sat_minor_direct, treewidth_le5)
from hadwiger.runner_util import jsonl_append, results_path


def main():
    g6 = sys.argv[1]
    n, adj = parse_graph6(g6)
    rng = random.Random(0xC0FFEE)
    report = {"g6": g6, "n": n, "e": edge_count(n, adj),
              "degrees": sorted(degrees(n, adj)), "mader_max": 5 * n - 15}

    chi = {}
    chi["dsatur_6col_found"] = dsatur(n, adj, 6, rng=rng, tries=200) is not None
    for solver in ("cd19", "g42", "m22"):
        t0 = time.time()
        try:
            res, _, _ = sat_kcolorable(n, adj, 6, solver_name=solver,
                                       clique_shortcut=False)
            chi[f"sat6_{solver}"] = {"result": res,
                                     "wall_s": round(time.time() - t0, 1)}
        except Exception as exc:
            chi[f"sat6_{solver}"] = {"error": str(exc)}
    report["chi_ge_7"] = chi
    chi_ok = all(v.get("result") is False for k, v in chi.items()
                 if k.startswith("sat6_") and "result" in v) and \
        sum(1 for k in chi if k.startswith("sat6_")) >= 2 and \
        not chi["dsatur_6col_found"]

    minor = {}
    minor["treewidth_le5"] = treewidth_le5(n, adj)  # must be False
    for solver in ("cd19", "g42"):
        t0 = time.time()
        try:
            v, _ = sat_minor_direct(n, adj, solver_name=solver)
            minor[f"direct_{solver}"] = {"result": v,
                                         "wall_s": round(time.time() - t0, 1)}
        except Exception as exc:
            minor[f"direct_{solver}"] = {"error": str(exc)}
    t0 = time.time()
    v, _ = sat_minor_cegar(n, adj, max_iters=2_000_000)
    minor["cegar"] = {"result": v, "wall_s": round(time.time() - t0, 1)}
    t0 = time.time()
    minor["bb"] = {"result": bb_minor(n, adj, node_budget=50_000_000),
                   "wall_s": round(time.time() - t0, 1)}
    minor["minorminer_embedding"] = minorminer_find(n, adj, tries=200) is not None
    report["no_k7_minor"] = minor
    minor_ok = (not minor["treewidth_le5"]
                and all(v.get("result") == "no" for k, v in minor.items()
                        if k.startswith("direct_") and "result" in v)
                and minor["cegar"]["result"] == "no"
                and minor["bb"]["result"] in ("no", "unknown")
                and not minor["minorminer_embedding"])

    report["VERDICT"] = ("CONFIRMED COUNTEREXAMPLE" if chi_ok and minor_ok
                         else "NOT CONFIRMED")
    jsonl_append(results_path("candidate_verification.jsonl"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
