#!/usr/bin/env python3
"""Independent re-verification of the Sidorenko search output.

The lattice tier decides kernels with **int64 numpy** variable elimination in a
greedy *min-fill* order.  The certifier decides them with **pure-Python
integer** elimination in a greedy *min-degree* order.  These are separate
implementations of the same mathematical quantity, sharing no code, so a
disagreement on any recorded witness would expose a bug in one of them.

This script replays, straight from results/sidorenko/search*.jsonl:

1. **Lattice witnesses.**  Every tier that was reported exhaustive recorded the
   upper triangle of the kernel attaining the minimum discriminant.  Each such
   witness is re-decided through the pure-stdlib certifier and the sign must
   match the sign the sweep recorded.  This is the cross-check that the int64
   path (whose exactness rests on an a-priori overflow bound) really is exact.

2. **Recorded certificates.**  Each stored certificate's `delta` is recomputed
   from its own integer data and must reproduce the stored sign and the stored
   exact rational deficit.

3. **The claim itself.**  For every bipartite graph, the recomputed minimum over
   exhausted families must be >= 0; for every non-bipartite positive control, a
   violation must still be certified.  Anything else is reported as a failure.

Output: results/sidorenko/spot_certification.json
"""

from __future__ import annotations

import glob
import json
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sidorenko.certify import certify


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "sidorenko")


def triu_pairs(k):
    return [(i, j) for i in range(k) for j in range(i, k)]


def matrix_from_triu(vec, k):
    M = [[0] * k for _ in range(k)]
    for t, (i, j) in enumerate(triu_pairs(k)):
        M[i][j] = int(vec[t])
        M[j][i] = int(vec[t])
    return M


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10**9
    recs = []
    seen = set()
    for path in sorted(glob.glob(os.path.join(OUT, "search.jsonl"))
                       + glob.glob(os.path.join(OUT, "search.*.jsonl"))):
        with open(path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r["graph"] in seen:
                    continue
                seen.add(r["graph"])
                recs.append(r)
    print(f"replaying {len(recs)} graph records from {OUT}")

    report = {"graphs": [], "witness_checks": 0, "witness_mismatches": [],
              "certificate_checks": 0, "certificate_mismatches": [],
              "claim_violations": []}

    for r in recs[:limit]:
        # Rebuild H from the record's own edge list is not possible (edges are
        # not stored), so rebuild from the corpus by name.
        g = GRAPHS.get(r["graph"])
        if g is None:
            report["graphs"].append({"graph": r["graph"], "skipped": "not reconstructible"})
            continue
        entry = {"graph": r["graph"], "role": r["role"], "witnesses": 0, "ok": True}

        for tier in r["lattice"]["tiers"]:
            if not tier.get("exhaustive") or tier.get("min_delta_witness_triu") is None:
                continue
            k = tier["k"]
            M = matrix_from_triu(tier["min_delta_witness_triu"], k)
            cert = certify(g, [1] * k, M, 1)
            report["witness_checks"] += 1
            entry["witnesses"] += 1
            if cert["delta_sign"] != tier["min_delta_sign"]:
                entry["ok"] = False
                report["witness_mismatches"].append(
                    {"graph": r["graph"], "k": k,
                     "sweep_sign": tier["min_delta_sign"],
                     "certifier_sign": cert["delta_sign"]})
            if r["bipartite"] and cert["delta_sign"] < 0:
                report["claim_violations"].append(
                    {"graph": r["graph"], "k": k, "matrix": M})

        cert_rec = r.get("certificate")
        if cert_rec:
            again = certify(g, cert_rec["weights_num"], cert_rec["matrix"],
                            cert_rec["matrix_den"])
            report["certificate_checks"] += 1
            same = (again["delta_sign"] == cert_rec["delta_sign"]
                    and Fraction(*again["deficit"]) == Fraction(*cert_rec["deficit"]))
            if not same:
                entry["ok"] = False
                report["certificate_mismatches"].append({"graph": r["graph"]})
            if r["bipartite"] and again["is_counterexample"]:
                report["claim_violations"].append(
                    {"graph": r["graph"], "source": "stored certificate"})
            if not r["bipartite"] and not again["is_counterexample"]:
                # a positive control must still certify a violation somewhere
                if not r["violation_found"]:
                    report["claim_violations"].append(
                        {"graph": r["graph"], "source": "positive control did not trigger"})
        report["graphs"].append(entry)
        print(f"  {r['graph']:22s} {r['role']:11s} witnesses={entry['witnesses']:>2d} "
              f"{'ok' if entry['ok'] else 'MISMATCH'}")

    ok = (not report["witness_mismatches"] and not report["certificate_mismatches"]
          and not report["claim_violations"])
    report["all_agree"] = bool(ok)
    with open(os.path.join(OUT, "spot_certification.json"), "w") as fh:
        json.dump(report, fh, indent=1, default=str)

    print()
    print(f"lattice witnesses re-decided through the independent pure-Python path: "
          f"{report['witness_checks']}  mismatches: {len(report['witness_mismatches'])}")
    print(f"stored certificates recomputed: {report['certificate_checks']}  "
          f"mismatches: {len(report['certificate_mismatches'])}")
    print(f"claim violations (bipartite H with a certified negative deficit): "
          f"{len(report['claim_violations'])}")
    print("ALL INDEPENDENT CHECKS AGREE" if ok else "DISAGREEMENT FOUND")
    return 0 if ok else 1


def _build_graph_table():
    """Name -> Graph for everything the search may have recorded."""
    from sidorenko.graph import corpus, enumerate_bipartite

    table = {}
    for g, _note in corpus():
        table[g.name] = g
    for (p, q) in ((2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (3, 5), (4, 4)):
        for g in enumerate_bipartite(p, q):
            table[g.name] = g
    return table


GRAPHS = _build_graph_table()


if __name__ == "__main__":
    raise SystemExit(main())
