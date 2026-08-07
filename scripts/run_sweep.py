#!/usr/bin/env python3
"""Prong 1: exhaustive minimal-counterexample sweep.

A lexicographically minimal (order, then size) counterexample to
Hadwiger t=7 is 7-contraction-critical, hence (Mader) 7-connected, so
delta >= 7; K7-minor-freeness gives e <= 5n-15 (Mader), so n >= 10 and
e(complement) is small: Delta(comp) <= n-8, with
    e(comp) in [C(n,2)-(5n-15), C(n,2)-ceil(7n/2)].
We enumerate the complements with geng (searching a SUPERSET of the
required space: 7-connectivity itself is not imposed, only delta>=7 via
the complement degree bound), take complements, and require a verified
6-coloring to discard. Survivors get exact treatment. Additional sound
pruning: a counterexample has no dominating vertex (else deleting it
yields a t=6 counterexample, contradicting Robertson-Seymour-Thomas).

Usage: run_sweep.py --n 12 [--res R --mod M] [--budget-hours H]
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hadwiger.coloring import dsatur, sat_kcolorable
from hadwiger.graphs import (complement, edge_count, has_dominating_vertex,
                             is_connected, parse_graph6, to_graph6)
from hadwiger.minor import exact_minor_decision
from hadwiger.runner_util import (done_shards, jsonl_append, record_candidate,
                                  results_path, stop_requested)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--res", type=int, default=0)
    ap.add_argument("--mod", type=int, default=1)
    ap.add_argument("--budget-hours", type=float, default=None)
    args = ap.parse_args()
    n = args.n
    assert n >= 10

    tot = n * (n - 1) // 2
    e_lo = (7 * n + 1) // 2  # ceil(7n/2), from delta >= 7
    e_hi = 5 * n - 15        # Mader
    if e_lo > e_hi:
        print(f"n={n}: empty class (e_lo {e_lo} > e_hi {e_hi})")
        return
    ce_lo, ce_hi = tot - e_hi, tot - e_lo
    cmax_deg = n - 8  # delta(G) >= 7

    manifest = results_path("sweep", f"n{n}_manifest.jsonl")
    shard = (n, args.res, args.mod)
    if tuple(shard) in done_shards(manifest):
        print(f"shard {shard} already done")
        return

    rng = random.Random(7_000_003 * args.res + n)
    # note: nauty-geng requires fused flag values ("-D4", not "-D 4")
    cmd = ["nauty-geng", "-q", f"-D{cmax_deg}", str(n), f"{ce_lo}:{ce_hi}"]
    if args.mod > 1:
        cmd.append(f"{args.res}/{args.mod}")
    print("generator:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1 << 20)

    t0 = time.time()
    stats = {"total": 0, "disconnected": 0, "dominating": 0, "colored6": 0,
             "sat_colored6": 0, "chi_ge_7": 0}
    deadline = t0 + args.budget_hours * 3600 if args.budget_hours else None
    survivors_path = results_path("sweep", f"n{n}_chi7_survivors.jsonl")

    for line in proc.stdout:
        stats["total"] += 1
        if stats["total"] % 5000 == 0:
            if stop_requested() or (deadline and time.time() > deadline):
                why = "stop" if stop_requested() else "budget"
                proc.terminate()
                jsonl_append(manifest, {"shard": shard, "status": why,
                                        "stats": stats,
                                        "wall_s": round(time.time() - t0, 1)})
                print(f"exiting shard incomplete ({why})", flush=True)
                return
        cn, cadj = parse_graph6(line)
        adj = complement(cn, cadj)
        if not is_connected(cn, adj):
            stats["disconnected"] += 1
            continue
        if has_dominating_vertex(cn, adj):
            stats["dominating"] += 1
            continue
        if dsatur(cn, adj, 6, rng=rng, tries=3) is not None:
            stats["colored6"] += 1  # verified coloring inside dsatur
            continue
        status, coloring, meta = sat_kcolorable(cn, adj, 6)
        if status is True:
            stats["sat_colored6"] += 1
            continue
        # chi >= 7 — extraordinarily rare; log it whatever happens next
        stats["chi_ge_7"] += 1
        g6 = to_graph6(cn, adj)
        verdict, model, detail = exact_minor_decision(cn, adj, rng,
                                                      require_agreement=True)
        jsonl_append(survivors_path,
                     {"g6": g6, "e": edge_count(cn, adj),
                      "chi_proof": meta.get("proof"), "minor": verdict,
                      "detail": detail})
        if verdict == "no":
            record_candidate("sweep", cn, g6,
                             {"chi_reason": "SAT UNSAT on 6-colorability",
                              "minor_detail": detail})
            proc.terminate()
            jsonl_append(manifest, {"shard": shard, "status": "candidate",
                                    "stats": stats, "g6": g6})
            print(f"CANDIDATE FOUND: {g6}", flush=True)
            return
        if verdict not in ("yes",):
            record_candidate("sweep-unresolved", cn, g6, {"detail": detail})
            proc.terminate()
            jsonl_append(manifest, {"shard": shard, "status": "unresolved",
                                    "stats": stats, "g6": g6})
            return

    rc = proc.wait()
    wall = round(time.time() - t0, 1)
    status = "done" if rc == 0 else "generator-error"
    jsonl_append(manifest, {"shard": shard, "status": status, "rc": rc,
                            "stats": stats, "wall_s": wall,
                            "rate_per_s": round(stats["total"] / max(wall, 1e-9))})
    print(f"shard {shard} {status}: {json.dumps(stats)} in {wall}s", flush=True)
    if rc != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
