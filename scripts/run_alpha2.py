#!/usr/bin/env python3
"""Prong 2: exhaustive alpha<=2 sweep (Seymour's key case).

Every graph G on n>=13 vertices with independence number <=2 has
chi(G) >= ceil(n/2) >= 7, so a K7-minor-free one is immediately a
Hadwiger t=7 counterexample. alpha(G)<=2 <=> complement(G) triangle-free,
and Mader's bound (K7-minor-free => e(G) <= 5n-15) forces
e(complement) >= C(n,2)-(5n-15), so only near-Turan-extremal
triangle-free complements can matter. Generation:

    nauty-geng -q -t n <minE>:<maxE> <res>/<mod>

then complement each graph and hunt K7 minors: witnessed fast path
first, exact SAT for the resistant. Sharded, resumable via manifest.

Usage: run_alpha2.py --n 13 [--res 0 --mod 4] [--budget-hours H]
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hadwiger.graphs import complement, edge_count, parse_graph6, to_graph6
from hadwiger.minor import (exact_minor_decision, fast_minor_search,
                            greedy_contraction_minor, minorminer_find)
from hadwiger.runner_util import (done_shards, jsonl_append, record_candidate,
                                  results_path, stop_requested)


def alpha_at_most_2(n, cadj) -> bool:
    """cadj is the complement (must be triangle-free for alpha(G)<=2)."""
    for u in range(n):
        nb = cadj[u]
        while nb:
            wb = nb & -nb
            nb ^= wb
            v = wb.bit_length() - 1
            if v > u and cadj[u] & cadj[v]:
                return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--res", type=int, default=0)
    ap.add_argument("--mod", type=int, default=1)
    ap.add_argument("--budget-hours", type=float, default=None)
    ap.add_argument("--emin", type=int, default=None,
                    help="override complement min edges (default: Mader bound)")
    args = ap.parse_args()
    n = args.n
    assert n >= 13, "alpha<=2 forces chi>=7 only from n=13 up"

    mader_max_g = 5 * n - 15
    emin = args.emin if args.emin is not None else n * (n - 1) // 2 - mader_max_g
    emax = (n * n) // 4  # Turan: max edges of a triangle-free graph
    manifest = results_path("alpha2", f"n{n}_manifest.jsonl")
    shard = (n, args.res, args.mod, emin, emax)
    if tuple(shard) in done_shards(manifest):
        print(f"shard {shard} already done; nothing to do")
        return

    rng = random.Random(1_000_003 * args.res + n)
    cmd = ["nauty-geng", "-q", "-t", str(n), f"{emin}:{emax}"]
    if args.mod > 1:
        cmd.append(f"{args.res}/{args.mod}")
    print("generator:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1 << 20)

    t0 = time.time()
    stats = {"total": 0, "fast_clique": 0, "fast_greedy": 0, "fast_mm": 0,
             "exact_yes": 0, "sanity_alpha_fail": 0}
    deadline = t0 + args.budget_hours * 3600 if args.budget_hours else None
    hard_path = results_path("alpha2", f"n{n}_hard_cases.jsonl")

    for line in proc.stdout:
        stats["total"] += 1
        if stats["total"] % 2000 == 0:
            if stop_requested():
                print("STOP flag seen; exiting shard incomplete", flush=True)
                proc.terminate()
                jsonl_append(manifest, {"shard": shard, "status": "interrupted",
                                        "stats": stats,
                                        "wall_s": round(time.time() - t0, 1)})
                return
            if deadline and time.time() > deadline:
                print("budget exhausted; exiting shard incomplete", flush=True)
                proc.terminate()
                jsonl_append(manifest, {"shard": shard, "status": "budget",
                                        "stats": stats,
                                        "wall_s": round(time.time() - t0, 1)})
                return
        cn, cadj = parse_graph6(line)
        # sanity: geng -t output must be triangle-free (checked cheaply on a
        # deterministic 1/512 sample; a failure means broken generation)
        if stats["total"] % 512 == 1 and not alpha_at_most_2(cn, cadj):
            stats["sanity_alpha_fail"] += 1
        adj = complement(cn, cadj)
        sets = fast_minor_search(cn, adj, rng, greedy_tries=4)
        if sets is not None:
            stats["fast_clique" if all(bin(s).count("1") == 1 for s in sets)
                  else "fast_greedy"] += 1
            continue
        # escalation: only the *additional* work (more greedy restarts,
        # then minorminer) — the exact clique search already ran above
        sets = greedy_contraction_minor(cn, adj, rng, tries=20)
        if sets is None:
            sets = minorminer_find(cn, adj, tries=20)
        if sets is not None:
            stats["fast_mm"] += 1
            continue
        # resistant graph: exact decision, logged regardless of outcome
        g6 = to_graph6(cn, adj)
        verdict, model, detail = exact_minor_decision(cn, adj, rng,
                                                      require_agreement=True)
        jsonl_append(hard_path, {"g6": g6, "e": edge_count(cn, adj),
                                 "verdict": verdict, "detail": detail})
        if verdict == "yes":
            stats["exact_yes"] += 1
        elif verdict == "no":
            # chi >= 7 by alpha<=2 counting; K7-minor-free confirmed by two
            # independent exact deciders => COUNTEREXAMPLE CANDIDATE
            record_candidate("alpha2", cn, g6,
                             {"chi_reason": f"alpha<=2, n={cn} => chi>=7",
                              "minor_detail": detail})
            proc.terminate()
            jsonl_append(manifest, {"shard": shard, "status": "candidate",
                                    "stats": stats, "g6": g6})
            print(f"CANDIDATE FOUND: {g6}", flush=True)
            return
        else:
            record_candidate("alpha2-unresolved", cn, g6, {"detail": detail})
            proc.terminate()
            jsonl_append(manifest, {"shard": shard, "status": "unresolved",
                                    "stats": stats, "g6": g6})
            return

    rc = proc.wait()
    wall = round(time.time() - t0, 1)
    if rc != 0:
        jsonl_append(manifest, {"shard": shard, "status": "generator-error",
                                "rc": rc, "stats": stats, "wall_s": wall})
        sys.exit(1)
    if stats["sanity_alpha_fail"]:
        jsonl_append(manifest, {"shard": shard, "status": "sanity-fail",
                                "stats": stats, "wall_s": wall})
        sys.exit(1)
    jsonl_append(manifest, {"shard": shard, "status": "done", "stats": stats,
                            "wall_s": wall,
                            "rate_per_s": round(stats["total"] / max(wall, 1e-9))})
    print(f"shard {shard} done: {json.dumps(stats)} in {wall}s", flush=True)


if __name__ == "__main__":
    main()
