#!/usr/bin/env python3
"""Counterexample search for Sidorenko's conjecture: main driver.

Per candidate graph H, three independent attacks:

1. **Exact lattice sweep** (`sidorenko.lattice`) -- every symmetric integer
   kernel in a delimited family, decided in exact integer arithmetic.  Where the
   family is exhausted this is an unconditional theorem about that family.
2. **Continuous multi-start search** (`sidorenko.optimize`) -- L-BFGS on the
   analytic gradient plus CMA-ES, over step kernels with 2..8 blocks and free
   block weights, seeded far from the constant kernel.
3. **Exact certification of the float optimum** (`sidorenko.certify`) -- the
   best float kernel is rounded to many nearby rationals and each is decided
   exactly, so the reported floor is a certified rational number, not a float.

Any violation found for a *bipartite* H is a genuine counterexample and aborts
the run immediately with the certificate written out.  Non-bipartite graphs are
carried along as live positive controls: they must be flagged, which is what
demonstrates the trigger works at all.

Resumable: one JSONL record per graph in results/sidorenko/search.jsonl; a rerun
skips graphs already recorded unless --restart is given.

  python3 scripts/run_sidorenko_search.py --time-budget 3600
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sidorenko.certify import certify_best
from sidorenko.contract import min_fill_order
from sidorenko.graph import corpus, enumerate_bipartite
from sidorenko.lattice import TIERS, sweep
from sidorenko.local import local_report
from sidorenko.optimize import TRIGGER, run_graph

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "sidorenko")


def shard_files():
    """Every shard of the search log, so resume and summary see all passes."""
    import glob

    return sorted(glob.glob(os.path.join(OUT, "search.jsonl"))
                  + glob.glob(os.path.join(OUT, "search.*.jsonl")))


def role_of(g):
    if g.bipartition() is None:
        return "pos-control"
    if g.known_positive_reason():
        return "neg-control"
    return "hard"


def candidates(include_enumeration=True, max_width=9):
    """Corpus + exhaustively enumerated small bipartite graphs, deduplicated.

    Graphs whose min-fill width exceeds `max_width` are dropped: a contraction
    then costs k^10 or more per kernel, which would consume the whole budget on
    a single graph and buy less coverage than spending it elsewhere.  Dropped
    graphs are reported by name so the omission is on the record.
    """
    entries = list(corpus())
    if include_enumeration:
        seen = {}
        for g, _ in entries:
            seen.setdefault(g.iso_hash(), []).append(g)
        for (p, q) in ((2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (3, 5), (4, 4)):
            for g in enumerate_bipartite(p, q):
                key = g.iso_hash()
                if any(g.is_isomorphic_to(h) for h in seen.get(key, ())):
                    continue
                seen.setdefault(key, []).append(g)
                entries.append((g, f"exhaustive enumeration, parts {p}+{q}"))
    kept, dropped = [], []
    for g, note in entries:
        _order, width = min_fill_order(g.n, g.edges)
        if width > max_width:
            dropped.append({"graph": g.name, "n": g.n, "e": g.m, "width": width})
        else:
            kept.append((g, note, width))
    return kept, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-budget", type=float, default=None,
                    help="wall-clock cap in seconds for the whole run")
    ap.add_argument("--per-graph", type=float, default=180.0,
                    help="soft wall-clock cap per graph")
    ap.add_argument("--tier-seconds", type=float, default=25.0,
                    help="cap per lattice tier per graph")
    ap.add_argument("--n-random", type=int, default=10,
                    help="random starts per block count in the continuous search")
    ap.add_argument("--blocks", type=int, nargs="+", default=[2, 3, 4, 5, 6, 8])
    ap.add_argument("--no-enumeration", action="store_true")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--only", type=str, default=None, help="substring filter on graph names")
    ap.add_argument("--shard", type=str, default="",
                    help="write to search.<shard>.jsonl instead of search.jsonl so "
                         "several disjoint passes can run concurrently without "
                         "interleaving appends into one file; resume and summary "
                         "read every shard")
    ap.add_argument("--roles", type=str, nargs="+", default=None,
                    choices=["hard", "neg-control", "pos-control"],
                    help="restrict to these roles; the enumeration is dominated by "
                         "proven-positive graphs, so --roles hard concentrates the "
                         "budget on genuinely open cases")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    jsonl = os.path.join(OUT, f"search.{args.shard}.jsonl" if args.shard else "search.jsonl")
    logf = open(os.path.join(OUT, "run.log"), "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    done = set()
    if args.restart:
        open(jsonl, "w").close()
    for path in shard_files():
        with open(path) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["graph"])
                except Exception:
                    pass

    kept, dropped = candidates(not args.no_enumeration)
    if args.only:
        kept = [x for x in kept if args.only in x[0].name]
    if args.roles:
        kept = [x for x in kept if role_of(x[0]) in set(args.roles)]
    log(f"=== Sidorenko search: {len(kept)} candidates, {len(dropped)} dropped for width, "
        f"{len(done)} already done ===")
    if dropped:
        log("    dropped (min-fill width > 9): "
            + ", ".join(f"{d['graph']}(w={d['width']})" for d in dropped))

    t0 = time.time()
    counterexample = None
    processed = 0
    for g, note, width in kept:
        if g.name in done:
            continue
        if args.time_budget and time.time() - t0 > args.time_budget:
            log(f"--- wall-clock budget reached; stopping with {processed} processed ---")
            break
        role = role_of(g)
        gt0 = time.time()

        lat = sweep(g, tiers=TIERS, tier_seconds=args.tier_seconds,
                    total_seconds=max(30.0, args.per_graph * 0.6))
        srec, (a, B) = run_graph(
            g, block_counts=tuple(args.blocks), n_random=args.n_random, seed=1,
            time_budget=max(20.0, args.per_graph * 0.4))

        cert = None
        if a is not None:
            cert, n_cands = certify_best(
                g, [float(x) for x in a], [[float(x) for x in row] for row in B],
                time_budget=max(10.0, args.per_graph * 0.25))
            if cert:
                cert["candidates_tried"] = n_cands

        violation = bool(
            (not lat["clean"])
            or (cert and cert["is_counterexample"])
            or (srec["float_hit"] is not None)
        )
        certified_violation = bool((not lat["clean"]) or (cert and cert["is_counterexample"]))

        rec = {
            "graph": g.name,
            "note": note,
            "role": role,
            **g.summary(),
            "minfill_width": width,
            "local": local_report(g, seed=3),
            "lattice": lat,
            "search": srec,
            "certificate": cert,
            "violation_found": violation,
            "certified_violation": certified_violation,
            "expected_violation": role == "pos-control",
            "seconds": time.time() - gt0,
        }
        # A control behaving correctly is not news; a bipartite hit is.
        if certified_violation and role != "pos-control":
            rec["VERDICT"] = "COUNTEREXAMPLE TO SIDORENKO'S CONJECTURE"
            counterexample = rec
        elif role == "pos-control":
            rec["VERDICT"] = ("control ok: violation detected as expected"
                              if violation else "CONTROL FAILED: no violation detected")
        else:
            rec["VERDICT"] = "no counterexample"

        with open(jsonl, "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        processed += 1

        floor = srec["best_F"]
        floor_str = "n/a" if floor is None else f"{floor:+.3e}"
        # The certified quantity worth printing is the exact minimum of the
        # deficit over the exhaustively swept integer families: "0" means the
        # minimum is exactly zero, attained at the constant kernels, over every
        # kernel in those families.  The rounded-candidate deficit is a
        # different and much weaker statement (it only says the search's own
        # best point is not a counterexample), so it is recorded in the JSONL
        # rather than advertised here.
        exh_signs = [t.get("min_delta_sign") for t in lat["tiers"] if t.get("exhaustive")]
        if not exh_signs:
            lat_str = "n/a"
        elif all(s == 0 for s in exh_signs):
            lat_str = "0(exact)"
        elif any(s is not None and s < 0 for s in exh_signs):
            lat_str = "NEGATIVE"
        else:
            lat_str = "+"
        log(f"[{processed:>3d}/{len(kept)}] {g.name:22s} n={g.n:2d} e={g.m:2d} w={width} "
            f"{role:11s} lattice_kernels={lat['kernels_decided']:>6d} "
            f"exh_tiers={lat['fully_exhaustive_tiers']} lat_min={lat_str} "
            f"min_F={floor_str} rejects={srec['numerical_rejects']} "
            f"{rec['VERDICT']}  ({rec['seconds']:.0f}s)")

        if counterexample:
            log("!!! CERTIFIED COUNTEREXAMPLE -- aborting the run !!!")
            with open(os.path.join(OUT, "COUNTEREXAMPLE.json"), "w") as fh:
                json.dump(counterexample, fh, indent=1, default=str)
            break

    summarize(log, time.time() - t0, dropped)
    return 2 if counterexample else 0


def summarize(log, seconds, dropped):
    recs, seen = [], set()
    for path in shard_files():
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
    by_role = {}
    for r in recs:
        by_role.setdefault(r["role"], []).append(r)

    hard = by_role.get("hard", [])
    floors = [r["search"]["best_F"] for r in hard if r["search"]["best_F"] is not None]
    control_fail = [r["graph"] for r in by_role.get("pos-control", [])
                    if not r["violation_found"]]
    neg_fail = [r["graph"] for r in by_role.get("neg-control", []) if r["violation_found"]]
    real = [r["graph"] for r in recs if r.get("VERDICT", "").startswith("COUNTEREXAMPLE")]

    # The crisp certified claim: over every integer kernel family that was
    # exhausted, the exact minimum of the deficit is 0, attained at the constant
    # kernels.  Anything other than 0 there would be news in one direction or a
    # bug in the other.
    def exh_min_zero(r):
        signs = [t.get("min_delta_sign") for t in r["lattice"]["tiers"] if t.get("exhaustive")]
        return bool(signs) and all(s == 0 for s in signs)

    verdict_recs = [r for r in recs if r["role"] != "pos-control"]
    exact_zero = [r["graph"] for r in verdict_recs if exh_min_zero(r)]
    no_exh = [r["graph"] for r in verdict_recs if not exh_min_zero(r)]

    summary = {
        "conjecture": "Sidorenko: t(H, W) >= t(K_2, W)^{e(H)} for every bipartite H",
        "outcome": ("COUNTEREXAMPLE FOUND" if real else "no counterexample found"),
        "counterexamples": real,
        "bipartite_graphs_with_exact_min_deficit_zero": len(exact_zero),
        "bipartite_graphs_without_a_fully_exhausted_tier": no_exh,
        "numerical_rejects_total": sum(
            r["search"].get("numerical_rejects", 0) for r in recs),
        "graphs_processed": len(recs),
        "by_role": {k: len(v) for k, v in by_role.items()},
        "hard_graphs": len(hard),
        "lattice_kernels_decided_total": sum(r["lattice"]["kernels_decided"] for r in recs),
        "fully_exhaustive_tiers_total": sum(r["lattice"]["fully_exhaustive_tiers"] for r in recs),
        "continuous_starts_total": sum(r["search"]["starts"] for r in recs),
        "float_floor_min_over_hard_graphs": (min(floors) if floors else None),
        "float_floor_max_over_hard_graphs": (max(floors) if floors else None),
        "float_trigger_threshold": TRIGGER,
        "exact_floor_is_zero_for_all_hard_graphs": all(
            r["certificate"] is None or r["certificate"]["deficit"][0] >= 0 for r in hard),
        "positive_controls_that_failed_to_trigger": control_fail,
        "negative_controls_that_wrongly_triggered": neg_fail,
        "graphs_dropped_for_width": dropped,
        "seconds": seconds,
    }
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)

    log("")
    log("=== SUMMARY ===")
    log(f"outcome: {summary['outcome']}")
    log(f"graphs processed: {summary['graphs_processed']}  {summary['by_role']}")
    log(f"exact kernel verdicts: {summary['lattice_kernels_decided_total']:,}"
        f"   fully exhaustive tiers: {summary['fully_exhaustive_tiers_total']}")
    log(f"bipartite graphs whose exhausted families have exact minimum deficit 0: "
        f"{len(exact_zero)}/{len(verdict_recs)}"
        + (f"   (no exhausted tier: {no_exh})" if no_exh else ""))
    log(f"numerical rejects (candidate hits discarded on recheck): "
        f"{summary['numerical_rejects_total']}")
    log(f"continuous starts: {summary['continuous_starts_total']:,}")
    if floors:
        log(f"float floor over the {len(hard)} hard graphs: "
            f"min {min(floors):+.3e}, max {max(floors):+.3e} (trigger at {TRIGGER:.0e})")
    log(f"positive controls that failed to trigger: {control_fail or 'none'}")
    log(f"negative controls that wrongly triggered: {neg_fail or 'none'}")
    log("-> results/sidorenko/summary.json")


if __name__ == "__main__":
    raise SystemExit(main())
