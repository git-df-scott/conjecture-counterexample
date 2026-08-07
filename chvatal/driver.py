#!/usr/bin/env python3
"""Parallel, checkpointed, guard-aware driver for the reduced-form search.

Splits ./chv_enum test runs into work units (k, second-generator mask), runs
them on a process pool, writes one JSON result file per unit (so completed
units are never re-run on resume), aggregates a summary, and re-runs the
usage guard between batches.

Exit codes: 0 all clean | 17 VIOLATION found (counterexample!) | 3 guard trip
(pause requested) | 1 other error.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ENUM = os.path.join(HERE, "chv_enum")


def second_masks(n, k):
    """Valid second-generator masks after first = {1..k}; mirrors enumerate.c."""
    g0 = (1 << k) - 1
    out = []
    for t in range(g0 + 1, 1 << n):
        if bin(t).count("1") < k:
            continue
        u = t & g0
        if u and u != g0 and u != t:
            out.append(t)
    return out


def unit_priority(n, k, s):
    """Star-distant-first ordering: prefer mid-size second generators with
    small overlap with the first generator (far from star/principal shape)."""
    overlap = bin(s & ((1 << k) - 1)).count("1")
    size = bin(s).count("1")
    return (overlap, abs(size - (n + 1) // 2), s)


def run_unit(n, k, s):
    p = subprocess.run([ENUM, "test", str(n), str(k), str(s)],
                       capture_output=True, text=True, timeout=None)
    return k, s, p.returncode, p.stdout, p.stderr


def guard_check(guard, threshold):
    if not guard:
        return 0, ""
    p = subprocess.run(["bash", guard, str(threshold)], capture_output=True, text=True)
    return p.returncode, p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--ks", default="3,4,5,6,2",
                    help="comma-separated k values, in run order (K-M-informed "
                         "default: k>=3 first for discovery odds, k=2 last)")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--results", default=None)
    ap.add_argument("--guard", default=None)
    ap.add_argument("--guard-threshold", type=int, default=80)
    ap.add_argument("--guard-every-sec", type=float, default=120.0)
    a = ap.parse_args()

    rdir = a.results or os.path.join(HERE, "results", f"n{a.n}")
    os.makedirs(rdir, exist_ok=True)

    units = []
    for k in [int(x) for x in a.ks.split(",")]:
        for s in sorted(second_masks(a.n, k), key=lambda s: unit_priority(a.n, k, s)):
            units.append((k, s))
    todo = [(k, s) for k, s in units
            if not os.path.exists(os.path.join(rdir, f"k{k}_s{s}.json"))]
    print(f"units total={len(units)} todo={len(todo)} procs={a.procs}", flush=True)

    rc, out = guard_check(a.guard, a.guard_threshold)
    if rc in (3, 4):
        print(f"GUARD_TRIP pre-start exit={rc}\n{out}", flush=True)
        return 3

    last_guard = time.time()
    done = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.procs) as ex:
        futs = {ex.submit(run_unit, a.n, k, s): (k, s) for k, s in todo}
        for fut in as_completed(futs):
            k, s, code, sout, serr = fut.result()
            if "VIOLATION" in sout:
                print(f"\n*** VIOLATION FOUND in unit k={k} s={s} ***\n{sout}", flush=True)
                with open(os.path.join(rdir, "VIOLATION.txt"), "w") as f:
                    f.write(sout)
                for other in futs:
                    other.cancel()
                return 17
            if code != 0:
                print(f"unit k={k} s={s} failed rc={code}: {serr[:300]}", flush=True)
                return 1
            line = [l for l in sout.splitlines() if l.startswith("{")][-1]
            with open(os.path.join(rdir, f"k{k}_s{s}.json"), "w") as f:
                f.write(line + "\n")
            done += 1
            if done % 50 == 0 or done == len(todo):
                el = time.time() - t0
                print(f"progress {done}/{len(todo)} units, {el:.0f}s elapsed", flush=True)
            if time.time() - last_guard > a.guard_every_sec:
                last_guard = time.time()
                rc, out = guard_check(a.guard, a.guard_threshold)
                if rc in (3, 4):
                    print(f"GUARD_TRIP mid-run exit={rc}\n{out}", flush=True)
                    for other in futs:
                        other.cancel()
                    return 3

    tot = {"n": a.n, "units": 0, "nodes": 0, "tested": 0, "ties": 0,
           "violations": 0, "best_excess": -10 ** 9, "best_example": None,
           "per_k": {}}
    for fn in os.listdir(rdir):
        if not fn.endswith(".json") or fn == "summary.json":
            continue
        d = json.load(open(os.path.join(rdir, fn)))
        tot["units"] += 1
        for f2 in ("nodes", "tested", "ties", "violations"):
            tot[f2] += d[f2]
        pk = tot["per_k"].setdefault(str(d["k"]), {"nodes": 0, "tested": 0})
        pk["nodes"] += d["nodes"]
        pk["tested"] += d["tested"]
        if d["tested"] and d["best_excess"] > tot["best_excess"]:
            tot["best_excess"] = d["best_excess"]
            tot["best_example"] = d["best_example"]
    with open(os.path.join(rdir, "summary.json"), "w") as f:
        json.dump(tot, f, indent=1)
    print(json.dumps(tot, indent=1), flush=True)
    return 0 if tot["violations"] == 0 else 17


if __name__ == "__main__":
    sys.exit(main())
