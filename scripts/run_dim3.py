#!/usr/bin/env python3
"""Dim-3 checkpoint: confirm the pipeline finds the cube/octahedron floor at 32/3.

Start families:
  * gauss     random Gaussian generators, m in {3,4,5,6,8,10}   (NM + CMA subset)
  * hanner    cube3/oct3 + large noise (Kim 2014: every Hanner polytope is a
              local min, so small-noise starts have no discovery value)
  * homotopy  descent from the ridge maximum on the cube<->octahedron path

Persistence: one JSONL line per completed start (immediately flushed);
re-running skips completed seeds.  Wall-clock capped; the cap truncates
coverage, it never loses completed work.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import json
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from mahler.evaluate import EvalStats, make_objective, CONJECTURED_MIN
from mahler.hanner import HANNER
from mahler.optimize import run_start

D = 3
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "dim3")


def build_tasks():
    tasks = []
    sid = 0

    def add(**kw):
        nonlocal sid
        kw["seed"] = 1_000_000 + sid
        sid += 1
        tasks.append(kw)

    for m in (3, 4, 5, 6, 8, 10):
        cap = 15_000 if m <= 6 else 30_000
        for _ in range(30):
            add(family="gauss", method="nm", m=m, d=D, max_evals=cap)
    for m in (4, 6, 8, 10):
        cap = 15_000 if m <= 6 else 30_000
        for _ in range(10):
            add(family="gauss", method="cma", m=m, d=D, max_evals=cap)
    for base in ("cube3", "oct3"):
        m = HANNER[D][base]().shape[0]
        for noise in (0.3, 0.6, 1.0, 1.5):
            for _ in range(10):
                add(family="hanner", method="nm", base=base, noise=noise,
                    m=m, d=D, max_evals=15_000)
    return tasks


def homotopy_probe():
    """Scan P along the cube->octahedron generator path; descend from the peak."""
    cube = HANNER[D]["cube3"]()
    octa = np.vstack([np.eye(3), [[0.25, 0.25, 0.25]]])  # oct + 1 interior gen
    stats = EvalStats()
    f = make_objective(4, D, stats)
    path = []
    for s in np.linspace(0.0, 1.0, 41):
        V = (1 - s) * octa + s * cube
        path.append((float(s), float(np.exp(f(V.ravel())))))
    s_peak, p_peak = max(path, key=lambda t: t[1])
    descents = []
    for k in range(3):
        V0 = ((1 - s_peak) * octa + s_peak * cube)
        task = dict(seed=9_000_000 + k, family="explicit", method="nm",
                    m=4, d=D, max_evals=15_000,
                    V0=(V0 + 0.01 * np.random.default_rng(k).standard_normal(V0.shape)).tolist())
        descents.append(run_start(task))
    return {"path": path, "s_peak": s_peak, "P_peak": p_peak,
            "descents": [{k: r[k] for k in ("P_final", "hanner", "converged", "nfev")}
                         for r in descents]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-seconds", type=float, default=1080.0,
                    help="wall-clock cap for the multi-start phase (default 18 min)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    jsonl = os.path.join(RESULTS, "starts.jsonl")
    done = set()
    if os.path.exists(jsonl):
        with open(jsonl) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["seed"])
                except Exception:
                    pass

    tasks = [t for t in build_tasks() if t["seed"] not in done]
    print(f"dim-3 checkpoint: {len(tasks)} starts to run "
          f"({len(done)} already done), cap {args.cap_seconds:.0f}s, "
          f"{args.workers} workers", flush=True)

    t0 = time.perf_counter()
    n_done = 0
    run_cpu = 0.0
    capped = False
    with open(jsonl, "a") as out:
        with Pool(processes=args.workers, maxtasksperchild=50) as pool:
            wave = 32
            for i in range(0, len(tasks), wave):
                if capped or time.perf_counter() - t0 > args.cap_seconds:
                    capped = True
                    break
                for rec in pool.imap_unordered(run_start, tasks[i:i + wave]):
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    n_done += 1
                    run_cpu += rec["wall_s"]
                    if time.perf_counter() - t0 > args.cap_seconds:
                        capped = True
                        break  # in-flight tasks are dropped; resume covers them
                elapsed = time.perf_counter() - t0
                print(f"  wave: {n_done}/{len(tasks)} starts done, "
                      f"{elapsed:.0f}s elapsed", flush=True)
    search_wall = time.perf_counter() - t0

    print("running homotopy probe...", flush=True)
    hres = homotopy_probe()

    # ---- summary -----------------------------------------------------------
    recs = []
    with open(jsonl) as fh:
        for line in fh:
            recs.append(json.loads(line))
    ok = [r for r in recs if r.get("P_final") is not None]
    summary = {
        "n_starts_planned": len(build_tasks()),
        "n_starts_completed": len(recs),
        "n_errors": len(recs) - len(ok),
        "capped": capped,
        "search_wall_s": round(search_wall, 1),
        "cpu_s_this_run": round(run_cpu, 1),
        "cpu_s_sum_alltime": round(sum(r["wall_s"] for r in recs), 1),
        "total_evals": sum(r["nfev"] for r in recs),
        "min_P_final": min((r["P_final"] for r in ok), default=None),
        "min_P_seen": min((r["min_P_seen"] for r in ok if r["min_P_seen"]),
                          default=None),
        "n_triggers_total": sum(r["n_trigger"] for r in recs),
        "n_faults_total": sum(r["n_fault"] for r in recs),
        "conjectured_min": CONJECTURED_MIN,
        "homotopy": hres,
    }
    summary["speedup_this_run"] = (
        round(run_cpu / search_wall, 2) if n_done and search_wall > 0 else None)

    def agg(pred, recs):
        sel = [r for r in recs if pred(r) and r.get("P_final") is not None]
        if not sel:
            return None
        nf = sorted(r["nfev"] for r in sel)
        return {
            "n": len(sel),
            "converged_frac": round(np.mean([r["converged"] for r in sel]), 3),
            "min_P": round(min(r["P_final"] for r in sel), 9),
            "median_P": round(float(np.median([r["P_final"] for r in sel])), 9),
            "nfev_q50": nf[len(nf) // 2],
            "nfev_q90": nf[int(len(nf) * 0.9)],
            "hanner_hist": {h: sum(1 for r in sel if r["hanner"] == h)
                            for h in ("cube3", "oct3", None)},
        }

    summary["by_family"] = {}
    for fam in ("gauss", "hanner"):
        for meth in ("nm", "cma"):
            key = f"{fam}/{meth}"
            a = agg(lambda r, fam=fam, meth=meth:
                    r["family"] == fam and r["method"] == meth, recs)
            if a:
                summary["by_family"][key] = a
    summary["by_m"] = {}
    for m in (3, 4, 5, 6, 8, 10):
        a = agg(lambda r, m=m: r["m"] == m, recs)
        if a:
            summary["by_m"][str(m)] = a

    with open(os.path.join(RESULTS, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "homotopy"},
                     indent=1), flush=True)
    print(f"homotopy: ridge peak P = {hres['P_peak']:.6f} at s = {hres['s_peak']:.3f}; "
          f"descents -> {[d1['hanner'] for d1 in hres['descents']]}")

    # top-5 lowest terminal bodies, for exact spot-certification
    best = sorted(ok, key=lambda r: r["P_final"])[:5]
    with open(os.path.join(RESULTS, "best_bodies.json"), "w") as fh:
        json.dump(best, fh, indent=1)
    print("wrote summary.json and best_bodies.json")


if __name__ == "__main__":
    main()
