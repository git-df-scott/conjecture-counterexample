#!/usr/bin/env python3
"""Dim-4 Mahler counterexample search (run only after the dim-3 checkpoint).

Start families (calibrated on the dim-3 checkpoint):
  * gauss/nm    random Gaussian generators, m in {4,5,6}    (NM fine at <=24 params)
  * gauss/cma   random Gaussian generators, m in {6,8,10,12,16}
  * hanner      all FOUR dim-4 Hanner types + large noise (0.3..1.5); small
                noise has no discovery value since every Hanner polytope is a
                proven local minimizer (Kim 2014)
  * robust      CMA at m in {20,24}: sensitivity of best-found P to the m-cap
  * homotopy    ridge probes: P along generator paths between every pair of
                the four Hanner types (endpoints whitened, smaller endpoint
                padded with interior points), then descents from each peak

Trigger protocol: if any evaluation reports P < 32/3 - 1e-6 the run ABORTS
immediately (no further searching), preserving the candidate generators in its
JSONL record for exact certification.  Persistence and caps as in run_dim3.
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

from mahler.evaluate import EvalStats, make_objective, whiten, CONJECTURED_MIN
from mahler.hanner import HANNER
from mahler.optimize import run_start

D = 4
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "dim4")
HANNER_M = {"cube4": 8, "cross4": 4, "octprism4": 6, "cubebipyr4": 5}


def build_tasks(smoke=False):
    tasks = []
    sid = 0
    rep = (lambda n: 2 if smoke else n)
    cap_scale = 0.05 if smoke else 1.0

    def add(**kw):
        nonlocal sid
        kw["seed"] = 2_000_000 + sid
        kw["max_evals"] = int(kw["max_evals"] * cap_scale)
        sid += 1
        tasks.append(kw)

    for m in (4, 5, 6):
        for _ in range(rep(35)):
            add(family="gauss", method="nm", m=m, d=D, max_evals=25_000)
    for m in (6, 8, 10, 12, 16):
        cap = 40_000 if m <= 8 else 60_000
        for _ in range(rep(40)):
            add(family="gauss", method="cma", m=m, d=D, max_evals=cap)
    for base, m in HANNER_M.items():
        method = "cma" if m >= 6 else "nm"
        for noise in (0.3, 0.6, 1.0, 1.5):
            for _ in range(rep(12)):
                add(family="hanner", method=method, base=base, noise=noise,
                    m=m, d=D, max_evals=40_000)
    for m in (20, 24):
        for _ in range(rep(12)):
            add(family="robust", method="cma", m=m, d=D, max_evals=60_000)
    return tasks


def _pad_generators(V, m):
    """Pad a generator matrix to m rows with strictly interior points
    (scaled copies of existing generators) — the body is unchanged."""
    V = np.asarray(V, dtype=float)
    k = 0
    while V.shape[0] < m:
        V = np.vstack([V, 0.5 * V[k % V.shape[0]]])
        k += 1
    return V


def homotopy_probes(smoke=False):
    """P along generator paths between every Hanner pair + descents from peaks."""
    names = list(HANNER_M)
    npath = 11 if smoke else 41
    descent_evals = 2_000 if smoke else 40_000
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            m = max(HANNER_M[a], HANNER_M[b])
            A = whiten(_pad_generators(HANNER[D][a](), m))
            B = whiten(_pad_generators(HANNER[D][b](), m))
            stats = EvalStats()
            f = make_objective(m, D, stats)
            path = []
            for s in np.linspace(0.0, 1.0, npath):
                V = (1 - s) * A + s * B
                path.append((float(s), float(np.exp(f(V.ravel())))))
            valid = [(s, P) for s, P in path if P < 30.0]
            s_peak, p_peak = max(valid, key=lambda t: t[1]) if valid else (None, None)
            descents = []
            if s_peak is not None:
                V0 = (1 - s_peak) * A + s_peak * B
                for k in range(2):
                    task = dict(seed=9_100_000 + 100 * len(out) + k,
                                family="explicit", method="cma", m=m, d=D,
                                max_evals=descent_evals,
                                V0=(V0 + 0.01 * np.random.default_rng(k).standard_normal(V0.shape)).tolist())
                    # keep the FULL record: V_final / V_at_min are the
                    # certification payload if a descent ever triggers
                    r = run_start(task)
                    r.pop("V0", None)
                    descents.append(r)
            out.append({"pair": [a, b], "m": m, "path": path, "s_peak": s_peak,
                        "P_peak": p_peak, "path_triggers": stats.n_trigger,
                        "descents": descents})
            if s_peak is not None:
                print(f"  homotopy {a}<->{b}: peak P = {p_peak:.6f} at "
                      f"s = {s_peak:.3f}; descents -> "
                      f"{[d1['hanner'] for d1 in descents]}", flush=True)
            else:
                print(f"  homotopy {a}<->{b}: no valid path points!", flush=True)
    return out


def main():
    global RESULTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-seconds", type=float, default=10_800.0,
                    help="wall-clock cap for the multi-start phase (default 3 h)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end shakeout in results/dim4-smoke")
    args = ap.parse_args()

    if args.smoke:
        RESULTS = RESULTS + "-smoke"
    os.makedirs(RESULTS, exist_ok=True)
    jsonl = os.path.join(RESULTS, "starts.jsonl")
    done = set()
    prior_triggers = []
    if os.path.exists(jsonl):
        with open(jsonl) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    done.add(rec["seed"])
                    if rec.get("n_trigger", 0) > 0:
                        prior_triggers.append(rec["seed"])
                except Exception:
                    pass

    tasks = [t for t in build_tasks(args.smoke) if t["seed"] not in done]
    if prior_triggers:
        # trigger protocol: a previous invocation recorded a candidate below
        # the bound — do NOT search further; regenerate outputs and exit 3
        print(f"prior TRIGGER record(s) present (seeds {prior_triggers}); "
              f"skipping all further searching per protocol", flush=True)
        tasks = []
    else:
        print(f"dim-4 search: {len(tasks)} starts to run "
              f"({len(done)} already done), cap {args.cap_seconds:.0f}s, "
              f"{args.workers} workers", flush=True)

    t0 = time.perf_counter()
    n_done = 0
    run_cpu = 0.0
    capped = False
    aborted_on_trigger = prior_triggers[0] if prior_triggers else None
    with open(jsonl, "a") as out:
        with Pool(processes=args.workers, maxtasksperchild=50) as pool:
            wave = 32
            for i in range(0, len(tasks), wave):
                if capped or aborted_on_trigger or \
                        time.perf_counter() - t0 > args.cap_seconds:
                    capped = capped or not aborted_on_trigger
                    break
                for rec in pool.imap_unordered(run_start, tasks[i:i + wave]):
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    n_done += 1
                    run_cpu += rec["wall_s"]
                    if rec.get("n_trigger", 0) > 0:
                        aborted_on_trigger = rec["seed"]
                        print(f"!!! TRIGGER: seed {rec['seed']} saw "
                              f"P = {rec['min_P_seen']:.12f} < 32/3 - 1e-6; "
                              f"ABORTING SEARCH for certification", flush=True)
                        break
                    if time.perf_counter() - t0 > args.cap_seconds:
                        capped = True
                        break
                elapsed = time.perf_counter() - t0
                print(f"  wave: {n_done}/{len(tasks)} starts done, "
                      f"{elapsed:.0f}s elapsed", flush=True)
    search_wall = time.perf_counter() - t0

    hres = []
    hjson = os.path.join(RESULTS, "homotopy.json")
    if not aborted_on_trigger:
        if os.path.exists(hjson):
            print("reusing persisted homotopy probes", flush=True)
            with open(hjson) as fh:
                hres = json.load(fh)
        else:
            print("running homotopy probes (all Hanner pairs)...", flush=True)
            hres = homotopy_probes(args.smoke)
            with open(hjson, "w") as fh:
                json.dump(hres, fh, indent=1)
        homotopy_triggered = any(
            probe.get("path_triggers", 0) > 0
            or any(d.get("n_trigger", 0) > 0 for d in probe["descents"])
            for probe in hres)
        if homotopy_triggered:
            aborted_on_trigger = "homotopy"
            print("!!! TRIGGER during homotopy phase: certify before "
                  "reporting anything else.", flush=True)

    # ---- summary -----------------------------------------------------------
    recs = []
    n_corrupt = 0
    with open(jsonl) as fh:
        for line in fh:
            try:
                recs.append(json.loads(line))
            except Exception:
                n_corrupt += 1
    ok = [r for r in recs if r.get("P_final") is not None]
    trig_recs = [r for r in recs if r.get("n_trigger", 0) > 0]
    summary = {
        "n_starts_planned": len(build_tasks(args.smoke)),
        "n_starts_completed": len(recs),
        "n_errors": len(recs) - len(ok),
        "n_corrupt_lines": n_corrupt,
        "capped": capped,
        "aborted_on_trigger": aborted_on_trigger,
        "search_wall_s": round(search_wall, 1),
        "cpu_s_this_run": round(run_cpu, 1),
        "cpu_s_sum_alltime": round(sum(r["wall_s"] for r in recs), 1),
        "total_evals": sum(r["nfev"] for r in recs),
        "min_P_final": min((r["P_final"] for r in ok), default=None),
        # over ALL records: an errored start may still have seen a low P
        "min_P_seen": min((r["min_P_seen"] for r in recs if r.get("min_P_seen")),
                          default=None),
        "n_triggers_total": sum(r["n_trigger"] for r in recs),
        "n_faults_total": sum(r["n_fault"] for r in recs),
        "conjectured_min": CONJECTURED_MIN,
        "homotopy": hres,
    }
    summary["speedup_this_run"] = (
        round(run_cpu / search_wall, 2) if n_done and search_wall > 0 else None)

    def agg(pred):
        sel = [r for r in recs if pred(r) and r.get("P_final") is not None]
        if not sel:
            return None
        nf = sorted(r["nfev"] for r in sel)
        return {
            "n": len(sel),
            "converged_frac": round(float(np.mean([r["converged"] for r in sel])), 3),
            "min_P": round(min(r["P_final"] for r in sel), 9),
            "median_P": round(float(np.median([r["P_final"] for r in sel])), 9),
            "nfev_q50": nf[len(nf) // 2],
            "nfev_q90": nf[int(len(nf) * 0.9)],
            "hanner_hist": {str(h): sum(1 for r in sel if r["hanner"] == h)
                            for h in ("cube4", "cross4", "octprism4",
                                      "cubebipyr4", None)},
        }

    summary["by_family"] = {}
    for fam in ("gauss", "hanner", "robust"):
        for meth in ("nm", "cma"):
            a = agg(lambda r, fam=fam, meth=meth:
                    r["family"] == fam and r["method"] == meth)
            if a:
                summary["by_family"][f"{fam}/{meth}"] = a
    summary["by_m"] = {}
    for m in (4, 5, 6, 8, 10, 12, 16, 20, 24):
        a = agg(lambda r, m=m: r["m"] == m)
        if a:
            summary["by_m"][str(m)] = a

    # converged non-Hanner terminals: candidate other critical points
    others = [
        {k: r[k] for k in ("seed", "family", "m", "P_final", "n_vertices",
                           "n_facets", "eff_m", "converged")}
        for r in ok
        if r["converged"] and r["hanner"] is None
    ]
    summary["converged_non_hanner"] = sorted(
        others, key=lambda r: r["P_final"])[:40]

    with open(os.path.join(RESULTS, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("homotopy", "converged_non_hanner")},
                     indent=1), flush=True)

    best = sorted(ok, key=lambda r: r["P_final"])[:8]
    if trig_recs:
        trig_sorted = sorted(trig_recs,
                             key=lambda r: r.get("min_P_seen") or float("inf"))
        tseeds = {t["seed"] for t in trig_sorted}
        best = trig_sorted + [r for r in best if r["seed"] not in tseeds]
    with open(os.path.join(RESULTS, "best_bodies.json"), "w") as fh:
        json.dump(best, fh, indent=1)
    print("wrote summary.json and best_bodies.json")
    if aborted_on_trigger:
        print(f"ABORTED ON TRIGGER seed={aborted_on_trigger}: certify before "
              f"reporting anything else.")
        sys.exit(3)


if __name__ == "__main__":
    main()
