"""Multi-start local minimization of log P over generator coordinates.

Local methods: restarted Nelder-Mead (restart from the terminal point with a
fresh simplex until improvement stalls — breaks NM stagnation cheaply) and
CMA-ES followed by an NM polish.  Between restarts the point is retracted onto
the whitened gauge slice (same body, better conditioning).

Every start returns a JSON-serializable record; the driver persists one JSONL
line per start so interrupted runs lose nothing.
"""
import time

import numpy as np
from scipy.optimize import minimize

from .classify import classify_body
from .evaluate import (EvalStats, make_objective, whiten, CONJECTURED_MIN,
                       PENALTY_BASE)
from .hanner import HANNER


def _retract(x, m, d):
    Vw = whiten(x.reshape(m, d))
    return x if Vw is None else Vw.ravel()


def nm_with_restarts(f, x0, m, d, stats, max_evals,
                     restart_tol=1e-9, max_restarts=25):
    x = np.asarray(x0, dtype=float).ravel()
    f_best = f(x)
    converged = False
    for _ in range(max_restarts):
        remaining = max_evals - stats.nfev
        if remaining < 200:
            break
        res = minimize(
            f, x, method="Nelder-Mead",
            options=dict(maxfev=int(remaining), xatol=1e-9, fatol=1e-11,
                         adaptive=True),
        )
        improve = f_best - res.fun
        if res.fun < f_best:
            f_best = res.fun
            x = res.x
        x = _retract(np.asarray(x), m, d)
        if improve < restart_tol:
            converged = True
            break
    return x, f_best, converged


def cma_then_polish(f, x0, m, d, stats, max_evals, seed):
    import cma

    es = cma.CMAEvolutionStrategy(
        np.asarray(x0, dtype=float).ravel(), 0.25,
        dict(maxfevals=int(max_evals * 0.7), tolfun=1e-11, tolx=1e-11,
             verbose=-9, verb_log=0, seed=int(seed) % (2**31 - 1) + 1),
    )
    es.optimize(f)
    x = np.asarray(es.result.xbest, dtype=float)
    return nm_with_restarts(f, _retract(x, m, d), m, d, stats, max_evals)


def make_start(task, rng):
    family = task["family"]
    m, d = task["m"], task["d"]
    if family in ("gauss", "robust"):
        return rng.standard_normal((m, d))
    if family == "hanner":
        base = HANNER[d][task["base"]]()
        return base + task["noise"] * rng.standard_normal(base.shape)
    if family == "explicit":
        return np.array(task["V0"], dtype=float)
    raise ValueError(f"unknown family {family}")


def run_start(task):
    """Execute one local-search start; returns a JSON-serializable record."""
    t0 = time.perf_counter()
    m, d = task["m"], task["d"]
    stats = EvalStats()
    try:  # record failures, never kill the pool worker
        rng = np.random.default_rng(task["seed"])
        f = make_objective(m, d, stats)
        V0 = make_start(task, rng)
        if task["method"] == "cma":
            x, f_best, converged = cma_then_polish(
                f, V0.ravel(), m, d, stats, task["max_evals"], task["seed"])
        else:
            x, f_best, converged = nm_with_restarts(
                f, V0.ravel(), m, d, stats, task["max_evals"])
        Vf = _retract(np.asarray(x), m, d).reshape(m, d)
        logP = f(Vf.ravel())
        if logP >= PENALTY_BASE:
            Vf = None
            P_final = None
            cls = {"n_vertices": None, "n_facets": None, "eff_m": None,
                   "hanner": None}
            converged = False
            err = "terminated in penalty region"
        else:
            P_final = float(np.exp(logP))
            cls = classify_body(Vf)
            err = None
    except Exception as exc:
        Vf = None
        P_final = None
        cls = {"n_vertices": None, "n_facets": None, "eff_m": None, "hanner": None}
        converged = False
        err = f"{type(exc).__name__}: {exc}"
    rec = {
        **{k: task[k] for k in ("seed", "family", "method", "m", "d")},
        "noise": task.get("noise"),
        "base": task.get("base"),
        "nfev": stats.nfev,
        "wall_s": round(time.perf_counter() - t0, 3),
        "converged": bool(converged),
        "P_final": P_final,
        "min_P_seen": (None if not np.isfinite(stats.min_P) else float(stats.min_P)),
        "n_trigger": stats.n_trigger,
        "n_fault": stats.n_fault,
        "error": err,
        **cls,
        "V_final": (None if Vf is None else [[round(v, 12) for v in row] for row in Vf.tolist()]),
    }
    if stats.n_trigger > 0 and stats.argmin_V is not None:
        rec["V_at_min"] = stats.argmin_V.tolist()  # full precision for certification
    return rec
