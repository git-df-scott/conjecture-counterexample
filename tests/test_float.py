"""Float-tier unit tests: exact Hanner values, GL-invariance under whitening,
classification, and degeneracy handling.  Run: python tests/test_float.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from mahler.evaluate import (EvalStats, make_objective, volume_product, whiten,
                             CONJECTURED_MIN, PENALTY_BASE)
from mahler.classify import classify_body
from mahler.hanner import HANNER, HANNER_FVEC

rng = np.random.default_rng(7)
n_pass = 0

# 1. all six Hanner constructions evaluate to exactly 32/3 (to float precision)
for d, family in HANNER.items():
    for name, ctor in family.items():
        V = ctor()
        P, volK, volKo, _ = volume_product(whiten(V))
        assert abs(P - CONJECTURED_MIN) < 1e-10, (name, P)
        n_pass += 1
        print(f"PASS {name}: P = {P:.12f} (32/3 = {CONJECTURED_MIN:.12f})")

# 2. GL-invariance contract: moderate conditioning evaluates accurately;
#    extreme conditioning gets a clean penalty; NEITHER may ever fire a trigger.
for d in (3, 4):
    name, ctor = next(iter(HANNER[d].items()))
    V = ctor()
    m = V.shape[0]
    for cond, expect_accurate in ((1e2, True), (1e4, True), (1e6, True),
                                  (1e10, False)):
        s = np.linspace(0, 1, d)
        D = np.diag(cond ** (s - 0.5))
        Q1, _ = np.linalg.qr(rng.standard_normal((d, d)))
        Q2, _ = np.linalg.qr(rng.standard_normal((d, d)))
        T = Q1 @ D @ Q2
        stats = EvalStats()
        f = make_objective(m, d, stats)
        val = f((V @ T).ravel())
        if expect_accurate:
            assert abs(val - np.log(CONJECTURED_MIN)) < 1e-7, (name, cond, val)
        else:
            assert val >= PENALTY_BASE, (name, cond, val)
        assert stats.n_trigger == 0, (name, cond, "false trigger!")
        n_pass += 1
    print(f"PASS {name}: accurate to cond 1e6, clean penalty at 1e10, "
          f"zero false triggers")

# 3. classification recovers every Hanner type
for d, family in HANNER.items():
    for name, ctor in family.items():
        cls = classify_body(ctor())
        assert cls["hanner"] == name, (name, cls)
        assert (cls["n_vertices"], cls["n_facets"]) == HANNER_FVEC[d][name]
        n_pass += 1
        print(f"PASS classify {name}: {cls}")

# 4. degenerate input gets a penalty, not a crash or a fake value
stats = EvalStats()
f = make_objective(4, 3, stats)
V_flat = rng.standard_normal((4, 3))
V_flat[:, 2] = 0.0
val = f(V_flat.ravel())
assert val >= PENALTY_BASE, val
assert stats.n_fault == 1
n_pass += 1
print(f"PASS degenerate input -> penalty {val}")

# 5. random bodies stay inside the [Kuperberg, Santalo] window
for d in (3, 4):
    stats = EvalStats()
    f = make_objective(8, d, stats)
    vals = [f(rng.standard_normal(8 * d)) for _ in range(200)]
    assert max(vals) < PENALTY_BASE, "unexpected fault on random body"
    assert stats.n_trigger == 0, "false trigger on random body"
    assert stats.min_P > CONJECTURED_MIN - 1e-6
    n_pass += 1
    print(f"PASS 200 random bodies d={d}: P in "
          f"[{stats.min_P:.4f}, {np.exp(max(vals)):.4f}], no faults/triggers")

print(f"\nALL FLOAT TESTS PASSED ({n_pass} checks)")
