"""Exact-tier unit tests.  Run: python tests/test_certify.py

Checks, in both the qhull-hinted and hint-free canonical paths:
  * exact P = 32/3 for all six Hanner constructions (multiplicity-1 evidence);
  * exact GL-invariance, including an ill-conditioned rational map (the float
    trap the exact tier must be immune to);
  * hint path == brute path exactly on random rational bodies, and both agree
    with float qhull to 1e-9 relative;
  * bipolar + winding + closure internals via the public API;
  * clean failures on degenerate input;
  * certify_candidate round-trip on a float body.
"""
import os
import sys
import time
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from mahler.certify import (CertificationError, certified_volume,
                            certify_volume_product, certify_candidate,
                            round_generators, _to_exact_points,
                            CONJECTURED_MIN_EXACT)
from mahler.evaluate import volume_product, whiten
from mahler.hanner import HANNER

rng = np.random.default_rng(11)
t_start = time.perf_counter()

# 1. exact Hanner values, both paths
for d, family in HANNER.items():
    for name, ctor in family.items():
        V = [[Fraction(int(x)) for x in row] for row in ctor().astype(int)]
        for hint in (True, False):
            volK, volKo, P = certify_volume_product(V, d, use_hint=hint)
            assert P == CONJECTURED_MIN_EXACT, (name, hint, P)
        print(f"PASS exact {name}: P = 32/3 exactly (hint and brute paths)")

# 2. exact GL-invariance incl. ill-conditioned rational map (d=3 cube)
V3 = [[Fraction(int(x)) for x in row] for row in HANNER[3]["cube3"]().astype(int)]
T_bad = [[Fraction(10**4), Fraction(1), Fraction(0)],
         [Fraction(0), Fraction(1, 10**4), Fraction(1)],
         [Fraction(0), Fraction(0), Fraction(1)]]
V3t = [[sum(V3[i][k] * T_bad[k][j] for k in range(3)) for j in range(3)]
       for i in range(len(V3))]
_, _, P = certify_volume_product(V3t, 3)
assert P == CONJECTURED_MIN_EXACT, P
print("PASS exact GL-invariance at condition ~1e8: P = 32/3 exactly "
      "(no float drift in the exact tier)")

# 3. random rational bodies: hint == brute exactly; both match float to 1e-9
for d, m, ntest in ((3, 6, 4), (4, 8, 3)):
    for k in range(ntest):
        Vf = rng.standard_normal((m, d))
        Vr = round_generators(Vf, 2**10)
        r1 = certify_volume_product(Vr, d, use_hint=True)
        r2 = certify_volume_product(Vr, d, use_hint=False)
        assert r1 == r2, (d, k, "hint vs brute mismatch")
        Pf, _, _, _ = volume_product(np.array([[float(x) for x in row] for row in Vr]))
        rel = abs(float(r1[2]) - Pf) / Pf
        assert rel < 1e-9, (d, k, rel)
    print(f"PASS d={d}: {ntest} random rational bodies, hint==brute exactly, "
          f"float agreement <1e-9")

# 4. degenerate input fails cleanly
try:
    certify_volume_product([[Fraction(1), Fraction(0), Fraction(0)],
                            [Fraction(0), Fraction(1), Fraction(0)],
                            [Fraction(1), Fraction(1), Fraction(0)]], 3)
    raise AssertionError("degenerate input did not raise")
except CertificationError as e:
    print(f"PASS degenerate input raises cleanly: {e}")

# 5. perturbed cube certifies P > 32/3 (cube is a local min; and no false positive)
Vp = round_generators(HANNER[3]["cube3"]() + 0.05 * rng.standard_normal((4, 3)), 2**10)
volK, volKo, P = certify_volume_product(Vp, 3)
assert P > CONJECTURED_MIN_EXACT, P
print(f"PASS perturbed cube: exact P = {float(P):.9f} > 32/3 (no false positive)")

# 6. certify_candidate on a float body (whitened random) round-trips
Vw = whiten(rng.standard_normal((6, 3)))
res = certify_candidate(Vw.tolist(), 3)
assert res["beats_conjecture"] is False
assert abs(res["P_float"] - volume_product(Vw)[0]) < 1e-4
print(f"PASS certify_candidate: exact P = {res['P_float']:.9f} "
      f"(rounded body, den<={res['max_denominator']}), beats=False")

print(f"\nALL EXACT-TIER TESTS PASSED  ({time.perf_counter()-t_start:.1f}s)")
