#!/usr/bin/env python3
"""Exact spot-certification of the best terminal bodies from a search run.

Reads results/<run>/best_bodies.json, rounds each body to rationals, and runs
the exact certifier end-to-end. On a run with no counterexample this validates
the full candidate path on real search output (expected: exact P >= 32/3,
beats_conjecture False for every body).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fractions import Fraction

from mahler.certify import certify_candidate, CertificationError, CONJECTURED_MIN_EXACT


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "dim3"
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results", run, "best_bodies.json")
    with open(path) as fh:
        bodies = json.load(fh)
    out = []
    for i, rec in enumerate(bodies):
        V = rec.get("V_at_min") or rec["V_final"]
        fp = rec.get("P_final")
        fp_s = f"{fp:.9f}" if fp is not None else f"min_seen={rec.get('min_P_seen')}"
        t0 = time.perf_counter()
        try:
            res = certify_candidate(V, rec["d"])
            res["certify_wall_s"] = round(time.perf_counter() - t0, 2)
            res["float_P_final"] = fp
            res["hanner"] = rec["hanner"]
            gap = Fraction(res["P_exact"]) - CONJECTURED_MIN_EXACT
            print(f"[{i}] float P={fp_s} ({rec['hanner']}): "
                  f"exact P = {res['P_exact']} = {res['P_float']:.9f}, "
                  f"beats={res['beats_conjecture']}, exact gap to 32/3 = "
                  f"{float(gap):+.3e}, {res['certify_wall_s']}s")
        except CertificationError as e:
            res = {"error": str(e), "float_P_final": fp}
            print(f"[{i}] float P={fp_s}: CERTIFICATION FAILED: {e}")
        out.append(res)
    dest = os.path.join(os.path.dirname(path), "spot_certification.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
