#!/usr/bin/env python3
"""Prong 4 (Inversion 2): structure-theorem-shaped candidate generator.

Sweeps parameterized families shaped like what the Graph Minor Structure
Theorem permits K7-minor-free graphs to look like, tuned toward the only
known density-free chromatic mechanism (odd-quadrangulation topology):

  * projective quadrangulation grids proj_quad(m,k), non-bipartite cases
  * + 1..2 apex vertices joined to varied subsets (never dominating both:
    a counterexample has no dominating vertex)
  * generalized Mycielski levels over odd cycles, with apex variants

For each candidate: exact chi bracketing via SAT (is it 6-colorable?);
if chi >= 7, exact K7-minor decision. Everything logged — the chi/minor
landscape of these families is the point, the counterexample the jackpot.

Usage: run_structgen.py [--hours 2]
"""

import argparse
import itertools
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hadwiger import family
from hadwiger.coloring import dsatur, sat_kcolorable
from hadwiger.graphs import edge_count, to_graph6
from hadwiger.minor import exact_minor_decision, fast_minor_search
from hadwiger.runner_util import (jsonl_append, record_candidate,
                                  results_path, stop_requested)


def candidates(rng):
    # projective quadrangulations, bare and apexed
    for m, k in itertools.product((3, 5, 7, 9), (2, 3, 4, 5)):
        n, adj = family.proj_quad(m, k)
        yield f"projquad({m},{k})", n, adj
        full = (1 << n) - 1
        n1, a1 = family.apex_to_subset(n, adj, full)
        yield f"projquad({m},{k})+apex", n1, a1
        for trial in range(3):
            sub = rng.getrandbits(n) | 1
            n2, a2 = family.apex_to_subset(n, adj, sub)
            n3, a3 = family.apex_to_subset(
                n2, a2, rng.getrandbits(n2 - 1) | (1 << (n2 - 1)))
            yield f"projquad({m},{k})+2apex[{trial}]", n3, a3
    # generalized Mycielski towers over odd cycles (+ optional apex)
    for c, r, levels in itertools.product((5, 7, 9), (2, 3), (1, 2, 3)):
        n, adj = family.cycle(c)
        for _ in range(levels):
            n, adj = family.gen_mycielski(n, adj, r)
            if n > 60:
                break
        yield f"genmyc(C{c},r{r})^{levels}", n, adj
        if n <= 50:
            n1, a1 = family.apex_to_subset(n, adj, (1 << n) - 1)
            yield f"genmyc(C{c},r{r})^{levels}+apex", n1, a1
    # Schrijver graph — chi=7 vertex-critical; where does its minor live?
    n, adj = family.schrijver(9, 2)
    yield "schrijver(9,2)", n, adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=2.0)
    args = ap.parse_args()
    rng = random.Random(0x57A6)
    log = results_path("structgen", "families.jsonl")
    deadline = time.time() + args.hours * 3600

    for name, n, adj in candidates(rng):
        if time.time() > deadline or stop_requested():
            break
        t0 = time.time()
        e = edge_count(n, adj)
        rec = {"family": name, "n": n, "e": e, "mader_max": 5 * n - 15}
        if e > 5 * n - 15:
            rec["skip"] = "over Mader bound: K7 minor guaranteed"
            jsonl_append(log, rec)
            continue
        six, coloring, meta = sat_kcolorable(n, adj, 6, conf_budget=500_000)
        if six is True:
            rec["chi"] = "<=6"
        elif six is None:
            rec["chi"] = "unknown(budget)"
        else:
            rec["chi"] = ">=7"
            verdict, model, detail = exact_minor_decision(
                n, adj, rng, require_agreement=True)
            rec["minor"] = verdict
            rec["minor_detail"] = detail
            g6 = to_graph6(n, adj)
            rec["g6"] = g6
            if verdict == "no":
                jsonl_append(log, rec)
                record_candidate("structgen", n, g6,
                                 {"family": name,
                                  "chi_reason": "SAT UNSAT (6-colorability)",
                                  "minor_detail": detail})
                return
        rec["wall_s"] = round(time.time() - t0, 1)
        jsonl_append(log, rec)
        print(f"{name}: n={n} e={e} chi={rec['chi']} "
              f"minor={rec.get('minor', '-')} [{rec['wall_s']}s]", flush=True)


if __name__ == "__main__":
    main()
