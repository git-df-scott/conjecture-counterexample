#!/usr/bin/env python3
"""Prong 3 (Inversion 1): annealing inside K7-minor-free space.

State: a graph certified K7-minor-free (exact SAT at batch boundaries).
Moves: add-edge (screened by the greedy minor heuristic; witnessed K7
=> reject), remove-edge, vertex split. Objective: 6-coloring hardness
(DSATUR failure fraction + SAT conflict count under budget). Any state
where 6-colorability is exactly UNSAT and both exact minor deciders say
'no' is a counterexample.

Soundness note: heuristic screening can miss minors, so provisional
states may silently acquire one; the periodic exact re-certification
rolls back to the last certified state when that happens. A candidate is
only ever declared from the full require_agreement exact protocol.

Usage: run_anneal.py --island 0 --hours 2 [--seed-kind 5tree --n0 24]
"""

import argparse
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hadwiger import family
from hadwiger.coloring import dsatur, sat_kcolorable
from hadwiger.graphs import (add_edge, edge_count, is_connected,
                             remove_edge, to_graph6)
from hadwiger.minor import (exact_minor_decision, fast_minor_search,
                            sat_minor_direct)


def split_vertex(n, adj, v, keep_mask, rng):
    """Split v: v keeps neighbors in keep_mask, a new vertex n gets the
    rest, plus the edge v-n. Splits are un-contractions, so they CAN
    create K7 minors — callers must screen and re-certify."""
    a = list(adj) + [0]
    move = adj[v] & ~keep_mask
    a[v] = (adj[v] & keep_mask) | (1 << n)
    a[n] = move | (1 << v)
    m = move
    while m:
        wb = m & -m
        m ^= wb
        w = wb.bit_length() - 1
        a[w] = (a[w] & ~(1 << v)) | (1 << n)
    return n + 1, tuple(a)
from hadwiger.runner_util import (jsonl_append, record_candidate,
                                  results_path, stop_requested)


def fitness(n, adj, rng):
    """Higher = harder to 6-color. Returns (score, exact_unsat_flag)."""
    fails = 0
    tries = 6
    for _ in range(tries):
        if dsatur(n, adj, 6, rng=rng, tries=1) is None:
            fails += 1
    conf = 0
    exact_unsat = False
    if fails == tries:
        # DSATUR can't do it at all: ask SAT (budgeted)
        status, _, meta = sat_kcolorable(n, adj, 6, conf_budget=20000, rng=rng)
        if status is False:
            exact_unsat = True
        elif status is None:
            conf = 20000
        else:
            conf = meta.get("stats", {}).get("conflicts", 0)
    else:
        status, _, meta = sat_kcolorable(n, adj, 6, conf_budget=2000, rng=rng)
        if status is False:
            exact_unsat = True
        conf = min(2000, meta.get("stats", {}).get("conflicts", 0) or 0)
    score = fails / tries + conf / 2000.0
    return score, exact_unsat


def seed_graph(kind, n0, rng):
    if kind == "5tree":
        return family.random_5tree(n0, rng)
    if kind == "projquad":
        n, adj = family.proj_quad(rng.choice([3, 5, 7]), rng.choice([3, 4, 5]))
        mask = (1 << n) - 1
        n, adj = family.apex_to_subset(n, adj, mask)
        return n, adj
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--island", type=int, default=0)
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--seed-kind", default="5tree")
    ap.add_argument("--n0", type=int, default=24)
    ap.add_argument("--recert-every", type=int, default=12,
                    help="accepted growth moves between exact re-certifications")
    ap.add_argument("--nmax", type=int, default=40,
                    help="vertex cap (exact deciders degrade beyond this)")
    args = ap.parse_args()
    rng = random.Random(0xA11CE + 7919 * args.island)
    log = results_path("anneal", f"island{args.island}.jsonl")

    n, adj = seed_graph(args.seed_kind, args.n0, rng)
    v, _ = sat_minor_direct(n, adj)
    assert v == "no", "seed must be certifiably K7-minor-free"
    cert = (n, adj)  # last exactly-certified state
    score, _ = fitness(n, adj, rng)
    best = score
    temp0, temp1 = 0.30, 0.02
    t0 = time.time()
    deadline = t0 + args.hours * 3600
    it = accepted_grow = 0
    while time.time() < deadline:
        it += 1
        if it % 50 == 0 and stop_requested():
            break
        frac = (time.time() - t0) / (args.hours * 3600)
        temp = temp0 * (temp1 / temp0) ** frac
        kind = rng.random()
        u, w = rng.randrange(n), rng.randrange(n)
        cn = n
        if kind < 0.5 and u != w and not (adj[u] >> w) & 1:
            if edge_count(n, adj) + 1 > 5 * n - 15:
                continue  # Mader ceiling: adding would force a K7 minor
            cand = add_edge(adj, u, w)
            if fast_minor_search(n, cand, rng, greedy_tries=6) is not None:
                continue  # witnessed K7 minor => reject
            is_grow = True
        elif kind < 0.75 and u != w and (adj[u] >> w) & 1:
            cand = remove_edge(adj, u, w)
            if not is_connected(n, cand):
                continue
            is_grow = False
        elif kind < 0.87 and n < args.nmax and bin(adj[u]).count("1") >= 4:
            nb = adj[u]
            keep = 0
            m = nb
            while m:
                wb = m & -m
                m ^= wb
                if rng.random() < 0.5:
                    keep |= wb
            if keep == 0 or keep == nb:
                continue
            cn, cand = split_vertex(n, adj, u, keep, rng)
            if fast_minor_search(cn, cand, rng, greedy_tries=6) is not None:
                continue
            is_grow = True
        else:
            continue
        new_score, exact_unsat = fitness(cn, cand, rng)
        if exact_unsat:
            # chi(cand) >= 7 exactly; is it minor-free, exactly?
            verdict, model, detail = exact_minor_decision(
                cn, cand, rng, require_agreement=True)
            g6 = to_graph6(cn, cand)
            jsonl_append(log, {"event": "chi7_state", "g6": g6,
                               "minor": verdict, "detail": detail})
            if verdict == "no":
                record_candidate("anneal", cn, g6,
                                 {"chi_reason": "SAT UNSAT (6-colorability)",
                                  "minor_detail": detail})
                return
            continue  # has a minor: not acceptable as a state (chi too high
                      # only via minor-ful structure); keep searching
        delta = new_score - score
        if delta >= 0 or rng.random() < math.exp(delta / max(temp, 1e-9)):
            n, adj = cn, cand
            score = new_score
            if is_grow:
                accepted_grow += 1
                if accepted_grow % args.recert_every == 0:
                    v, model = sat_minor_direct(n, adj)
                    if v == "yes":
                        jsonl_append(log, {"event": "rollback",
                                           "it": it, "score": score})
                        n, adj = cert
                        score, _ = fitness(n, adj, rng)
                    else:
                        cert = (n, adj)
            if score > best:
                best = score
                jsonl_append(log, {"event": "best", "it": it,
                                   "score": round(score, 4), "n": n,
                                   "e": edge_count(n, adj),
                                   "g6": to_graph6(n, adj)})
    jsonl_append(log, {"event": "finished", "iters": it,
                       "best": round(best, 4),
                       "wall_s": round(time.time() - t0, 1)})
    print(f"island {args.island}: {it} iters, best {best:.4f}")


if __name__ == "__main__":
    main()
