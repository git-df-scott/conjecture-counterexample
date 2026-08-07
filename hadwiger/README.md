# Hadwiger conjecture (t=7): counterexample search

Hadwiger's conjecture (1943): every graph with chromatic number >= t has
a K_t minor. Proven for t <= 6 (t=5,6 depend on the Four Color Theorem);
**t=7 is the smallest open case**. A counterexample is a graph with
chi >= 7 and no K7 minor. This directory searches for one. Expected
outcome, honestly: none found — the durable value of a clean run is the
pair of sharp exhaustive statements from lanes 1-2 below.

## What is known, and how it shapes the search

* Kawarabayashi–Toft: every 7-chromatic graph has a K7 or K4,4 minor.
  A 2025 result (arXiv:2507.03244): every graph with no K7^v minor (K7
  minus two adjacent edges) is 6-colorable — a counterexample fails at
  literally the last edge of a K7 model. K7-minor-free graphs are
  8-colorable, so a counterexample has chi in {7, 8}.
* Mader: K7-minor-free graphs have e <= 5n-15; 5-trees achieve equality
  and are 6-chromatic — the density frontier sits one chromatic unit
  from a counterexample.
* A lex-minimal (order, then size) counterexample is 7-contraction-
  critical, hence 7-connected (Mader), hence delta >= 7, hence n >= 10.
  It also has no dominating vertex v: otherwise G-v would be a
  6-chromatic K6-minor-free graph, contradicting the proven t=6 case.
* alpha(G) <= 2 (Seymour's designated key case): then chi >= ceil(n/2),
  so for n >= 13 every such graph is automatically >= 7-chromatic, and
  the *only* question is K7-minor-freeness. Known minor guarantees for
  alpha=2 (Duchet–Meyniel line) give only ~K_{n/3}: for n in 13..15 the
  conjecture demands strictly more than any theorem provides.

## Search lanes

1. **Exhaustive minimal-counterexample sweep** (`scripts/run_sweep.py`),
   n = 10..13: all connected graphs with delta >= 7, no dominating
   vertex, ceil(7n/2) <= e <= 5n-15 — a *superset* of the space a
   lex-minimal counterexample can inhabit (7-connectivity is not
   imposed). Generated as geng complements (Delta <= n-8, narrow edge
   window). Clean completion of order n proves: **no counterexample to
   Hadwiger t=7 exists on <= n vertices.**
2. **Exhaustive alpha<=2 sweep** (`scripts/run_alpha2.py`), n = 13..18:
   complements of triangle-free graphs in the Mader-permitted window
   e(complement) >= C(n,2)-(5n-15) (only near-Turan-extremal
   triangle-free graphs survive; the window collapses the class by
   ~100x). chi >= 7 is free by counting; each graph needs only the
   minor test. The window is EMPTY from n=19 on (then
   C(n,2) - n^2/4 > 5n-15, so every alpha<=2 graph has a K7 minor by
   Mader density alone), and orders <= 12 are covered by lane 1, so
   clean completion of lanes 1+2 together proves: **Hadwiger's
   conjecture for t=7 holds for every graph with independence number
   <= 2** — the full special case Seymour singled out as key, all
   orders. (Novelty vs. the literature to be checked before any such
   claim is made outside this repo.)
3. **Annealing inside K7-minor-free space** (`scripts/run_anneal.py`):
   islands whose states stay certifiably K7-minor-free (witnessed-
   rejection of minor-creating moves + periodic exact re-certification
   with rollback), climbing a 6-coloring-hardness objective (DSATUR
   failure fraction + budgeted SAT conflict counts). Seeds at the Mader
   frontier (5-trees) and topological (projective-quadrangulation)
   corners.
4. **Structure-theorem-shaped generator** (`scripts/run_structgen.py`):
   parameterized families at the intersection of "what the Graph Minor
   Structure Theorem permits" and "the only density-free chromatic
   mechanism known" (Youngs-style odd quadrangulations of projective
   surfaces), with non-dominating apex decorations; exact chi and minor
   analysis of every member.

## Soundness invariants (the part that matters)

* A graph is discarded as 6-colorable **only** on an explicit coloring
  checked by `verify_coloring`. DSATUR failure decides nothing.
* A graph is discarded as containing a K7 minor **only** on an explicit
  7-branch-set model checked by `verify_k7_model` (disjoint, nonempty,
  connected, pairwise adjacent). Heuristics (clique, greedy contraction,
  minorminer) *propose*; the verifier *decides*.
* "No K7 minor" is only ever concluded from an exact decider: the
  direct SAT encoding (canonical roots + layered reachability), the
  independent CEGAR encoding (partition + lazy connectivity cuts), or
  the treewidth shortcut (a greedy elimination order of width <= 5 is a
  certificate that tw <= 5 < tw(K7) = 6). Budget exhaustion returns
  'unknown' and escalates; it is never treated as 'no'.
* chi >= 7 is only ever concluded from SAT UNSAT on 6-colorability
  (or, in lane 2, from the alpha <= 2 counting bound, with alpha <= 2
  sample-checked against the generator).
* Any candidate triggers a global STOP flag and
  `scripts/verify_candidate.py`: 6-colorability UNSAT under three
  independent solver families; minor-freeness under two solver families
  x two encodings + branch-and-bound + minorminer must-fail.

## Validation gate (`scripts/hadwiger_validate.py`)

There is no small-case known answer to checkpoint against (Hadwiger
t<=6 proofs don't yield a cheap machine-checkable target), so the gate
substitutes theorem-backed controls: must-detect graphs (K7, K8,
K10-PM, the 1-subdivision of K7 — minor but no subgraph — Mader-
supercritical randoms at e=5n-14, Kneser(9,2)), must-clear graphs
(5-trees, planar+2apex, K6), three-way cross-validation of the exact
deciders on hundreds of randoms, chromatic controls (Kneser, Mycielski,
Grotzsch, Petersen), and corruption tests of the witness verifiers
themselves. The gate must pass before any search lane runs.

## Running

    python3 scripts/hadwiger_validate.py          # gate (must pass)
    python3 scripts/measure_classes.py            # class sizes
    HOURS=10 scripts/launch_overnight.sh          # all lanes, resumable

All lanes persist one JSONL record per shard/event under
`results/hadwiger/` and honor a `results/hadwiger/STOP` flag; re-running
a lane skips manifest-completed shards, so an interrupted session
resumes without loss.

## Honest scope

Lanes 1-2 are exhaustive over sharply-defined but small slices (n <= 13
resp. 15). Lanes 3-4 are heuristic exploration. None of this
constitutes evidence about Hadwiger's conjecture at large: if a
counterexample exists it is widely expected to be large, and this
search covers a vanishing fraction of the space a name like "Hadwiger
search" might suggest. The claims a clean run supports are exactly the
two bold statements under "Search lanes", nothing more.
