# Chvátal's Ideal Conjecture — Counterexample Search

**Conjecture** (V. Chvátal, 1974, open): for every ideal (downward-closed family) `F`
over a finite ground set, the maximum size of an intersecting subfamily of `F` equals
the maximum size of a *star* (all members through one fixed element):
`max_G |G| = max_x |{A ∈ F : x ∈ A}|`.

A counterexample is an ideal whose best intersecting subfamily **strictly beats every
star**. Note that non-star *ties* are common and not counterexamples (e.g. the triangle
`{12,13,23}` in its down-closure ties the star at 3; in the full power set of `[3]`,
`{12,13,23,123}` ties `2^{3-1}`) — the open question is strict excess only.

## Reduction (what we enumerate)

If `(F, G)` is a counterexample, then WLOG:
1. `G` is up-closed within `F` (supersets of intersecting sets still intersect), and
2. `F = ↓G` (shrinking `F` to the down-closure of `G` only lowers star degrees).

Hence a minimal counterexample has the canonical form: an **intersecting antichain**
`A = {A_1,…,A_m}` with `F = ↓A` and `G = {S : A_i ⊆ S ⊆ A_j}`, where every
`|A_i| ≥ 2`, `⋂A_i = ∅`, `⋃A_i = [n]`. It therefore suffices to sweep intersecting
antichains and compare `|G|` against `max_x deg_{↓A}(x)` — no per-ideal max-clique
computation is ever needed.

`excess(A) := |G| − max_x deg(A)`. Conjecture ⇔ `excess ≤ 0` always. We track the
maximum excess observed; `excess = 0` are ties (expected), `> 0` would be a
counterexample and halts everything.

## Implementation

- `enumerate.c` (`chv_enum`): orderly DFS over bitmask generators in increasing mask
  order (each antichain visited exactly once), pairwise-intersect + incomparability
  filters, WLOG first generator `{1..k}` for `k = min |A_i|` (a valid symmetry
  restriction; subsequent generators have size ≥ k). Per-node test is O(m+n) via
  precomputed 2^n-bit down-set/up-set/element bitsets. Modes: `count` (unrestricted
  intersecting-antichain census) and `test` (reduced-form counterexample scan),
  optional `second`-generator pinning (work units) and `maxm` (generator-count cap).
- `brute_small.py`: independent verification path — enumerates ALL ideals (as
  antichains of maximal sets) for n ≤ 5 and computes true max intersecting subfamily
  by exact branch-and-bound, no reduction assumed.
- `search8`: simulated annealing over intersecting antichains on [8] maximizing
  excess (moves: add/remove/replace generator; exact incremental evaluation).
- `driver.py`, `resume_all.sh`, `run_slices8*.sh`: parallel work units, restart-proof
  relaunching, per-unit JSON results with skip-on-complete.

### Correctness anchors
- Census mode matches the intersecting-antichain counts n ≤ 6 exactly
  (1, 2, 4, 12, 81, 2646, 1422564 — OEIS A001206 shifted; n=6 = 1,422,564).
- `brute_small.py` agrees with the reduced scan for all n ≤ 5 (no counterexample,
  max excess 0 at n ≥ 3, tie families as expected).
- n = 7 result agrees with the independent IP-based verification of
  Eifler–Gleixner–Pulaj (2018/2022).

## Results

### Step 2 — reproduction of the known verified range (n ≤ 7): CONFIRMED CLEAN

Literature: Eifler, Gleixner, Pulaj, *A Safe Computational Framework for Integer
Programming applied to Chvátal's Conjecture* (arXiv:1809.01572; ACM TOMS 48(2), 2022;
ZIB-Report 18-49: "Chvátal's Conjecture Holds for Ground Sets of Seven Elements").
Their exact rational IP + certificates verified all downsets with `|⋃| ≤ 7`; the n=8
IP timed out and remains open. Earlier theory: Chvátal (compressed ideals),
Schönheim (maximal sets with common element), Snevily 1992 (shifted-type ideals),
Friedgut–Kahn–Kalai–Keller (arXiv:1608.08954, related correlation results),
"Chvátal's conjecture for downsets of small rank" (arXiv:1703.00494).

This work reproduces n ≤ 7 by direct reduced-form enumeration (method independent of
EGP's IP): **41,528,613,715 antichains visited, 41,520,284,073 tested, 0 violations**
(`results/n7/summary.json`):

| branch (min gen size k) | nodes | best excess |
|---|---|---|
| k=2 | 7,828,352 | −4 |
| k=3 | 8,597,009,005 | **−2** |
| k=4 | 32,922,640,337 | −7 |
| k=5 | 1,135,957 | −32 |
| k=6 | 64 | −56 |

All smaller n likewise clean (results/n3..n6): global best excess 0 (ties only,
e.g. triangle down-closures), never positive.

### Step 3 — one step past the verified range (n = 8, open territory): partial, clean so far

Exhaustive n=8 reduced-form space is ~10^17+ (the k=5 branch alone contains a free
2^55 subtree of pairwise-intersecting 5-sets) — beyond any full sweep, consistent
with EGP's n=8 timeout. Coverage achieved (`results/n8/`), **all with 0 violations**:

| slice | scope | nodes | best excess |
|---|---|---|---|
| k=7 | exhaustive | 128 | −119 |
| k=6 | exhaustive | 140,765,689 | −70 |
| k=5, m≤8 | exhaustive | 1,337,156,188 | −27 |
| k=4, m≤7 | exhaustive | 3,539,523,424 | −11 |
| k=3, m≤7 | exhaustive | 4,062,064,666 | −5 |
| k=2, m≤8 | exhaustive | 1,306,127,394 | −6 |
| k=3, m≤8 | 171-unit sweep (in progress) | ~7.7×10^10 est. | pending |
| annealing | ~2×10^10+ steps, many chains | — | 0 (ties only) |

Summary of the certainty region at n=8: **no counterexample exists with ≤ 7
generators (any k), ≤ 8 generators for k ∈ {2,5}, or min generator size ≥ 6.**
Annealing chains repeatedly converge to excess-0 tie families and never cross 0 —
the boundary behaves exactly as the conjecture predicts everywhere sampled.

The excess trend (−7 → −6 → −5 for k=3 as m grows 4→7) keeps the large-m, small-k
region the only live one; the k=3 m≤8 sweep and k=4 m≤8 are the next certainty
rings, with annealing covering the unrestricted remainder stochastically.

## Reproduce

```
gcc -O2 -o chv_enum enumerate.c && gcc -O2 -o search8 search8.c
./chv_enum test 7 3            # n=7, k=3 branch (12 min single core)
./chv_enum count 6 0           # census cross-check: prints 1422564
python3 brute_small.py 5       # independent all-ideals check, n<=5
./chv_enum test 8 3 11 8       # one n=8 work unit (k=3, second gen mask 11, m<=8)
./search8 8 1000000000 42 100000000   # annealer, 1e9 steps, seed 42
```

Any strict violation prints the antichain immediately and exits 17.

## Status

- No counterexample found anywhere (n ≤ 7 exhaustive; n = 8 within the region above).
- Environment note: the search ran through repeated container restarts; all long jobs
  are unit-split and resumable (`resume_all.sh`), results land as per-unit JSONs.
