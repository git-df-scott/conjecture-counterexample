# Counterexample searches for open conjectures, with exact certification

Each subproject picks an open conjecture, searches hard for a
counterexample, and certifies whatever it finds in exact arithmetic — so
that a hit would be a proof rather than a numerical suggestion, and a
miss is reported as a miss, with coverage statistics attached. Neither
search has found a counterexample. That is the expected outcome and it is
reported as such; the deliverable is the certified negative result plus
the machinery.

| subproject | conjecture | status | certification |
| --- | --- | --- | --- |
| `sidorenko/` | Sidorenko's conjecture (open) | no counterexample found; 10.1M kernels decided exactly, minimum deficit exactly 0 | exact integer, complete in both directions |
| `mahler/` | Mahler conjecture in dim 4 (open) | no counterexample found; floor at 32/3 | exact rational, sound in the counterexample direction |

## Sidorenko's conjecture

For every bipartite graph `H` and every graphon `W`,

```
t(H, W)  >=  t(K_2, W)^e(H)
```

i.e. no bipartite `H` can appear less often than in a random graph of the
same edge density. Open in general. Proven for trees, even cycles,
complete bipartite graphs, bipartite graphs with a vertex adjacent to the
whole other part (Conlon–Fox–Sudakov), and hypercubes (Hatami); the
literature cites `K_{5,5}` minus a Hamilton cycle as the smallest
bipartite graph for which it was open. This subproject searches for a
counterexample over step kernels and certifies every verdict exactly.

### Two reductions that shape the search

* **The search space is complete.** For fixed `H`, `t(H, ·)` and
  `t(K_2, ·)` are cut-metric continuous and step kernels are dense, so the
  infimum of the deficit over all graphons equals its infimum over step
  kernels. Finitely many blocks is a resolution limit, not a structural
  one — a counterexample, if one exists, is witnessed by some step kernel.

* **Scale invariance.** The deficit is homogeneous of degree `e(H)` under
  `W -> cW`, so its *sign* is scale-invariant. Two payoffs: the graphon
  constraint `W <= 1` can be dropped during the search and restored at
  the end for free, and the objective can be evaluated after rescaling to
  `t(K_2, W) = 1`, which removes the catastrophic cancellation that
  `t(H, W) - p^e(H)` suffers near the constant kernel (the two terms
  agree to 16 digits there). Sidorenko for `H` is then exactly
  `min log t(H, W) >= 0`.

### Why the search must be global

Perturb the constant kernel, `W = p(1 + f)` with `f` mean zero. Every
edge-subset term whose support has a degree-1 vertex integrates to zero,
so the lowest surviving order is the girth `2g`, and it contributes
`c_2g(H) · tr(f^2g) = c_2g(H) · Σ λ_i^2g >= 0`. **The constant kernel is a
strict local minimum of the deficit for every `H` of even girth.** No
locally-started method can ever succeed; a counterexample must live far
from quasirandomness. This is why the search seeds from 0/1 blowups and
wide log-normal kernels, and why the lattice tier sweeps extreme points.

`sidorenko/local.py` measures the prediction per graph, and has to do it
in **exact rational arithmetic**: the deficit at `eps = 1/32` with girth 8
is around `1e-12` while float cancellation error is `1e-16`, leaving four
usable digits and a fitted exponent of 6.5 instead of 8. Using integer
mean-zero directions (`f_ij = k² A_ij − k(r_i + r_j) + s` has zero row
sums by construction, so the edge density is pinned at exactly 1) with
`eps = 1/(2^j max|f|)` makes `W` rational and the deficit an exact
`Fraction`. Measured orders then match the girth on every graph checked:
`C_8 → 8.0000`, `C_10 → 10.0000`, `C_12 → 12.0000`,
`theta_4-4-4 → 8.0000`, Heawood `→ 6.001`, Desargues `→ 6.002`.

### Numerical robustness is load-bearing here

A search is only evidence of absence if it does not manufacture hits. Two
proven-positive graphs initially produced "violations" — `C_8` at
`F = −6.4e−06` and `Q_3` at `F = −0.405` — and both were artifacts of
L-BFGS drifting to degenerate boundary points (block weights around
`1e-86`, matrix entries spanning `1e83`):

* the unnormalized product over `e(H)` edges reaches `1e664` and
  **overflows**. Fixed by max-normalizing the matrix before contracting,
  which is free because `F` and `dF/dtheta` are both scale-invariant.
* max-normalization does not bound the *results*: at `Q_3`'s point `t` and
  `p^12` were both `6.4e-321`, **subnormal**, carrying four significant
  digits, so `log t − e log p` had `1e-3` of error. Anything leaving the
  normal float range is now rejected rather than believed.
* `t == 0` was read as a hard violation, but underflow produces the same
  `0`. A genuine structural zero is now distinguished from underflow by an
  **exact integer homomorphism count** into the kernel's support — so odd
  cycles still certify while underflow is discarded.

Neither artifact ever produced a false certificate; the exact tier refused
both. Both are pinned by regression tests. Candidate hits are additionally
re-evaluated through an independent path before being believed, and
counted as `numerical_rejects` when they do not survive.

### Architecture

* **Float search tier** (`sidorenko/contract.py`, `density.py`,
  `optimize.py`). `t(H, W)` on a `k`-block kernel is a tensor network with
  one variable per vertex of `H`, contracted by greedy min-fill variable
  elimination in `O(n · k^(w+1))` — the `k^n` sum is hopeless at `n = 20`.
  Analytic gradients come from per-edge and per-vertex pinned
  contractions and are pinned by both Euler identities implied by
  homogeneity plus finite differences. Multi-start L-BFGS-B and CMA-ES
  over `k = 2..8` blocks with free block weights.

* **Exact lattice tier** (`sidorenko/lattice.py`). Enumerates *every*
  symmetric integer kernel `M ∈ {0..m}^(k×k)` up to simultaneous
  permutation, with uniform weights, and decides each exactly. With
  uniform weights the deficit sign is the sign of the integer
  `Δ = T·k^2e − S^e·k^n`; `T` is computed by int64 elimination, which is
  *exact* (not merely accurate) under an a-priori bound `k^n · m^e < 2^62`
  that the code checks in Python integers before choosing the dtype,
  falling back to unbounded ints otherwise. Where a tier is exhausted
  this is an unconditional theorem about that family. Tiers that do not
  fit the time budget are subsampled and **relabelled non-exhaustive**,
  with the covered fraction recorded.

* **Exact certification tier** (`sidorenko/certify.py`). Pure stdlib, no
  numpy, no float, and no code shared with the search tier — the
  contraction is re-implemented over Python integers with a *different*
  elimination heuristic, so the two paths agree only if both are right.
  For rational data `a_i = c_i/C`, `W = M/D`, the deficit sign is the sign
  of `Δ = T·C^2e − S^e·C^n`. Float output is used only as an uncertified
  combinatorial hint: the best kernel is rounded to ~130 nearby rationals
  (including exact zeros) and each is decided from scratch. Unlike the
  geometric certification in `mahler/`, this is a **complete decision
  procedure with no error direction** — `Δ < 0` would be an unconditional
  disproof, and `Δ >= 0` is an unconditional verification of that kernel.

### Checkpoints

* `scripts/run_sidorenko_controls.py` — run first. Establishes that the
  trigger fires when it must (non-bipartite `H`, where Sidorenko provably
  fails, must yield an exact certificate), that it does not fire when it
  must not (proven-positive classes, exhaustively swept), and that the
  measured local vanishing order equals the girth.
* `scripts/run_sidorenko_search.py` — the search. Resumable (one JSONL
  record per graph), aborts on a certified bipartite violation. `--roles
  hard` concentrates the budget on graphs in no proven class; `--shard`
  lets disjoint passes run concurrently without interleaving appends.
* `scripts/spot_certify_sidorenko.py` — replays every recorded lattice
  witness and certificate through the independent pure-stdlib path.

### Results

**No counterexample found.** 88 graphs, 67 of them in no known-positive
class, up to 24 vertices and 42 edges.

| | |
| --- | --- |
| exactly decided integer kernels | 10,117,114 |
| fully exhausted kernel families | 327 |
| bipartite graphs whose exhausted families have exact minimum deficit **0** | 84 / 84 |
| continuous multi-start optimizations | 5,834 |
| float floor over the 67 hard graphs | −2.8e−14 … −1.1e−15 (trigger at −1e−09) |
| candidate hits discarded on recheck | 0 |
| local vanishing order ≠ girth | 0 graphs |

The certified statement per graph is `lat_min = 0(exact)`: over every
integer kernel family that was exhausted, the exact minimum of the deficit
is **exactly zero**, attained at the constant kernels — computed as the
sign of an integer, with no tolerance. The float floors sit at `1e-14`,
which is contraction round-off at the constant kernel, five orders of
magnitude above the trigger.

Notable hard cases, all clean:

| H | n | e | girth | kernels decided | exhausted tiers |
| --- | --- | --- | --- | --- | --- |
| `K_{5,5}` − `C_10` (cited smallest open case) | 10 | 15 | 4 | 131,571 | 4 |
| Heawood = PG(2,2) incidence | 14 | 21 | 6 | 99,296 | 4 |
| Möbius–Kantor | 16 | 24 | 6 | 60,413 | 3 |
| Pappus | 18 | 27 | 6 | 21,999 | 3 |
| Desargues | 20 | 30 | 6 | 13,195 | 3 |
| Nauru | 24 | 36 | 6 | 7,217 | 3 |
| `K_{t,t}` − PM / − Hamilton cycle, t = 5,6,7 | 10–14 | 15–42 | 4 | 12k–132k | 3–4 |
| grids, theta graphs, subdivisions | 8–16 | 9–24 | 4–8 | 132k | 4 |

Plus all 46 connected bipartite graphs with parts up to 4+4 that lie in
**no** known-positive class (the other 166 enumerated graphs are proven
cases and were skipped as redundant with the 17 negative controls).

Independent re-verification (`scripts/spot_certify_sidorenko.py`): the
lattice tier decides kernels with int64 numpy elimination in min-fill
order; the certifier decides them with pure-Python integer elimination in
min-degree order — separate implementations sharing no code. All **327**
recorded lattice witnesses and all **88** stored certificates agree, with
0 mismatches.

`results/sidorenko/`: `controls.json`, `search.jsonl` +
`search.enum.jsonl` (one record per graph: local analysis, lattice
verdicts, search trace, exact certificate), `summary.json`,
`spot_certification.json`, `run.log`.

### What this does and does not establish

It does not resolve Sidorenko's conjecture, and no search of this kind
could: the conjecture quantifies over all graphons and all bipartite `H`,
while this covers finitely many `H` at finite block resolution. What it
does establish, unconditionally and in exact arithmetic, is that the
deficit is nonnegative on every kernel in 327 fully enumerated families
across 84 bipartite graphs — including every case the literature flags as
hard — with a pipeline demonstrated to detect violations when they exist
(6 positive controls certified, with exact deficits −1/8 through
−1/32768). The honest reading of a negative result at this scale is that
it is consistent with the conjecture and locates no weak spot.

### Tests

```
python3 tests/test_sidorenko.py
```

Cross-checks the min-fill contraction against the direct `k^n` sum, the
analytic gradient against finite differences and both Euler identities,
the pure-stdlib exact certifier against the float tier, the predicted
local order against a log-log fit, and the isomorphism dedup against an
independent pairwise-VF2 count. Plus sign controls in both directions.

Requirements: Python 3.11+, numpy, scipy, cma, networkx (search tier
only; the exact certifier is dependency-free by design).

## Mahler conjecture (dim 4)

The Mahler conjecture states that for a centrally symmetric convex body
K in R^n, the volume product P(K) = vol(K) * vol(K°) is minimized by the
cube / cross-polytope (more generally, every Hanner polytope), with
minimum 4^n/n!. It is proven for n=2 and n=3 (Iriyeh–Shibata); n=4 is
open. This repository searches for a counterexample among symmetric
polytopes in R^4 — with the realistic expectation, given the proven
local minimality of all Hanner polytopes (Kim 2014) and the proven
subclasses (unconditional bodies, zonoids), that the outcome is a
certified negative result: coverage statistics plus best-found P vs. the
bound 4^4/4! = 32/3.

### Architecture

Two tiers:

* **Float search tier** (`mahler/evaluate.py`, `mahler/optimize.py`):
  bodies are `K = conv(±v_1..±v_m)`; P(K) is evaluated with two qhull
  convex-hull calls (facets of K, scaled to `<u,x> <= 1`, are the polar's
  vertices). The volume product is GL(d)-invariant, so every evaluation
  first gauge-normalizes ("whitens") the generators via SVD — measured
  float error on ill-conditioned GL-images of the cube otherwise drifts
  signed-negative and would fire false counterexample triggers.
  Multi-start local minimization: restarted Nelder–Mead and CMA-ES.

* **Exact certification tier** (`mahler/certify.py`): pure stdlib
  `fractions.Fraction`. Floating-point output is used only as an
  uncertified combinatorial hint; every certified simplex is exactly
  verified to lie on a supporting hyperplane, ridge closure makes the
  certified surface a mod-2 cycle, and one exact generic-ray winding test
  certifies boundary completeness (hence also completeness of the facet
  normal list feeding the polar step). The possible failure direction
  only ever over-estimates P, so a certified P < 32/3 would be a sound
  counterexample certificate; exactness in practice is confirmed by an
  independent hint-free canonical triangulation path that must agree to
  exact equality. See the module docstring for the full argument.

### Checkpoints

* `scripts/run_dim3.py` — dim-3 checkpoint (conjecture proven there):
  confirms the pipeline finds the cube/octahedron floor at exactly 32/3,
  and calibrates costs for dim 4. Wall-clock capped; one JSONL record
  per start; resumable.
* dim-4 search: only run after the dim-3 checkpoint is reviewed.

### Tests

```
python3 tests/test_float.py
python3 tests/test_certify.py
```

Both tiers reproduce the exact value 32/3 on all six Hanner
constructions (d=3: cube, octahedron; d=4: cube, cross-polytope,
octahedral prism, cube bipyramid — note 4^3/3! = 4^4/4! = 32/3, a
numerical coincidence).

Requirements: Python 3.11+, numpy, scipy, cma (float tier only; the
exact tier is dependency-free by design).
