# Conjecture counterexample searches

Two independent projects share this repository:

* **Mahler conjecture (dim 4)** — below, the original project.
* **Hadwiger conjecture (t=7)** — `hadwiger/` package,
  `scripts/hadwiger_validate.py`, `scripts/run_{sweep,alpha2,anneal,structgen}.py`;
  methodology, soundness invariants, and honest-scope statement in
  `hadwiger/README.md`.

# Mahler conjecture (dim 4): counterexample search with exact certification

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

## Architecture

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

## Checkpoints

* `scripts/run_dim3.py` — dim-3 checkpoint (conjecture proven there):
  confirms the pipeline finds the cube/octahedron floor at exactly 32/3,
  and calibrates costs for dim 4. Wall-clock capped; one JSONL record
  per start; resumable.
* dim-4 search: only run after the dim-3 checkpoint is reviewed.

## Tests

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
