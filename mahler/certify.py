"""Exact rational certification of the volume product of a symmetric polytope.

Given rational generators v_1..v_m, let K = conv({+-v_i}).  This module computes
vol(K), vol(K polar) and P = vol(K) * vol(K polar) EXACTLY over Q, using
floating-point qhull only as an uncertified combinatorial hint.  Nothing float
participates in the certified claims; all verified quantities are stdlib
fractions.Fraction.

Soundness architecture (per certified point set S, 0 assumed and verified
interior via central symmetry + full rank):

1. Boundary certificate.  A candidate list of d-point simplices is checked so
   that each kept simplex T = (p_1..p_d) satisfies, exactly:
     (a) det[p_1..p_d] != 0  (affinely independent; hyperplane not through 0,
         so the normalization <u,x> = 1 exists, and u is then automatically the
         OUTWARD normal since 0 is interior);
     (b) <u, p> <= 1 for ALL p in S  (supporting hyperplane).
   Hence every kept simplex lies in a genuine facet of conv(S) (its d affinely
   independent points span a (d-1)-face), and u is a genuine facet normal.

2. Closure.  Every ridge ((d-1)-subset of a kept simplex) must occur in exactly
   2 kept simplices.  The union SIGMA of kept simplices is then a mod-2
   (d-1)-cycle, so the parity of transversal crossings with a generic ray from
   the origin is the same for every generic direction (mod-2 degree theory).

3. Winding.  One exact generic-ray test (rational direction; directions exactly
   parallel to a kept simplex's hyperplane, or hitting a simplex's relative
   boundary, are detected exactly and redrawn) must give an ODD crossing count.
   Since SIGMA is contained in the boundary of conv(S) and a generic ray from
   the interior point 0 crosses that boundary exactly once, odd parity forces
   SIGMA to cover the whole boundary up to measure zero, with odd multiplicity
   (a missing facet region would give parity 0 through that region).

4. Volume.  sum_T |det T| / d! equals the integral of the covering multiplicity
   over the boundary cone decomposition, which is >= vol(conv S), with equality
   iff the multiplicity is 1 a.e.  Over-coverage can only OVER-estimate volume;
   under-coverage is excluded by step 3.  Both vol(K) and vol(K polar) are
   therefore certified UPPER BOUNDS that are exact under multiplicity 1, so a
   certified P < 4^d/d! is a sound counterexample certificate (the possible
   error direction only ever inflates P — the extra simplices still carry
   genuine facet normals by step 1, so the polar step is unaffected).  A single
   run does NOT re-verify multiplicity 1 at runtime; therefore any candidate
   that would beat the conjecture is re-certified through the hint-free
   canonical path and both triples must agree EXACTLY before the claim is
   emitted (certify_candidate).  The closure count treats ridges as index
   (d-1)-subsets; the mod-2 cycle argument therefore relies on candidates
   coming from a triangulation whose shared ridges match combinatorially,
   which holds for both candidate generators used here (qhull simplicial
   output; canonical fan triangulation with global-lexicographic anchors).

5. Facet-normal completeness.  Step 3 also certifies that the deduped set of
   kept-simplex normals is the COMPLETE facet-normal set of conv(S) (a missing
   facet is excluded).  The polar body is then exactly conv of those normals,
   and its volume is certified by re-running the same procedure on them.  A
   bipolar cross-check verifies that every facet normal of the polar is one of
   the original points (K polar polar = K).

The hint-free path (use_hint=False) builds candidate simplices with no qhull
input: facets are enumerated by brute force over d-subsets (every facet of a
polytope contains d affinely independent input points, so this enumeration is
complete; a fast float prefilter may discard subsets, which can only cause a
certification FAILURE, never a false certificate, since steps 2-3 re-certify
completeness), and each facet is triangulated canonically so that shared ridges
receive identical triangulations from both sides.
"""
import math
import random
from collections import Counter
from fractions import Fraction
from itertools import combinations

from .exactlin import mat_rank, solve, det, dot

CONJECTURED_MIN_EXACT = Fraction(32, 3)  # 4^3/3! = 4^4/4!


class CertificationError(Exception):
    pass


class NongenericRay(Exception):
    pass


# ---------------------------------------------------------------------------
# point-set helpers


def _to_exact_points(V_rows):
    pts = []
    seen = set()
    for row in V_rows:
        for sign in (1, -1):
            p = tuple(Fraction(sign) * Fraction(x) for x in row)
            if p not in seen:
                seen.add(p)
                pts.append(p)
    return pts


def _check_symmetric(pts):
    pset = set(pts)
    for p in pts:
        if tuple(-c for c in p) not in pset:
            raise CertificationError("point set not centrally symmetric")


# ---------------------------------------------------------------------------
# candidate simplices


def _hint_simplices(pts):
    """Uncertified qhull candidate simplices.  ANY failure (qhull error, float
    overflow on huge rationals, ...) is converted to CertificationError so the
    caller's fallback to the hint-free path always engages."""
    try:
        import numpy as np
        from scipy.spatial import ConvexHull

        scale = max(abs(c) for p in pts for c in p)
        fpts = np.array([[float(c / scale) for c in p] for p in pts], dtype=float)
        hull = ConvexHull(fpts)
        return sorted({tuple(sorted(int(i) for i in s)) for s in hull.simplices})
    except Exception as exc:
        raise CertificationError(f"qhull hint failed: {exc}") from exc


def _float_supported_subsets(pts, d):
    """Float-prefiltered d-subsets that plausibly span supporting hyperplanes.

    Only a speedup: a wrongly discarded facet subset makes closure/winding FAIL
    (safe direction).  Kept loose: margin 1e-4 on points scaled to O(1); the
    scaling is done exactly over Fraction BEFORE float conversion so huge
    rational coordinates cannot overflow.  Any failure disables the prefilter.
    """
    import numpy as np

    try:
        scale = max(abs(c) for p in pts for c in p)
        fpts = np.array([[float(c / scale) for c in p] for p in pts], dtype=float)
    except Exception:
        return list(combinations(range(len(pts)), len(pts[0])))
    n = len(pts)
    out = []
    for combo in combinations(range(n), d):
        M = fpts[list(combo)]
        try:
            u = np.linalg.solve(M, np.ones(d))
        except np.linalg.LinAlgError:
            continue
        prod = fpts @ u
        if prod.max() <= 1.0 + 1e-4:
            out.append(combo)
    return out


# ---------------------------------------------------------------------------
# canonical (hint-free) facet triangulation


def _facet_normals_bruteforce(pts, d, prefilter=True):
    """Complete facet list [(u, touchset)] by exact enumeration of d-subsets."""
    cand = _float_supported_subsets(pts, d) if prefilter else combinations(range(len(pts)), d)
    facets = {}
    for combo in cand:
        M = [list(pts[i]) for i in combo]
        u = solve(M, [Fraction(1)] * d)
        if u is None:
            continue
        if all(dot(u, p) <= 1 for p in pts):
            facets.setdefault(tuple(u), None)
    out = []
    for u in facets:
        touch = tuple(i for i, p in enumerate(pts) if dot(u, p) == 1)
        out.append((u, touch))
    return out


def _affine_rank(points):
    if len(points) <= 1:
        return 0
    p0 = points[0]
    return mat_rank([[c - c0 for c, c0 in zip(p, p0)] for p in points[1:]])


def _injective_coordinate_projection(points, k):
    """Indices of k coordinates whose projection is injective/affine-rank-preserving."""
    p0 = points[0]
    diffs = [[c - c0 for c, c0 in zip(p, p0)] for p in points[1:]]
    dim = len(p0)
    for cols in combinations(range(dim), k):
        sub = [[row[c] for c in cols] for row in diffs]
        if mat_rank(sub) == k:
            return cols
    raise CertificationError("no injective coordinate projection found")


def _segment_triangulation(idx, points):
    """Extreme two indices of collinear points (exact parameter order)."""
    p0 = points[0]
    direction = None
    for p in points[1:]:
        v = [c - c0 for c, c0 in zip(p, p0)]
        if any(x != 0 for x in v):
            direction = v
            break
    if direction is None:
        raise CertificationError("degenerate segment")
    params = [(dot([c - c0 for c, c0 in zip(p, p0)], direction), i)
              for p, i in zip(points, idx)]
    lo = min(params)[1]
    hi = max(params)[1]
    if lo == hi:
        raise CertificationError("degenerate segment")
    return [(lo, hi)]


def _polygon_triangulation(idx, points):
    """Canonical fan triangulation of a convex polygon (2-dim point set).

    points: exact GLOBAL coordinate tuples (any ambient dim), affine rank 2.
    Canonicality: anchor = lexicographically smallest point tuple; edges =
    maximal boundary segments (deduped by their supporting line).  Both are
    projection-independent, so the two facets sharing this polygon as a ridge
    produce the identical triangulation.
    """
    cols = _injective_coordinate_projection(points, 2)
    proj = [tuple(p[c] for c in cols) for p in points]
    n = len(points)
    # boundary lines by brute force over pairs
    lines = {}
    for i, j in combinations(range(n), 2):
        (x1, y1), (x2, y2) = proj[i], proj[j]
        a = (y1 - y2, x2 - x1)  # normal to the segment direction
        if a[0] == 0 and a[1] == 0:
            continue
        b = a[0] * x1 + a[1] * y1
        vals = [a[0] * x + a[1] * y for (x, y) in proj]
        if all(v <= b for v in vals):
            key = _canon_hyperplane(a, b)
        elif all(v >= b for v in vals):
            key = _canon_hyperplane((-a[0], -a[1]), -b)
        else:
            continue
        lines.setdefault(key, set()).update(
            k for k, v in enumerate(vals) if v == b
        )
    # maximal edge per boundary line = extreme two points on it
    edges = set()
    for key, onset in lines.items():
        onlist = sorted(onset)
        if len(onlist) < 2:
            continue
        seg = _segment_triangulation([idx[k] for k in onlist],
                                     [points[k] for k in onlist])
        edges.add(seg[0])
    if not edges:
        raise CertificationError("polygon has no edges")
    anchor_local = min(range(n), key=lambda k: points[k])
    anchor = idx[anchor_local]
    tris = []
    for (i, j) in sorted(edges):
        if anchor != i and anchor != j:
            tris.append(tuple(sorted((anchor, i, j))))
    return tris


def _canon_hyperplane(a, b):
    """Canonical form of a hyperplane (a, b), invariant to scaling incl. sign."""
    vals = list(a) + [b]
    lead = next(v for v in vals if v != 0)
    return tuple(v / lead for v in vals)


def _polytope3_triangulation(idx, points):
    """Canonical triangulation of a 3-dim convex point set (any ambient dim)."""
    cols = _injective_coordinate_projection(points, 3)
    proj = [tuple(p[c] for c in cols) for p in points]
    n = len(points)
    # brute-force face planes via triples (general (a, b) form: a.x = b)
    planes = {}
    for combo in combinations(range(n), 3):
        p1, p2, p3 = (proj[i] for i in combo)
        u = [c - c0 for c, c0 in zip(p2, p1)]
        v = [c - c0 for c, c0 in zip(p3, p1)]
        a = (u[1] * v[2] - u[2] * v[1],
             u[2] * v[0] - u[0] * v[2],
             u[0] * v[1] - u[1] * v[0])
        if a == (0, 0, 0) or all(x == 0 for x in a):
            continue
        b = dot(a, p1)
        vals = [dot(a, q) for q in proj]
        if all(v_ <= b for v_ in vals):
            key = _canon_hyperplane(a, b)
        elif all(v_ >= b for v_ in vals):
            key = _canon_hyperplane(tuple(-x for x in a), -b)
        else:
            continue
        planes.setdefault(key, set()).update(
            k for k, v_ in enumerate(vals) if v_ == b
        )
    if not planes:
        raise CertificationError("3-polytope has no faces")
    anchor_local = min(range(n), key=lambda k: points[k])
    anchor = idx[anchor_local]
    tets = []
    for key, onset in planes.items():
        onlist = sorted(onset)
        if anchor_local in onset:
            continue
        if len(onlist) < 3 or _affine_rank([points[k] for k in onlist]) != 2:
            continue
        tris = _polygon_triangulation([idx[k] for k in onlist],
                                      [points[k] for k in onlist])
        for t in tris:
            tets.append(tuple(sorted(t + (anchor,))))
    return tets


def _triangulate_facet(global_idx, pts, d):
    """Canonical triangulation of a facet's touching set into (d-1)-simplices."""
    points = [pts[i] for i in global_idx]
    if d == 3:
        return _polygon_triangulation(list(global_idx), points)
    if d == 4:
        return _polytope3_triangulation(list(global_idx), points)
    raise CertificationError(f"unsupported dimension {d}")


def _bruteforce_simplices(pts, d, prefilter=True):
    simplices = set()
    for u, touch in _facet_normals_bruteforce(pts, d, prefilter=prefilter):
        if _affine_rank([pts[i] for i in touch]) != d - 1:
            raise CertificationError("facet touching set has wrong rank")
        for s in _triangulate_facet(touch, pts, d):
            simplices.add(tuple(sorted(s)))
    return sorted(simplices)


# ---------------------------------------------------------------------------
# the certificate


def _verify_simplices(pts, cand, d):
    """Exact per-simplex checks (support + nondegeneracy); returns {simplex: u}."""
    kept = {}
    for s in cand:
        M = [list(pts[i]) for i in s]
        u = solve(M, [Fraction(1)] * d)
        if u is None:
            continue
        ok = True
        for p in pts:
            if dot(u, p) > 1:
                ok = False
                break
        if ok:
            kept[s] = tuple(u)
    return kept


def _check_closure(kept, d):
    ridge_count = Counter()
    for s in kept:
        for ridge in combinations(s, d - 1):
            ridge_count[ridge] += 1
    bad = [r for r, c in ridge_count.items() if c != 2]
    if bad:
        raise CertificationError(
            f"closure failure: {len(bad)} ridges not shared by exactly 2 simplices")


def _winding_crossings(pts, kept, d, ray_seed):
    """Exact crossing count of a generic rational ray from 0 with the surface."""
    rnd = random.Random(ray_seed)
    for _ in range(60):
        r = tuple(Fraction(rnd.randint(1, 1 << 20) - (1 << 19), 1 << 10)
                  for _ in range(d))
        if all(x == 0 for x in r):
            continue
        try:
            count = 0
            for s, u in kept.items():
                ur = dot(u, r)
                if ur == 0:
                    raise NongenericRay("ray parallel to a facet hyperplane")
                if ur < 0:
                    continue  # crossing would be at t<0
                t = 1 / ur
                x = tuple(t * c for c in r)
                # barycentric coords: sum lam_i p_i = x, sum lam_i = 1
                rows = [[pts[i][j] for i in s] for j in range(d)]
                lam = solve(rows, list(x))
                if lam is None:
                    raise NongenericRay("unexpected singular barycentric system")
                if sum(lam) != 1:
                    raise CertificationError("barycentric normalization failed")
                if any(l == 0 for l in lam):
                    raise NongenericRay("ray meets a simplex boundary")
                if all(l > 0 for l in lam):
                    count += 1
            return count
        except NongenericRay:
            continue
    raise CertificationError("no generic ray found in 60 attempts")


def certified_volume(pts, d, use_hint=True, ray_seed=20260806):
    """Certified volume of conv(pts) and its complete facet-normal list.

    Returns (vol, normals, n_crossings).  vol is exact under multiplicity one
    (guaranteed >= truth otherwise; see module docstring).  Raises
    CertificationError if any check fails.
    """
    _check_symmetric(pts)
    if mat_rank([list(p) for p in pts]) < d:
        raise CertificationError("point set is not full-dimensional")
    cand = _hint_simplices(pts) if use_hint else _bruteforce_simplices(pts, d)
    kept = _verify_simplices(pts, cand, d)
    if not kept:
        raise CertificationError("no certified simplices")
    _check_closure(kept, d)
    ncross = _winding_crossings(pts, kept, d, ray_seed)
    if ncross % 2 != 1:
        raise CertificationError(f"winding parity failed (crossings={ncross})")
    vol = Fraction(0)
    for s in kept:
        vol += abs(det([list(pts[i]) for i in s]))
    vol /= math.factorial(d)
    normals = sorted(set(kept.values()))
    nset = set(normals)
    for u in normals:
        if tuple(-c for c in u) not in nset:
            raise CertificationError("facet normals not centrally symmetric")
    return vol, normals, ncross


def certify_volume_product(V_rows, d, use_hint=True, ray_seed=20260806):
    """Exact volume product of K = conv(+-rows).  Returns (volK, volKo, P).

    Falls back from the qhull-hinted path to the hint-free canonical path on
    certification failure.  The polar step and bipolar cross-check run in both.
    """
    pts = _to_exact_points(V_rows)

    def _run(hint):
        volK, U, _ = certified_volume(pts, d, use_hint=hint, ray_seed=ray_seed)
        volKo, W, _ = certified_volume(list(U), d, use_hint=hint, ray_seed=ray_seed + 1)
        ptset = set(pts)
        for w in W:
            if w not in ptset:
                raise CertificationError("bipolar check failed: K polar polar != K")
        return volK, volKo, volK * volKo

    if use_hint:
        try:
            return _run(True)
        except CertificationError:
            return _run(False)
    return _run(False)


# ---------------------------------------------------------------------------
# candidate handling


def round_generators(V_float, max_den):
    return [[Fraction(float(x)).limit_denominator(max_den) for x in row]
            for row in V_float]


def certify_candidate(V_float, d, max_dens=(2**12, 2**16, 2**20)):
    """Round a float candidate to rationals and certify each rounding.

    Returns a dict for the first rounding that certifies with exact
    P < 32/3, else the last certified result (beats=False), else raises.
    The certified object is the ROUNDED rational polytope, never the float one.
    """
    last = None
    for md in max_dens:
        Vr = round_generators(V_float, md)
        try:
            volK, volKo, P = certify_volume_product(Vr, d)
        except CertificationError:
            continue
        if P < Fraction(129, 25):
            # below the Kuperberg lower bound (~5.16 <= true P always): a bug
            raise CertificationError(
                f"certified P={P} is below the Kuperberg bound - certifier bug")
        if P < CONJECTURED_MIN_EXACT:
            # would-be counterexample: require the independent hint-free path
            # to reproduce the exact same triple before emitting the claim
            volK2, volKo2, P2 = certify_volume_product(Vr, d, use_hint=False)
            if (volK, volKo, P) != (volK2, volKo2, P2):
                raise CertificationError(
                    "hint and canonical paths disagree on a would-be "
                    f"counterexample: {(volK, volKo, P)} vs {(volK2, volKo2, P2)}")
        result = {
            "max_denominator": md,
            "generators": [[str(x) for x in row] for row in Vr],
            "volK": str(volK),
            "volKpolar": str(volKo),
            "P_exact": str(P),
            "P_float": float(P),
            "beats_conjecture": P < CONJECTURED_MIN_EXACT,
        }
        if result["beats_conjecture"]:
            return result
        last = result
    if last is None:
        raise CertificationError("no rounding certified successfully")
    return last
