"""Heuristic (float) classification of terminal bodies against Hanner types.

Reporting layer only — the exact certifier is the rigorous layer.  Bodies are
whitened, scaled, and their merged (vertex count, facet count) compared with
the known Hanner f-vector data.

Facet counting is measure-weighted: qhull's triangulated output of a perturbed
product polytope contains genuine "sliver" simplices (through nearly-coplanar
vertex quadruples) whose hyperplane tilt is O(1)-arbitrary rather than O(eps),
so clustering equations alone can never merge them.  Instead, simplex
equations are clustered and each cluster weighted by its total (d-1)-measure;
clusters carrying < 1e-6 of the surface measure are slivers and are dropped.
"""
import math

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .evaluate import whiten
from .hanner import HANNER_FVEC

_EMPTY = {"n_vertices": None, "n_facets": None, "eff_m": None, "hanner": None}


def _cluster_points(rows, tol):
    out = []
    for r in rows:
        if not any(np.linalg.norm(r - c) < tol for c in out):
            out.append(r)
    return out


def _count_facets(hull, pts, tol, measure_floor=1e-6):
    d = pts.shape[1]
    reps = []   # list of [equation, accumulated (d-1)-measure]
    for k, simplex in enumerate(hull.simplices):
        q = pts[simplex]
        E = q[1:] - q[0]
        gram = E @ E.T
        area = math.sqrt(max(float(np.linalg.det(gram)), 0.0)) / math.factorial(d - 1)
        e = hull.equations[k]
        for rep in reps:
            if np.linalg.norm(e - rep[0]) < tol:
                rep[1] += area
                break
        else:
            reps.append([e, area])
    total = sum(r[1] for r in reps)
    if total <= 0.0:
        return 0
    return sum(1 for r in reps if r[1] > measure_floor * total)


def classify_body(V, tol=1e-3):
    d = np.asarray(V).shape[1]
    Vw = whiten(np.asarray(V, dtype=float))
    if Vw is None:
        return dict(_EMPTY)
    pts = np.vstack([Vw, -Vw])
    pts = pts / np.abs(pts).max()
    try:
        hull = ConvexHull(pts)
    except QhullError:
        return dict(_EMPTY)
    verts = _cluster_points([pts[i] for i in hull.vertices], tol)
    nV = len(verts)
    nF = _count_facets(hull, pts, tol)
    label = None
    for name, (ev, ef) in HANNER_FVEC[d].items():
        if (nV, nF) == (ev, ef):
            label = name
            break
    return {"n_vertices": nV, "n_facets": nF, "eff_m": nV // 2, "hanner": label}
