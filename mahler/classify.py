"""Heuristic (float) classification of terminal bodies against Hanner types.

Reporting layer only — the exact certifier is the rigorous layer.  Bodies are
whitened, scaled, and their merged (vertex count, facet count) compared with
the known Hanner f-vector data.
"""
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .evaluate import whiten
from .hanner import HANNER_FVEC

_EMPTY = {"n_vertices": None, "n_facets": None, "eff_m": None, "hanner": None}


def _cluster(rows, tol):
    out = []
    for r in rows:
        if not any(np.linalg.norm(r - c) < tol for c in out):
            out.append(r)
    return out


def classify_body(V, tol=1e-3):
    d = V.shape[1]
    Vw = whiten(np.asarray(V, dtype=float))
    if Vw is None:
        return dict(_EMPTY)
    pts = np.vstack([Vw, -Vw])
    pts = pts / np.abs(pts).max()
    try:
        hull = ConvexHull(pts)
    except QhullError:
        return dict(_EMPTY)
    verts = _cluster([pts[i] for i in hull.vertices], tol)
    facets = _cluster([hull.equations[i] for i in range(len(hull.equations))], tol)
    nV, nF = len(verts), len(facets)
    label = None
    for name, (ev, ef) in HANNER_FVEC[d].items():
        if (nV, nF) == (ev, ef):
            label = name
            break
    return {"n_vertices": nV, "n_facets": nF, "eff_m": nV // 2, "hanner": label}
