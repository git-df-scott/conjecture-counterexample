"""Hanner polytopes: the conjectured minimizers of the volume product.

Each function returns a generator matrix V (m x d, float); the body is
K = conv({rows} U {-rows}).  All have volume product exactly 4^d/d!
(= 32/3 in both d=3 and d=4).

Combinatorial types:
  d=3: cube (8 vertices, 6 facets), octahedron (6, 8).
  d=4: cube (16, 8), cross-polytope (8, 16),
       octahedral prism  = B1^3 (+)_inf segment (12, 10),
       cube bipyramid    = Binf^3 (+)_1  segment (10, 12).
"""
import numpy as np


def cube3():
    return np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1]], dtype=float)


def oct3():
    return np.eye(3)


def cube4():
    return np.array(
        [[s1, s2, s3, 1] for s1 in (1, -1) for s2 in (1, -1) for s3 in (1, -1)],
        dtype=float,
    )


def cross4():
    return np.eye(4)


def octprism4():
    gens = []
    for i in range(3):
        e = [0.0, 0.0, 0.0]
        e[i] = 1.0
        gens.append(e + [1.0])
        gens.append(e + [-1.0])
    return np.array(gens)


def cubebipyr4():
    gens = [[s1, s2, s3, 0.0] for (s1, s2, s3) in
            [(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1)]]
    gens.append([0.0, 0.0, 0.0, 1.0])
    return np.array(gens)


HANNER = {
    3: {"cube3": cube3, "oct3": oct3},
    4: {"cube4": cube4, "cross4": cross4,
        "octprism4": octprism4, "cubebipyr4": cubebipyr4},
}

# (vertex count, facet count) after merging coplanar facets
HANNER_FVEC = {
    3: {"cube3": (8, 6), "oct3": (6, 8)},
    4: {"cube4": (16, 8), "cross4": (8, 16),
        "octprism4": (12, 10), "cubebipyr4": (10, 12)},
}
