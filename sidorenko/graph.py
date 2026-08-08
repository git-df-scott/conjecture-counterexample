"""Graph plumbing and the candidate corpus for the Sidorenko search.

Sidorenko's conjecture: for every bipartite graph H and every graphon
W: [0,1]^2 -> [0,1] (symmetric measurable),

    t(H, W) >= t(K_2, W)^{e(H)}.

Reductions used to choose what to search over:

* **Connectivity.**  t(H, .) is multiplicative over connected components and
  e(H) is additive, so H = H_1 + H_2 satisfies Sidorenko whenever H_1 and H_2
  do; a disconnected counterexample forces a connected one.  Isolated vertices
  contribute a factor 1 to both sides.  The corpus is therefore restricted to
  connected graphs on >= 2 vertices without loss of generality.

* **Bipartiteness is necessary, not an assumption to test.**  Any H with an odd
  cycle fails: take W = the bipartite-blowup kernel, which has t(K_2, W) > 0
  and t(H, W) = 0.  Odd cycles are kept in the corpus precisely as *positive
  controls* -- the pipeline must flag them.

Known-positive classes (used only to label results, never to skip work):
trees, even cycles, complete bipartite graphs, bipartite graphs with a vertex
adjacent to every vertex of the other part (Conlon-Fox-Sudakov), and hypercube
graphs (Hatami).  Labels are informational: the search runs on every candidate,
and a "known positive" that produced a violation would indicate a bug in this
repository, which is exactly why they are retained as negative controls.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

__all__ = ["Graph", "corpus", "enumerate_bipartite", "KNOWN_POSITIVE_REASONS"]


@dataclass(frozen=True)
class Graph:
    name: str
    n: int
    edges: tuple  # tuple of sorted (u, v) pairs, u < v

    @staticmethod
    def make(name, n, edges):
        norm = sorted({(min(u, v), max(u, v)) for u, v in edges})
        for u, v in norm:
            if u == v:
                raise ValueError(f"{name}: loops not allowed")
            if not (0 <= u < n and 0 <= v < n):
                raise ValueError(f"{name}: vertex out of range")
        return Graph(name, n, tuple(norm))

    @property
    def m(self):
        return len(self.edges)

    def adj(self):
        a = [set() for _ in range(self.n)]
        for u, v in self.edges:
            a[u].add(v)
            a[v].add(u)
        return a

    def degrees(self):
        return [len(s) for s in self.adj()]

    def is_connected(self):
        if self.n == 0:
            return False
        a = self.adj()
        seen = {0}
        stack = [0]
        while stack:
            for w in a[stack.pop()]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        return len(seen) == self.n

    def bipartition(self):
        """Return a 2-colouring list, or None if H has an odd cycle."""
        a = self.adj()
        colour = [-1] * self.n
        for s in range(self.n):
            if colour[s] != -1:
                continue
            colour[s] = 0
            stack = [s]
            while stack:
                x = stack.pop()
                for y in a[x]:
                    if colour[y] == -1:
                        colour[y] = 1 - colour[x]
                        stack.append(y)
                    elif colour[y] == colour[x]:
                        return None
        return colour

    def girth(self):
        """Length of a shortest cycle, or None if acyclic (BFS from each vertex)."""
        a = self.adj()
        best = None
        for s in range(self.n):
            dist = {s: 0}
            parent = {s: None}
            queue = [s]
            qi = 0
            while qi < len(queue):
                x = queue[qi]
                qi += 1
                for y in a[x]:
                    if y not in dist:
                        dist[y] = dist[x] + 1
                        parent[y] = x
                        queue.append(y)
                    elif y != parent[x]:
                        cand = dist[x] + dist[y] + 1
                        best = cand if best is None else min(best, cand)
        return best

    def count_cycles_of_length(self, length):
        """Number of cycle subgraphs of H with exactly `length` edges."""
        if length < 3:
            return 0
        a = self.adj()
        total = 0
        for start in range(self.n):
            # Count closed walks that are cycles with `start` the least vertex.
            def walk(path, visited):
                nonlocal total
                x = path[-1]
                if len(path) == length:
                    if start in a[x]:
                        total += 1
                    return
                for y in sorted(a[x]):
                    if y <= start or y in visited:
                        continue
                    visited.add(y)
                    path.append(y)
                    walk(path, visited)
                    path.pop()
                    visited.discard(y)

            walk([start], {start})
        # Each cycle is traversed in 2 directions from its least vertex.
        return total // 2

    def to_networkx(self):
        import networkx as nx

        G = nx.Graph()
        G.add_nodes_from(range(self.n))
        G.add_edges_from(self.edges)
        return G

    def iso_hash(self):
        """Cheap isomorphism-invariant bucket key (Weisfeiler-Leman).

        Complete invariants are not needed: `iso_hash` buckets, and
        `is_isomorphic_to` decides inside a bucket.  A brute-force canonical
        form is not an option here -- most interesting graphs in the corpus are
        vertex-transitive, so orbit refinement leaves n! relabellings.
        """
        import networkx as nx

        return (self.n, self.m, tuple(sorted(self.degrees())),
                nx.weisfeiler_lehman_graph_hash(self.to_networkx(), iterations=4))

    def is_isomorphic_to(self, other):
        import networkx as nx

        if (self.n, self.m, sorted(self.degrees())) != (other.n, other.m, sorted(other.degrees())):
            return False
        return nx.is_isomorphic(self.to_networkx(), other.to_networkx())

    def known_positive_reason(self):
        """A literature class guaranteeing Sidorenko for H, or None."""
        colour = self.bipartition()
        if colour is None:
            return None
        deg = self.degrees()
        if self.m == self.n - 1:
            return "tree (Sidorenko 1991)"
        if all(d == 2 for d in deg) and self.is_connected():
            return "even cycle (Sidorenko 1991)"
        left = [v for v in range(self.n) if colour[v] == 0]
        right = [v for v in range(self.n) if colour[v] == 1]
        if self.m == len(left) * len(right):
            return "complete bipartite (Sidorenko 1991)"
        for v in range(self.n):
            other = len(right) if colour[v] == 0 else len(left)
            if deg[v] == other:
                return "vertex complete to the other part (Conlon-Fox-Sudakov 2010)"
        for d in range(1, 7):
            if self.n == 2**d and self.m == d * 2 ** (d - 1) and _is_hypercube(self, d):
                return f"hypercube Q_{d} (Hatami 2010)"
        return None

    def summary(self):
        colour = self.bipartition()
        return {
            "name": self.name,
            "n": self.n,
            "e": self.m,
            "bipartite": colour is not None,
            "degrees": sorted(self.degrees()),
            "girth": self.girth(),
            "known_positive_reason": self.known_positive_reason(),
        }


KNOWN_POSITIVE_REASONS = (
    "tree",
    "even cycle",
    "complete bipartite",
    "vertex complete to the other part",
    "hypercube",
)


def _is_hypercube(g, d):
    """True if g is isomorphic to Q_d."""
    if g.n != 2**d:
        return False
    edges = [
        (x, x ^ (1 << b)) for x in range(2**d) for b in range(d) if x < (x ^ (1 << b))
    ]
    return g.is_isomorphic_to(Graph.make(f"Q{d}", 2**d, edges))


# --------------------------------------------------------------------------
# named constructions
# --------------------------------------------------------------------------


def cycle(n, name=None):
    return Graph.make(name or f"C_{n}", n, [(i, (i + 1) % n) for i in range(n)])


def path(n):
    return Graph.make(f"P_{n}", n, [(i, i + 1) for i in range(n - 1)])


def complete_bipartite(p, q):
    return Graph.make(f"K_{{{p},{q}}}", p + q, [(i, p + j) for i in range(p) for j in range(q)])


def crown(t):
    """K_{t,t} minus a perfect matching."""
    return Graph.make(
        f"K_{{{t},{t}}}-PM", 2 * t, [(i, t + j) for i in range(t) for j in range(t) if i != j]
    )


def kt_minus_hamilton_cycle(t):
    """K_{t,t} minus a Hamilton cycle u_0 v_0 u_1 v_1 ... u_{t-1} v_{t-1} u_0.

    For t = 5 this is the graph cited in the literature as the smallest
    bipartite graph for which Sidorenko's conjecture was open.
    """
    removed = {(i, t + i) for i in range(t)} | {(i, t + (i - 1) % t) for i in range(t)}
    edges = [
        (i, t + j) for i in range(t) for j in range(t) if (i, t + j) not in removed
    ]
    return Graph.make(f"K_{{{t},{t}}}-C_{{{2*t}}}", 2 * t, edges)


def hypercube(d):
    return Graph.make(
        f"Q_{d}", 2**d, [(x, x ^ (1 << b)) for x in range(2**d) for b in range(d) if x < x ^ (1 << b)]
    )


def grid(p, q):
    idx = lambda i, j: i * q + j
    edges = []
    for i in range(p):
        for j in range(q):
            if i + 1 < p:
                edges.append((idx(i, j), idx(i + 1, j)))
            if j + 1 < q:
                edges.append((idx(i, j), idx(i, j + 1)))
    return Graph.make(f"grid_{p}x{q}", p * q, edges)


def lcf(n, shifts, repeats, name):
    """Lederberg-Coxeter-Frucht notation for a cubic Hamiltonian graph."""
    edges = [(i, (i + 1) % n) for i in range(n)]
    k = len(shifts)
    for r in range(repeats):
        for s in range(k):
            i = r * k + s
            edges.append((i % n, (i + shifts[s]) % n))
    return Graph.make(name, n, edges)


def incidence_graph(points, blocks, name):
    """Bipartite incidence graph of a set system."""
    p = len(points)
    index = {x: i for i, x in enumerate(points)}
    edges = [(index[x], p + b) for b, blk in enumerate(blocks) for x in blk]
    return Graph.make(name, p + len(blocks), edges)


def projective_plane_incidence(q):
    """Incidence graph of PG(2, q) for prime q (q^2+q+1 points and lines).

    Uses the classical difference-set construction only for q = 2 (Fano); for
    general prime q the lines are built from the point coordinates directly.
    """
    pts = []
    for a in range(q):
        for b in range(q):
            pts.append((a, b, 1))
    for a in range(q):
        pts.append((a, 1, 0))
    pts.append((1, 0, 0))
    index = {p: i for i, p in enumerate(pts)}

    def norm(v):
        for c in v:
            if c % q:
                inv = pow(c % q, q - 2, q) if q > 2 else 1
                return tuple((x * inv) % q for x in v)
        return None

    lines = []
    seen = set()
    for coef in pts:  # a line is also given by homogeneous coefficients
        key = norm(coef)
        if key in seen:
            continue
        seen.add(key)
        member = [
            index[p]
            for p in pts
            if sum(c * x for c, x in zip(key, p)) % q == 0
        ]
        lines.append(member)
    n_pts = len(pts)
    edges = [(v, n_pts + li) for li, mem in enumerate(lines) for v in mem]
    return Graph.make(f"PG(2,{q})-incidence", n_pts + len(lines), edges)


def theta(paths):
    """Generalised theta graph: two hub vertices joined by internally disjoint
    paths with the given edge lengths."""
    edges = []
    nxt = 2
    for L in paths:
        prev = 0
        for _ in range(L - 1):
            edges.append((prev, nxt))
            prev = nxt
            nxt += 1
        edges.append((prev, 1))
    return Graph.make("theta_" + "-".join(map(str, paths)), nxt, edges)


def subdivision(g, times=1):
    """Subdivide every edge of g `times` times (bipartite double-ish blowup)."""
    n = g.n
    edges = []
    for u, v in g.edges:
        prev = u
        for _ in range(times):
            edges.append((prev, n))
            prev = n
            n += 1
        edges.append((prev, v))
    return Graph.make(f"sub{times}({g.name})", n, edges)


def bipartite_double_cover(g):
    edges = []
    for u, v in g.edges:
        edges.append((u, g.n + v))
        edges.append((v, g.n + u))
    return Graph.make(f"double({g.name})", 2 * g.n, edges)


def petersen():
    outer = [(i, (i + 1) % 5) for i in range(5)]
    inner = [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
    spokes = [(i, 5 + i) for i in range(5)]
    return Graph.make("Petersen", 10, outer + inner + spokes)


def mobius_kantor():
    return lcf(16, [5, -5], 8, "Moebius-Kantor")


def heawood():
    return lcf(14, [5, -5], 7, "Heawood")


def pappus():
    return lcf(18, [5, 7, -7, 7, -7, -5], 3, "Pappus")


def desargues():
    return lcf(20, [5, -5, 9, -9], 5, "Desargues")


def nauru():
    return lcf(24, [5, -9, 7, -7, 9, -5], 4, "Nauru")


def corpus(max_vertices=24):
    """Flagged candidate list: hard/open cases, plus controls in both directions.

    Every entry carries a `role`:
      * "hard"      -- no known-positive class applies; a real search target.
      * "neg-control" -- a known-positive bipartite H; a violation means a bug.
      * "pos-control" -- non-bipartite; the pipeline MUST report a violation.
    """
    out = []

    def add(g, note=""):
        if g.n > max_vertices:
            return
        out.append((g, note))

    # ---- non-bipartite positive controls (Sidorenko provably fails) ----
    add(cycle(3), "positive control: odd cycle")
    add(cycle(5), "positive control: odd cycle")
    add(Graph.make("K_4", 4, [(i, j) for i in range(4) for j in range(i + 1, 4)]), "positive control")
    add(petersen(), "positive control: odd girth 5")

    # ---- known-positive bipartite negative controls ----
    for g in (path(3), path(5), cycle(4), cycle(6), cycle(8)):
        add(g, "negative control: proven class")
    for p, q in ((2, 2), (2, 3), (3, 3), (3, 4), (4, 4), (4, 5)):
        add(complete_bipartite(p, q), "negative control: proven class")
    add(hypercube(3), "negative control: proven class (= K_{4,4}-PM)")
    add(hypercube(4), "negative control: proven class")
    add(Graph.make("star+edge", 5, [(0, 1), (0, 2), (0, 3), (3, 4)]), "negative control: tree")

    # ---- hard / historically-cited cases ----
    add(kt_minus_hamilton_cycle(5), "cited as the smallest open bipartite case")
    add(kt_minus_hamilton_cycle(6), "K_{6,6} minus a Hamilton cycle")
    add(kt_minus_hamilton_cycle(7), "K_{7,7} minus a Hamilton cycle")
    add(crown(5), "K_{5,5} minus a perfect matching")
    add(crown(6), "K_{6,6} minus a perfect matching")
    add(crown(7), "K_{7,7} minus a perfect matching")
    add(heawood(), "incidence graph of the Fano plane, girth 6")
    add(mobius_kantor(), "cubic, girth 6, vertex-transitive")
    add(pappus(), "cubic, girth 6")
    add(desargues(), "cubic, girth 6, bipartite double cover of Petersen")
    add(nauru(), "cubic, girth 6")
    add(grid(3, 3), "grid")
    add(grid(3, 4), "grid")
    add(grid(4, 4), "grid")
    add(theta([3, 3, 3]), "theta graph, girth 6")
    add(theta([2, 4, 4]), "theta graph")
    add(theta([3, 3, 5]), "theta graph")
    add(theta([4, 4, 4]), "theta graph, girth 8")
    add(subdivision(petersen()), "subdivided Petersen")
    add(bipartite_double_cover(cycle(5)), "= C_10")
    add(subdivision(Graph.make("K_4", 4, [(i, j) for i in range(4) for j in range(i + 1, 4)])),
        "subdivided K_4")
    add(projective_plane_incidence(2), "= Heawood, independent construction")
    add(projective_plane_incidence(3), "incidence graph of PG(2,3), girth 6")

    # C_6 with a chord-path family: bipartite, min degree 2, not in any class
    add(Graph.make("C_6+diag-path", 8,
                   [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 6), (6, 7), (7, 3)]),
        "two hexagon vertices joined by an extra even path")
    add(Graph.make("cube-minus-edge", 8,
                   [e for e in hypercube(3).edges if e != (0, 1)]),
        "Q_3 minus an edge")
    add(Graph.make("K33-plus-subdiv-edge", 8,
                   list(complete_bipartite(3, 3).edges) + [(0, 6), (6, 7), (7, 3)]),
        "K_{3,3} with an attached even path between the parts")
    return out


def enumerate_bipartite(p, q, min_degree=1, limit=None, seed=None, sample=None):
    """Connected bipartite graphs with parts of size p and q, up to isomorphism,
    with every degree >= min_degree.

    Iterates over the 2^(p*q) bipartite adjacency matrices (feasible for
    p*q <= 20); if `sample` is given, that many random matrices are drawn
    instead.  Deduplication buckets by a Weisfeiler-Leman hash and decides
    inside a bucket with VF2, which is what makes the vertex-transitive cases
    tractable.
    """
    import random

    slots = [(i, p + j) for i in range(p) for j in range(q)]
    buckets: dict = {}
    out = []
    rng = random.Random(seed)

    def consider(mask):
        edges = [slots[b] for b in range(len(slots)) if mask >> b & 1]
        if len(edges) < max(p, q):
            return
        g = Graph.make(f"B{p}x{q}_{len(out)}", p + q, edges)
        if not g.is_connected() or min(g.degrees()) < min_degree:
            return
        key = g.iso_hash()
        for other in buckets.get(key, ()):
            if g.is_isomorphic_to(other):
                return
        buckets.setdefault(key, []).append(g)
        out.append(g)

    if sample is None:
        for mask in range(1 << len(slots)):
            consider(mask)
            if limit and len(out) >= limit:
                break
    else:
        for _ in range(sample):
            density = rng.uniform(0.25, 0.85)
            mask = 0
            for b in range(len(slots)):
                if rng.random() < density:
                    mask |= 1 << b
            consider(mask)
            if limit and len(out) >= limit:
                break
    return out
