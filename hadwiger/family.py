"""Graph constructions: validation controls and structgen candidate families."""

from __future__ import annotations

from itertools import combinations

from .graphs import add_edge, complement, random_gnm


def complete(n):
    full = (1 << n) - 1
    return n, tuple((full ^ (1 << v)) for v in range(n))


def complete_minus_perfect_matching(n):
    assert n % 2 == 0
    _, adj = complete(n)
    a = list(adj)
    for i in range(0, n, 2):
        a[i] &= ~(1 << (i + 1))
        a[i + 1] &= ~(1 << i)
    return n, tuple(a)


def cycle(n):
    adj = [0] * n
    for i in range(n):
        adj[i] |= 1 << ((i + 1) % n)
        adj[(i + 1) % n] |= 1 << i
    return n, tuple(adj)


def petersen():
    return kneser(5, 2)


def kneser(m, k):
    verts = list(combinations(range(m), k))
    n = len(verts)
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if not set(verts[i]) & set(verts[j]):
                adj[i] |= 1 << j
                adj[j] |= 1 << i
    return n, tuple(adj)


def schrijver(m, k):
    """Stable k-subsets of Z_m (no two cyclically consecutive elements)."""
    def stable(s):
        st = set(s)
        return all((x + 1) % m not in st for x in st)

    verts = [s for s in combinations(range(m), k) if stable(s)]
    n = len(verts)
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if not set(verts[i]) & set(verts[j]):
                adj[i] |= 1 << j
                adj[j] |= 1 << i
    return n, tuple(adj)


def mycielski(n, adj):
    """Classic Mycielskian: vertices 0..n-1 original, n..2n-1 shadows, 2n apex."""
    nn = 2 * n + 1
    a = [0] * nn
    for v in range(n):
        nb = adj[v]
        while nb:
            wb = nb & -nb
            nb ^= wb
            w = wb.bit_length() - 1
            if w > v:
                for (p, q) in ((v, w), (v, n + w), (n + v, w)):
                    a[p] |= 1 << q
                    a[q] |= 1 << p
    for v in range(n):
        a[n + v] |= 1 << (2 * n)
        a[2 * n] |= 1 << (n + v)
    return nn, tuple(a)


def gen_mycielski(n, adj, r):
    """Generalized Mycielskian M_r: r levels + apex. r=1 is a cone-ish level."""
    nn = r * n + 1
    apex = nn - 1
    a = [0] * nn
    edge_list = [(u, v) for u in range(n) for v in range(u + 1, n)
                 if (adj[u] >> v) & 1]
    for (u, v) in edge_list:
        a[u] |= 1 << v
        a[v] |= 1 << u
    for lvl in range(1, r):
        for (u, v) in edge_list:
            for (p, q) in (((lvl - 1) * n + u, lvl * n + v),
                           ((lvl - 1) * n + v, lvl * n + u)):
                a[p] |= 1 << q
                a[q] |= 1 << p
    top = r - 1
    for v in range(n):
        a[top * n + v] |= 1 << apex
        a[apex] |= 1 << (top * n + v)
    return nn, tuple(a)


def subdivide_all_edges(n, adj):
    """1-subdivision: one new vertex per edge."""
    edge_list = [(u, v) for u in range(n) for v in range(u + 1, n)
                 if (adj[u] >> v) & 1]
    nn = n + len(edge_list)
    a = [0] * nn
    for idx, (u, v) in enumerate(edge_list):
        w = n + idx
        a[u] |= 1 << w
        a[w] |= 1 << u
        a[v] |= 1 << w
        a[w] |= 1 << v
    return nn, tuple(a)


def random_5tree(n, rng):
    """5-tree: K6 seed; each new vertex joined to a random existing 5-clique.
    Treewidth 5 => certifiably K7-minor-free; e = 5n-15 (Mader extremal)."""
    assert n >= 6
    adj = [0] * n
    for u in range(6):
        for v in range(u + 1, 6):
            adj[u] |= 1 << v
            adj[v] |= 1 << u
    cliques5 = [tuple(c) for c in combinations(range(6), 5)]
    for w in range(6, n):
        c = rng.choice(cliques5)
        for u in c:
            adj[u] |= 1 << w
            adj[w] |= 1 << u
        for drop in c:
            newc = tuple(sorted((set(c) - {drop}) | {w}))
            cliques5.append(newc)
    return n, tuple(adj)


def planar_grid_2apex(rows, cols, rng):
    """Triangulated grid (planar, K5-minor-free) + 2 apexes => K7-minor-free."""
    n = rows * cols + 2
    adj = [0] * n

    def vid(r, c):
        return r * cols + c

    def link(u, v):
        adj[u] |= 1 << v
        adj[v] |= 1 << u

    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                link(vid(r, c), vid(r, c + 1))
            if r + 1 < rows:
                link(vid(r, c), vid(r + 1, c))
            if r + 1 < rows and c + 1 < cols:
                if rng.random() < 0.5:
                    link(vid(r, c), vid(r + 1, c + 1))
                else:
                    link(vid(r, c + 1), vid(r + 1, c))
    a1, a2 = n - 2, n - 1
    for v in range(rows * cols):
        link(a1, v)
        link(a2, v)
    link(a1, a2)
    return n, tuple(adj)


def mader_supercritical(n, rng):
    """Random graph with exactly 5n-14 edges: guaranteed K7 minor (Mader)."""
    return n, random_gnm(n, 5 * n - 14, rng)


def proj_quad(m, k, close_bottom=True):
    """Quadrangular (2m x k) cylinder grid with antipodal top identification
    (a crosscap): the Youngs-style odd-quadrangulation chromatic mechanism.
    Vertices (i,j), i in Z_{2m}, j in 0..k-1; top row i identified with i+m."""
    width = 2 * m
    ids = {}
    n = 0
    for j in range(k):
        for i in range(width):
            key = (i, j)
            if j == k - 1 and i >= m:
                key = (i - m, j)
            if key not in ids:
                ids[key] = n
                n += 1
    adj = [0] * n

    def vid(i, j):
        i %= width
        key = (i, j)
        if j == k - 1 and i >= m:
            key = (i - m, j)
        return ids[key]

    def link(u, v):
        if u != v:
            adj[u] |= 1 << v
            adj[v] |= 1 << u

    for j in range(k):
        for i in range(width):
            link(vid(i, j), vid(i + 1, j))
            if j + 1 < k:
                link(vid(i, j), vid(i, j + 1))
    if close_bottom:
        # close the bottom boundary cycle with antipodal chords too
        for i in range(m):
            link(vid(i, 0), vid(i + m, 0))
    return n, tuple(adj)


def apex_to_subset(n, adj, mask):
    a = list(adj) + [0]
    w = n
    m = mask
    while m:
        vb = m & -m
        m ^= vb
        v = vb.bit_length() - 1
        a[v] |= 1 << w
        a[w] |= 1 << v
    return n + 1, tuple(a)
