"""Bitset graph core for the Hadwiger t=7 search.

A graph is (n, adj) where adj is a tuple of n ints; bit u of adj[v] set
iff uv is an edge. No self-loops. All soundness-critical helpers live
here and are exercised by the validation gate.
"""

from __future__ import annotations


def parse_graph6(line: str):
    s = line.strip()
    data = [ord(c) - 63 for c in s]
    if any(d < 0 or d > 63 for d in data):
        raise ValueError(f"bad graph6: {s!r}")
    if data[0] == 63:
        n = (data[1] << 12) | (data[2] << 6) | data[3]
        bits_data = data[4:]
    else:
        n = data[0]
        bits_data = data[1:]
    need = n * (n - 1) // 2
    if len(bits_data) * 6 < need:
        raise ValueError(f"graph6 too short for n={n}: {s!r}")
    adj = [0] * n
    idx = 0
    for j in range(1, n):
        for i in range(j):
            d = bits_data[idx // 6]
            if (d >> (5 - idx % 6)) & 1:
                adj[i] |= 1 << j
                adj[j] |= 1 << i
            idx += 1
    return n, tuple(adj)


def to_graph6(n: int, adj) -> str:
    if n < 63:
        head = [n]
    elif n <= 258047:
        head = [63, (n >> 12) & 63, (n >> 6) & 63, n & 63]
    else:
        raise ValueError("n too large")
    bits = []
    for j in range(1, n):
        for i in range(j):
            bits.append((adj[i] >> j) & 1)
    while len(bits) % 6:
        bits.append(0)
    out = head + [
        (bits[k] << 5) | (bits[k + 1] << 4) | (bits[k + 2] << 3)
        | (bits[k + 3] << 2) | (bits[k + 4] << 1) | bits[k + 5]
        for k in range(0, len(bits), 6)
    ]
    return "".join(chr(63 + d) for d in out)


def edge_count(n: int, adj) -> int:
    return sum(bin(a).count("1") for a in adj) // 2


def edges(n: int, adj):
    return [(u, v) for u in range(n) for v in range(u + 1, n) if (adj[u] >> v) & 1]


def degrees(n: int, adj):
    return [bin(a).count("1") for a in adj]


def complement(n: int, adj):
    full = (1 << n) - 1
    return tuple((full ^ adj[v]) & ~(1 << v) for v in range(n))


def add_edge(adj, u: int, v: int):
    a = list(adj)
    a[u] |= 1 << v
    a[v] |= 1 << u
    return tuple(a)


def remove_edge(adj, u: int, v: int):
    a = list(adj)
    a[u] &= ~(1 << v)
    a[v] &= ~(1 << u)
    return tuple(a)


def connected_component(adj, mask: int, start_bit: int) -> int:
    """Vertices of `mask` reachable from start_bit through vertices in mask."""
    comp = start_bit
    frontier = start_bit
    while frontier:
        nxt = 0
        f = frontier
        while f:
            vb = f & -f
            f ^= vb
            nxt |= adj[vb.bit_length() - 1]
        frontier = nxt & mask & ~comp
        comp |= frontier
    return comp


def is_connected_induced(adj, mask: int) -> bool:
    if mask == 0:
        return False
    return connected_component(adj, mask, mask & -mask) == mask


def is_connected(n: int, adj) -> bool:
    if n == 0:
        return False
    return is_connected_induced(adj, (1 << n) - 1)


def components_of(adj, mask: int):
    comps = []
    rest = mask
    while rest:
        c = connected_component(adj, mask, rest & -rest)
        comps.append(c)
        rest &= ~c
    return comps


def has_dominating_vertex(n: int, adj) -> bool:
    full = (1 << n) - 1
    return any(adj[v] | (1 << v) == full for v in range(n))


def contract(n: int, adj, u: int, v: int):
    """Contract edge/pair (u,v): v merges into u; returns (n-1, adj') with
    vertices relabeled to drop index v."""
    merged = (adj[u] | adj[v]) & ~(1 << u) & ~(1 << v)
    a = list(adj)
    a[u] = merged
    for w in range(n):
        if (merged >> w) & 1:
            a[w] |= 1 << u
        a[w] &= ~(1 << v)
    # drop index v: relabel w>v -> w-1
    out = []
    for w in range(n):
        if w == v:
            continue
        m = a[w]
        low = m & ((1 << v) - 1)
        high = m >> (v + 1)
        out.append(low | (high << v))
    return n - 1, tuple(out)


def induced_subgraph(n: int, adj, mask: int):
    verts = [v for v in range(n) if (mask >> v) & 1]
    idx = {v: i for i, v in enumerate(verts)}
    m = len(verts)
    a = [0] * m
    for v in verts:
        nb = adj[v] & mask
        while nb:
            wb = nb & -nb
            nb ^= wb
            w = wb.bit_length() - 1
            a[idx[v]] |= 1 << idx[w]
    return m, tuple(a)


def relabel_sorted_by_degree(n: int, adj):
    order = sorted(range(n), key=lambda v: -bin(adj[v]).count("1"))
    pos = {v: i for i, v in enumerate(order)}
    a = [0] * n
    for v in range(n):
        for w in range(n):
            if (adj[v] >> w) & 1:
                a[pos[v]] |= 1 << pos[w]
    return tuple(a)


def random_gnp(n: int, p: float, rng):
    adj = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u] |= 1 << v
                adj[v] |= 1 << u
    return tuple(adj)


def random_gnm(n: int, m: int, rng):
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    if m > len(all_pairs):
        raise ValueError("too many edges")
    chosen = rng.sample(all_pairs, m)
    adj = [0] * n
    for u, v in chosen:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return tuple(adj)
