"""Exact linear algebra over the rationals (stdlib fractions only)."""
from fractions import Fraction


def mat_rank(rows):
    """Rank of a matrix given as a list of rows of Fractions/ints."""
    M = [[Fraction(x) for x in r] for r in rows]
    if not M:
        return 0
    nr, nc = len(M), len(M[0])
    rank = 0
    for col in range(nc):
        piv = None
        for r in range(rank, nr):
            if M[r][col] != 0:
                piv = r
                break
        if piv is None:
            continue
        M[rank], M[piv] = M[piv], M[rank]
        prow = M[rank]
        pval = prow[col]
        for r in range(nr):
            if r != rank and M[r][col] != 0:
                fac = M[r][col] / pval
                M[r] = [a - fac * b for a, b in zip(M[r], prow)]
        rank += 1
        if rank == min(nr, nc):
            break
    return rank


def solve(rows, rhs):
    """Solve the square system M x = rhs exactly; return None if singular."""
    n = len(rows)
    A = [[Fraction(x) for x in rows[i]] + [Fraction(rhs[i])] for i in range(n)]
    for c in range(n):
        piv = None
        for r in range(c, n):
            if A[r][c] != 0:
                piv = r
                break
        if piv is None:
            return None
        A[c], A[piv] = A[piv], A[c]
        pval = A[c][c]
        A[c] = [a / pval for a in A[c]]
        prow = A[c]
        for r in range(n):
            if r != c and A[r][c] != 0:
                fac = A[r][c]
                A[r] = [a - fac * b for a, b in zip(A[r], prow)]
    return [A[i][n] for i in range(n)]


def det(rows):
    """Exact determinant of a square matrix."""
    n = len(rows)
    A = [[Fraction(x) for x in r] for r in rows]
    dv = Fraction(1)
    for c in range(n):
        piv = None
        for r in range(c, n):
            if A[r][c] != 0:
                piv = r
                break
        if piv is None:
            return Fraction(0)
        if piv != c:
            A[c], A[piv] = A[piv], A[c]
            dv = -dv
        pval = A[c][c]
        dv *= pval
        prow = A[c]
        for r in range(c + 1, n):
            if A[r][c] != 0:
                fac = A[r][c] / pval
                A[r] = [a - fac * b for a, b in zip(A[r], prow)]
    return dv


def dot(u, v):
    return sum((a * b for a, b in zip(u, v)), Fraction(0))
