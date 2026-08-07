/*
 * Chvátal's conjecture (1974) — exhaustive counterexample search, reduced form.
 *
 * Conjecture: for every ideal (downward-closed family) F over a finite ground
 * set, the maximum size of an intersecting subfamily of F equals the maximum
 * star size max_x |{A in F : x in A}|.  A counterexample is an ideal whose
 * maximum intersecting subfamily STRICTLY beats every star.
 *
 * Reduction (standard):  if (F, G) is a counterexample, replace G by its
 * up-closure within F (supersets of pairwise-intersecting sets still pairwise
 * intersect, so size only grows), then replace F by the downward closure of G
 * (degrees only shrink).  Hence WLOG a counterexample has the canonical form
 *     A = {A_1..A_m}  an intersecting antichain,   F = downclosure(A),
 *     G = { S : A_i <= S <= A_j for some i,j },
 * where additionally every |A_i| >= 2 (a singleton generator confines G to a
 * star), the A_i have no common element (a common element confines G to a
 * star), and union(A_i) = ground set (unused elements are dropped).
 * Testing all such A on ground sets of size <= n verifies the conjecture for
 * ALL ideals on <= n elements; any violation found is a genuine counterexample.
 *
 * Modes:
 *   ./chv_enum count N            count ALL intersecting antichains of
 *                                 nonempty subsets of [N] (no constraints).
 *                                 Validation anchor: count+1 must equal OEIS
 *                                 A001206(N+1) = 2,4,12,81,2646,1422564,
 *                                 229809982112 for N=1..7.
 *   ./chv_enum test N k [second]  test reduced-form antichains whose minimum
 *                                 generator size is k, with the symmetry
 *                                 normalization first-generator = {1..k}
 *                                 (lowest k bits).  Optional 'second' fixes
 *                                 the second generator mask (work unit for
 *                                 parallel runs).  Every antichain with >=3
 *                                 generators, empty common intersection and
 *                                 full union is tested:  |G| vs max degree.
 *
 * Soundness of the normalization: let k be the minimum generator size of a
 * counterexample antichain; permute so that one minimum-size generator maps to
 * {1..k}.  Every other generator then has size >= k, and every set of size
 * >= k other than {1..k} has mask value > 2^k-1, so the normalized antichain
 * is reachable by this enumeration (first generator {1..k}, all later
 * generators have mask > 2^k-1 and popcount >= k).  k ranges over 2..N-1.
 *
 * Output: one JSON summary line on stdout.  Any violation prints immediately
 * to stdout as "VIOLATION ..." (flushed) — the driver halts everything.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>

#define MAXN 8
#define MAXW 4                     /* bitset words over 2^N masks (<=256 bits) */
#define MAXDEPTH 100               /* max antichain size is C(8,4)=70 */

static int N, NM, ALLMASK, NW;
static uint64_t down_bs[256][MAXW]; /* down_bs[s]: bitset of subsets of s      */
static uint64_t up_bs[256][MAXW];   /* up_bs[s]:   bitset of supersets of s    */
static uint64_t elem_bs[MAXN][MAXW];/* elem_bs[x]: bitset of masks containing x*/

static int gens[MAXDEPTH];
static uint64_t downA[MAXDEPTH + 1][MAXW];
static int and_st[MAXDEPTH + 1], or_st[MAXDEPTH + 1];
static int candbuf[MAXDEPTH][300];

static int mode_test, fixk;
static unsigned long long nodes, tested, ties, violations;
static long long best_excess = LLONG_MIN;
static int best_gens[MAXDEPTH], best_m;

static inline int bs_popcnt(const uint64_t *b) {
    int r = 0;
    for (int w = 0; w < NW; w++) r += __builtin_popcountll(b[w]);
    return r;
}

static void print_antichain(FILE *f, const int *g, int m) {
    fprintf(f, "[");
    for (int i = 0; i < m; i++) {
        fprintf(f, i ? ",{" : "{");
        int first = 1;
        for (int x = 0; x < N; x++)
            if (g[i] >> x & 1) { fprintf(f, first ? "%d" : ",%d", x + 1); first = 0; }
        fprintf(f, "}");
    }
    fprintf(f, "]");
}

static void test_node(int m) {          /* gens[0..m-1] set, downA[m] current */
    tested++;
    uint64_t G[MAXW] = {0};
    for (int i = 0; i < m; i++)
        for (int w = 0; w < NW; w++)
            G[w] |= up_bs[gens[i]][w] & downA[m][w];
    int g = bs_popcnt(G);
    int maxdeg = 0;
    for (int x = 0; x < N; x++) {
        uint64_t t[MAXW]; int d;
        for (int w = 0; w < NW; w++) t[w] = downA[m][w] & elem_bs[x][w];
        d = bs_popcnt(t);
        if (d > maxdeg) maxdeg = d;
    }
    long long ex = (long long)g - maxdeg;
    if (ex > best_excess) {
        best_excess = ex; best_m = m;
        memcpy(best_gens, gens, m * sizeof(int));
    }
    if (ex == 0) ties++;
    if (ex > 0) {
        violations++;
        printf("VIOLATION n=%d |G|=%d maxdeg=%d antichain=", N, g, maxdeg);
        print_antichain(stdout, gens, m);
        printf("\n");
        fflush(stdout);
    }
}

static void dfs(int m, const int *cand, int ncand) {
    for (int ci = 0; ci < ncand; ci++) {
        int s = cand[ci];
        gens[m] = s;
        nodes++;
        for (int w = 0; w < NW; w++) downA[m + 1][w] = downA[m][w] | down_bs[s][w];
        and_st[m + 1] = and_st[m] & s;
        or_st[m + 1]  = or_st[m]  | s;
        if (mode_test && m + 1 >= 3 && and_st[m + 1] == 0 && or_st[m + 1] == ALLMASK)
            test_node(m + 1);
        /* children: later candidates still compatible with s */
        int *nc = candbuf[m], nn = 0;
        for (int cj = ci + 1; cj < ncand; cj++) {
            int t = cand[cj], u = t & s;
            if (u && u != s && u != t) nc[nn++] = t;
        }
        if (nn) dfs(m + 1, nc, nn);
    }
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s count N | %s test N k [second]\n", argv[0], argv[0]); return 2; }
    mode_test = !strcmp(argv[1], "test");
    N = atoi(argv[2]);
    if (N < 1 || N > MAXN) { fprintf(stderr, "bad N\n"); return 2; }
    NM = 1 << N; ALLMASK = NM - 1; NW = (NM + 63) / 64;
    int second = -1;
    if (mode_test) {
        if (argc < 4) { fprintf(stderr, "test mode needs k\n"); return 2; }
        fixk = atoi(argv[3]);
        if (fixk < 2 || fixk > N - 1) { fprintf(stderr, "bad k\n"); return 2; }
        if (argc > 4) second = atoi(argv[4]);
    }

    for (int s = 0; s < NM; s++) {
        int t = s;                          /* subsets of s, including 0 and s */
        for (;;) {
            down_bs[s][t >> 6] |= 1ULL << (t & 63);
            if (!t) break;
            t = (t - 1) & s;
        }
        for (int u = 0; u < NM; u++)
            if ((u & s) == s) up_bs[s][u >> 6] |= 1ULL << (u & 63);
    }
    for (int x = 0; x < N; x++)
        for (int u = 0; u < NM; u++)
            if (u >> x & 1) elem_bs[x][u >> 6] |= 1ULL << (u & 63);

    memset(downA[0], 0, sizeof downA[0]);
    and_st[0] = ALLMASK; or_st[0] = 0;

    if (!mode_test) {                       /* count ALL intersecting antichains */
        int root[300], nr = 0;
        for (int s = 1; s < NM; s++) root[nr++] = s;
        dfs(0, root, nr);
        printf("{\"mode\":\"count\",\"n\":%d,\"antichains_nonempty\":%llu,"
               "\"antichains_total_incl_empty\":%llu}\n", N, nodes, nodes + 1);
        return 0;
    }

    int g0 = (1 << fixk) - 1;               /* first generator = {1..k} */
    gens[0] = g0; nodes = 1;
    for (int w = 0; w < NW; w++) downA[1][w] = down_bs[g0][w];
    and_st[1] = g0; or_st[1] = g0;
    int cand1[300], nc1 = 0;
    for (int t = g0 + 1; t < NM; t++) {
        if (__builtin_popcount(t) < fixk) continue;
        int u = t & g0;
        if (u && u != g0 && u != t) {
            if (second >= 0 && t != second) continue;
            cand1[nc1++] = t;
        }
    }
    if (second >= 0 && nc1 == 0) { fprintf(stderr, "invalid second mask\n"); return 2; }
    if (second >= 0) {                      /* work unit: fix second, free below */
        int s = cand1[0];
        gens[1] = s; nodes++;
        for (int w = 0; w < NW; w++) downA[2][w] = downA[1][w] | down_bs[s][w];
        and_st[2] = and_st[1] & s; or_st[2] = or_st[1] | s;
        int c2[300], n2 = 0;
        for (int t = g0 + 1; t < NM; t++) {
            if (t <= s || __builtin_popcount(t) < fixk) continue;
            int u1 = t & g0, u2 = t & s;
            if (u1 && u1 != g0 && u1 != t && u2 && u2 != s && u2 != t) c2[n2++] = t;
        }
        if (n2) dfs(2, c2, n2);
    } else if (nc1) {
        dfs(1, cand1, nc1);
    }

    printf("{\"mode\":\"test\",\"n\":%d,\"k\":%d,\"second\":%d,\"nodes\":%llu,"
           "\"tested\":%llu,\"ties\":%llu,\"violations\":%llu,\"best_excess\":%lld,"
           "\"best_example\":\"", N, fixk, second, nodes, tested, ties, violations,
           tested ? best_excess : -9999);
    if (tested) print_antichain(stdout, best_gens, best_m);
    printf("\"}\n");
    return violations ? 17 : 0;
}
