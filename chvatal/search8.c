/*
 * Stochastic counterexample hunt for Chvátal's conjecture on [N] (N<=8).
 *
 * State: an intersecting antichain A (pairwise intersecting, pairwise
 * incomparable, sizes >= 2).  Objective (exact, recomputed per move):
 *     excess = |G| - maxdeg,   G = {S : A_i <= S <= A_j},  over F = down(A);
 * score = excess*1024 + |G|  (ties broken toward denser G).  Any state with
 * excess > 0 is a genuine counterexample (prints + exit 17 immediately).
 *
 * Simulated annealing with geometric cooling, periodic reheats, and random
 * restarts.  Deterministic xorshift RNG seeded from --seed (reproducible).
 *
 * usage: ./search8 N steps seed [report_every]
 * Output: improvement lines + final JSON summary.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define MAXW 4
#define MAXM 72

static int N, NM, ALLMASK, NW;
static uint64_t down_bs[256][MAXW], up_bs[256][MAXW], elem_bs[8][MAXW];

static int gens[MAXM], m;

static uint64_t rng_s;
static inline uint64_t rnd(void) {
    rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17;
    return rng_s;
}

static inline int bs_popcnt(const uint64_t *b) {
    int r = 0;
    for (int w = 0; w < NW; w++) r += __builtin_popcountll(b[w]);
    return r;
}

/* exact objective for gens[0..m-1]; returns excess, sets *gsize */
static int eval(int *gsize) {
    uint64_t dA[MAXW] = {0}, G[MAXW] = {0};
    for (int i = 0; i < m; i++)
        for (int w = 0; w < NW; w++) dA[w] |= down_bs[gens[i]][w];
    for (int i = 0; i < m; i++)
        for (int w = 0; w < NW; w++) G[w] |= up_bs[gens[i]][w] & dA[w];
    int g = bs_popcnt(G), maxdeg = 0;
    for (int x = 0; x < N; x++) {
        uint64_t t[MAXW];
        for (int w = 0; w < NW; w++) t[w] = dA[w] & elem_bs[x][w];
        int d = bs_popcnt(t);
        if (d > maxdeg) maxdeg = d;
    }
    *gsize = g;
    return g - maxdeg;
}

static int compatible(int t) {              /* can t join the antichain? */
    if (__builtin_popcount(t) < 2) return 0;
    for (int i = 0; i < m; i++) {
        int u = t & gens[i];
        if (!u || u == t || u == gens[i]) return 0;
    }
    return 1;
}

static void print_state(FILE *f, const int *g, int mm) {
    fprintf(f, "[");
    for (int i = 0; i < mm; i++) {
        fprintf(f, i ? ",{" : "{");
        int first = 1;
        for (int x = 0; x < N; x++)
            if (g[i] >> x & 1) { fprintf(f, first ? "%d" : ",%d", x + 1); first = 0; }
        fprintf(f, "}");
    }
    fprintf(f, "]");
}

static void random_restart(void) {
    m = 0;
    int target = 3 + (int)(rnd() % 6);      /* 3..8 seed generators */
    for (int tries = 0; tries < 4000 && m < target; tries++) {
        int sz = 3 + (int)(rnd() % 3);      /* sizes 3..5: star-distant zone */
        int t = 0;
        while (__builtin_popcount(t) < sz) t |= 1 << (rnd() % N);
        if (compatible(t)) gens[m++] = t;
    }
    if (m == 0) { gens[0] = 7; m = 1; }     /* fallback {1,2,3} */
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s N steps seed [report_every]\n", argv[0]); return 2; }
    N = atoi(argv[1]);
    long long steps = atoll(argv[2]);
    rng_s = (uint64_t)atoll(argv[3]) * 2654435761u + 88172645463325252ull;
    long long report_every = argc > 4 ? atoll(argv[4]) : 0;
    if (N < 3 || N > 8) { fprintf(stderr, "bad N\n"); return 2; }
    NM = 1 << N; ALLMASK = NM - 1; NW = (NM + 63) / 64;

    for (int s = 0; s < NM; s++) {
        int t = s;
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

    random_restart();
    int gsize;
    int cur_ex = eval(&gsize);
    long long cur_score = (long long)cur_ex * 1024 + gsize;
    long long best_score = cur_score;
    int best_ex = cur_ex, best_g = gsize, best_gens[MAXM], best_m = m;
    memcpy(best_gens, gens, sizeof gens);
    double T = 8.0;
    long long since_best = 0;

    for (long long it = 0; it < steps; it++) {
        T *= 0.999999;
        if (++since_best > 400000) {        /* stuck: reheat or restart */
            if (rnd() & 1) T = 8.0;
            else { random_restart(); cur_ex = eval(&gsize); cur_score = (long long)cur_ex * 1024 + gsize; }
            since_best = 0;
        }
        int save_gens[MAXM], save_m = m;
        memcpy(save_gens, gens, m * sizeof(int));
        unsigned op = rnd() % 3;
        if (op == 0 || m >= MAXM - 2) {     /* remove */
            if (m > 1) { int i = rnd() % m; gens[i] = gens[m - 1]; m--; }
        } else if (op == 1) {               /* add */
            int done = 0;
            for (int tr = 0; tr < 30 && !done; tr++) {
                int t = 1 + (int)(rnd() % (NM - 1));
                if (compatible(t)) { gens[m++] = t; done = 1; }
            }
        } else {                            /* replace */
            if (m > 0) {
                int i = rnd() % m;
                int old = gens[i];
                gens[i] = gens[m - 1]; m--;
                int done = 0;
                for (int tr = 0; tr < 30 && !done; tr++) {
                    int t = 1 + (int)(rnd() % (NM - 1));
                    if (t != old && compatible(t)) { gens[m++] = t; done = 1; }
                }
                if (!done) { gens[m++] = old; }
            }
        }
        int ex = eval(&gsize);
        long long sc = (long long)ex * 1024 + gsize;
        long long d = sc - cur_score;
        if (d >= 0 || exp((double)d / (T * 64.0)) * 18446744073709551616.0 > (double)rnd()) {
            cur_score = sc; cur_ex = ex;
            if (sc > best_score) {
                best_score = sc; best_ex = ex; best_g = gsize;
                best_m = m; memcpy(best_gens, gens, m * sizeof(int));
                since_best = 0;
                if (ex > 0) {
                    printf("VIOLATION n=%d excess=%d |G|=%d antichain=", N, ex, gsize);
                    print_state(stdout, gens, m);
                    printf("\n");
                    fflush(stdout);
                    return 17;
                }
            }
        } else {
            memcpy(gens, save_gens, save_m * sizeof(int));
            m = save_m;
        }
        if (report_every && it % report_every == 0) {
            printf("it=%lld best_excess=%d best_G=%d cur_excess=%d m=%d T=%.3f\n",
                   it, best_ex, best_g, cur_ex, m, T);
            fflush(stdout);
        }
    }
    printf("{\"mode\":\"anneal\",\"n\":%d,\"steps\":%lld,\"best_excess\":%d,"
           "\"best_G\":%d,\"best_example\":\"", N, steps, best_ex, best_g);
    print_state(stdout, best_gens, best_m);
    printf("\"}\n");
    return 0;
}
