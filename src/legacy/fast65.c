/*
 * pomerance_search.c
 *
 * Fast search for Pomerance triples (p, A, x0): the shortest known primality
 * certificates, defined for the DANGER3 data challenge
 * (https://github.com/AndrewVSutherland/DANGER3).
 *
 * A Pomerance triple exhibits a point of exact order 2^k on the Montgomery
 * curve B*y^2 = x^3 + A*x^2 + x over Z/pZ, where k is the least integer with
 * 2^k > floor(sqrt(p)) + 1 + 2*sqrt(floor(sqrt(p))). Such a point is larger
 * than the Hasse bound allows for any prime factor of p up to sqrt(p), so its
 * existence proves p prime. Verification takes k doublings; the work is in
 * the search, since a random curve admits such a point with probability on
 * the order of 1/sqrt(p).
 *
 * This program implements the X1(16) prescribed-torsion search (the method
 * behind the 10^22 through 10^26 challenge records) with three improvements:
 *
 *   1. Reciprocal-mark deduplication. The X1(16) sampler derives each curve
 *      from a quadratic equation whose two roots yield two marked points on
 *      the same curve. The two marks have reciprocal x-coordinates, and
 *      x -> 1/x is translation by the rational 2-torsion point (0,0); on the
 *      nonsplit family the rational 2-power subgroup is cyclic, so the two
 *      marks always have equal halving depth and return the same verdict.
 *      Testing both is therefore pure double work. This implementation keeps
 *      one mark per curve, an exact halving of the dominant cost.
 *
 *   2. Centered 65-bit Barrett arithmetic. For 2^64 < p < 2^65, residues
 *      centered in (-p/2, p/2] have 64-bit magnitude, so a modular product
 *      needs one 64x64 multiply and a one-step Barrett correction. All field
 *      arithmetic here runs on plain residues through this multiplier; no
 *      Montgomery-domain conversions are needed anywhere.
 *
 *   3. Batched inversions. Marked points are produced in batches of 32
 *      curves whose 2 inversions each are combined into a single field
 *      inversion via Montgomery's product trick.
 *
 * Scope: this implementation requires 2^64 < p < 2^65 and p mod 8 in
 * {3, 5, 7}. The residue restriction is inherited from the square-root
 * formulas of the X1(16) method (p = 1 mod 8 has no fast square root here);
 * the size restriction is where the centered multiplier applies and where
 * this program's speedup over the prior state of the art was measured.
 *
 * Usage:
 *     ./pomerance_search <p> [seed] [max_curves]
 *
 * On success, prints exactly one line to stdout:
 *     <A> <x0> <curves_tested>
 * and exits 0. Progress and diagnostics go to stderr. Exits 1 if the curve
 * budget is exhausted, 2 on unsupported input.
 *
 * Provenance: derived from the MIT-licensed record implementations by
 * Fabian Ruehle (2-Sylow projection base), Alexa McLain (X1(16) sampler,
 * successive halving, nonsplit filter), and Jane Shi (p = 3 mod 4 square
 * roots), with the three improvements above contributed by an autonomous
 * GPT-5.6 research session in the pomerance-bench environment (2026).
 * See README.md and LICENSE.
 *
 * Build:
 *     gcc -O3 -fopenmp -o pomerance_search pomerance_search.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

typedef uint64_t u64;
typedef __uint128_t u128;

/* ------------------------------------------------------------------------
 * Small utilities: 128-bit decimal I/O, PRNG, wall clock
 * ------------------------------------------------------------------------ */

static u128 parse128(const char *s) {
    u128 v = 0;
    while (*s >= '0' && *s <= '9') v = v * 10 + (u128)(*s++ - '0');
    return v;
}

static void sprint128(char *buf, u128 v) {
    char tmp[50];
    int i = 49;
    tmp[i] = '\0';
    if (v == 0) { strcpy(buf, "0"); return; }
    while (v > 0) { tmp[--i] = (char)('0' + (int)(v % 10)); v /= 10; }
    strcpy(buf, tmp + i);
}

/* xorshift128+, seeded per thread. */
typedef struct { u64 s0, s1; } Rng;

static inline u64 rng64(Rng *r) {
    u64 s1 = r->s0, s0 = r->s1;
    r->s0 = s0;
    s1 ^= s1 << 23;
    r->s1 = s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26);
    return r->s1 + s0;
}

/* Uniform value below p by masked rejection (p has 65 bits here). */
static inline u128 rand_below(Rng *rng, u128 p, u128 mask) {
    for (;;) {
        u128 v = (((u128)rng64(rng) << 64) | (u128)rng64(rng)) & mask;
        if (v < p) return v;
    }
}

static double now_sec(void) {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

/* ------------------------------------------------------------------------
 * Field arithmetic for 2^64 < p < 2^65: centered residues + Barrett
 *
 * A residue r in [0, p) is centered by replacing r > p/2 with p - r and a
 * sign flip. Centered magnitudes are below p/2 < 2^64, so they fit one
 * 64-bit word and their product fits u128. With mu = floor(2^128 / p),
 * the quotient estimate floor(x * mu / 2^128) computed from the two high
 * 64x64 partial products is off by at most one for x < p^2/4, so a single
 * conditional subtraction completes the reduction.
 * ------------------------------------------------------------------------ */

typedef struct {
    u128 p;
    u64 recip64;   /* floor(2^128 / p); note 2^128 / p < 2^64 since p > 2^64 */
} Field;

static void field_init(Field *f, u128 p) {
    f->p = p;
    f->recip64 = (u64)(((u128)-1) / p);
}

static inline u128 addmod(u128 a, u128 b, u128 p) {
    u128 s = a + b;
    return s >= p ? s - p : s;
}

static inline u128 submod(u128 a, u128 b, u128 p) {
    return a >= b ? a - b : p - b + a;
}

/* Modular product of plain residues a, b in [0, p). */
static inline u128 mulmod(u128 a, u128 b, const Field *f) {
    const u128 p = f->p;
    const u128 half = p >> 1;
    int neg = 0;
    u64 aa, bb;
    if (a > half) { aa = (u64)(p - a); neg ^= 1; } else aa = (u64)a;
    if (b > half) { bb = (u64)(p - b); neg ^= 1; } else bb = (u64)b;

    u128 x = (u128)aa * bb;                      /* x < p^2/4 < 2^128 */
    u64 xl = (u64)x, xh = (u64)(x >> 64);
    u64 mu = f->recip64;
    u128 cross = (u128)xh * mu + (((u128)xl * mu) >> 64);
    u64 q = (u64)(cross >> 64);                  /* q = floor(x/p) or one less */
    u128 qp = (u128)q * (u64)p + ((u128)q << 64);  /* q * p, using p = 2^64 + low(p) */
    u128 r = x - qp;
    if (r >= p) r -= p;
    if (neg && r) r = p - r;
    return r;
}

static u128 powmod(u128 a, u128 e, const Field *f) {
    u128 r = 1, b = a % f->p;
    while (e > 0) {
        if (e & 1) r = mulmod(r, b, f);
        b = mulmod(b, b, f);
        e >>= 1;
    }
    return r;
}

/*
 * Square root mod p for the two supported residue classes.
 *
 *   p = 3 mod 4: sqrt(n) = n^((p+1)/4) whenever n is a quadratic residue.
 *   p = 5 mod 8: sqrt(n) = n^((p+3)/8), corrected by sqrt(-1) when the
 *                first candidate squares to -n. sqrtm1 = 2^((p-1)/4) is
 *                a square root of -1 because 2 is a nonresidue mod such p.
 *
 * Returns 1 and writes the root when n is a residue, else returns 0. The
 * failure branch doubles as the algorithm's nonresidue test: it costs the
 * same as a Legendre symbol computed by Euler's criterion, and the halving
 * search needs the root anyway whenever the test would pass.
 */
static int sqrtmod(u128 *root, u128 n, u128 p, u128 sqrtm1, const Field *f) {
    n %= p;
    if (n == 0) { *root = 0; return 1; }
    if ((p & 3) == 3) {
        u128 x = powmod(n, (p + 1) >> 2, f);
        if (mulmod(x, x, f) == n) { *root = x; return 1; }
        return 0;
    }
    u128 x = powmod(n, (p + 3) >> 3, f);
    if (mulmod(x, x, f) == n) { *root = x; return 1; }
    x = mulmod(x, sqrtm1, f);
    if (mulmod(x, x, f) == n) { *root = x; return 1; }
    return 0;
}

static u128 invert(u128 a, u128 p, const Field *f) {
    return powmod(a, p - 2, f);
}

/*
 * Montgomery's product trick: invert n values with a single field inversion
 * plus 3(n-1) multiplications. Returns 0 if any input is zero.
 */
static int invert_batch(u128 *out, const u128 *vals, int n, u128 p, const Field *f) {
    u128 prefix[128];
    if (n < 1 || n >= (int)(sizeof(prefix) / sizeof(prefix[0]))) return 0;
    prefix[0] = 1;
    for (int i = 0; i < n; i++) {
        if (vals[i] == 0) return 0;
        prefix[i + 1] = mulmod(prefix[i], vals[i], f);
    }
    u128 acc = invert(prefix[n], p, f);
    for (int i = n - 1; i >= 0; i--) {
        out[i] = mulmod(acc, prefix[i], f);
        acc = mulmod(acc, vals[i], f);
    }
    return 1;
}

/* ------------------------------------------------------------------------
 * Independent verification
 *
 * Runs the definition directly: k projective doublings of (x0 : 1) with the
 * generic doubling law, requiring Z nonzero before step k and zero at step
 * k. Uses schoolbook shift-and-add multiplication only, so a bug in the
 * fast multiplier above cannot produce a false positive.
 * ------------------------------------------------------------------------ */

static u128 mulmod_schoolbook(u128 a, u128 b, u128 p) {
    u128 r = 0;
    a %= p;
    b %= p;
    while (b > 0) {
        if (b & 1) r = addmod(r, a, p);
        a = addmod(a, a, p);
        b >>= 1;
    }
    return r;
}

static int compute_k(u128 p) {
    u64 q = (u64)sqrtl((long double)p);
    while ((u128)(q + 1) * (q + 1) <= p) q++;
    while ((u128)q * q > p) q--;
    u64 sq = (u64)sqrtl((long double)q);
    while ((u64)(sq + 1) * (sq + 1) <= q) sq++;
    while ((u64)sq * sq > q) sq--;
    u64 bound = q + 1 + 2 * sq;
    int k = 0;
    u64 v = 1;
    while (v <= bound) { k++; v <<= 1; }
    return k;
}

static int verify_triple(u128 p, u128 A, u128 x0) {
    int k = compute_k(p);
    if (A % p == 2 || A % p == p - 2) return 0;
    u128 X = x0 % p, Z = 1;
    for (int i = 1; i <= k; i++) {
        u128 X2 = mulmod_schoolbook(X, X, p);
        u128 Z2 = mulmod_schoolbook(Z, Z, p);
        u128 XZ = mulmod_schoolbook(X, Z, p);
        u128 d = submod(X2, Z2, p);
        u128 Xn = mulmod_schoolbook(d, d, p);
        u128 inner = addmod(addmod(X2, mulmod_schoolbook(A, XZ, p), p), Z2, p);
        u128 fourXZ = addmod(addmod(XZ, XZ, p), addmod(XZ, XZ, p), p);
        u128 Zn = mulmod_schoolbook(fourXZ, inner, p);
        X = Xn;
        Z = Zn;
        if (i < k && Z == 0) return 0;
        if (i == k && Z != 0) return 0;
    }
    return 1;
}

/* ------------------------------------------------------------------------
 * The X1(16) nonsplit sampler with reciprocal-mark deduplication
 *
 * Points of order 16 on Montgomery curves are parameterized by the modular
 * curve X1(16) in Tate normal form: a random field element y determines a
 * curve coefficient A(y) (a degree-8 rational function) together with the
 * x-coordinate of a point of order 16 on that curve. Sampling y instead of
 * A restricts the search to curves whose order is divisible by 16, which
 * multiplies the hit rate by roughly 16 at negligible cost.
 *
 * For each y, the marked x-coordinate satisfies the quadratic
 *     qa * x^2 + qb * x + qc = 0,
 *     qa = y^2 - 2y,  qb = 2y^2 - y^3,  qc = 1 - y,
 * solved by one square root of the discriminant. The prior state of the art
 * emitted both roots as two candidates and ran the full halving test on
 * each. The two roots are reciprocal points differing by the 2-torsion
 * point (0,0), so on this family (cyclic rational 2-power subgroup) they
 * have identical halving depth: testing the second mark can never change
 * the curve's verdict. Only the first root is kept here.
 *
 * The nonsplit filter: the curve's rational 2-power subgroup is cyclic
 * exactly when (y^2 - 2)(y^2 - 4y + 2) is a nonresidue. Cyclic 2-power
 * torsion concentrates the halving survivors, and the filter also makes
 * the depth equality above hold. The nonresidue test is expressed as a
 * failed square root, which shares its code with the halving steps.
 *
 * Marked points are converted to the Montgomery x-line projectively:
 * with root numerator r = sqrt(D) - qb over the denominator 2*qa, the
 * marked coordinate is
 *     xP = x / (x - y) = r / (r - 2*qa*y),
 * so only two inversions are needed per curve: the coefficient denominator
 * 4*(y-1)^4 and the marked-point denominator. Batches of 32 curves share
 * one field inversion across all 64 denominators.
 * ------------------------------------------------------------------------ */

/* Numerator of A(y): y^8 - 8y^7 + 24y^6 - 32y^5 + 8y^4 + 32y^3 - 48y^2 + 32y - 8,
 * evaluated by Horner's rule. */
static u128 x16_A_numerator(u128 y, u128 p, const Field *f) {
    u128 num = 1;
    num = submod(mulmod(num, y, f), 8, p);
    num = addmod(mulmod(num, y, f), 24, p);
    num = submod(mulmod(num, y, f), 32, p);
    num = addmod(mulmod(num, y, f), 8, p);
    num = addmod(mulmod(num, y, f), 32, p);
    num = submod(mulmod(num, y, f), 48, p);
    num = addmod(mulmod(num, y, f), 32, p);
    num = submod(mulmod(num, y, f), 8, p);
    return num;
}

static int x16_y_is_nonsplit(u128 p, u128 y, u128 y2, u128 sqrtm1, const Field *f) {
    u128 f1 = submod(y2, 2, p);
    u128 four_y = addmod(addmod(y, y, p), addmod(y, y, p), p);
    u128 f2 = addmod(submod(y2, four_y, p), 2, p);
    u128 prod = mulmod(f1, f2, f);
    u128 root;
    if (prod == 0) return 0;
    return !sqrtmod(&root, prod, p, sqrtm1, f);
}

/*
 * Fill up to cap (curve, marked point) pairs, one marked point per curve.
 * Returns the number produced (normally cap; 0 only if the shared batch
 * inversion hits a zero, which callers treat as a retry).
 */
#define BATCH_CURVES 32

static int x16_fill_batch(u128 *As, u128 *xPs, int cap, Rng *rng,
                          u128 p, u128 rand_mask, u128 sqrtm1, const Field *f) {
    u128 ys[BATCH_CURVES], rnums[BATCH_CURVES];
    u128 vals[2 * BATCH_CURVES], invs[2 * BATCH_CURVES];
    int ny = 0;
    if (cap > BATCH_CURVES) cap = BATCH_CURVES;

    while (ny < cap) {
        u128 y = rand_below(rng, p, rand_mask);
        if (y == 0) continue;
        u128 y2 = mulmod(y, y, f);
        if (!x16_y_is_nonsplit(p, y, y2, sqrtm1, f)) continue;

        u128 y3 = mulmod(y2, y, f);
        u128 qa = submod(y2, addmod(y, y, p), p);
        if (qa == 0) continue;
        u128 qb = submod(addmod(y2, y2, p), y3, p);
        u128 qc = submod(1, y, p);
        u128 D = submod(mulmod(qb, qb, f),
                        mulmod(addmod(qa, qa, p), addmod(qc, qc, p), f), p);
        u128 sd;
        if (!sqrtmod(&sd, D, p, sqrtm1, f)) continue;

        /* One root only: the reciprocal-mark cut. */
        u128 rnum = submod(sd, qb, p);
        u128 twoqa_y = mulmod(addmod(qa, qa, p), y, f);
        u128 mark_den = submod(rnum, twoqa_y, p);

        u128 ym1 = submod(y, 1, p);
        u128 ym1_2 = mulmod(ym1, ym1, f);
        u128 coef_den = mulmod(4, mulmod(ym1_2, ym1_2, f), f);
        if (coef_den == 0 || mark_den == 0) continue;

        ys[ny] = y;
        rnums[ny] = rnum;
        vals[2 * ny] = coef_den;
        vals[2 * ny + 1] = mark_den;
        ny++;
    }

    if (!invert_batch(invs, vals, 2 * ny, p, f)) return 0;

    int nout = 0;
    for (int i = 0; i < ny; i++) {
        u128 A = mulmod(x16_A_numerator(ys[i], p, f), invs[2 * i], f);
        if (A <= 2 || A >= p - 2) continue;
        u128 xP = mulmod(rnums[i], invs[2 * i + 1], f);
        if (xP == 0) continue;
        As[nout] = A;
        xPs[nout++] = xP;
    }
    return nout;
}

/* ------------------------------------------------------------------------
 * Successive halving
 *
 * The sampler hands over a point of order 16, i.e. 2-power depth 4. One
 * halving step lifts depth d to d + 1 by solving [2]Q' = Q on the x-line:
 * with x = x(Q), first take sd = sqrt(x^2 + A*x + 1); the halved
 * x-coordinates then satisfy a second quadratic solved by
 * sw = sqrt(u^2 - 4) with u = 2x + 2*sd, giving x(Q') = (u +/- sw) / 2.
 * If the first sd sign yields no residue, the other sign is tried. A
 * failed step proves the point cannot be halved: the curve is abandoned.
 * Reaching depth k yields a candidate of order dividing 2^k which the
 * independent verifier then checks for exactness.
 *
 * Each step costs 2 to 3 square roots and succeeds with probability about
 * one half, so an average curve dies after a couple of steps and the full
 * k-deep climbs concentrate on the curves that matter.
 * ------------------------------------------------------------------------ */

static int halve_once(u128 *xo, u128 p, u128 A, u128 x, u128 sqrtm1, const Field *f) {
    const u128 inv2 = (p + 1) >> 1;   /* (p+1)/2 = 1/2 mod p */
    u128 x2 = mulmod(x, x, f);
    u128 d = addmod(addmod(x2, mulmod(A, x, f), p), 1, p);
    u128 sd;
    if (!sqrtmod(&sd, d, p, sqrtm1, f)) return 0;

    u128 signs[2] = { sd, submod(0, sd, p) };
    for (int i = 0; i < 2; i++) {
        u128 u = addmod(addmod(x, x, p), addmod(signs[i], signs[i], p), p);
        u128 w = submod(mulmod(u, u, f), 4, p);
        u128 sw;
        if (!sqrtmod(&sw, w, p, sqrtm1, f)) continue;
        u128 cand[2] = {
            mulmod(addmod(u, sw, p), inv2, f),
            mulmod(submod(u, sw, p), inv2, f)
        };
        for (int j = 0; j < 2; j++) {
            if (cand[j] != 0) { *xo = cand[j]; return 1; }
        }
    }
    return 0;
}

static int halve_chain(u128 *xout, u128 p, u128 A, u128 x, int depth, int k,
                       u128 sqrtm1, const Field *f) {
    for (; depth < k; depth++) {
        if (!halve_once(&x, p, A, x, sqrtm1, f)) return 0;
    }
    if (!verify_triple(p, A, x)) return 0;
    *xout = x;
    return 1;
}

/* ------------------------------------------------------------------------
 * Search driver
 * ------------------------------------------------------------------------ */

static int search(u128 p, u64 seed, u64 max_curves,
                  u128 *out_A, u128 *out_x0, u64 *out_curves) {
    int k = compute_k(p);
    Field f;
    field_init(&f, p);

    int pbits = 65;
    u128 rand_mask = ((u128)1 << pbits) - 1;
    u128 sqrtm1 = 0;
    if ((u64)(p & 7) == 5) sqrtm1 = powmod(2, (p - 1) >> 2, &f);

    volatile int found = 0;
    u128 found_A = 0, found_x0 = 0;
    u64 thread_curves[256] = {0};
    double t0 = now_sec();

#pragma omp parallel
    {
        int tid = 0, nthr = 1;
#ifdef _OPENMP
        tid = omp_get_thread_num();
        nthr = omp_get_num_threads();
#endif
        Rng rng = {
            .s0 = 7364529176530163ULL ^ ((u64)tid * 6364136223846793005ULL) ^ seed,
            .s1 = 1442695040888963407ULL ^ ((u64)(tid + 1) * 2862933555777941757ULL)
                  ^ (seed << 1)
        };
        for (int i = 0; i < 200; i++) rng64(&rng);

        u64 budget = max_curves / (u64)nthr + 1, lc = 0;
        u128 batch_A[BATCH_CURVES], batch_x[BATCH_CURVES];
        int pos = 0, count = 0;

        while (!found && lc < budget) {
            if (pos >= count) {
                count = x16_fill_batch(batch_A, batch_x, BATCH_CURVES,
                                       &rng, p, rand_mask, sqrtm1, &f);
                pos = 0;
                if (count == 0) continue;
            }
            u128 A = batch_A[pos];
            u128 xP = batch_x[pos++];

            u128 xR;
            if (halve_chain(&xR, p, A, xP, 4, k, sqrtm1, &f)) {
#pragma omp critical
                {
                    if (!found) { found = 1; found_A = A; found_x0 = xR; }
                }
            }
            lc++;

            if (tid == 0 && lc % 500000 == 0 && !found) {
                double el = now_sec() - t0;
                u64 est = lc * (u64)nthr;
                fprintf(stderr, "  curves=%llu elapsed=%.1fs rate=%.3fM/s\n",
                        (unsigned long long)est, el, est / el / 1e6);
            }
        }
        if (tid >= 0 && tid < 256) thread_curves[tid] = lc;
    }

    u64 total = 0;
    for (int i = 0; i < 256; i++) total += thread_curves[i];
    *out_curves = total;
    if (!found) return 0;
    *out_A = found_A;
    *out_x0 = found_x0;
    return 1;
}

/* Match OpenMP workers to the cgroup CPU quota when the caller has not set
 * a thread count: on shared machines nproc reports the node, not the quota,
 * and oversubscription costs about 2x at quota 4 on a 48-core node. */
static void set_threads_from_quota(void) {
#ifdef _OPENMP
    if (getenv("OMP_NUM_THREADS")) return;
    FILE *fp = fopen("/sys/fs/cgroup/cpu.max", "r");
    if (fp) {
        char word[32];
        long quota = 0, period = 0;
        if (fscanf(fp, "%31s %ld", word, &period) == 2 && strcmp(word, "max") != 0) {
            quota = atol(word);
            if (quota > 0 && period > 0) {
                int n = (int)((quota + period - 1) / period);
                if (n >= 1) omp_set_num_threads(n);
            }
        }
        fclose(fp);
    }
#endif
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <p> [seed] [max_curves]\n", argv[0]);
        return 2;
    }
    u128 p = parse128(argv[1]);
    u64 seed = (argc > 2) ? strtoull(argv[2], NULL, 10) : (u64)time(NULL);
    u64 max_curves = (argc > 3) ? strtoull(argv[3], NULL, 10)
                                : 4000000000000000000ULL;

    if (p <= ((u128)1 << 64) || p >= ((u128)1 << 65)) {
        fprintf(stderr, "unsupported: this implementation requires "
                        "2^64 < p < 2^65 (got a %s-bit value); see README, "
                        "Scope and limitations\n",
                (p > ((u128)1 << 64)) ? ">65" : "<=64");
        return 2;
    }
    u64 r8 = (u64)(p & 7);
    if (p % 2 == 0 || (r8 != 3 && r8 != 5 && r8 != 7)) {
        fprintf(stderr, "unsupported: p mod 8 must be 3, 5, or 7 "
                        "(the X1(16) square-root formulas need it); got %llu\n",
                (unsigned long long)r8);
        return 2;
    }

    set_threads_from_quota();

    u128 A, x0;
    u64 curves;
    if (!search(p, seed, max_curves, &A, &x0, &curves)) {
        fprintf(stderr, "no triple found within %llu curves\n",
                (unsigned long long)max_curves);
        return 1;
    }
    if (!verify_triple(p, A, x0)) {
        fprintf(stderr, "internal error: candidate failed verification\n");
        return 1;
    }

    char as[50], xs[50];
    sprint128(as, A);
    sprint128(xs, x0);
    printf("%s %s %llu\n", as, xs, (unsigned long long)curves);
    return 0;
}
