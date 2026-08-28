/*
 * pomerance.c — Find Pomerance triples (p, A, x0) for a given prime p
 *
 * A Pomerance triple (p, A, x0) is defined as follows: p is an odd prime,
 * A and x0 are nonneg integers < p with A ≠ ±2 mod p, such that doubling
 * the projective point (x0:1) on the Montgomery curve By^2 = x^3+Ax^2+x
 * exactly k times yields Z ≡ 0 mod p, where k is the least integer with
 * 2^k > floor(sqrt(p)) + 1 + 2*floor(sqrt(floor(sqrt(p)))).
 *
 * Algorithm: 2-Sylow projection.  For each candidate group order N = 2^k·m
 * in the Hasse interval, multiply a random point by the odd part m first
 * (projecting into the 2-Sylow subgroup), then double.  This gives
 * O(1/sqrt(p)) success probability per random (A, x0) trial.
 *
 * Automatically uses u64 arithmetic for p < 2^63, u128 for p < 2^127.
 *
 * Compile:
 *   gcc -O3 -fopenmp -o pomerance pomerance.c -lm
 *   gcc -O3 -o pomerance pomerance.c -lm           (single-threaded)
 *
 * Usage:
 *   ./pomerance <p> [seed_offset] [max_trials]
 *
 * Reference: https://github.com/AndrewVSutherland/DANGER3
 *
 * ----------------------------------------------------------------------
 * PROVENANCE
 *
 * Base implementation (2-Sylow projection search, u64/u128 Montgomery
 * arithmetic): Fabian Ruehle with Claude Opus 4.6, for the p = 10^21+117
 * record.  https://github.com/ruehlef/Danger2026DataChallenge (MIT)
 *
 * X1(16) prescribed-torsion sampler, successive-halving descent, and
 * nonsplit-discriminant filter (modes x16*): Alexa McLain with GPT 5.5
 * Codex, for the p = 10^22+9 and p = 10^23+117 records.
 * https://github.com/alexamclain/Danger2026DataChallenge (MIT)
 *
 * p ≡ 3 (mod 4) support (June 2026, this repository, for p = 10^24+7):
 * the upstream X1(16) sampler required p ≡ 5 (mod 8), using
 * sqrt(n) = n^((p+3)/8) with a sqrt(-1) correction.  10^24+7 ≡ 7 (mod 8),
 * so square roots are instead taken as sqrt(n) = n^((p+1)/4), which is
 * exact for quadratic residues when p ≡ 3 (mod 4).  Three sites changed,
 * each tagged "PATCH(p3mod4)" below:
 *   1. sqrtmod_p5_128       — added p ≡ 3 (mod 4) branch
 *   2. sqrtmod_p5_flag_128  — same, with used_sqrtm1 always 0
 *   3. search128 setup      — admit p ≡ 3 (mod 4) into X1(16) modes
 *                             (sqrtm1 unused in that case)
 * Found (10^24+7, 38923582678463553756710, 843367907077058108520461)
 * on 2026-06-11 in mode x16halvenonsplit; verified by DANGER3 vpp.py.
 * No other functional changes relative to McLain's version; benchmark
 * and statistics modes retain their original p ≡ 5 (mod 8) gates.
 * ----------------------------------------------------------------------
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

typedef uint64_t u64;
typedef __uint128_t u128;

static u64 g_seed_offset = 0;
static u64 g_max_trials_override = 0;
static int g_x16_mode = 0;
static int g_x16_halve_mode = 0;
static int g_x16_halve_full_mode = 0;
static int g_x16_nonsplit_filter_mode = 0;
static int g_x16_first_d_y_gate_mode = 0;
static int g_x16_first_d_skip_mode = 0;
static int g_x16_ell3_bench_mode = 0;
static int g_x16_ell3_direct_y_bench_mode = 0;
static int g_x16_atkin5_bench_mode = 0;
static int g_x16_atkin7_bench_mode = 0;
static int g_x16_halve_stats_mode = 0;
static int g_x16_branch_stats_mode = 0;
static int g_x16_gate_stats_mode = 0;
static int g_x16_lookahead_stats_mode = 0;
static int g_x16_label_stats_mode = 0;
static int g_x16_split_stats_mode = 0;
static int g_x16_split_branch_stats_mode = 0;
static int g_x16_split_translate_stats_mode = 0;
static int g_x16_pair_stats_mode = 0;
static int g_x16_torsion_stats_mode = 0;
static int g_x16_cubic3_stats_mode = 0;
static int g_x16_quartic_stats_mode = 0;
static int g_x1_32_root_bench_mode = 0;
static int g_x16_lookahead_depth = 2;
static int g_x16_branch_target = 12;
static int g_x16_gate_target = 12;
static int g_x16_lookahead_target = 12;
static int g_x16_label_target = 14;
static int g_x16_split_target = 20;
static int g_x16_split_branch_target = 14;
static int g_x16_split_translate_target = 16;
static int g_x16_cubic3_target = 20;
static int g_x16_quartic_target = 14;

typedef struct {
    unsigned char ydeg;
    unsigned char xdeg;
    signed char coeff;
} X132Term;

/* ================================================================
 * Parsing / printing u128
 * ================================================================ */

static u128 parse128(const char *s) {
    u128 v = 0;
    while (*s >= '0' && *s <= '9') { v = v * 10 + (*s - '0'); s++; }
    return v;
}

static void sprint128(char *buf, u128 v) {
    if (v == 0) { buf[0] = '0'; buf[1] = '\0'; return; }
    char tmp[50]; int i = 49; tmp[i] = '\0';
    while (v > 0) { tmp[--i] = '0' + (int)(v % 10); v /= 10; }
    strcpy(buf, tmp + i);
}

static void print128(u128 v) { char b[50]; sprint128(b, v); fputs(b, stdout); }

static int digits128(u128 v) { char b[50]; sprint128(b, v); return (int)strlen(b); }

/* ================================================================
 * PRNG (xorshift128+)
 * ================================================================ */

typedef struct { u64 s0, s1; } Rng;

static inline u64 rng64(Rng *r) {
    u64 s1 = r->s0, s0 = r->s1; r->s0 = s0;
    s1 ^= s1 << 23; r->s1 = s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26);
    return r->s1 + s0;
}

static inline int bitlen128(u128 x) {
    u64 hi = (u64)(x >> 64);
    if (hi) return 64 + (64 - __builtin_clzll(hi));
    u64 lo = (u64)x;
    return lo ? 64 - __builtin_clzll(lo) : 0;
}

static inline u128 rand_below128(Rng *rng, u128 p, u128 mask) {
    for (;;) {
        u128 v = ((u128)rng64(rng) << 64) | (u128)rng64(rng);
        v &= mask;
        if (v < p) return v;
    }
}

/* ================================================================
 * Timer
 * ================================================================ */

static double now_sec(void) {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

/* ================================================================
 * u64 code path (p < 2^63)
 * ================================================================ */

static inline u64 addmod64(u64 a, u64 b, u64 p) { return a >= p-b ? a-(p-b) : a+b; }
static inline u64 submod64(u64 a, u64 b, u64 p) { return a >= b ? a-b : p-b+a; }
static inline u64 mulmod64(u64 a, u64 b, u64 p) { return (u64)((u128)a*b%(u128)p); }

typedef struct { u64 p, ni, R2, one; } Mont64;

static void m64_init(Mont64 *m, u64 p) {
    m->p = p; u64 x = 1;
    for (int i = 0; i < 6; i++) x *= 2 - p * x;
    m->ni = (u64)(0ULL - x);
    m->one = (u64)(((u128)1 << 64) % (u128)p);
    m->R2  = (u64)(((u128)m->one * m->one) % (u128)p);
}

static inline u64 mred64(u128 t, const Mont64 *m) {
    u64 q = (u64)t * m->ni;
    u64 r = (u64)((t + (u128)q * m->p) >> 64);
    return r >= m->p ? r - m->p : r;
}
static inline u64 mm64(u64 a, u64 b, const Mont64 *m) { return mred64((u128)a*b, m); }
static inline u64 toM64(u64 a, const Mont64 *m) { return mm64(a % m->p, m->R2, m); }
static inline u64 frM64(u64 a, const Mont64 *m) { return mred64((u128)a, m); }

static inline void xDBL64(u64 *Xo, u64 *Zo, u64 X, u64 Z, u64 a24, const Mont64 *m) {
    u64 p = m->p;
    u64 u = addmod64(X,Z,p), v = submod64(X,Z,p);
    u = mm64(u,u,m); v = mm64(v,v,m);
    *Xo = mm64(u,v,m);
    u64 w = submod64(u,v,p);
    *Zo = mm64(w, addmod64(v, mm64(a24,w,m), p), m);
}

static inline void xADD64(u64 *Xo, u64 *Zo, u64 X0, u64 Z0, u64 X1, u64 Z1,
                            u64 xP, const Mont64 *m) {
    u64 p = m->p;
    u64 u = mm64(submod64(X0,Z0,p), addmod64(X1,Z1,p), m);
    u64 v = mm64(addmod64(X0,Z0,p), submod64(X1,Z1,p), m);
    u64 s = addmod64(u,v,p), d = submod64(u,v,p);
    *Xo = mm64(s,s,m);
    *Zo = mm64(xP, mm64(d,d,m), m);
}

static void xMUL64(u64 *Xo, u64 *Zo, u64 xP, u64 n, u64 a24, const Mont64 *m) {
    if (n==0) { *Xo=0; *Zo=0; return; }
    if (n==1) { *Xo=xP; *Zo=m->one; return; }
    u64 X0=xP, Z0=m->one, X1, Z1;
    xDBL64(&X1,&Z1,X0,Z0,a24,m);
    int bits = 64 - __builtin_clzll(n);
    for (int i=bits-2; i>=0; i--) {
        if ((n>>i)&1) { xADD64(&X0,&Z0,X0,Z0,X1,Z1,xP,m); xDBL64(&X1,&Z1,X1,Z1,a24,m); }
        else          { xADD64(&X1,&Z1,X0,Z0,X1,Z1,xP,m); xDBL64(&X0,&Z0,X0,Z0,a24,m); }
    }
    *Xo=X0; *Zo=Z0;
}

static int verify64(u64 p, u64 A, u64 x0) {
    u64 q = (u64)sqrtl((long double)p);
    while ((u128)(q+1)*(q+1)<=(u128)p) q++;
    while ((u128)q*q>(u128)p) q--;
    u64 sq = (u64)sqrtl((long double)q);
    while ((sq+1)*(sq+1)<=q) sq++;
    while (sq*sq>q) sq--;
    u64 bound = q+1+2*sq;
    int k=0; u64 v=1; while(v<=bound){k++;v<<=1;}

    if (A%p==2||A%p==p-2) return 0;
    u64 X=x0%p, Z=1;
    for (int i=1; i<=k; i++) {
        u64 X2=mulmod64(X,X,p), Z2=mulmod64(Z,Z,p), XZ=mulmod64(X,Z,p);
        u64 d=submod64(X2,Z2,p), Xn=mulmod64(d,d,p);
        u64 inn=addmod64(addmod64(X2,mulmod64(A,XZ,p),p),Z2,p);
        u64 f4=addmod64(addmod64(XZ,XZ,p),addmod64(XZ,XZ,p),p);
        u64 Zn=mulmod64(f4,inn,p); X=Xn; Z=Zn;
        if (i<k&&Z==0) return 0;
        if (i==k&&Z!=0) return 0;
    }
    return 1;
}

/* ================================================================
 * u128 code path (p < 2^127)
 * ================================================================ */

typedef struct { u128 lo, hi; } u256;

static inline u256 wide_mul(u128 a, u128 b) {
    u64 a0=(u64)a, a1=(u64)(a>>64), b0=(u64)b, b1=(u64)(b>>64);
    u128 ll=(u128)a0*b0, lh=(u128)a0*b1, hl=(u128)a1*b0, hh=(u128)a1*b1;
    u128 mid=lh+hl; u128 carry_mid=(mid<lh)?1:0;
    u128 lo=ll+(mid<<64); u128 carry_lo=(lo<ll)?1:0;
    return (u256){lo, hh+(mid>>64)+(carry_mid<<64)+carry_lo};
}

static inline u256 wide_add(u256 a, u256 b) {
    u128 lo=a.lo+b.lo; return (u256){lo, a.hi+b.hi+((lo<a.lo)?1:0)};
}

typedef struct { u128 p, ni, R2, one; } Mont128;

static void m128_init(Mont128 *m, u128 p) {
    m->p = p;
    u128 x = 1; for (int i = 0; i < 7; i++) x *= 2 - p * x;
    m->ni = (u128)0 - x;
    u128 r = 1;
    for (int i = 0; i < 128; i++) { r<<=1; if(r>=p) r-=p; }
    m->one = r;
    for (int i = 0; i < 128; i++) { r<<=1; if(r>=p) r-=p; }
    m->R2 = r;
}

static inline u128 mred128(u256 T, const Mont128 *m) {
    u128 q = T.lo * m->ni;
    u256 s = wide_add(T, wide_mul(q, m->p));
    u128 t = s.hi;
    return t >= m->p ? t - m->p : t;
}
static inline u128 mm128(u128 a, u128 b, const Mont128 *m) { return mred128(wide_mul(a,b), m); }
static inline u128 toM128(u128 a, const Mont128 *m) { return mm128(a % m->p, m->R2, m); }
static inline u128 frM128(u128 a, const Mont128 *m) { return mred128((u256){a,0}, m); }

static inline u128 addmod128(u128 a, u128 b, u128 p) { u128 s=a+b; return s>=p?s-p:s; }
static inline u128 submod128(u128 a, u128 b, u128 p) { return a>=b?a-b:p-b+a; }

static inline void xDBL128(u128 *Xo, u128 *Zo, u128 X, u128 Z, u128 a24, const Mont128 *m) {
    u128 p=m->p;
    u128 u=addmod128(X,Z,p), v=submod128(X,Z,p);
    u=mm128(u,u,m); v=mm128(v,v,m);
    *Xo=mm128(u,v,m);
    u128 w=submod128(u,v,p);
    *Zo=mm128(w, addmod128(v, mm128(a24,w,m), p), m);
}

static inline void xADD128(u128 *Xo, u128 *Zo, u128 X0, u128 Z0, u128 X1, u128 Z1,
                             u128 xP, const Mont128 *m) {
    u128 p=m->p;
    u128 u=mm128(submod128(X0,Z0,p), addmod128(X1,Z1,p), m);
    u128 v=mm128(addmod128(X0,Z0,p), submod128(X1,Z1,p), m);
    u128 s=addmod128(u,v,p), d=submod128(u,v,p);
    *Xo=mm128(s,s,m);
    *Zo=mm128(xP, mm128(d,d,m), m);
}

static void xMUL128(u128 *Xo, u128 *Zo, u128 xP, u64 n, u128 a24, const Mont128 *m) {
    if (n==0) { *Xo=0; *Zo=0; return; }
    if (n==1) { *Xo=xP; *Zo=m->one; return; }
    u128 X0=xP, Z0=m->one, X1, Z1;
    xDBL128(&X1,&Z1,X0,Z0,a24,m);
    int bits = 64 - __builtin_clzll(n);
    for (int i=bits-2; i>=0; i--) {
        if ((n>>i)&1) { xADD128(&X0,&Z0,X0,Z0,X1,Z1,xP,m); xDBL128(&X1,&Z1,X1,Z1,a24,m); }
        else          { xADD128(&X1,&Z1,X0,Z0,X1,Z1,xP,m); xDBL128(&X0,&Z0,X0,Z0,a24,m); }
    }
    *Xo=X0; *Zo=Z0;
}

static void xMULPAIR128(u128 *Xn, u128 *Zn, u128 *Xnp1, u128 *Znp1,
                         u128 xP, u64 n, u128 a24, const Mont128 *m) {
    if (n==0) { *Xn=0; *Zn=0; *Xnp1=xP; *Znp1=m->one; return; }
    u128 X0=xP, Z0=m->one, X1, Z1;
    xDBL128(&X1,&Z1,X0,Z0,a24,m);
    if (n==1) { *Xn=X0; *Zn=Z0; *Xnp1=X1; *Znp1=Z1; return; }
    int bits = 64 - __builtin_clzll(n);
    for (int i=bits-2; i>=0; i--) {
        if ((n>>i)&1) { xADD128(&X0,&Z0,X0,Z0,X1,Z1,xP,m); xDBL128(&X1,&Z1,X1,Z1,a24,m); }
        else          { xADD128(&X1,&Z1,X0,Z0,X1,Z1,xP,m); xDBL128(&X0,&Z0,X0,Z0,a24,m); }
    }
    *Xn=X0; *Zn=Z0; *Xnp1=X1; *Znp1=Z1;
}

static u128 mulmod_slow(u128 a, u128 b, u128 p) {
    u128 r=0; a%=p; b%=p;
    while (b>0) { if(b&1){r+=a;if(r>=p)r-=p;} a+=a;if(a>=p)a-=p; b>>=1; }
    return r;
}

static int verify128(u128 p, u128 A, u128 x0) {
    u64 q = (u64)sqrtl((long double)p);
    while ((u128)(q+1)*(q+1)<=p) q++;
    while ((u128)q*q>p) q--;
    u64 sq = (u64)sqrtl((long double)q);
    while ((sq+1)*(sq+1)<=q) sq++;
    while (sq*sq>q) sq--;
    u64 bound = q+1+2*sq;
    int k=0; u64 v=1; while(v<=bound){k++;v<<=1;}

    if (A%p==2||A%p==p-2) return 0;
    u128 X=x0%p, Z=1;
    for (int i=1; i<=k; i++) {
        u128 X2=mulmod_slow(X,X,p), Z2=mulmod_slow(Z,Z,p), XZ=mulmod_slow(X,Z,p);
        u128 d=submod128(X2,Z2,p), Xn=mulmod_slow(d,d,p);
        u128 inn=addmod128(addmod128(X2,mulmod_slow(A,XZ,p),p),Z2,p);
        u128 f4=addmod128(addmod128(XZ,XZ,p),addmod128(XZ,XZ,p),p);
        u128 Zn=mulmod_slow(f4,inn,p); X=Xn; Z=Zn;
        if (i<k&&Z==0) return 0;
        if (i==k&&Z!=0) return 0;
    }
    return 1;
}

static int projected_hit128(u128 *xRo, u128 p, u128 A, int k, int max_s,
                            u128 QX, u128 QZ, u128 a24m, const Mont128 *mt) {
    if (QZ == 0) return 0;
    u128 CX=QX, CZ=QZ;
    int zs = -1;
    for (int s=1; s<=max_s && s<50; s++) {
        xDBL128(&CX,&CZ,CX,CZ,a24m,mt);
        if (CZ==0) { zs=s; break; }
    }
    if (zs < k) return 0;
    int target = zs - k;
    CX=QX; CZ=QZ;
    for (int s=0; s<target; s++) xDBL128(&CX,&CZ,CX,CZ,a24m,mt);
    u128 cz = frM128(CZ, mt);
    if (cz == 0) return 0;
    u128 czinv_m = mt->one, base = CZ; u128 e2 = p-2;
    while (e2>0) { if(e2&1) czinv_m=mm128(czinv_m,base,mt); base=mm128(base,base,mt); e2>>=1; }
    u128 xR = frM128(mm128(CX, czinv_m, mt), mt);
    if (!verify128(p, A, xR)) return 0;
    *xRo = xR;
    return 1;
}

static inline u128 mulmod128_mont(u128 a, u128 b, const Mont128 *mt) {
    return frM128(mm128(toM128(a, mt), toM128(b, mt), mt), mt);
}

static u128 powmod128_mont(u128 a, u128 e, const Mont128 *mt) {
    u128 r = mt->one, b = toM128(a, mt);
    while (e > 0) {
        if (e & 1) r = mm128(r, b, mt);
        b = mm128(b, b, mt);
        e >>= 1;
    }
    return frM128(r, mt);
}

static int sqrtmod_p5_128(u128 *root, u128 n, u128 p, u128 sqrtm1, const Mont128 *mt) {
    n %= p;
    if (n == 0) { *root = 0; return 1; }
    if ((p & 3) == 3) {
        /* PATCH(p3mod4): sqrt(n) = n^((p+1)/4) when n is a quadratic residue */
        u128 x = powmod128_mont(n, (p + 1) >> 2, mt);
        if (mulmod128_mont(x, x, mt) == n) { *root = x; return 1; }
        return 0;
    }
    u128 x = powmod128_mont(n, (p + 3) >> 3, mt);
    if (mulmod128_mont(x, x, mt) == n) { *root = x; return 1; }
    x = mulmod128_mont(x, sqrtm1, mt);
    if (mulmod128_mont(x, x, mt) == n) { *root = x; return 1; }
    return 0;
}

static int sqrtmod_p5_flag_128(u128 *root, int *used_sqrtm1,
                               u128 n, u128 p, u128 sqrtm1,
                               const Mont128 *mt) {
    n %= p;
    *used_sqrtm1 = 0;
    if (n == 0) { *root = 0; return 1; }
    if ((p & 3) == 3) {
        /* PATCH(p3mod4): sqrt(n) = n^((p+1)/4); sqrt(-1) correction unused */
        u128 xr = powmod128_mont(n, (p + 1) >> 2, mt);
        if (mulmod128_mont(xr, xr, mt) == n) { *root = xr; return 1; }
        return 0;
    }
    u128 x = powmod128_mont(n, (p + 3) >> 3, mt);
    if (mulmod128_mont(x, x, mt) == n) { *root = x; return 1; }
    x = mulmod128_mont(x, sqrtm1, mt);
    if (mulmod128_mont(x, x, mt) == n) {
        *used_sqrtm1 = 1;
        *root = x;
        return 1;
    }
    return 0;
}

static u128 invert128_mont(u128 a, u128 p, const Mont128 *mt) {
    return powmod128_mont(a, p - 2, mt);
}

static int invert_batch128(u128 *out, const u128 *vals, int n, u128 p, const Mont128 *mt) {
    u128 prefix[8];
    prefix[0] = 1;
    for (int i = 0; i < n; i++) {
        if (vals[i] == 0) return 0;
        prefix[i + 1] = mulmod128_mont(prefix[i], vals[i], mt);
    }
    u128 acc = invert128_mont(prefix[n], p, mt);
    for (int i = n - 1; i >= 0; i--) {
        out[i] = mulmod128_mont(acc, prefix[i], mt);
        acc = mulmod128_mont(acc, vals[i], mt);
    }
    return 1;
}

static int poly_deg128(const u128 *a, int max_deg) {
    for (int i = max_deg; i >= 0; i--) if (a[i]) return i;
    return -1;
}

static void poly_monic128(u128 *a, int d, u128 p, const Mont128 *mt) {
    if (d < 0 || a[d] == 1) return;
    u128 inv_lc = invert128_mont(a[d], p, mt);
    for (int i = 0; i <= d; i++) a[i] = mulmod128_mont(a[i], inv_lc, mt);
}

static void poly_mulmod128(u128 out[5], const u128 aa[5], const u128 bb[5],
                           const u128 mod[5], int md, u128 p, const Mont128 *mt) {
    u128 tmp[9] = {0};
    int ad = poly_deg128(aa, 4), bd = poly_deg128(bb, 4);
    if (ad < 0 || bd < 0) { memset(out, 0, 5 * sizeof(u128)); return; }
    for (int i = 0; i <= ad; i++) {
        for (int j = 0; j <= bd; j++) {
            u128 prod = mulmod128_mont(aa[i], bb[j], mt);
            tmp[i+j] = addmod128(tmp[i+j], prod, p);
        }
    }
    for (int d = ad + bd; d >= md; d--) {
        u128 c = tmp[d];
        if (!c) continue;
        int shift = d - md;
        for (int j = 0; j < md; j++) {
            tmp[j+shift] = submod128(tmp[j+shift], mulmod128_mont(c, mod[j], mt), p);
        }
        tmp[d] = 0;
    }
    for (int i = 0; i < 5; i++) out[i] = (i < md) ? tmp[i] : 0;
}

static void poly_mod128(u128 out[5], const u128 aa[5], const u128 bb[5],
                        int bd, u128 p, const Mont128 *mt) {
    u128 r[5];
    for (int i = 0; i < 5; i++) r[i] = aa[i];
    int rd = poly_deg128(r, 4);
    u128 inv_lc = invert128_mont(bb[bd], p, mt);
    while (rd >= bd && rd >= 0) {
        int shift = rd - bd;
        u128 c = mulmod128_mont(r[rd], inv_lc, mt);
        for (int j = 0; j <= bd; j++) {
            r[j+shift] = submod128(r[j+shift], mulmod128_mont(c, bb[j], mt), p);
        }
        rd = poly_deg128(r, 4);
    }
    for (int i = 0; i < 5; i++) out[i] = r[i];
}

static int poly_gcd_monic128(u128 out[5], const u128 aa[5], const u128 bb[5],
                             u128 p, const Mont128 *mt) {
    u128 a[5], b[5], r[5];
    for (int i = 0; i < 5; i++) { a[i] = aa[i]; b[i] = bb[i]; }
    int bd = poly_deg128(b, 4);
    while (bd >= 0) {
        poly_mod128(r, a, b, bd, p, mt);
        for (int i = 0; i < 5; i++) { a[i] = b[i]; b[i] = r[i]; }
        bd = poly_deg128(b, 4);
    }
    int ad = poly_deg128(a, 4);
    if (ad >= 0) poly_monic128(a, ad, p, mt);
    for (int i = 0; i < 5; i++) out[i] = a[i];
    return ad;
}

static void poly_powmod128(u128 out[5], const u128 base[5], u128 exp,
                           const u128 mod[5], int md, u128 p, const Mont128 *mt) {
    u128 r[5] = {1,0,0,0,0};
    u128 b[5] = {0};
    for (int i = 0; i < 5; i++) b[i] = base[i];
    if (poly_deg128(b, 4) >= md) {
        u128 zero[5] = {0}, tmp[5];
        zero[md] = 1;
        for (int i = 0; i < md; i++) zero[i] = mod[i];
        poly_mod128(tmp, b, zero, md, p, mt);
        for (int i = 0; i < 5; i++) b[i] = tmp[i];
    }
    while (exp) {
        if (exp & 1) {
            u128 tmp[5]; poly_mulmod128(tmp, r, b, mod, md, p, mt);
            for (int i = 0; i < 5; i++) r[i] = tmp[i];
        }
        u128 tmp[5]; poly_mulmod128(tmp, b, b, mod, md, p, mt);
        for (int i = 0; i < 5; i++) b[i] = tmp[i];
        exp >>= 1;
    }
    for (int i = 0; i < 5; i++) out[i] = r[i];
}

static void psi3_mulmod4_128(u128 out[4], const u128 aa[4], const u128 bb[4],
                             u128 c3, u128 inv3, u128 p, const Mont128 *mt) {
    u128 tmp[7] = {0};
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            tmp[i + j] = addmod128(tmp[i + j], mulmod128_mont(aa[i], bb[j], mt), p);
        }
    }

    for (int d = 6; d >= 4; d--) {
        u128 c = tmp[d];
        if (!c) continue;
        int s = d - 4;
        tmp[s] = addmod128(tmp[s], mulmod128_mont(c, inv3, mt), p);
        tmp[s + 2] = submod128(tmp[s + 2], mulmod128_mont(c, 2, mt), p);
        tmp[s + 3] = submod128(tmp[s + 3], mulmod128_mont(c, c3, mt), p);
        tmp[d] = 0;
    }

    for (int i = 0; i < 4; i++) out[i] = tmp[i];
}

static void psi3_zpowmod4_128(u128 out[4], u128 exp, u128 c3, u128 inv3,
                              u128 p, const Mont128 *mt) {
    u128 r[4] = {1,0,0,0};
    u128 b[4] = {0,1,0,0};
    while (exp) {
        if (exp & 1) {
            u128 tmp[4];
            psi3_mulmod4_128(tmp, r, b, c3, inv3, p, mt);
            for (int i = 0; i < 4; i++) r[i] = tmp[i];
        }
        u128 tmp[4];
        psi3_mulmod4_128(tmp, b, b, c3, inv3, p, mt);
        for (int i = 0; i < 4; i++) b[i] = tmp[i];
        exp >>= 1;
    }
    for (int i = 0; i < 4; i++) out[i] = r[i];
}

static int montgomery_trace_mod3_p1_128(u128 A, u128 p, const Mont128 *mt) {
    if (p % 3 != 1) return -1;
    u128 inv3 = invert128_mont(3, p, mt);
    u128 psi[5] = {
        submod128(0, inv3, p),
        0,
        2 % p,
        mulmod128_mont((4 * (A % p)) % p, inv3, mt),
        1
    };
    u128 xpoly[5] = {0,1,0,0,0};
    u128 xp[5], xp_minus_x[5], root_factor[5];
    poly_powmod128(xp, xpoly, p, psi, 4, p, mt);
    for (int i = 0; i < 5; i++) xp_minus_x[i] = xp[i];
    xp_minus_x[1] = submod128(xp_minus_x[1], 1, p);
    int rd = poly_gcd_monic128(root_factor, psi, xp_minus_x, p, mt);
    if (rd <= 0) return 0;

    u128 f[5] = {0,1,A % p,1,0};
    u128 leg[5], leg_plus_one[5], nonsquare_factor[5];
    poly_powmod128(leg, f, (p - 1) >> 1, root_factor, rd, p, mt);
    for (int i = 0; i < 5; i++) leg_plus_one[i] = leg[i];
    leg_plus_one[0] = addmod128(leg_plus_one[0], 1, p);
    int nd = poly_gcd_monic128(nonsquare_factor, root_factor, leg_plus_one, p, mt);
    return nd > 0 ? 1 : 2;
}

static int montgomery_trace_mod3_reject_shortcut_p1_128(u128 A, u128 p, const Mont128 *mt) {
    if (p % 3 != 1) return -1;
    u128 inv3 = invert128_mont(3, p, mt);
    u128 psi[5] = {
        submod128(0, inv3, p),
        0,
        2 % p,
        mulmod128_mont((4 * (A % p)) % p, inv3, mt),
        1
    };
    u128 xpoly[5] = {0,1,0,0,0};
    u128 leg[5], nonsquare_roots[5];
    poly_powmod128(leg, xpoly, (p - 1) >> 1, psi, 4, p, mt);
    leg[0] = addmod128(leg[0], 1, p);
    int nd = poly_gcd_monic128(nonsquare_roots, psi, leg, p, mt);
    return nd > 0;
}

static int montgomery_trace_mod3_reject_fast_p1_128(u128 A, u128 p, const Mont128 *mt) {
    if (p % 3 != 1) return -1;
    u128 inv3 = invert128_mont(3, p, mt);
    u128 c3 = mulmod128_mont((4 * (A % p)) % p, inv3, mt);
    u128 leg4[4];
    psi3_zpowmod4_128(leg4, (p - 1) >> 1, c3, inv3, p, mt);

    u128 psi[5] = {
        submod128(0, inv3, p),
        0,
        2 % p,
        c3,
        1
    };
    u128 leg[5] = {leg4[0], leg4[1], leg4[2], leg4[3], 0};
    leg[0] = addmod128(leg[0], 1, p);
    u128 nonsquare_roots[5];
    int nd = poly_gcd_monic128(nonsquare_roots, psi, leg, p, mt);
    return nd > 0;
}

static int montgomery_trace_mod3_reject_fast_inv3_p1_128(u128 A, u128 p,
                                                         const Mont128 *mt,
                                                         u128 inv3) {
    if (p % 3 != 1) return -1;
    u128 c3 = mulmod128_mont((4 * (A % p)) % p, inv3, mt);
    u128 leg4[4];
    psi3_zpowmod4_128(leg4, (p - 1) >> 1, c3, inv3, p, mt);

    u128 psi[5] = {
        submod128(0, inv3, p),
        0,
        2 % p,
        c3,
        1
    };
    u128 leg[5] = {leg4[0], leg4[1], leg4[2], leg4[3], 0};
    leg[0] = addmod128(leg[0], 1, p);
    u128 nonsquare_roots[5];
    int nd = poly_gcd_monic128(nonsquare_roots, psi, leg, p, mt);
    return nd > 0;
}

static u128 x16_A_numerator_from_y128(u128 y, u128 p, const Mont128 *mt) {
    u128 num = 1;
    num = submod128(mulmod128_mont(num, y, mt), 8, p);
    num = addmod128(mulmod128_mont(num, y, mt), 24, p);
    num = submod128(mulmod128_mont(num, y, mt), 32, p);
    num = addmod128(mulmod128_mont(num, y, mt), 8, p);
    num = addmod128(mulmod128_mont(num, y, mt), 32, p);
    num = submod128(mulmod128_mont(num, y, mt), 48, p);
    num = addmod128(mulmod128_mont(num, y, mt), 32, p);
    num = submod128(mulmod128_mont(num, y, mt), 8, p);
    return num;
}

static int x16_y_ell3_reject_direct_g_p1_128(u128 y, u128 y2, u128 p,
                                             u128 inv3, const Mont128 *mt) {
    if (p % 3 != 1) return -1;
    u128 ym1 = submod128(y, 1, p);
    if (ym1 == 0) return -1;
    u128 ym1_2 = mulmod128_mont(ym1, ym1, mt);
    u128 d = mulmod128_mont(ym1_2, ym1_2, mt);
    if (d == 0) return -1;

    (void)y2;
    u128 den3 = mulmod128_mont(3, d, mt);
    u128 c3 = mulmod128_mont(x16_A_numerator_from_y128(y, p, mt),
                             invert128_mont(den3, p, mt), mt);

    u128 leg4[4];
    psi3_zpowmod4_128(leg4, (p - 1) >> 1, c3, inv3, p, mt);

    u128 psi[5] = {
        submod128(0, inv3, p),
        0,
        2 % p,
        c3,
        1
    };
    u128 leg[5] = {leg4[0], leg4[1], leg4[2], leg4[3], 0};
    leg[0] = addmod128(leg[0], 1, p);
    u128 nonsquare_roots[5];
    int nd = poly_gcd_monic128(nonsquare_roots, psi, leg, p, mt);
    return nd > 0;
}

static int x16_root_to_montgomery_A128(u128 *Ao, u128 *xPo,
                                       u128 p, u128 x, u128 y,
                                       const Mont128 *mt) {
    u128 num = x16_A_numerator_from_y128(y, p, mt);

    u128 ym1 = submod128(y, 1, p);
    u128 ym1_2 = mulmod128_mont(ym1, ym1, mt);
    u128 denA = mulmod128_mont(4, mulmod128_mont(ym1_2, ym1_2, mt), mt);
    u128 denx = submod128(x, y, p);
    u128 vals[2] = { denA, denx };
    u128 invs[2];
    if (!invert_batch128(invs, vals, 2, p, mt)) return 0;

    u128 A = mulmod128_mont(num, invs[0], mt);
    u128 xP = mulmod128_mont(x, invs[1], mt);
    if (A <= 2 || A >= p - 2) return 0;
    *Ao = A;
    *xPo = xP;
    return 1;
}

static int split_discriminant128(u128 p, u128 A, u128 sqrtm1, const Mont128 *mt) {
    u128 disc = submod128(mulmod128_mont(A, A, mt), 4, p);
    u128 root;
    return sqrtmod_p5_128(&root, disc, p, sqrtm1, mt);
}

static int x16_y_predicts_nonsplit128(u128 p, u128 y, u128 y2, const Mont128 *mt) {
    u128 f1 = submod128(y2, 2, p);
    u128 four_y = addmod128(addmod128(y, y, p), addmod128(y, y, p), p);
    u128 f2 = addmod128(submod128(y2, four_y, p), 2, p);
    u128 f = mulmod128_mont(f1, f2, mt);
    if (f == 0) return 0;
    u128 leg = powmod128_mont(f, (p - 1) >> 1, mt);
    return leg != 1;
}

static u128 x16_y_first_d_value128(u128 p, u128 y, u128 y2, const Mont128 *mt) {
    u128 two_y = addmod128(y, y, p);
    u128 f = submod128(y, 1, p);
    f = mulmod128_mont(f, submod128(y2, 2, p), mt);
    f = mulmod128_mont(f, addmod128(submod128(y2, two_y, p), 2, p), mt);
    return f;
}

static int x16_y_first_d_gate128(u128 p, u128 y, u128 y2, const Mont128 *mt) {
    u128 f = x16_y_first_d_value128(p, y, y2, mt);
    if (f == 0) return 0;
    return powmod128_mont(f, (p - 1) >> 1, mt) == 1;
}

static int x16_y_first_d_root128(u128 *zo, u128 p, u128 y, u128 y2,
                                 u128 sqrtm1, const Mont128 *mt) {
    u128 f = x16_y_first_d_value128(p, y, y2, mt);
    if (f == 0) return 0;
    return sqrtmod_p5_128(zo, f, p, sqrtm1, mt);
}

static int x16_y_quartic_class128(u128 p, u128 y, u128 y2,
                                  u128 sqrtm1, const Mont128 *mt) {
    u128 f1 = submod128(y2, 2, p);
    u128 four_y = addmod128(addmod128(y, y, p), addmod128(y, y, p), p);
    u128 f2 = addmod128(submod128(y2, four_y, p), 2, p);
    u128 f = mulmod128_mont(f1, f2, mt);
    if (f == 0) return 4;
    u128 q = powmod128_mont(f, (p - 1) >> 2, mt);
    if (q == 1) return 0;                       /* +1 */
    if (q == p - 1) return 1;                   /* -1 */
    if (q == sqrtm1) return 2;                  /* +sqrt(-1) */
    if (q == submod128(0, sqrtm1, p)) return 3; /* -sqrt(-1) */
    return 5;
}

static int x16_cubic3_class128(u128 p, u128 y, u128 y2, const Mont128 *mt) {
    if (p % 3 != 1) return 0;
    u128 ym1 = submod128(y, 1, p);
    u128 ym2 = submod128(y, 2, p);
    u128 two_y = addmod128(y, y, p);
    u128 four_y = addmod128(two_y, two_y, p);
    u128 f1 = submod128(y2, 2, p);
    u128 f2 = addmod128(submod128(y2, four_y, p), 2, p);
    u128 f3 = addmod128(submod128(y2, two_y, p), 2, p);
    u128 D16 = mulmod128_mont(mulmod128_mont(y, ym2, mt),
                              mulmod128_mont(f1, f3, mt), mt);
    u128 C3 = mulmod128_mont(mulmod128_mont(D16, ym1, mt),
                             mulmod128_mont(f3, f2, mt), mt);
    if (C3 == 0) return 0;
    u128 chi = powmod128_mont(C3, (p - 1) / 3, mt);
    return chi == 1 ? 1 : 2; /* 1=cube, 2=noncube */
}

static int x16_montgomery_A128(u128 *Ao, u128 *xPo,
                               u128 *pending_A, u128 *pending_xP, int *have_pending_A,
                               Rng *rng, u128 p, u128 rand_mask,
                               u128 sqrtm1, const Mont128 *mt) {
    if (*have_pending_A) {
        *Ao = *pending_A;
        *xPo = *pending_xP;
        *have_pending_A = 0;
        return 1;
    }

    for (;;) {
        u128 y = rand_below128(rng, p, rand_mask);
        if (y == 0) continue;

        u128 y2 = mulmod128_mont(y, y, mt);
        u128 y3 = mulmod128_mont(y2, y, mt);
        u128 qa = submod128(y2, addmod128(y, y, p), p);
        if (qa == 0) continue;
        u128 qb = submod128(addmod128(y2, y2, p), y3, p);
        u128 qc = submod128(1, y, p);
        u128 D = submod128(mulmod128_mont(qb, qb, mt),
                           mulmod128_mont(addmod128(qa, qa, p), addmod128(qc, qc, p), mt),
                           p);
        u128 sd;
        if (!sqrtmod_p5_128(&sd, D, p, sqrtm1, mt)) continue;

        u128 inv_2qa = invert128_mont(addmod128(qa, qa, p), p, mt);
        u128 roots[2] = {
            mulmod128_mont(submod128(sd, qb, p), inv_2qa, mt),
            mulmod128_mont(submod128(p - sd, qb, p), inv_2qa, mt)
        };

        int got_A = 0;
        u128 first_A = 0;
        u128 first_xP = 0;
        for (int ri = 0; ri < 2; ri++) {
            u128 A, xP;
            if (!x16_root_to_montgomery_A128(&A, &xP, p, roots[ri], y, mt)) continue;
            if (!got_A) {
                first_A = A;
                first_xP = xP;
                got_A = 1;
            } else {
                *Ao = first_A;
                *xPo = first_xP;
                *pending_A = A;
                *pending_xP = xP;
                *have_pending_A = 1;
                return 1;
            }
        }
        if (got_A) {
            *Ao = first_A;
            *xPo = first_xP;
            return 1;
        }
    }
}

static int x16_montgomery_yA128(u128 *Ao, u128 *yo, u128 *y2o,
                                Rng *rng, u128 p, u128 rand_mask,
                                u128 sqrtm1, const Mont128 *mt) {
    for (;;) {
        u128 y = rand_below128(rng, p, rand_mask);
        if (y == 0) continue;

        u128 y2 = mulmod128_mont(y, y, mt);
        u128 y3 = mulmod128_mont(y2, y, mt);
        u128 qa = submod128(y2, addmod128(y, y, p), p);
        if (qa == 0) continue;
        u128 qb = submod128(addmod128(y2, y2, p), y3, p);
        u128 qc = submod128(1, y, p);
        u128 D = submod128(mulmod128_mont(qb, qb, mt),
                           mulmod128_mont(addmod128(qa, qa, p), addmod128(qc, qc, p), mt),
                           p);
        u128 sd;
        if (!sqrtmod_p5_128(&sd, D, p, sqrtm1, mt)) continue;

        u128 inv_2qa = invert128_mont(addmod128(qa, qa, p), p, mt);
        u128 roots[2] = {
            mulmod128_mont(submod128(sd, qb, p), inv_2qa, mt),
            mulmod128_mont(submod128(p - sd, qb, p), inv_2qa, mt)
        };

        for (int ri = 0; ri < 2; ri++) {
            u128 A, xP;
            if (!x16_root_to_montgomery_A128(&A, &xP, p, roots[ri], y, mt)) continue;
            (void)xP;
            *Ao = A;
            *yo = y;
            *y2o = y2;
            return 1;
        }
    }
}

static int x16_montgomery_A_nonsplit128(u128 *Ao, u128 *xPo,
                                        u128 *pending_A, u128 *pending_xP, int *have_pending_A,
                                        Rng *rng, u128 p, u128 rand_mask,
                                        u128 sqrtm1, const Mont128 *mt) {
    if (*have_pending_A) {
        *Ao = *pending_A;
        *xPo = *pending_xP;
        *have_pending_A = 0;
        return 1;
    }

    for (;;) {
        u128 y = rand_below128(rng, p, rand_mask);
        if (y == 0) continue;

        u128 y2 = mulmod128_mont(y, y, mt);
        if (!x16_y_predicts_nonsplit128(p, y, y2, mt)) continue;
        if (g_x16_first_d_y_gate_mode && !x16_y_first_d_gate128(p, y, y2, mt)) continue;
        u128 y3 = mulmod128_mont(y2, y, mt);
        u128 qa = submod128(y2, addmod128(y, y, p), p);
        if (qa == 0) continue;
        u128 qb = submod128(addmod128(y2, y2, p), y3, p);
        u128 qc = submod128(1, y, p);
        u128 D = submod128(mulmod128_mont(qb, qb, mt),
                           mulmod128_mont(addmod128(qa, qa, p), addmod128(qc, qc, p), mt),
                           p);
        u128 sd;
        if (!sqrtmod_p5_128(&sd, D, p, sqrtm1, mt)) continue;

        u128 inv_2qa = invert128_mont(addmod128(qa, qa, p), p, mt);
        u128 roots[2] = {
            mulmod128_mont(submod128(sd, qb, p), inv_2qa, mt),
            mulmod128_mont(submod128(p - sd, qb, p), inv_2qa, mt)
        };

        int got_A = 0;
        u128 first_A = 0;
        u128 first_xP = 0;
        for (int ri = 0; ri < 2; ri++) {
            u128 A, xP;
            if (!x16_root_to_montgomery_A128(&A, &xP, p, roots[ri], y, mt)) continue;
            if (!got_A) {
                first_A = A;
                first_xP = xP;
                got_A = 1;
            } else {
                *Ao = first_A;
                *xPo = first_xP;
                *pending_A = A;
                *pending_xP = xP;
                *have_pending_A = 1;
                return 1;
            }
        }
        if (got_A) {
            *Ao = first_A;
            *xPo = first_xP;
            return 1;
        }
    }
}

static int halve_once_known_d128(u128 *xo, u128 p, u128 x,
                                 u128 sd, u128 sqrtm1, const Mont128 *mt) {
    const u128 inv2 = (p + 1) >> 1;
    u128 roots_d[2] = { sd, submod128(0, sd, p) };
    for (int i = 0; i < 2; i++) {
        u128 u = addmod128(addmod128(x, x, p), addmod128(roots_d[i], roots_d[i], p), p);
        u128 w = submod128(mulmod128_mont(u, u, mt), 4, p);
        u128 sw;
        if (!sqrtmod_p5_128(&sw, w, p, sqrtm1, mt)) continue;
        u128 candidates[2] = {
            mulmod128_mont(addmod128(u, sw, p), inv2, mt),
            mulmod128_mont(submod128(u, sw, p), inv2, mt)
        };
        for (int j = 0; j < 2; j++) {
            if (candidates[j] != 0) {
                *xo = candidates[j];
                return 1;
            }
        }
    }
    return 0;
}

static int x16_first_half_from_yroot128(u128 *x32o, u128 p, u128 A,
                                        u128 x_model, u128 y, u128 z,
                                        u128 xP16, u128 sqrtm1,
                                        const Mont128 *mt) {
    u128 ym1 = submod128(y, 1, p);
    u128 den = mulmod128_mont(2 % p, submod128(x_model, y, p), mt);
    den = mulmod128_mont(den, mulmod128_mont(ym1, ym1, mt), mt);
    if (den == 0) return 0;
    u128 inv_den = invert128_mont(den, p, mt);
    u128 sd = mulmod128_mont(mulmod128_mont(y, z, mt), inv_den, mt);

    u128 d = addmod128(addmod128(mulmod128_mont(xP16, xP16, mt),
                                 mulmod128_mont(A, xP16, mt), p), 1, p);
    if (mulmod128_mont(sd, sd, mt) != d) return 0;
    return halve_once_known_d128(x32o, p, xP16, sd, sqrtm1, mt);
}

static int x16_montgomery_A_nonsplit_dgate_skip128(u128 *Ao, u128 *x32o,
                                                   u128 *pending_A, u128 *pending_x32,
                                                   int *have_pending_A,
                                                   Rng *rng, u128 p, u128 rand_mask,
                                                   u128 sqrtm1, const Mont128 *mt) {
    if (*have_pending_A) {
        *Ao = *pending_A;
        *x32o = *pending_x32;
        *have_pending_A = 0;
        return 1;
    }

    for (;;) {
        u128 y = rand_below128(rng, p, rand_mask);
        if (y == 0) continue;

        u128 y2 = mulmod128_mont(y, y, mt);
        if (!x16_y_predicts_nonsplit128(p, y, y2, mt)) continue;
        u128 z;
        if (!x16_y_first_d_root128(&z, p, y, y2, sqrtm1, mt)) continue;

        u128 y3 = mulmod128_mont(y2, y, mt);
        u128 qa = submod128(y2, addmod128(y, y, p), p);
        if (qa == 0) continue;
        u128 qb = submod128(addmod128(y2, y2, p), y3, p);
        u128 qc = submod128(1, y, p);
        u128 D = submod128(mulmod128_mont(qb, qb, mt),
                           mulmod128_mont(addmod128(qa, qa, p), addmod128(qc, qc, p), mt),
                           p);
        u128 sd;
        if (!sqrtmod_p5_128(&sd, D, p, sqrtm1, mt)) continue;

        u128 inv_2qa = invert128_mont(addmod128(qa, qa, p), p, mt);
        u128 roots[2] = {
            mulmod128_mont(submod128(sd, qb, p), inv_2qa, mt),
            mulmod128_mont(submod128(p - sd, qb, p), inv_2qa, mt)
        };

        int got_A = 0;
        u128 first_A = 0;
        u128 first_x32 = 0;
        for (int ri = 0; ri < 2; ri++) {
            u128 A, xP16, x32;
            if (!x16_root_to_montgomery_A128(&A, &xP16, p, roots[ri], y, mt)) continue;
            if (!x16_first_half_from_yroot128(&x32, p, A, roots[ri], y, z, xP16,
                                              sqrtm1, mt)) continue;
            if (!got_A) {
                first_A = A;
                first_x32 = x32;
                got_A = 1;
            } else {
                *Ao = first_A;
                *x32o = first_x32;
                *pending_A = A;
                *pending_x32 = x32;
                *have_pending_A = 1;
                return 1;
            }
        }
        if (got_A) {
            *Ao = first_A;
            *x32o = first_x32;
            return 1;
        }
    }
}

static int x16_montgomery_A_quartic128(u128 *Ao, u128 *xPo,
                                       u128 *yo, u128 *y2o,
                                       int *d16_flag, int *g4_class,
                                       u128 *pending_A, u128 *pending_xP,
                                       u128 *pending_y, u128 *pending_y2,
                                       int *pending_d16_flag,
                                       int *pending_g4_class,
                                       int *have_pending_A,
                                       int nonsplit_only,
                                       Rng *rng, u128 p, u128 rand_mask,
                                       u128 sqrtm1, const Mont128 *mt) {
    if (*have_pending_A) {
        *Ao = *pending_A;
        *xPo = *pending_xP;
        *yo = *pending_y;
        *y2o = *pending_y2;
        *d16_flag = *pending_d16_flag;
        *g4_class = *pending_g4_class;
        *have_pending_A = 0;
        return 1;
    }

    for (;;) {
        u128 y = rand_below128(rng, p, rand_mask);
        if (y == 0) continue;

        u128 y2 = mulmod128_mont(y, y, mt);
        int g4 = x16_y_quartic_class128(p, y, y2, sqrtm1, mt);
        if (nonsplit_only && !(g4 == 2 || g4 == 3)) continue;

        u128 y3 = mulmod128_mont(y2, y, mt);
        u128 qa = submod128(y2, addmod128(y, y, p), p);
        if (qa == 0) continue;
        u128 qb = submod128(addmod128(y2, y2, p), y3, p);
        u128 qc = submod128(1, y, p);
        u128 D = submod128(mulmod128_mont(qb, qb, mt),
                           mulmod128_mont(addmod128(qa, qa, p), addmod128(qc, qc, p), mt),
                           p);
        u128 sd;
        int d16_used_i = 0;
        if (!sqrtmod_p5_flag_128(&sd, &d16_used_i, D, p, sqrtm1, mt)) continue;

        u128 inv_2qa = invert128_mont(addmod128(qa, qa, p), p, mt);
        u128 roots[2] = {
            mulmod128_mont(submod128(sd, qb, p), inv_2qa, mt),
            mulmod128_mont(submod128(p - sd, qb, p), inv_2qa, mt)
        };

        int got_A = 0;
        u128 first_A = 0;
        u128 first_xP = 0;
        for (int ri = 0; ri < 2; ri++) {
            u128 A, xP;
            if (!x16_root_to_montgomery_A128(&A, &xP, p, roots[ri], y, mt)) continue;
            if (!got_A) {
                first_A = A;
                first_xP = xP;
                got_A = 1;
            } else {
                *Ao = first_A;
                *xPo = first_xP;
                *d16_flag = d16_used_i;
                *g4_class = g4;
                *yo = y;
                *y2o = y2;
                *pending_A = A;
                *pending_xP = xP;
                *pending_y = y;
                *pending_y2 = y2;
                *pending_d16_flag = d16_used_i;
                *pending_g4_class = g4;
                *have_pending_A = 1;
                return 1;
            }
        }
        if (got_A) {
            *Ao = first_A;
            *xPo = first_xP;
            *d16_flag = d16_used_i;
            *g4_class = g4;
            *yo = y;
            *y2o = y2;
            return 1;
        }
    }
}

static int x16_montgomery_A_nonsplit_cubic3_128(u128 *Ao, u128 *xPo, int *cube_class,
                                                u128 *pending_A, u128 *pending_xP,
                                                int *pending_class, int *have_pending_A,
                                                Rng *rng, u128 p, u128 rand_mask,
                                                u128 sqrtm1, const Mont128 *mt) {
    if (*have_pending_A) {
        *Ao = *pending_A;
        *xPo = *pending_xP;
        *cube_class = *pending_class;
        *have_pending_A = 0;
        return 1;
    }

    for (;;) {
        u128 y = rand_below128(rng, p, rand_mask);
        if (y == 0) continue;

        u128 y2 = mulmod128_mont(y, y, mt);
        if (!x16_y_predicts_nonsplit128(p, y, y2, mt)) continue;
        u128 y3 = mulmod128_mont(y2, y, mt);
        u128 qa = submod128(y2, addmod128(y, y, p), p);
        if (qa == 0) continue;
        u128 qb = submod128(addmod128(y2, y2, p), y3, p);
        u128 qc = submod128(1, y, p);
        u128 D = submod128(mulmod128_mont(qb, qb, mt),
                           mulmod128_mont(addmod128(qa, qa, p), addmod128(qc, qc, p), mt),
                           p);
        u128 sd;
        if (!sqrtmod_p5_128(&sd, D, p, sqrtm1, mt)) continue;
        int cls = x16_cubic3_class128(p, y, y2, mt);
        if (cls == 0) continue;

        u128 inv_2qa = invert128_mont(addmod128(qa, qa, p), p, mt);
        u128 roots[2] = {
            mulmod128_mont(submod128(sd, qb, p), inv_2qa, mt),
            mulmod128_mont(submod128(p - sd, qb, p), inv_2qa, mt)
        };

        int got_A = 0;
        u128 first_A = 0;
        u128 first_xP = 0;
        for (int ri = 0; ri < 2; ri++) {
            u128 A, xP;
            if (!x16_root_to_montgomery_A128(&A, &xP, p, roots[ri], y, mt)) continue;
            if (!got_A) {
                first_A = A;
                first_xP = xP;
                got_A = 1;
            } else {
                *Ao = first_A;
                *xPo = first_xP;
                *cube_class = cls;
                *pending_A = A;
                *pending_xP = xP;
                *pending_class = cls;
                *have_pending_A = 1;
                return 1;
            }
        }
        if (got_A) {
            *Ao = first_A;
            *xPo = first_xP;
            *cube_class = cls;
            return 1;
        }
    }
}

static int bench_x16_ell3_128(u128 p) {
    if ((u64)(p & 7) != 5) {
        printf("x16ell3bench requires p ≡ 5 mod 8 for the X1(16) sampler.\n");
        return 1;
    }
    if (p % 3 != 1) {
        printf("x16ell3bench currently implements the p ≡ 1 mod 3 trace classification.\n");
        return 1;
    }

    Mont128 mt; m128_init(&mt, p);
    int pbits = bitlen128(p);
    u128 rand_mask = pbits >= 128 ? (u128)0 - 1 : (((u128)1 << pbits) - 1);
    u128 sqrtm1 = powmod128_mont(2, (p - 1) >> 2, &mt);
    u64 trials = g_max_trials_override ? g_max_trials_override : 100000ULL;
    if (trials == 0) trials = 100000ULL;

    u128 *As = (u128 *)malloc((size_t)trials * sizeof(u128));
    unsigned char *tm3s = (unsigned char *)malloc((size_t)trials * sizeof(unsigned char));
    if (!As || !tm3s) {
        printf("Could not allocate benchmark array for %llu samples.\n", (unsigned long long)trials);
        free(As);
        free(tm3s);
        return 1;
    }

    Rng rng = {
        .s0 = 7364529176530163ULL ^ g_seed_offset,
        .s1 = 1442695040888963407ULL ^ (g_seed_offset << 1)
    };
    for (int i = 0; i < 200; i++) rng64(&rng);

    u128 pending_A = 0, pending_xP = 0, xP16 = 0;
    int have_pending_A = 0;
    double t0 = now_sec();
    for (u64 i = 0; i < trials; i++) {
        x16_montgomery_A128(&As[i], &xP16, &pending_A, &pending_xP, &have_pending_A,
                             &rng, p, rand_mask, sqrtm1, &mt);
    }
    double t_sample = now_sec() - t0;

    u64 counts[3] = {0,0,0};
    u64 accepted = 0;
    t0 = now_sec();
    for (u64 i = 0; i < trials; i++) {
        int tm3 = montgomery_trace_mod3_p1_128(As[i], p, &mt);
        if (tm3 >= 0 && tm3 < 3) counts[tm3]++;
        if (tm3 == 0 || tm3 == 2) accepted++;
        tm3s[i] = (unsigned char)(tm3 < 0 ? 255 : tm3);
    }
    double t_filter = now_sec() - t0;

    u64 shortcut_reject = 0;
    u64 shortcut_mismatch = 0;
    t0 = now_sec();
    for (u64 i = 0; i < trials; i++) {
        int reject = montgomery_trace_mod3_reject_shortcut_p1_128(As[i], p, &mt);
        if (reject == 1) shortcut_reject++;
        if (reject != (tm3s[i] == 1)) shortcut_mismatch++;
    }
    double t_shortcut = now_sec() - t0;

    u64 fast_reject = 0;
    u64 fast_mismatch = 0;
    t0 = now_sec();
    for (u64 i = 0; i < trials; i++) {
        int reject = montgomery_trace_mod3_reject_fast_p1_128(As[i], p, &mt);
        if (reject == 1) fast_reject++;
        if (reject != (tm3s[i] == 1)) fast_mismatch++;
    }
    double t_fast = now_sec() - t0;

    u128 inv3 = invert128_mont(3, p, &mt);
    u64 fast_inv3_reject = 0;
    u64 fast_inv3_mismatch = 0;
    t0 = now_sec();
    for (u64 i = 0; i < trials; i++) {
        int reject = montgomery_trace_mod3_reject_fast_inv3_p1_128(As[i], p, &mt, inv3);
        if (reject == 1) fast_inv3_reject++;
        if (reject != (tm3s[i] == 1)) fast_inv3_mismatch++;
    }
    double t_fast_inv3 = now_sec() - t0;

    u128 pow_checksum = 0;
    t0 = now_sec();
    for (u64 i = 0; i < trials; i++) {
        u128 c3 = mulmod128_mont((4 * (As[i] % p)) % p, inv3, &mt);
        u128 leg4[4];
        psi3_zpowmod4_128(leg4, (p - 1) >> 1, c3, inv3, p, &mt);
        pow_checksum ^= leg4[0] ^ (leg4[1] << 1) ^ (leg4[2] << 2) ^ (leg4[3] << 3);
    }
    double t_powonly = now_sec() - t0;

    printf("X1(16) ell=3 trace-filter benchmark\n");
    printf("samples = %llu\n", (unsigned long long)trials);
    printf("x16_sample_seconds = %.6f  rate_Mps = %.6f\n",
           t_sample, (double)trials / t_sample / 1e6);
    printf("ell3_filter_seconds = %.6f  rate_Mps = %.6f\n",
           t_filter, (double)trials / t_filter / 1e6);
    printf("ell3_seconds_per_curve = %.9e\n", t_filter / (double)trials);
    printf("trace_mod3_counts = {0:%llu, 1:%llu, 2:%llu}\n",
           (unsigned long long)counts[0],
           (unsigned long long)counts[1],
           (unsigned long long)counts[2]);
    printf("accepted_mod3_0_or_2 = %llu/%llu  rate = %.6f\n",
           (unsigned long long)accepted,
           (unsigned long long)trials,
           (double)accepted / (double)trials);
    printf("ell3_shortcut_seconds = %.6f  rate_Mps = %.6f\n",
           t_shortcut, (double)trials / t_shortcut / 1e6);
    printf("ell3_shortcut_seconds_per_curve = %.9e\n", t_shortcut / (double)trials);
    printf("ell3_shortcut_reject_trace1 = %llu/%llu  rate = %.6f\n",
           (unsigned long long)shortcut_reject,
           (unsigned long long)trials,
           (double)shortcut_reject / (double)trials);
    printf("ell3_shortcut_mismatches_vs_exact_trace1 = %llu\n",
           (unsigned long long)shortcut_mismatch);
    printf("ell3_fast_shortcut_seconds = %.6f  rate_Mps = %.6f\n",
           t_fast, (double)trials / t_fast / 1e6);
    printf("ell3_fast_shortcut_seconds_per_curve = %.9e\n", t_fast / (double)trials);
    printf("ell3_fast_shortcut_reject_trace1 = %llu/%llu  rate = %.6f\n",
           (unsigned long long)fast_reject,
           (unsigned long long)trials,
           (double)fast_reject / (double)trials);
    printf("ell3_fast_shortcut_mismatches_vs_exact_trace1 = %llu\n",
           (unsigned long long)fast_mismatch);
    printf("ell3_fast_inv3_seconds = %.6f  rate_Mps = %.6f\n",
           t_fast_inv3, (double)trials / t_fast_inv3 / 1e6);
    printf("ell3_fast_inv3_seconds_per_curve = %.9e\n", t_fast_inv3 / (double)trials);
    printf("ell3_fast_inv3_reject_trace1 = %llu/%llu  rate = %.6f\n",
           (unsigned long long)fast_inv3_reject,
           (unsigned long long)trials,
           (double)fast_inv3_reject / (double)trials);
    printf("ell3_fast_inv3_mismatches_vs_exact_trace1 = %llu\n",
           (unsigned long long)fast_inv3_mismatch);
    printf("ell3_fast_powonly_seconds = %.6f  rate_Mps = %.6f\n",
           t_powonly, (double)trials / t_powonly / 1e6);
    printf("ell3_fast_powonly_seconds_per_curve = %.9e checksum_low64=%llu\n",
           t_powonly / (double)trials, (unsigned long long)pow_checksum);

    free(As);
    free(tm3s);
    return 0;
}

static int bench_x16_ell3_direct_y_128(u128 p) {
    if ((u64)(p & 7) != 5) {
        printf("x16ell3directybench requires p ≡ 5 mod 8 for the X1(16) sampler.\n");
        return 1;
    }
    if (p % 3 != 1) {
        printf("x16ell3directybench currently implements the p ≡ 1 mod 3 trace classification.\n");
        return 1;
    }

    Mont128 mt; m128_init(&mt, p);
    int pbits = bitlen128(p);
    u128 rand_mask = pbits >= 128 ? (u128)0 - 1 : (((u128)1 << pbits) - 1);
    u128 sqrtm1 = powmod128_mont(2, (p - 1) >> 2, &mt);
    u128 inv3 = invert128_mont(3, p, &mt);
    u64 trials = g_max_trials_override ? g_max_trials_override : 20000ULL;
    if (trials == 0) trials = 20000ULL;

    u128 *As = (u128 *)malloc((size_t)trials * sizeof(u128));
    u128 *ys = (u128 *)malloc((size_t)trials * sizeof(u128));
    u128 *y2s = (u128 *)malloc((size_t)trials * sizeof(u128));
    unsigned char *tm3s = (unsigned char *)malloc((size_t)trials * sizeof(unsigned char));
    unsigned char *nonsplit = (unsigned char *)malloc((size_t)trials * sizeof(unsigned char));
    if (!As || !ys || !y2s || !tm3s || !nonsplit) {
        printf("Could not allocate direct-y ell=3 benchmark arrays for %llu samples.\n",
               (unsigned long long)trials);
        free(As);
        free(ys);
        free(y2s);
        free(tm3s);
        free(nonsplit);
        return 1;
    }

    Rng rng = {
        .s0 = 9364529176530163ULL ^ g_seed_offset,
        .s1 = 2442695040888963407ULL ^ (g_seed_offset << 1)
    };
    for (int i = 0; i < 200; i++) rng64(&rng);

    double t0 = now_sec();
    for (u64 i = 0; i < trials; i++) {
        x16_montgomery_yA128(&As[i], &ys[i], &y2s[i],
                             &rng, p, rand_mask, sqrtm1, &mt);
    }
    double t_sample = now_sec() - t0;

    u64 counts[3] = {0,0,0};
    u64 accepted = 0;
    t0 = now_sec();
    for (u64 i = 0; i < trials; i++) {
        int tm3 = montgomery_trace_mod3_p1_128(As[i], p, &mt);
        if (tm3 >= 0 && tm3 < 3) counts[tm3]++;
        if (tm3 == 0 || tm3 == 2) accepted++;
        tm3s[i] = (unsigned char)(tm3 < 0 ? 255 : tm3);
    }
    double t_filter = now_sec() - t0;

    u64 nonsplit_count = 0;
    for (u64 i = 0; i < trials; i++) {
        nonsplit[i] = (unsigned char)x16_y_predicts_nonsplit128(p, ys[i], y2s[i], &mt);
        if (nonsplit[i]) nonsplit_count++;
    }

    u64 direct_reject = 0;
    u64 direct_mismatch = 0;
    t0 = now_sec();
    for (u64 i = 0; i < trials; i++) {
        int reject = x16_y_ell3_reject_direct_g_p1_128(ys[i], y2s[i], p, inv3, &mt);
        if (reject == 1) direct_reject++;
        if (reject != (tm3s[i] == 1)) direct_mismatch++;
    }
    double t_direct = now_sec() - t0;

    u64 direct_nonsplit_reject = 0;
    u64 direct_nonsplit_mismatch = 0;
    t0 = now_sec();
    for (u64 i = 0; i < trials; i++) {
        if (!nonsplit[i]) continue;
        int reject = x16_y_ell3_reject_direct_g_p1_128(ys[i], y2s[i], p, inv3, &mt);
        if (reject == 1) direct_nonsplit_reject++;
        if (reject != (tm3s[i] == 1)) direct_nonsplit_mismatch++;
    }
    double t_direct_nonsplit = now_sec() - t0;

    printf("X1(16) direct-y ell=3 quartic benchmark\n");
    printf("samples = %llu accepted_y_values\n", (unsigned long long)trials);
    printf("nonsplit_y_values = %llu/%llu  rate = %.6f\n",
           (unsigned long long)nonsplit_count,
           (unsigned long long)trials,
           (double)nonsplit_count / (double)trials);
    printf("x16_y_sample_seconds = %.6f  rate_Mps = %.6f\n",
           t_sample, (double)trials / t_sample / 1e6);
    printf("ell3_filter_seconds = %.6f  rate_Mps = %.6f\n",
           t_filter, (double)trials / t_filter / 1e6);
    printf("ell3_seconds_per_y = %.9e\n", t_filter / (double)trials);
    printf("trace_mod3_counts = {0:%llu, 1:%llu, 2:%llu}\n",
           (unsigned long long)counts[0],
           (unsigned long long)counts[1],
           (unsigned long long)counts[2]);
    printf("accepted_mod3_0_or_2 = %llu/%llu  rate = %.6f\n",
           (unsigned long long)accepted,
           (unsigned long long)trials,
           (double)accepted / (double)trials);
    printf("ell3_direct_y_seconds = %.6f  rate_Mps = %.6f\n",
           t_direct, (double)trials / t_direct / 1e6);
    printf("ell3_direct_y_seconds_per_y = %.9e\n", t_direct / (double)trials);
    printf("ell3_direct_y_reject_trace1 = %llu/%llu  rate = %.6f\n",
           (unsigned long long)direct_reject,
           (unsigned long long)trials,
           (double)direct_reject / (double)trials);
    printf("ell3_direct_y_mismatches_vs_exact_trace1 = %llu\n",
           (unsigned long long)direct_mismatch);
    printf("ell3_direct_y_nonsplit_seconds = %.6f  rate_Mps = %.6f\n",
           t_direct_nonsplit,
           nonsplit_count ? (double)nonsplit_count / t_direct_nonsplit / 1e6 : 0.0);
    printf("ell3_direct_y_nonsplit_seconds_per_y = %.9e\n",
           nonsplit_count ? t_direct_nonsplit / (double)nonsplit_count : 0.0);
    printf("ell3_direct_y_nonsplit_reject_trace1 = %llu/%llu  rate = %.6f\n",
           (unsigned long long)direct_nonsplit_reject,
           (unsigned long long)nonsplit_count,
           nonsplit_count ? (double)direct_nonsplit_reject / (double)nonsplit_count : 0.0);
    printf("ell3_direct_y_nonsplit_mismatches_vs_exact_trace1 = %llu\n",
           (unsigned long long)direct_nonsplit_mismatch);

    free(As);
    free(ys);
    free(y2s);
    free(tm3s);
    free(nonsplit);
    return 0;
}

static int poly8_deg128(const u128 *a) {
    for (int i = 8; i >= 0; i--) if (a[i]) return i;
    return -1;
}

static void poly8_monic128(u128 *a, int d, u128 p, const Mont128 *mt) {
    if (d < 0 || a[d] == 1) return;
    u128 inv_lc = invert128_mont(a[d], p, mt);
    for (int i = 0; i <= d; i++) a[i] = mulmod128_mont(a[i], inv_lc, mt);
}

static void poly8_mod128(u128 out[9], const u128 aa[17], int ad,
                         const u128 bb[9], int bd,
                         u128 p, const Mont128 *mt) {
    u128 r[17] = {0};
    for (int i = 0; i <= ad; i++) r[i] = aa[i];
    u128 inv_lc = invert128_mont(bb[bd], p, mt);
    int rd = ad;
    while (rd >= bd) {
        u128 c = mulmod128_mont(r[rd], inv_lc, mt);
        int shift = rd - bd;
        for (int j = 0; j <= bd; j++) {
            r[j + shift] = submod128(r[j + shift], mulmod128_mont(c, bb[j], mt), p);
        }
        while (rd >= 0 && !r[rd]) rd--;
    }
    for (int i = 0; i < 9; i++) out[i] = r[i];
}

static void poly8_rem128(u128 out[9], const u128 aa[9], const u128 bb[9],
                         int bd, u128 p, const Mont128 *mt) {
    u128 wide[17] = {0};
    for (int i = 0; i < 9; i++) wide[i] = aa[i];
    poly8_mod128(out, wide, 8, bb, bd, p, mt);
}

static int poly8_gcd_monic128(u128 out[9], const u128 aa[9], const u128 bb[9],
                              u128 p, const Mont128 *mt) {
    u128 a[9], b[9], r[9];
    for (int i = 0; i < 9; i++) { a[i] = aa[i]; b[i] = bb[i]; }
    int bd = poly8_deg128(b);
    while (bd >= 0) {
        poly8_rem128(r, a, b, bd, p, mt);
        for (int i = 0; i < 9; i++) { a[i] = b[i]; b[i] = r[i]; }
        bd = poly8_deg128(b);
    }
    int ad = poly8_deg128(a);
    if (ad >= 0) poly8_monic128(a, ad, p, mt);
    for (int i = 0; i < 9; i++) out[i] = a[i];
    return ad;
}

static void poly8_mulmod128(u128 out[9], const u128 aa[9], const u128 bb[9],
                            const u128 mod[9], int md,
                            u128 p, const Mont128 *mt) {
    u128 tmp[17] = {0};
    int ad = poly8_deg128(aa), bd = poly8_deg128(bb);
    if (ad < 0 || bd < 0) { memset(out, 0, 9 * sizeof(u128)); return; }
    for (int i = 0; i <= ad; i++) {
        for (int j = 0; j <= bd; j++) {
            tmp[i + j] = addmod128(tmp[i + j], mulmod128_mont(aa[i], bb[j], mt), p);
        }
    }
    poly8_mod128(out, tmp, ad + bd, mod, md, p, mt);
}

static void poly8_powmod128(u128 out[9], const u128 base[9], u128 exp,
                            const u128 mod[9], int md, u128 p, const Mont128 *mt) {
    u128 r[9] = {1,0,0,0,0,0,0,0,0};
    u128 b[9] = {0};
    for (int i = 0; i < 9; i++) b[i] = base[i];
    if (poly8_deg128(b) >= md) {
        u128 tmp[9];
        poly8_rem128(tmp, b, mod, md, p, mt);
        for (int i = 0; i < 9; i++) b[i] = tmp[i];
    }
    while (exp) {
        if (exp & 1) {
            u128 tmp[9]; poly8_mulmod128(tmp, r, b, mod, md, p, mt);
            for (int i = 0; i < 9; i++) r[i] = tmp[i];
        }
        u128 tmp[9]; poly8_mulmod128(tmp, b, b, mod, md, p, mt);
        for (int i = 0; i < 9; i++) b[i] = tmp[i];
        exp >>= 1;
    }
    for (int i = 0; i < 9; i++) out[i] = r[i];
}

static int poly10_deg128(const u128 *a) {
    for (int i = 10; i >= 0; i--) if (a[i]) return i;
    return -1;
}

static void poly10_monic128(u128 *a, int d, u128 p, const Mont128 *mt) {
    if (d < 0 || a[d] == 1) return;
    u128 inv_lc = invert128_mont(a[d], p, mt);
    for (int i = 0; i <= d; i++) a[i] = mulmod128_mont(a[i], inv_lc, mt);
}

static void poly10_mod128(u128 out[11], const u128 aa[21], int ad,
                          const u128 bb[11], int bd,
                          u128 p, const Mont128 *mt) {
    u128 r[21] = {0};
    for (int i = 0; i <= ad; i++) r[i] = aa[i];
    u128 inv_lc = invert128_mont(bb[bd], p, mt);
    int rd = ad;
    while (rd >= bd) {
        u128 c = mulmod128_mont(r[rd], inv_lc, mt);
        int shift = rd - bd;
        for (int j = 0; j <= bd; j++) {
            r[j + shift] = submod128(r[j + shift], mulmod128_mont(c, bb[j], mt), p);
        }
        while (rd >= 0 && !r[rd]) rd--;
    }
    for (int i = 0; i < 11; i++) out[i] = r[i];
}

static void poly10_rem128(u128 out[11], const u128 aa[11], const u128 bb[11],
                          int bd, u128 p, const Mont128 *mt) {
    u128 wide[21] = {0};
    for (int i = 0; i < 11; i++) wide[i] = aa[i];
    poly10_mod128(out, wide, 10, bb, bd, p, mt);
}

static int poly10_gcd_monic128(u128 out[11], const u128 aa[11], const u128 bb[11],
                               u128 p, const Mont128 *mt) {
    u128 a[11], b[11], r[11];
    for (int i = 0; i < 11; i++) { a[i] = aa[i]; b[i] = bb[i]; }
    int bd = poly10_deg128(b);
    while (bd >= 0) {
        poly10_rem128(r, a, b, bd, p, mt);
        for (int i = 0; i < 11; i++) { a[i] = b[i]; b[i] = r[i]; }
        bd = poly10_deg128(b);
    }
    int ad = poly10_deg128(a);
    if (ad >= 0) poly10_monic128(a, ad, p, mt);
    for (int i = 0; i < 11; i++) out[i] = a[i];
    return ad;
}

static void poly10_mulmod128(u128 out[11], const u128 aa[11], const u128 bb[11],
                             const u128 mod[11], int md,
                             u128 p, const Mont128 *mt) {
    u128 tmp[21] = {0};
    int ad = poly10_deg128(aa), bd = poly10_deg128(bb);
    if (ad < 0 || bd < 0) { memset(out, 0, 11 * sizeof(u128)); return; }
    for (int i = 0; i <= ad; i++) {
        for (int j = 0; j <= bd; j++) {
            tmp[i + j] = addmod128(tmp[i + j],
                                   mulmod128_mont(aa[i], bb[j], mt), p);
        }
    }
    poly10_mod128(out, tmp, ad + bd, mod, md, p, mt);
}

static void poly10_powmod128(u128 out[11], const u128 base[11], u128 exp,
                             const u128 mod[11], int md, u128 p, const Mont128 *mt) {
    u128 r[11] = {1,0,0,0,0,0,0,0,0,0,0};
    u128 b[11] = {0};
    for (int i = 0; i < 11; i++) b[i] = base[i];
    if (poly10_deg128(b) >= md) {
        u128 tmp[11];
        poly10_rem128(tmp, b, mod, md, p, mt);
        for (int i = 0; i < 11; i++) b[i] = tmp[i];
    }
    while (exp) {
        if (exp & 1) {
            u128 tmp[11]; poly10_mulmod128(tmp, r, b, mod, md, p, mt);
            for (int i = 0; i < 11; i++) r[i] = tmp[i];
        }
        u128 tmp[11]; poly10_mulmod128(tmp, b, b, mod, md, p, mt);
        for (int i = 0; i < 11; i++) b[i] = tmp[i];
        exp >>= 1;
    }
    for (int i = 0; i < 11; i++) out[i] = r[i];
}

static const X132Term X1_32_TERMS[] = {
    {0,5,-1},
    {1,0,1},{1,1,1},{1,2,1},{1,3,1},{1,4,1},{1,5,1},{1,6,-3},{1,7,3},
    {2,0,-5},{2,1,-5},{2,2,-9},{2,3,-3},{2,4,3},{2,5,9},{2,6,1},{2,7,-6},{2,8,4},{2,9,-4},
    {3,0,10},{3,1,14},{3,2,24},{3,3,-4},{3,4,-12},{3,5,-20},{3,6,3},{3,7,-7},{3,8,6},{3,9,6},{3,10,-2},{3,11,2},
    {4,0,-10},{4,1,-25},{4,2,-24},{4,3,8},{4,4,16},{4,5,29},{4,6,3},{4,7,5},{4,8,-14},{4,9,5},{4,10,-6},{4,11,-2},
    {5,0,5},{5,1,25},{5,2,9},{5,3,7},{5,4,-33},{5,5,-13},{5,6,-4},{5,8,-1},{5,9,3},{5,10,7},{5,11,1},
    {6,0,-1},{6,1,-11},{6,2,-7},{6,3,-2},{6,4,25},{6,5,-15},{6,6,20},{6,7,-5},{6,8,5},{6,9,-5},{6,10,-5},
    {7,2,8},{7,3,-12},{7,4,14},{7,5,-10},{7,8,-10},{7,9,10},
    {8,1,1},{8,4,-6},{8,5,7},{8,6,-12},{8,7,20},{8,8,-10},
    {9,2,-2},{9,3,4},{9,4,-5},{9,5,11},{9,6,-13},{9,7,5},
    {10,3,1},{10,4,-3},{10,5,3},{10,6,-1}
};

static void x1_32_fiber_poly10_128(u128 out[11], u128 x, u128 p, const Mont128 *mt) {
    for (int i = 0; i < 11; i++) out[i] = 0;
    u128 xp[12];
    xp[0] = 1;
    for (int i = 1; i < 12; i++) xp[i] = mulmod128_mont(xp[i - 1], x, mt);
    size_t nterms = sizeof(X1_32_TERMS) / sizeof(X1_32_TERMS[0]);
    for (size_t i = 0; i < nterms; i++) {
        X132Term t = X1_32_TERMS[i];
        u128 mag = (u128)(t.coeff < 0 ? -t.coeff : t.coeff);
        u128 val = mulmod128_mont(mag, xp[t.xdeg], mt);
        if (t.coeff < 0) out[t.ydeg] = submod128(out[t.ydeg], val, p);
        else out[t.ydeg] = addmod128(out[t.ydeg], val, p);
    }
}

static int x1_32_fiber_root_degree128(const u128 f[11], u128 p, const Mont128 *mt) {
    int fd = poly10_deg128(f);
    if (fd <= 0) return 0;
    u128 ypoly[11] = {0,1,0,0,0,0,0,0,0,0,0};
    u128 yp[11], yp_minus_y[11], root_factor[11];
    poly10_powmod128(yp, ypoly, p, f, fd, p, mt);
    for (int i = 0; i < 11; i++) yp_minus_y[i] = yp[i];
    yp_minus_y[1] = submod128(yp_minus_y[1], 1, p);
    int rd = poly10_gcd_monic128(root_factor, f, yp_minus_y, p, mt);
    return rd > 0 ? rd : 0;
}

static int bench_x1_32_root_128(u128 p) {
    Mont128 mt; m128_init(&mt, p);
    int pbits = bitlen128(p);
    u128 rand_mask = pbits >= 128 ? (u128)0 - 1 : (((u128)1 << pbits) - 1);
    u64 trials = g_max_trials_override ? g_max_trials_override : 10000ULL;
    if (trials == 0) trials = 10000ULL;

    Rng rng = {
        .s0 = 7364529176530163ULL ^ g_seed_offset,
        .s1 = 1442695040888963407ULL ^ (g_seed_offset << 1)
    };
    for (int i = 0; i < 200; i++) rng64(&rng);

    u64 degree_hist[11] = {0};
    u64 root_degree_hist[11] = {0};
    u64 root_fibers = 0;
    u64 root_degree_sum = 0;
    double eval_seconds = 0.0;
    double root_seconds = 0.0;
    double t0 = now_sec();

    for (u64 i = 0; i < trials; i++) {
        u128 x;
        do {
            x = (((u128)rng64(&rng) << 64) ^ rng64(&rng)) & rand_mask;
        } while (x >= p);

        u128 f[11];
        double te = now_sec();
        x1_32_fiber_poly10_128(f, x, p, &mt);
        eval_seconds += now_sec() - te;
        int fd = poly10_deg128(f);
        if (fd >= 0 && fd <= 10) degree_hist[fd]++;

        double tr = now_sec();
        int rd = x1_32_fiber_root_degree128(f, p, &mt);
        root_seconds += now_sec() - tr;
        if (rd >= 0 && rd <= 10) root_degree_hist[rd]++;
        if (rd > 0) {
            root_fibers++;
            root_degree_sum += (u64)rd;
        }
    }

    double elapsed = now_sec() - t0;
    printf("X1(32) degree-10 fiber root-existence benchmark\n");
    printf("p = "); print128(p); printf("\n");
    printf("samples = %llu\n", (unsigned long long)trials);
    printf("source = Sutherland FFFc32 optimized X1(32) equation\n");
    printf("method = gcd(f(y), y^p - y) over Fp\n");
    printf("elapsed_seconds = %.6f  fiber_rate = %.6f/s  fiber_rate_Mps = %.9f\n",
           elapsed, (double)trials / elapsed, (double)trials / elapsed / 1e6);
    printf("eval_seconds = %.6f  eval_seconds_per_fiber = %.9e\n",
           eval_seconds, eval_seconds / (double)trials);
    printf("root_gcd_seconds = %.6f  root_gcd_seconds_per_fiber = %.9e\n",
           root_seconds, root_seconds / (double)trials);
    printf("root_fibers = %llu/%llu  rate = %.6f\n",
           (unsigned long long)root_fibers,
           (unsigned long long)trials,
           (double)root_fibers / (double)trials);
    printf("avg_root_degree_per_root_fiber = %.6f\n",
           root_fibers ? (double)root_degree_sum / (double)root_fibers : 0.0);
    printf("fiber_degree_hist:");
    for (int d = 0; d <= 10; d++) if (degree_hist[d]) {
        printf(" %d:%llu", d, (unsigned long long)degree_hist[d]);
    }
    printf("\n");
    printf("root_degree_hist:");
    for (int d = 0; d <= 10; d++) if (root_degree_hist[d]) {
        printf(" %d:%llu", d, (unsigned long long)root_degree_hist[d]);
    }
    printf("\n");
    return 0;
}

static u128 montgomery_j128(u128 A, u128 p, const Mont128 *mt) {
    u128 A2 = mulmod128_mont(A, A, mt);
    u128 den = submod128(A2, 4 % p, p);
    if (!den) return 0;
    u128 c = submod128(A2, 3 % p, p);
    u128 c3 = mulmod128_mont(mulmod128_mont(c, c, mt), c, mt);
    u128 num = mulmod128_mont(256 % p, c3, mt);
    return mulmod128_mont(num, invert128_mont(den, p, mt), mt);
}

static int montgomery_x0_7_has_root128(u128 A, u128 p, const Mont128 *mt) {
    u128 j = montgomery_j128(A, p, mt);
    if (!j) return 0;
    u128 f[9] = {
        49 % p,
        submod128(748 % p, j, p),
        4018 % p,
        8624 % p,
        5915 % p,
        1904 % p,
        322 % p,
        28 % p,
        1
    };
    u128 xpoly[9] = {0,1,0,0,0,0,0,0,0};
    u128 xp[9], xp_minus_x[9], root_factor[9];
    poly8_powmod128(xp, xpoly, p, f, 8, p, mt);
    for (int i = 0; i < 9; i++) xp_minus_x[i] = xp[i];
    xp_minus_x[1] = submod128(xp_minus_x[1], 1, p);
    int rd = poly8_gcd_monic128(root_factor, f, xp_minus_x, p, mt);
    return rd > 0;
}

static int montgomery_x0_5_has_root128(u128 A, u128 p, const Mont128 *mt) {
    u128 j = montgomery_j128(A, p, mt);
    if (!j) return 0;
    u128 f[9] = {
        125 % p,
        submod128(750 % p, j, p),
        1575 % p,
        1300 % p,
        315 % p,
        30 % p,
        1,
        0,
        0
    };
    u128 xpoly[9] = {0,1,0,0,0,0,0,0,0};
    u128 xp[9], xp_minus_x[9], root_factor[9];
    poly8_powmod128(xp, xpoly, p, f, 6, p, mt);
    for (int i = 0; i < 9; i++) xp_minus_x[i] = xp[i];
    xp_minus_x[1] = submod128(xp_minus_x[1], 1, p);
    int rd = poly8_gcd_monic128(root_factor, f, xp_minus_x, p, mt);
    return rd > 0;
}

static int bench_x16_atkin5_128(u128 p) {
    if ((u64)(p & 7) != 5) {
        printf("x16atkin5bench requires p ≡ 5 mod 8 for the X1(16) sampler.\n");
        return 1;
    }
    if (p % 5 != 2) {
        printf("x16atkin5bench p23 target-residue interpretation assumes p ≡ 2 mod 5.\n");
    }

    Mont128 mt; m128_init(&mt, p);
    int pbits = bitlen128(p);
    u128 rand_mask = pbits >= 128 ? (u128)0 - 1 : (((u128)1 << pbits) - 1);
    u128 sqrtm1 = powmod128_mont(2, (p - 1) >> 2, &mt);
    u64 trials = g_max_trials_override ? g_max_trials_override : 100000ULL;
    if (trials == 0) trials = 100000ULL;

    u128 *As = (u128 *)malloc((size_t)trials * sizeof(u128));
    if (!As) {
        printf("Could not allocate benchmark array for %llu samples.\n",
               (unsigned long long)trials);
        return 1;
    }

    Rng rng = {
        .s0 = 7364529176530163ULL ^ g_seed_offset,
        .s1 = 1442695040888963407ULL ^ (g_seed_offset << 1)
    };
    for (int i = 0; i < 200; i++) rng64(&rng);

    u128 pending_A = 0, pending_xP = 0, xP16 = 0;
    int have_pending_A = 0;
    double t0 = now_sec();
    for (u64 i = 0; i < trials; i++) {
        if (g_x16_nonsplit_filter_mode) {
            x16_montgomery_A_nonsplit128(&As[i], &xP16, &pending_A, &pending_xP,
                                         &have_pending_A, &rng, p, rand_mask,
                                         sqrtm1, &mt);
        } else {
            x16_montgomery_A128(&As[i], &xP16, &pending_A, &pending_xP,
                                 &have_pending_A, &rng, p, rand_mask,
                                 sqrtm1, &mt);
        }
    }
    double t_sample = now_sec() - t0;

    u64 has_root = 0;
    t0 = now_sec();
    for (u64 i = 0; i < trials; i++) {
        if (montgomery_x0_5_has_root128(As[i], p, &mt)) has_root++;
    }
    double t_filter = now_sec() - t0;

    printf("X1(16) Atkin/Elkies ell=5 status benchmark\n");
    printf("p = "); print128(p); printf("\n");
    printf("samples = %llu\n", (unsigned long long)trials);
    printf("sample_class = %s\n", g_x16_nonsplit_filter_mode ? "nonsplit_y_filtered" : "all_x16");
    printf("x16_sample_seconds = %.6f  rate_Mps = %.6f\n",
           t_sample, (double)trials / t_sample / 1e6);
    printf("atkin5_filter_seconds = %.6f  rate_Mps = %.6f\n",
           t_filter, (double)trials / t_filter / 1e6);
    printf("atkin5_seconds_per_curve = %.9e\n", t_filter / (double)trials);
    printf("x0_5_has_root_elkies_accept = %llu/%llu  rate = %.6f\n",
           (unsigned long long)has_root,
           (unsigned long long)trials,
           (double)has_root / (double)trials);
    printf("x0_5_no_root_atkin_reject = %llu/%llu  rate = %.6f\n",
           (unsigned long long)(trials - has_root),
           (unsigned long long)trials,
           (double)(trials - has_root) / (double)trials);

    free(As);
    return 0;
}

static int bench_x16_atkin7_128(u128 p) {
    if ((u64)(p & 7) != 5) {
        printf("x16atkin7bench requires p ≡ 5 mod 8 for the X1(16) sampler.\n");
        return 1;
    }

    Mont128 mt; m128_init(&mt, p);
    int pbits = bitlen128(p);
    u128 rand_mask = pbits >= 128 ? (u128)0 - 1 : (((u128)1 << pbits) - 1);
    u128 sqrtm1 = powmod128_mont(2, (p - 1) >> 2, &mt);
    u64 trials = g_max_trials_override ? g_max_trials_override : 100000ULL;
    if (trials == 0) trials = 100000ULL;

    u128 *As = (u128 *)malloc((size_t)trials * sizeof(u128));
    if (!As) {
        printf("Could not allocate benchmark array for %llu samples.\n",
               (unsigned long long)trials);
        return 1;
    }

    Rng rng = {
        .s0 = 7364529176530163ULL ^ g_seed_offset,
        .s1 = 1442695040888963407ULL ^ (g_seed_offset << 1)
    };
    for (int i = 0; i < 200; i++) rng64(&rng);

    u128 pending_A = 0, pending_xP = 0, xP16 = 0;
    int have_pending_A = 0;
    double t0 = now_sec();
    for (u64 i = 0; i < trials; i++) {
        x16_montgomery_A128(&As[i], &xP16, &pending_A, &pending_xP, &have_pending_A,
                             &rng, p, rand_mask, sqrtm1, &mt);
    }
    double t_sample = now_sec() - t0;

    u64 has_root = 0;
    t0 = now_sec();
    for (u64 i = 0; i < trials; i++) {
        if (montgomery_x0_7_has_root128(As[i], p, &mt)) has_root++;
    }
    double t_filter = now_sec() - t0;
    u64 atkin_accept = trials - has_root;

    printf("X1(16) Atkin/Elkies ell=7 status benchmark\n");
    printf("samples = %llu\n", (unsigned long long)trials);
    printf("x16_sample_seconds = %.6f  rate_Mps = %.6f\n",
           t_sample, (double)trials / t_sample / 1e6);
    printf("atkin7_filter_seconds = %.6f  rate_Mps = %.6f\n",
           t_filter, (double)trials / t_filter / 1e6);
    printf("atkin7_seconds_per_curve = %.9e\n", t_filter / (double)trials);
    printf("x0_7_has_root_elkies = %llu/%llu  rate = %.6f\n",
           (unsigned long long)has_root,
           (unsigned long long)trials,
           (double)has_root / (double)trials);
    printf("x0_7_no_root_atkin_accept = %llu/%llu  rate = %.6f\n",
           (unsigned long long)atkin_accept,
           (unsigned long long)trials,
           (double)atkin_accept / (double)trials);

    free(As);
    return 0;
}

static int halve_once_all128(u128 *xs, int max_xs, u128 p, u128 A, u128 x,
                             u128 sqrtm1, const Mont128 *mt) {
    const u128 inv2 = (p + 1) >> 1;
    u128 x2 = mulmod128_mont(x, x, mt);
    u128 d = addmod128(addmod128(x2, mulmod128_mont(A, x, mt), p), 1, p);
    u128 sd;
    if (!sqrtmod_p5_128(&sd, d, p, sqrtm1, mt)) return 0;

    int count = 0;
    u128 roots_d[2] = { sd, submod128(0, sd, p) };
    for (int i = 0; i < 2; i++) {
        u128 u = addmod128(addmod128(x, x, p), addmod128(roots_d[i], roots_d[i], p), p);
        u128 w = submod128(mulmod128_mont(u, u, mt), 4, p);
        u128 sw;
        if (!sqrtmod_p5_128(&sw, w, p, sqrtm1, mt)) continue;
        u128 candidates[2] = {
            mulmod128_mont(addmod128(u, sw, p), inv2, mt),
            mulmod128_mont(submod128(u, sw, p), inv2, mt)
        };
        for (int j = 0; j < 2; j++) {
            if (candidates[j] != 0) {
                int dup = 0;
                for (int k = 0; k < count; k++) if (xs[k] == candidates[j]) { dup = 1; break; }
                if (!dup && count < max_xs) xs[count++] = candidates[j];
            }
        }
    }
    return count;
}

static int halve_once_first128(u128 *xo, u128 p, u128 A, u128 x,
                               u128 sqrtm1, const Mont128 *mt) {
    const u128 inv2 = (p + 1) >> 1;
    u128 x2 = mulmod128_mont(x, x, mt);
    u128 d = addmod128(addmod128(x2, mulmod128_mont(A, x, mt), p), 1, p);
    u128 sd;
    if (!sqrtmod_p5_128(&sd, d, p, sqrtm1, mt)) return 0;

    u128 roots_d[2] = { sd, submod128(0, sd, p) };
    for (int i = 0; i < 2; i++) {
        u128 u = addmod128(addmod128(x, x, p), addmod128(roots_d[i], roots_d[i], p), p);
        u128 w = submod128(mulmod128_mont(u, u, mt), 4, p);
        u128 sw;
        if (!sqrtmod_p5_128(&sw, w, p, sqrtm1, mt)) continue;
        u128 candidates[2] = {
            mulmod128_mont(addmod128(u, sw, p), inv2, mt),
            mulmod128_mont(submod128(u, sw, p), inv2, mt)
        };
        for (int j = 0; j < 2; j++) {
            if (candidates[j] != 0) {
                *xo = candidates[j];
                return 1;
            }
        }
    }
    return 0;
}

static int halve_once_first_count128(u128 *xo, u64 *d_sqrt_calls,
                                     u64 *w_sqrt_calls,
                                     u128 p, u128 A, u128 x,
                                     u128 sqrtm1, const Mont128 *mt) {
    const u128 inv2 = (p + 1) >> 1;
    u128 x2 = mulmod128_mont(x, x, mt);
    u128 d = addmod128(addmod128(x2, mulmod128_mont(A, x, mt), p), 1, p);
    u128 sd;
    (*d_sqrt_calls)++;
    if (!sqrtmod_p5_128(&sd, d, p, sqrtm1, mt)) return 0;

    u128 roots_d[2] = { sd, submod128(0, sd, p) };
    for (int i = 0; i < 2; i++) {
        u128 u = addmod128(addmod128(x, x, p), addmod128(roots_d[i], roots_d[i], p), p);
        u128 w = submod128(mulmod128_mont(u, u, mt), 4, p);
        u128 sw;
        (*w_sqrt_calls)++;
        if (!sqrtmod_p5_128(&sw, w, p, sqrtm1, mt)) continue;
        u128 candidates[2] = {
            mulmod128_mont(addmod128(u, sw, p), inv2, mt),
            mulmod128_mont(submod128(u, sw, p), inv2, mt)
        };
        for (int j = 0; j < 2; j++) {
            if (candidates[j] != 0) {
                *xo = candidates[j];
                return 1;
            }
        }
    }
    return 0;
}

typedef struct {
    u64 d_sqrt_calls;
    u64 w_sqrt_calls;
    u64 legendre_calls;
    u64 gate_evals;
    u64 gate_ties;
    u64 gate_one_good;
    u64 gate_both_good;
    u64 gate_both_bad;
    u64 gate_prefer_second;
    u64 gate_chose_second;
} GateStats128;

static int inverse_gate_good128(u128 p, u128 t, u128 u,
                                const Mont128 *mt, GateStats128 *stats) {
    u128 f = mulmod128_mont(t, addmod128(u, 2, p), mt);
    stats->legendre_calls++;
    if (f == 0) return 0;
    return powmod128_mont(f, (p - 1) >> 1, mt) == 1;
}

static int halve_once_gate128(u128 *xo, u128 p, u128 A, u128 x,
                              u128 sqrtm1, const Mont128 *mt,
                              GateStats128 *stats) {
    const u128 inv2 = (p + 1) >> 1;
    u128 x2 = mulmod128_mont(x, x, mt);
    u128 d = addmod128(addmod128(x2, mulmod128_mont(A, x, mt), p), 1, p);
    u128 sd;
    stats->d_sqrt_calls++;
    if (!sqrtmod_p5_128(&sd, d, p, sqrtm1, mt)) return 0;

    u128 roots_d[2] = { sd, submod128(0, sd, p) };
    u128 us[2];
    int good[2];
    for (int i = 0; i < 2; i++) {
        us[i] = addmod128(addmod128(x, x, p), addmod128(roots_d[i], roots_d[i], p), p);
        good[i] = inverse_gate_good128(p, x, us[i], mt, stats);
    }

    stats->gate_evals++;
    if (good[0] && good[1]) stats->gate_both_good++;
    else if (!good[0] && !good[1]) stats->gate_both_bad++;
    else stats->gate_one_good++;
    if (good[0] == good[1]) stats->gate_ties++;

    int order[2] = {0, 1};
    if (good[1] && !good[0]) {
        order[0] = 1;
        order[1] = 0;
        stats->gate_prefer_second++;
    }

    for (int oi = 0; oi < 2; oi++) {
        int i = order[oi];
        u128 u = us[i];
        u128 w = submod128(mulmod128_mont(u, u, mt), 4, p);
        u128 sw;
        stats->w_sqrt_calls++;
        if (!sqrtmod_p5_128(&sw, w, p, sqrtm1, mt)) continue;
        u128 candidates[2] = {
            mulmod128_mont(addmod128(u, sw, p), inv2, mt),
            mulmod128_mont(submod128(u, sw, p), inv2, mt)
        };
        for (int j = 0; j < 2; j++) {
            if (candidates[j] != 0) {
                if (i == 1) stats->gate_chose_second++;
                *xo = candidates[j];
                return 1;
            }
        }
    }
    return 0;
}

static int halve_once_first_flags128(u128 *xo, int *d_class, int *w_class,
                                     u128 p, u128 A, u128 x,
                                     u128 sqrtm1, const Mont128 *mt) {
    const u128 inv2 = (p + 1) >> 1;
    *d_class = 2;
    *w_class = 2;
    u128 x2 = mulmod128_mont(x, x, mt);
    u128 d = addmod128(addmod128(x2, mulmod128_mont(A, x, mt), p), 1, p);
    u128 sd;
    int d_used_i = 0;
    if (!sqrtmod_p5_flag_128(&sd, &d_used_i, d, p, sqrtm1, mt)) return 0;
    *d_class = d_used_i;

    u128 roots_d[2] = { sd, submod128(0, sd, p) };
    for (int i = 0; i < 2; i++) {
        u128 u = addmod128(addmod128(x, x, p), addmod128(roots_d[i], roots_d[i], p), p);
        u128 w = submod128(mulmod128_mont(u, u, mt), 4, p);
        u128 sw;
        int w_used_i = 0;
        if (!sqrtmod_p5_flag_128(&sw, &w_used_i, w, p, sqrtm1, mt)) continue;
        u128 candidates[2] = {
            mulmod128_mont(addmod128(u, sw, p), inv2, mt),
            mulmod128_mont(submod128(u, sw, p), inv2, mt)
        };
        for (int j = 0; j < 2; j++) {
            if (candidates[j] != 0) {
                *w_class = w_used_i;
                *xo = candidates[j];
                return 1;
            }
        }
    }
    return 0;
}

static int halve_extend128(u128 *xout, u128 p, u128 A, u128 x, int depth, int k,
                           u128 sqrtm1, const Mont128 *mt) {
    if (depth == k) {
        if (!verify128(p, A, x)) return 0;
        *xout = x;
        return 1;
    }
    u128 xs[4];
    int n = halve_once_all128(xs, 4, p, A, x, sqrtm1, mt);
    for (int i = 0; i < n; i++) {
        if (halve_extend128(xout, p, A, xs[i], depth + 1, k, sqrtm1, mt)) return 1;
    }
    return 0;
}

static int halve_chain_from_depth128(u128 *xout, u128 p, u128 A, u128 x, int depth, int k,
                                     u128 sqrtm1, const Mont128 *mt) {
    if (g_x16_halve_full_mode) {
        return halve_extend128(xout, p, A, x, depth, k, sqrtm1, mt);
    }
    for (; depth < k; depth++) {
        if (!halve_once_first128(&x, p, A, x, sqrtm1, mt)) return 0;
    }
    if (!verify128(p, A, x)) return 0;
    *xout = x;
    return 1;
}

static int compute_k(u128 p);

static int bench_x16_halve_stats_128(u128 p) {
    if ((u64)(p & 7) != 5) {
        printf("x16halvestats requires p ≡ 5 mod 8 for the X1(16) sampler.\n");
        return 1;
    }

    int k = compute_k(p);
    Mont128 mt; m128_init(&mt, p);
    int pbits = bitlen128(p);
    u128 rand_mask = pbits >= 128 ? (u128)0 - 1 : (((u128)1 << pbits) - 1);
    u128 sqrtm1 = powmod128_mont(2, (p - 1) >> 2, &mt);
    u64 trials = g_max_trials_override ? g_max_trials_override : 100000ULL;
    if (trials == 0) trials = 100000ULL;

    u64 depth_counts[65] = {0};
    u64 next_branch_counts[5] = {0};
    u64 full_depth_hits = 0;

    Rng rng = {
        .s0 = 7364529176530163ULL ^ g_seed_offset,
        .s1 = 1442695040888963407ULL ^ (g_seed_offset << 1)
    };
    for (int i = 0; i < 200; i++) rng64(&rng);

    u128 pending_A = 0, pending_xP = 0;
    int have_pending_A = 0;
    double t0 = now_sec();
    for (u64 i = 0; i < trials; i++) {
        u128 A, x;
        x16_montgomery_A128(&A, &x, &pending_A, &pending_xP, &have_pending_A,
                             &rng, p, rand_mask, sqrtm1, &mt);
        int depth = 4;
        while (depth < k) {
            u128 nx;
            if (!halve_once_first128(&nx, p, A, x, sqrtm1, &mt)) break;
            x = nx;
            depth++;
        }
        if (depth >= 0 && depth < 65) depth_counts[depth]++;
        if (depth == k) {
            full_depth_hits++;
        } else {
            u128 xs[4];
            int n = halve_once_all128(xs, 4, p, A, x, sqrtm1, &mt);
            if (n < 0) n = 0;
            if (n > 4) n = 4;
            next_branch_counts[n]++;
        }
    }
    double elapsed = now_sec() - t0;

    printf("X1(16) first-branch halving survival stats\n");
    printf("p = "); print128(p); printf("\n");
    printf("k = %d\n", k);
    printf("samples = %llu\n", (unsigned long long)trials);
    printf("elapsed_seconds = %.6f  rate_Mps = %.6f\n",
           elapsed, (double)trials / elapsed / 1e6);
    printf("full_first_branch_depth_%d = %llu/%llu  rate = %.9f\n",
           k, (unsigned long long)full_depth_hits, (unsigned long long)trials,
           (double)full_depth_hits / (double)trials);
    printf("depth_histogram_nonzero:\n");
    for (int d = 0; d < 65; d++) {
        if (depth_counts[d]) {
            printf("  depth=%d count=%llu rate=%.9f\n",
                   d, (unsigned long long)depth_counts[d],
                   (double)depth_counts[d] / (double)trials);
        }
    }
    printf("all_branch_options_at_first_stop:\n");
    for (int n = 0; n <= 4; n++) {
        if (next_branch_counts[n]) {
            printf("  options=%d count=%llu rate=%.9f\n",
                   n, (unsigned long long)next_branch_counts[n],
                   (double)next_branch_counts[n] / (double)trials);
        }
    }
    return 0;
}

static int append_unique_x128(u128 *xs, int n, int cap, u128 x) {
    for (int i = 0; i < n; i++) if (xs[i] == x) return n;
    if (n < cap) xs[n++] = x;
    return n;
}

static int bench_x16_branch_stats_128(u128 p) {
    if ((u64)(p & 7) != 5) {
        printf("x16branchstats requires p ≡ 5 mod 8 for the X1(16) sampler.\n");
        return 1;
    }

    const int start_depth = 4;
    int target_depth = g_x16_branch_target;
    const int cap = 65536;
    int k = compute_k(p);
    if (target_depth < start_depth + 1) target_depth = 12;
    if (target_depth > k) target_depth = k;
    if (target_depth > 64) target_depth = 64;
    Mont128 mt; m128_init(&mt, p);
    int pbits = bitlen128(p);
    u128 rand_mask = pbits >= 128 ? (u128)0 - 1 : (((u128)1 << pbits) - 1);
    u128 sqrtm1 = powmod128_mont(2, (p - 1) >> 2, &mt);
    u64 trials = g_max_trials_override ? g_max_trials_override : 50000ULL;
    if (trials == 0) trials = 50000ULL;

    u128 *frontier = (u128 *)malloc((size_t)cap * sizeof(u128));
    u128 *next = (u128 *)malloc((size_t)cap * sizeof(u128));
    if (!frontier || !next) {
        printf("Could not allocate branch frontier arrays.\n");
        free(frontier); free(next);
        return 1;
    }

    u64 first_survive[65] = {0};
    u64 all_survive[65] = {0};
    u64 frontier_sum[65] = {0};
    u64 frontier_max[65] = {0};
    u64 truncations[65] = {0};

    Rng rng = {
        .s0 = 7364529176530163ULL ^ g_seed_offset,
        .s1 = 1442695040888963407ULL ^ (g_seed_offset << 1)
    };
    for (int i = 0; i < 200; i++) rng64(&rng);

    u128 pending_A = 0, pending_xP = 0;
    int have_pending_A = 0;
    double t0 = now_sec();
    for (u64 i = 0; i < trials; i++) {
        u128 A, x16;
        if (g_x16_nonsplit_filter_mode) {
            x16_montgomery_A_nonsplit128(&A, &x16, &pending_A, &pending_xP, &have_pending_A,
                                         &rng, p, rand_mask, sqrtm1, &mt);
        } else {
            x16_montgomery_A128(&A, &x16, &pending_A, &pending_xP, &have_pending_A,
                                &rng, p, rand_mask, sqrtm1, &mt);
        }

        u128 first_x = x16;
        int first_alive = 1;
        first_survive[start_depth]++;
        all_survive[start_depth]++;
        frontier[0] = x16;
        int nfront = 1;
        frontier_sum[start_depth] += 1;
        if (frontier_max[start_depth] < 1) frontier_max[start_depth] = 1;

        for (int depth = start_depth; depth < target_depth; depth++) {
            if (first_alive) {
                u128 nx;
                if (halve_once_first128(&nx, p, A, first_x, sqrtm1, &mt)) {
                    first_x = nx;
                    first_survive[depth + 1]++;
                } else {
                    first_alive = 0;
                }
            }

            int nnext = 0;
            int truncated = 0;
            for (int j = 0; j < nfront; j++) {
                u128 xs[4];
                int nh = halve_once_all128(xs, 4, p, A, frontier[j], sqrtm1, &mt);
                for (int h = 0; h < nh; h++) {
                    int before = nnext;
                    nnext = append_unique_x128(next, nnext, cap, xs[h]);
                    if (before == nnext && nnext == cap) truncated = 1;
                }
            }

            if (nnext > 0) {
                all_survive[depth + 1]++;
                frontier_sum[depth + 1] += (u64)nnext;
                if (frontier_max[depth + 1] < (u64)nnext) frontier_max[depth + 1] = (u64)nnext;
                if (truncated) truncations[depth + 1]++;
            }
            u128 *tmp = frontier; frontier = next; next = tmp;
            nfront = nnext;
            if (nfront == 0 && !first_alive) break;
        }
    }
    double elapsed = now_sec() - t0;

    printf("X1(16) bounded first-vs-all branch halving stats\n");
    printf("p = "); print128(p); printf("\n");
    printf("k = %d\n", k);
    printf("samples = %llu\n", (unsigned long long)trials);
    printf("sample_class = %s\n", g_x16_nonsplit_filter_mode ? "nonsplit_y_filtered" : "all_x16");
    printf("start_depth = %d  target_depth = %d  frontier_cap = %d\n", start_depth, target_depth, cap);
    printf("elapsed_seconds = %.6f  rate_Mps = %.6f\n",
           elapsed, (double)trials / elapsed / 1e6);
    printf("depth first_survive all_survive all_avg_frontier all_max_frontier truncations\n");
    for (int d = start_depth; d <= target_depth; d++) {
        double first_rate = (double)first_survive[d] / (double)trials;
        double all_rate = (double)all_survive[d] / (double)trials;
        double avg_frontier = all_survive[d] ? (double)frontier_sum[d] / (double)all_survive[d] : 0.0;
        printf("%d %llu %.9f %llu %.9f %.3f %llu %llu\n",
               d,
               (unsigned long long)first_survive[d], first_rate,
               (unsigned long long)all_survive[d], all_rate,
               avg_frontier,
               (unsigned long long)frontier_max[d],
               (unsigned long long)truncations[d]);
    }

    free(frontier); free(next);
    return 0;
}

static int bench_x16_split_branch_stats_128(u128 p) {
    if ((u64)(p & 7) != 5) {
        printf("x16splitbranchstats requires p ≡ 5 mod 8 for the X1(16) sampler.\n");
        return 1;
    }

    const int start_depth = 4;
    int target_depth = g_x16_split_branch_target;
    const int cap = 65536;
    int k = compute_k(p);
    if (target_depth < start_depth + 1) target_depth = 14;
    if (target_depth > k) target_depth = k;
    if (target_depth > 64) target_depth = 64;
    Mont128 mt; m128_init(&mt, p);
    int pbits = bitlen128(p);
    u128 rand_mask = pbits >= 128 ? (u128)0 - 1 : (((u128)1 << pbits) - 1);
    u128 sqrtm1 = powmod128_mont(2, (p - 1) >> 2, &mt);
    u64 trials = g_max_trials_override ? g_max_trials_override : 50000ULL;
    if (trials == 0) trials = 50000ULL;

    u128 *frontier = (u128 *)malloc((size_t)cap * sizeof(u128));
    u128 *next = (u128 *)malloc((size_t)cap * sizeof(u128));
    if (!frontier || !next) {
        printf("Could not allocate split branch frontier arrays.\n");
        free(frontier); free(next);
        return 1;
    }

    u64 class_total[3] = {0};
    u64 first_survive[3][65] = {{0}};
    u64 all_survive[3][65] = {{0}};
    u64 frontier_sum[3][65] = {{0}};
    u64 frontier_max[3][65] = {{0}};
    u64 truncations[3][65] = {{0}};

    Rng rng = {
        .s0 = 7364529176530163ULL ^ g_seed_offset,
        .s1 = 1442695040888963407ULL ^ (g_seed_offset << 1)
    };
    for (int i = 0; i < 200; i++) rng64(&rng);

    u128 pending_A = 0, pending_xP = 0;
    int have_pending_A = 0;
    double t0 = now_sec();
    for (u64 i = 0; i < trials; i++) {
        u128 A, x16;
        x16_montgomery_A128(&A, &x16, &pending_A, &pending_xP, &have_pending_A,
                            &rng, p, rand_mask, sqrtm1, &mt);

        int cls = split_discriminant128(p, A, sqrtm1, &mt) ? 1 : 2;
        class_total[cls]++;

        u128 first_x = x16;
        int first_alive = 1;
        first_survive[cls][start_depth]++;
        all_survive[cls][start_depth]++;
        frontier[0] = x16;
        int nfront = 1;
        frontier_sum[cls][start_depth] += 1;
        if (frontier_max[cls][start_depth] < 1) frontier_max[cls][start_depth] = 1;

        for (int depth = start_depth; depth < target_depth; depth++) {
            if (first_alive) {
                u128 nx;
                if (halve_once_first128(&nx, p, A, first_x, sqrtm1, &mt)) {
                    first_x = nx;
                    first_survive[cls][depth + 1]++;
                } else {
                    first_alive = 0;
                }
            }

            int nnext = 0;
            int truncated = 0;
            for (int j = 0; j < nfront; j++) {
                u128 xs[4];
                int nh = halve_once_all128(xs, 4, p, A, frontier[j], sqrtm1, &mt);
                for (int h = 0; h < nh; h++) {
                    int before = nnext;
                    nnext = append_unique_x128(next, nnext, cap, xs[h]);
                    if (before == nnext && nnext == cap) truncated = 1;
                }
            }

            if (nnext > 0) {
                all_survive[cls][depth + 1]++;
                frontier_sum[cls][depth + 1] += (u64)nnext;
                if (frontier_max[cls][depth + 1] < (u64)nnext) {
                    frontier_max[cls][depth + 1] = (u64)nnext;
                }
                if (truncated) truncations[cls][depth + 1]++;
            }
            u128 *tmp = frontier; frontier = next; next = tmp;
            nfront = nnext;
            if (nfront == 0 && !first_alive) break;
        }
    }
    double elapsed = now_sec() - t0;

    printf("X1(16) split-class bounded first-vs-all branch halving stats\n");
    printf("p = "); print128(p); printf("\n");
    printf("k = %d\n", k);
    printf("samples = %llu\n", (unsigned long long)trials);
    printf("start_depth = %d  target_depth = %d  frontier_cap = %d\n", start_depth, target_depth, cap);
    printf("split criterion: Legendre(A^2 - 4, p) = 1\n");
    printf("elapsed_seconds = %.6f  rate_Mps = %.6f\n",
           elapsed, (double)trials / elapsed / 1e6);
    printf("class totals: split=%llu nonsplit=%llu\n",
           (unsigned long long)class_total[1],
           (unsigned long long)class_total[2]);
    printf("depth class first_survive first_rate all_survive all_rate all_avg_frontier all_max_frontier truncations\n");
    for (int d = start_depth; d <= target_depth; d++) {
        for (int cls = 1; cls <= 2; cls++) {
            double first_rate = class_total[cls] ? (double)first_survive[cls][d] / (double)class_total[cls] : 0.0;
            double all_rate = class_total[cls] ? (double)all_survive[cls][d] / (double)class_total[cls] : 0.0;
            double avg_frontier = all_survive[cls][d] ? (double)frontier_sum[cls][d] / (double)all_survive[cls][d] : 0.0;
            printf("%d %s %llu %.9f %llu %.9f %.3f %llu %llu\n",
                   d,
                   cls == 1 ? "split" : "nonsplit",
                   (unsigned long long)first_survive[cls][d], first_rate,
                   (unsigned long long)all_survive[cls][d], all_rate,
                   avg_frontier,
                   (unsigned long long)frontier_max[cls][d],
                   (unsigned long long)truncations[cls][d]);
        }
    }

    free(frontier); free(next);
    return 0;
}

static int bench_x16_gate_stats_128(u128 p) {
    if ((u64)(p & 7) != 5) {
        printf("x16gatestats requires p ≡ 5 mod 8 for the X1(16) sampler.\n");
        return 1;
    }

    const int start_depth = 4;
    int target_depth = g_x16_gate_target;
    const int cap = 65536;
    int k = compute_k(p);
    if (target_depth < start_depth + 1) target_depth = 12;
    if (target_depth > k) target_depth = k;
    if (target_depth > 64) target_depth = 64;
    Mont128 mt; m128_init(&mt, p);
    int pbits = bitlen128(p);
    u128 rand_mask = pbits >= 128 ? (u128)0 - 1 : (((u128)1 << pbits) - 1);
    u128 sqrtm1 = powmod128_mont(2, (p - 1) >> 2, &mt);
    u64 trials = g_max_trials_override ? g_max_trials_override : 50000ULL;
    if (trials == 0) trials = 50000ULL;

    u128 *frontier = (u128 *)malloc((size_t)cap * sizeof(u128));
    u128 *next = (u128 *)malloc((size_t)cap * sizeof(u128));
    if (!frontier || !next) {
        printf("Could not allocate gate-stat frontier arrays.\n");
        free(frontier); free(next);
        return 1;
    }

    u64 first_survive[65] = {0};
    u64 gate_survive[65] = {0};
    u64 all_survive[65] = {0};
    u64 frontier_sum[65] = {0};
    u64 frontier_max[65] = {0};
    u64 truncations[65] = {0};
    u64 first_d_sqrt_calls = 0;
    u64 first_w_sqrt_calls = 0;
    u64 first_half_calls = 0;
    u64 gate_half_calls = 0;
    u64 all_half_calls = 0;
    GateStats128 gate_stats = {0};

    Rng rng = {
        .s0 = 7364529176530163ULL ^ g_seed_offset,
        .s1 = 1442695040888963407ULL ^ (g_seed_offset << 1)
    };
    for (int i = 0; i < 200; i++) rng64(&rng);

    u128 pending_A = 0, pending_xP = 0;
    int have_pending_A = 0;
    double t0 = now_sec();
    for (u64 i = 0; i < trials; i++) {
        u128 A, x16;
        if (g_x16_nonsplit_filter_mode) {
            x16_montgomery_A_nonsplit128(&A, &x16, &pending_A, &pending_xP, &have_pending_A,
                                         &rng, p, rand_mask, sqrtm1, &mt);
        } else {
            x16_montgomery_A128(&A, &x16, &pending_A, &pending_xP, &have_pending_A,
                                &rng, p, rand_mask, sqrtm1, &mt);
        }

        u128 first_x = x16;
        int first_alive = 1;
        u128 gate_x = x16;
        int gate_alive = 1;
        first_survive[start_depth]++;
        gate_survive[start_depth]++;
        all_survive[start_depth]++;

        frontier[0] = x16;
        int nfront = 1;
        frontier_sum[start_depth] += 1;
        if (frontier_max[start_depth] < 1) frontier_max[start_depth] = 1;

        for (int depth = start_depth; depth < target_depth; depth++) {
            if (first_alive) {
                u128 nx;
                first_half_calls++;
                if (halve_once_first_count128(&nx, &first_d_sqrt_calls, &first_w_sqrt_calls,
                                              p, A, first_x, sqrtm1, &mt)) {
                    first_x = nx;
                    first_survive[depth + 1]++;
                } else {
                    first_alive = 0;
                }
            }

            if (gate_alive) {
                u128 nx;
                gate_half_calls++;
                if (halve_once_gate128(&nx, p, A, gate_x, sqrtm1, &mt, &gate_stats)) {
                    gate_x = nx;
                    gate_survive[depth + 1]++;
                } else {
                    gate_alive = 0;
                }
            }

            int nnext = 0;
            int truncated = 0;
            for (int j = 0; j < nfront; j++) {
                u128 xs[4];
                all_half_calls++;
                int nh = halve_once_all128(xs, 4, p, A, frontier[j], sqrtm1, &mt);
                for (int h = 0; h < nh; h++) {
                    int before = nnext;
                    nnext = append_unique_x128(next, nnext, cap, xs[h]);
                    if (before == nnext && nnext == cap) truncated = 1;
                }
            }

            if (nnext > 0) {
                all_survive[depth + 1]++;
                frontier_sum[depth + 1] += (u64)nnext;
                if (frontier_max[depth + 1] < (u64)nnext) frontier_max[depth + 1] = (u64)nnext;
                if (truncated) truncations[depth + 1]++;
            }
            u128 *tmp = frontier; frontier = next; next = tmp;
            nfront = nnext;
            if (nfront == 0 && !first_alive && !gate_alive) break;
        }
    }
    double elapsed = now_sec() - t0;

    printf("X1(16) inverse-gate branch halving stats\n");
    printf("p = "); print128(p); printf("\n");
    printf("k = %d\n", k);
    printf("samples = %llu\n", (unsigned long long)trials);
    printf("sample_class = %s\n", g_x16_nonsplit_filter_mode ? "nonsplit_y_filtered" : "all_x16");
    printf("start_depth = %d  target_depth = %d  frontier_cap = %d\n", start_depth, target_depth, cap);
    printf("gate_score = Legendre(t*(u+2), p), where u = 2*t +/- 2*sqrt(t^2+A*t+1)\n");
    printf("elapsed_seconds = %.6f  rate_Mps = %.6f\n",
           elapsed, (double)trials / elapsed / 1e6);
    printf("half_calls first=%llu gate=%llu all=%llu\n",
           (unsigned long long)first_half_calls,
           (unsigned long long)gate_half_calls,
           (unsigned long long)all_half_calls);
    printf("sqrt_calls first_d=%llu first_w=%llu gate_d=%llu gate_w=%llu gate_legendre=%llu\n",
           (unsigned long long)first_d_sqrt_calls,
           (unsigned long long)first_w_sqrt_calls,
           (unsigned long long)gate_stats.d_sqrt_calls,
           (unsigned long long)gate_stats.w_sqrt_calls,
           (unsigned long long)gate_stats.legendre_calls);
    printf("gate_evals=%llu ties=%llu one_good=%llu both_good=%llu both_bad=%llu prefer_second=%llu chose_second=%llu\n",
           (unsigned long long)gate_stats.gate_evals,
           (unsigned long long)gate_stats.gate_ties,
           (unsigned long long)gate_stats.gate_one_good,
           (unsigned long long)gate_stats.gate_both_good,
           (unsigned long long)gate_stats.gate_both_bad,
           (unsigned long long)gate_stats.gate_prefer_second,
           (unsigned long long)gate_stats.gate_chose_second);
    printf("per_sample first_d=%.3f first_w=%.3f gate_d=%.3f gate_w=%.3f gate_legendre=%.3f\n",
           (double)first_d_sqrt_calls / (double)trials,
           (double)first_w_sqrt_calls / (double)trials,
           (double)gate_stats.d_sqrt_calls / (double)trials,
           (double)gate_stats.w_sqrt_calls / (double)trials,
           (double)gate_stats.legendre_calls / (double)trials);
    printf("depth first_survive gate_survive all_survive all_avg_frontier all_max_frontier truncations\n");
    for (int d = start_depth; d <= target_depth; d++) {
        double first_rate = (double)first_survive[d] / (double)trials;
        double gate_rate = (double)gate_survive[d] / (double)trials;
        double all_rate = (double)all_survive[d] / (double)trials;
        double avg_frontier = all_survive[d] ? (double)frontier_sum[d] / (double)all_survive[d] : 0.0;
        printf("%d %llu %.9f %llu %.9f %llu %.9f %.3f %llu %llu\n",
               d,
               (unsigned long long)first_survive[d], first_rate,
               (unsigned long long)gate_survive[d], gate_rate,
               (unsigned long long)all_survive[d], all_rate,
               avg_frontier,
               (unsigned long long)frontier_max[d],
               (unsigned long long)truncations[d]);
    }

    free(frontier); free(next);
    return 0;
}

static const char *sqrt_flag_name(int c) {
    switch (c) {
        case 0: return "sqrt_no_i";
        case 1: return "sqrt_used_i";
        case 2: return "no_sqrt";
        default: return "other";
    }
}

static const char *g4_class_name(int c) {
    switch (c) {
        case 0: return "+1";
        case 1: return "-1";
        case 2: return "+i";
        case 3: return "-i";
        case 4: return "zero";
        default: return "other";
    }
}

static int legendre_class128(u128 p, u128 f, const Mont128 *mt) {
    f %= p;
    if (f == 0) return 2;
    u128 q = powmod128_mont(f, (p - 1) >> 1, mt);
    return q == 1 ? 1 : 0; /* 0=-1, 1=+1, 2=zero */
}

static const char *legendre_class_name(int c) {
    switch (c) {
        case 0: return "-1";
        case 1: return "+1";
        case 2: return "zero";
        default: return "other";
    }
}

static void print_feature_row(const char *feature, const char *label,
                              u64 total, u64 d12_survive, u64 d14_survive,
                              u64 target_survive, int d12, int d14,
                              int target_depth) {
    double r12 = total ? (double)d12_survive / (double)total : 0.0;
    double r14 = total ? (double)d14_survive / (double)total : 0.0;
    double rt = total ? (double)target_survive / (double)total : 0.0;
    printf("%s %s %llu %llu %.9f %llu %.9f %llu %.9f\n",
           feature, label, (unsigned long long)total,
           (unsigned long long)d12_survive, r12,
           (unsigned long long)d14_survive, r14,
           (unsigned long long)target_survive, rt);
    (void)d12;
    (void)d14;
    (void)target_depth;
}

static int bench_x16_quartic_stats_128(u128 p) {
    if ((u64)(p & 7) != 5) {
        printf("x16quarticstats requires p ≡ 5 mod 8 for sqrt-correction telemetry.\n");
        return 1;
    }

    const int start_depth = 4;
    int target_depth = g_x16_quartic_target;
    int k = compute_k(p);
    if (target_depth < start_depth + 1) target_depth = 14;
    if (target_depth > k) target_depth = k;
    if (target_depth > 64) target_depth = 64;
    int d12 = target_depth < 12 ? target_depth : 12;
    int d14 = target_depth < 14 ? target_depth : 14;

    Mont128 mt; m128_init(&mt, p);
    int pbits = bitlen128(p);
    u128 rand_mask = pbits >= 128 ? (u128)0 - 1 : (((u128)1 << pbits) - 1);
    u128 sqrtm1 = powmod128_mont(2, (p - 1) >> 2, &mt);
    u64 trials = g_max_trials_override ? g_max_trials_override : 50000ULL;
    if (trials == 0) trials = 50000ULL;

    u64 d16_total[2] = {0};
    u64 d16_survive[2][65] = {{0}};
    u64 g4_total[6] = {0};
    u64 g4_survive[6][65] = {{0}};
    u64 d_total[3] = {0};
    u64 d_survive[3][65] = {{0}};
    u64 w_total[3] = {0};
    u64 w_survive[3][65] = {{0}};
    u64 combo_total[9] = {0};
    u64 combo_survive[9][65] = {{0}};
    enum { QFEATURES = 14 };
    const char *qfeature_names[QFEATURES] = {
        "A_minus_2", "A_plus_2", "A2_minus_3",
        "y", "y_minus_1", "y_minus_2", "y_plus_1",
        "y2_minus_2", "y2_minus_2y_plus_2", "y2_minus_4y_plus_2",
        "y2_minus_y_plus_1", "y2_plus_y_plus_1", "y2_minus_3y_plus_1",
        "first_d_y_gate"
    };
    u64 qfeature_total[QFEATURES][3] = {{0}};
    u64 qfeature_survive[QFEATURES][3][65] = {{{0}}};
    enum { QATOMS = QFEATURES * 2 };
    u64 qpair_total[QATOMS][QATOMS] = {{0}};
    u64 qpair_survive[QATOMS][QATOMS][65] = {{{0}}};

    Rng rng = {
        .s0 = 11995408973635179863ULL ^ g_seed_offset,
        .s1 = 1442695040888963407ULL ^ (g_seed_offset << 1)
    };
    for (int i = 0; i < 200; i++) rng64(&rng);

    u128 pending_A = 0, pending_xP = 0, pending_y = 0, pending_y2 = 0;
    int pending_d16_flag = 0, pending_g4_class = 0;
    int have_pending_A = 0;
    double t0 = now_sec();
    for (u64 i = 0; i < trials; i++) {
        u128 A, x16, y, y2;
        int d16_flag = 0, g4_class = 0;
        x16_montgomery_A_quartic128(&A, &x16, &y, &y2, &d16_flag, &g4_class,
                                    &pending_A, &pending_xP,
                                    &pending_y, &pending_y2,
                                    &pending_d16_flag, &pending_g4_class,
                                    &have_pending_A,
                                    g_x16_nonsplit_filter_mode,
                                    &rng, p, rand_mask, sqrtm1, &mt);
        if (g4_class < 0 || g4_class > 5) g4_class = 5;

        u128 A2 = mulmod128_mont(A, A, &mt);
        u128 two_y = addmod128(y, y, p);
        u128 three_y = addmod128(two_y, y, p);
        u128 four_y = addmod128(two_y, two_y, p);
        u128 qvals[QFEATURES] = {
            submod128(A, 2, p),
            addmod128(A, 2, p),
            submod128(A2, 3, p),
            y,
            submod128(y, 1, p),
            submod128(y, 2, p),
            addmod128(y, 1, p),
            submod128(y2, 2, p),
            addmod128(submod128(y2, two_y, p), 2, p),
            addmod128(submod128(y2, four_y, p), 2, p),
            addmod128(submod128(y2, y, p), 1, p),
            addmod128(addmod128(y2, y, p), 1, p),
            addmod128(submod128(y2, three_y, p), 1, p),
            mulmod128_mont(
                mulmod128_mont(submod128(y, 1, p), submod128(y2, 2, p), &mt),
                addmod128(submod128(y2, two_y, p), 2, p),
                &mt)
        };
        int qclasses[QFEATURES];
        for (int qf = 0; qf < QFEATURES; qf++) {
            qclasses[qf] = legendre_class128(p, qvals[qf], &mt);
            qfeature_total[qf][qclasses[qf]]++;
        }
        int qatoms[QFEATURES];
        int nqatoms = 0;
        for (int qf = 0; qf < QFEATURES; qf++) {
            if (qclasses[qf] <= 1) qatoms[nqatoms++] = 2 * qf + qclasses[qf];
        }
        for (int ai = 0; ai < nqatoms; ai++) {
            for (int bi = ai + 1; bi < nqatoms; bi++) {
                int a = qatoms[ai], b = qatoms[bi];
                if (a > b) { int tmp = a; a = b; b = tmp; }
                qpair_total[a][b]++;
            }
        }

        int d_class = 2, w_class = 2;
        int max_depth = start_depth;
        u128 x = x16, nx;
        if (halve_once_first_flags128(&nx, &d_class, &w_class, p, A, x, sqrtm1, &mt)) {
            x = nx;
            max_depth = start_depth + 1;
            for (int depth = start_depth + 1; depth < target_depth; depth++) {
                if (!halve_once_first128(&nx, p, A, x, sqrtm1, &mt)) break;
                x = nx;
                max_depth = depth + 1;
            }
        }
        if (d_class < 0 || d_class > 2) d_class = 2;
        if (w_class < 0 || w_class > 2) w_class = 2;
        int combo = d_class * 3 + w_class;

        d16_total[d16_flag ? 1 : 0]++;
        g4_total[g4_class]++;
        d_total[d_class]++;
        w_total[w_class]++;
        combo_total[combo]++;
        for (int depth = start_depth; depth <= max_depth; depth++) {
            d16_survive[d16_flag ? 1 : 0][depth]++;
            g4_survive[g4_class][depth]++;
            d_survive[d_class][depth]++;
            w_survive[w_class][depth]++;
            combo_survive[combo][depth]++;
            for (int qf = 0; qf < QFEATURES; qf++) {
                qfeature_survive[qf][qclasses[qf]][depth]++;
            }
            for (int ai = 0; ai < nqatoms; ai++) {
                for (int bi = ai + 1; bi < nqatoms; bi++) {
                    int a = qatoms[ai], b = qatoms[bi];
                    if (a > b) { int tmp = a; a = b; b = tmp; }
                    qpair_survive[a][b][depth]++;
                }
            }
        }
    }
    double elapsed = now_sec() - t0;

    printf("X1(16) quartic sqrt-correction survival stats\n");
    printf("p = "); print128(p); printf("\n");
    printf("k = %d\n", k);
    printf("samples = %llu\n", (unsigned long long)trials);
    printf("sample_class = %s\n", g_x16_nonsplit_filter_mode ? "nonsplit_y_filtered" : "all_x16");
    printf("start_depth = %d  target_depth = %d\n", start_depth, target_depth);
    printf("sqrt_flag_classes: 0=sqrt_no_i 1=sqrt_used_i 2=no_sqrt\n");
    printf("g4_classes: 0=+1 1=-1 2=+i 3=-i 4=zero 5=other\n");
    printf("elapsed_seconds = %.6f  rate_Mps = %.6f\n",
           elapsed, (double)trials / elapsed / 1e6);
    printf("feature class total survive_d%d rate_d%d survive_d%d rate_d%d survive_target rate_target\n",
           d12, d12, d14, d14);

    for (int c = 0; c < 2; c++) {
        print_feature_row("D16_sqrt", sqrt_flag_name(c), d16_total[c],
                          d16_survive[c][d12], d16_survive[c][d14],
                          d16_survive[c][target_depth], d12, d14, target_depth);
    }
    for (int c = 0; c < 6; c++) {
        print_feature_row("G4", g4_class_name(c), g4_total[c],
                          g4_survive[c][d12], g4_survive[c][d14],
                          g4_survive[c][target_depth], d12, d14, target_depth);
    }
    for (int c = 0; c < 3; c++) {
        print_feature_row("first_d_sqrt", sqrt_flag_name(c), d_total[c],
                          d_survive[c][d12], d_survive[c][d14],
                          d_survive[c][target_depth], d12, d14, target_depth);
    }
    for (int c = 0; c < 3; c++) {
        print_feature_row("first_w_sqrt", sqrt_flag_name(c), w_total[c],
                          w_survive[c][d12], w_survive[c][d14],
                          w_survive[c][target_depth], d12, d14, target_depth);
    }
    for (int d = 0; d < 3; d++) {
        for (int w = 0; w < 3; w++) {
            char label[64];
            snprintf(label, sizeof(label), "%s/%s", sqrt_flag_name(d), sqrt_flag_name(w));
            int c = d * 3 + w;
            print_feature_row("first_d_w_combo", label, combo_total[c],
                              combo_survive[c][d12], combo_survive[c][d14],
                              combo_survive[c][target_depth], d12, d14, target_depth);
        }
    }
    for (int qf = 0; qf < QFEATURES; qf++) {
        for (int c = 0; c < 3; c++) {
            print_feature_row(qfeature_names[qf], legendre_class_name(c),
                              qfeature_total[qf][c],
                              qfeature_survive[qf][c][d12],
                              qfeature_survive[qf][c][d14],
                              qfeature_survive[qf][c][target_depth],
                              d12, d14, target_depth);
        }
    }
    printf("qfeature_pair feature1 class1 feature2 class2 total survive_d%d rate_d%d survive_d%d rate_d%d survive_target rate_target\n",
           d12, d12, d14, d14);
    for (int a = 0; a < QATOMS; a++) {
        int qf1 = a / 2;
        int c1 = a % 2;
        for (int b = a + 1; b < QATOMS; b++) {
            int qf2 = b / 2;
            int c2 = b % 2;
            u64 total = qpair_total[a][b];
            if (!total) continue;
            double r12 = (double)qpair_survive[a][b][d12] / (double)total;
            double r14 = (double)qpair_survive[a][b][d14] / (double)total;
            double rt = (double)qpair_survive[a][b][target_depth] / (double)total;
            printf("qpair %s %s %s %s %llu %llu %.9f %llu %.9f %llu %.9f\n",
                   qfeature_names[qf1], legendre_class_name(c1),
                   qfeature_names[qf2], legendre_class_name(c2),
                   (unsigned long long)total,
                   (unsigned long long)qpair_survive[a][b][d12], r12,
                   (unsigned long long)qpair_survive[a][b][d14], r14,
                   (unsigned long long)qpair_survive[a][b][target_depth], rt);
        }
    }

    return 0;
}

static u64 count_halve_descendants128(u128 p, u128 A, u128 x, int levels,
                                      u64 cap, u64 *half_calls,
                                      u128 sqrtm1, const Mont128 *mt) {
    if (levels <= 0) return 1;
    if (cap == 0) return 0;

    u128 xs[4];
    (*half_calls)++;
    int n = halve_once_all128(xs, 4, p, A, x, sqrtm1, mt);
    if (n <= 0) return 0;

    u64 total = 0;
    for (int i = 0; i < n; i++) {
        u64 room = cap - total;
        u64 c = count_halve_descendants128(p, A, xs[i], levels - 1, room,
                                           half_calls, sqrtm1, mt);
        total += c;
        if (total >= cap) return cap;
    }
    return total;
}

static int choose_lookahead_half128(u128 *xo, u128 p, u128 A, u128 x,
                                    int lookahead_depth, u64 descendant_cap,
                                    u64 *half_calls, u128 sqrtm1,
                                    const Mont128 *mt) {
    u128 xs[4];
    (*half_calls)++;
    int n = halve_once_all128(xs, 4, p, A, x, sqrtm1, mt);
    if (n <= 0) return 0;

    int best = 0;
    u64 best_score = 0;
    for (int i = 0; i < n; i++) {
        u64 score = count_halve_descendants128(p, A, xs[i],
                                               lookahead_depth - 1,
                                               descendant_cap,
                                               half_calls, sqrtm1, mt);
        if (i == 0 || score > best_score) {
            best = i;
            best_score = score;
        }
    }
    *xo = xs[best];
    return 1;
}

static int bench_x16_lookahead_stats_128(u128 p) {
    if ((u64)(p & 7) != 5) {
        printf("x16lookaheadstats requires p ≡ 5 mod 8 for the X1(16) sampler.\n");
        return 1;
    }

    const int start_depth = 4;
    int target_depth = g_x16_lookahead_target;
    int lookahead_depth = g_x16_lookahead_depth;
    const int cap = 65536;
    const u64 descendant_cap = 65536;
    int k = compute_k(p);
    if (target_depth < start_depth + 1) target_depth = 12;
    if (target_depth > k) target_depth = k;
    if (target_depth > 64) target_depth = 64;
    if (lookahead_depth < 1) lookahead_depth = 1;
    if (lookahead_depth > 6) lookahead_depth = 6;

    Mont128 mt; m128_init(&mt, p);
    int pbits = bitlen128(p);
    u128 rand_mask = pbits >= 128 ? (u128)0 - 1 : (((u128)1 << pbits) - 1);
    u128 sqrtm1 = powmod128_mont(2, (p - 1) >> 2, &mt);
    u64 trials = g_max_trials_override ? g_max_trials_override : 50000ULL;
    if (trials == 0) trials = 50000ULL;

    u128 *frontier = (u128 *)malloc((size_t)cap * sizeof(u128));
    u128 *next = (u128 *)malloc((size_t)cap * sizeof(u128));
    if (!frontier || !next) {
        printf("Could not allocate branch frontier arrays.\n");
        free(frontier); free(next);
        return 1;
    }

    u64 first_survive[65] = {0};
    u64 lookahead_survive[65] = {0};
    u64 all_survive[65] = {0};
    u64 frontier_sum[65] = {0};
    u64 frontier_max[65] = {0};
    u64 truncations[65] = {0};
    u64 first_half_calls = 0;
    u64 lookahead_half_calls = 0;
    u64 all_half_calls = 0;

    Rng rng = {
        .s0 = 7364529176530163ULL ^ g_seed_offset,
        .s1 = 1442695040888963407ULL ^ (g_seed_offset << 1)
    };
    for (int i = 0; i < 200; i++) rng64(&rng);

    u128 pending_A = 0, pending_xP = 0;
    int have_pending_A = 0;
    double t0 = now_sec();
    for (u64 i = 0; i < trials; i++) {
        u128 A, x16;
        x16_montgomery_A128(&A, &x16, &pending_A, &pending_xP, &have_pending_A,
                             &rng, p, rand_mask, sqrtm1, &mt);

        u128 first_x = x16;
        int first_alive = 1;
        u128 lookahead_x = x16;
        int lookahead_alive = 1;
        first_survive[start_depth]++;
        lookahead_survive[start_depth]++;
        all_survive[start_depth]++;

        frontier[0] = x16;
        int nfront = 1;
        frontier_sum[start_depth] += 1;
        if (frontier_max[start_depth] < 1) frontier_max[start_depth] = 1;

        for (int depth = start_depth; depth < target_depth; depth++) {
            if (first_alive) {
                u128 nx;
                first_half_calls++;
                if (halve_once_first128(&nx, p, A, first_x, sqrtm1, &mt)) {
                    first_x = nx;
                    first_survive[depth + 1]++;
                } else {
                    first_alive = 0;
                }
            }

            if (lookahead_alive) {
                u128 nx;
                if (choose_lookahead_half128(&nx, p, A, lookahead_x,
                                             lookahead_depth, descendant_cap,
                                             &lookahead_half_calls, sqrtm1, &mt)) {
                    lookahead_x = nx;
                    lookahead_survive[depth + 1]++;
                } else {
                    lookahead_alive = 0;
                }
            }

            int nnext = 0;
            int truncated = 0;
            for (int j = 0; j < nfront; j++) {
                u128 xs[4];
                all_half_calls++;
                int nh = halve_once_all128(xs, 4, p, A, frontier[j], sqrtm1, &mt);
                for (int h = 0; h < nh; h++) {
                    int before = nnext;
                    nnext = append_unique_x128(next, nnext, cap, xs[h]);
                    if (before == nnext && nnext == cap) truncated = 1;
                }
            }

            if (nnext > 0) {
                all_survive[depth + 1]++;
                frontier_sum[depth + 1] += (u64)nnext;
                if (frontier_max[depth + 1] < (u64)nnext) frontier_max[depth + 1] = (u64)nnext;
                if (truncated) truncations[depth + 1]++;
            }
            u128 *tmp = frontier; frontier = next; next = tmp;
            nfront = nnext;
            if (nfront == 0 && !first_alive && !lookahead_alive) break;
        }
    }
    double elapsed = now_sec() - t0;

    printf("X1(16) bounded lookahead branch halving stats\n");
    printf("p = "); print128(p); printf("\n");
    printf("k = %d\n", k);
    printf("samples = %llu\n", (unsigned long long)trials);
    printf("start_depth = %d  target_depth = %d  frontier_cap = %d\n", start_depth, target_depth, cap);
    printf("lookahead_depth = %d  descendant_cap = %llu\n",
           lookahead_depth, (unsigned long long)descendant_cap);
    printf("elapsed_seconds = %.6f  rate_Mps = %.6f\n",
           elapsed, (double)trials / elapsed / 1e6);
    printf("half_calls first=%llu lookahead=%llu all=%llu\n",
           (unsigned long long)first_half_calls,
           (unsigned long long)lookahead_half_calls,
           (unsigned long long)all_half_calls);
    printf("half_calls_per_sample first=%.3f lookahead=%.3f all=%.3f\n",
           (double)first_half_calls / (double)trials,
           (double)lookahead_half_calls / (double)trials,
           (double)all_half_calls / (double)trials);
    printf("depth first_survive lookahead_survive all_survive all_avg_frontier all_max_frontier truncations\n");
    for (int d = start_depth; d <= target_depth; d++) {
        double first_rate = (double)first_survive[d] / (double)trials;
        double lookahead_rate = (double)lookahead_survive[d] / (double)trials;
        double all_rate = (double)all_survive[d] / (double)trials;
        double avg_frontier = all_survive[d] ? (double)frontier_sum[d] / (double)all_survive[d] : 0.0;
        printf("%d %llu %.9f %llu %.9f %llu %.9f %.3f %llu %llu\n",
               d,
               (unsigned long long)first_survive[d], first_rate,
               (unsigned long long)lookahead_survive[d], lookahead_rate,
               (unsigned long long)all_survive[d], all_rate,
               avg_frontier,
               (unsigned long long)frontier_max[d],
               (unsigned long long)truncations[d]);
    }

    free(frontier); free(next);
    return 0;
}

static int halve_once_ordered128(u128 *xo, int *labelo, u128 p, u128 A, u128 x,
                                 const int order[4], u64 *d_sqrt_calls,
                                 u64 *w_sqrt_calls, u128 sqrtm1,
                                 const Mont128 *mt) {
    const u128 inv2 = (p + 1) >> 1;
    u128 x2 = mulmod128_mont(x, x, mt);
    u128 d = addmod128(addmod128(x2, mulmod128_mont(A, x, mt), p), 1, p);
    u128 sd;
    (*d_sqrt_calls)++;
    if (!sqrtmod_p5_128(&sd, d, p, sqrtm1, mt)) return 0;

    u128 roots_d[2] = { sd, submod128(0, sd, p) };
    for (int oi = 0; oi < 4; oi++) {
        int label = order[oi];
        if (label < 0 || label > 3) continue;
        int i = label >> 1;
        int j = label & 1;
        u128 u = addmod128(addmod128(x, x, p), addmod128(roots_d[i], roots_d[i], p), p);
        u128 w = submod128(mulmod128_mont(u, u, mt), 4, p);
        u128 sw;
        (*w_sqrt_calls)++;
        if (!sqrtmod_p5_128(&sw, w, p, sqrtm1, mt)) continue;
        u128 candidate = j == 0
            ? mulmod128_mont(addmod128(u, sw, p), inv2, mt)
            : mulmod128_mont(submod128(u, sw, p), inv2, mt);
        if (candidate != 0) {
            *xo = candidate;
            *labelo = label;
            return 1;
        }
    }
    return 0;
}

static int bench_x16_label_stats_128(u128 p) {
    if ((u64)(p & 7) != 5) {
        printf("x16labelstats requires p ≡ 5 mod 8 for the X1(16) sampler.\n");
        return 1;
    }

    const int start_depth = 4;
    int target_depth = g_x16_label_target;
    int k = compute_k(p);
    if (target_depth < start_depth + 1) target_depth = 14;
    if (target_depth > k) target_depth = k;
    if (target_depth > 64) target_depth = 64;

    Mont128 mt; m128_init(&mt, p);
    int pbits = bitlen128(p);
    u128 rand_mask = pbits >= 128 ? (u128)0 - 1 : (((u128)1 << pbits) - 1);
    u128 sqrtm1 = powmod128_mont(2, (p - 1) >> 2, &mt);
    u64 trials = g_max_trials_override ? g_max_trials_override : 100000ULL;
    if (trials == 0) trials = 100000ULL;

    const int orders[4][4] = {
        {0, 1, 2, 3},
        {1, 0, 2, 3},
        {2, 0, 1, 3},
        {3, 0, 1, 2}
    };
    const char *names[4] = {
        "natural_0123",
        "prefer_1",
        "prefer_2",
        "prefer_3"
    };

    u64 survive[4][65] = {{0}};
    u64 chosen_label[4][4] = {{0}};
    u64 d_sqrt_calls[4] = {0};
    u64 w_sqrt_calls[4] = {0};

    Rng rng = {
        .s0 = 7364529176530163ULL ^ g_seed_offset,
        .s1 = 1442695040888963407ULL ^ (g_seed_offset << 1)
    };
    for (int i = 0; i < 200; i++) rng64(&rng);

    u128 pending_A = 0, pending_xP = 0;
    int have_pending_A = 0;
    double t0 = now_sec();
    for (u64 i = 0; i < trials; i++) {
        u128 A, x16;
        x16_montgomery_A128(&A, &x16, &pending_A, &pending_xP, &have_pending_A,
                             &rng, p, rand_mask, sqrtm1, &mt);

        u128 xs[4] = {x16, x16, x16, x16};
        int alive[4] = {1, 1, 1, 1};
        for (int s = 0; s < 4; s++) survive[s][start_depth]++;

        for (int depth = start_depth; depth < target_depth; depth++) {
            int any_alive = 0;
            for (int s = 0; s < 4; s++) {
                if (!alive[s]) continue;
                u128 nx;
                int label = -1;
                if (halve_once_ordered128(&nx, &label, p, A, xs[s], orders[s],
                                          &d_sqrt_calls[s], &w_sqrt_calls[s],
                                          sqrtm1, &mt)) {
                    xs[s] = nx;
                    survive[s][depth + 1]++;
                    if (label >= 0 && label < 4) chosen_label[s][label]++;
                    any_alive = 1;
                } else {
                    alive[s] = 0;
                }
            }
            if (!any_alive) break;
        }
    }
    double elapsed = now_sec() - t0;

    printf("X1(16) halving sign-label order stats\n");
    printf("p = "); print128(p); printf("\n");
    printf("k = %d\n", k);
    printf("samples = %llu\n", (unsigned long long)trials);
    printf("start_depth = %d  target_depth = %d\n", start_depth, target_depth);
    printf("label = 2*d_sign_index + w_sign_index, with natural order 0,1,2,3\n");
    printf("elapsed_seconds = %.6f  rate_Mps = %.6f\n",
           elapsed, (double)trials / elapsed / 1e6);
    printf("strategy half_call_counts d_sqrt w_sqrt d_per_sample w_per_sample chosen_labels_0_1_2_3\n");
    for (int s = 0; s < 4; s++) {
        printf("%s %llu %llu %.3f %.3f %llu %llu %llu %llu\n",
               names[s],
               (unsigned long long)d_sqrt_calls[s],
               (unsigned long long)w_sqrt_calls[s],
               (double)d_sqrt_calls[s] / (double)trials,
               (double)w_sqrt_calls[s] / (double)trials,
               (unsigned long long)chosen_label[s][0],
               (unsigned long long)chosen_label[s][1],
               (unsigned long long)chosen_label[s][2],
               (unsigned long long)chosen_label[s][3]);
    }
    printf("depth natural prefer_1 prefer_2 prefer_3\n");
    for (int d = start_depth; d <= target_depth; d++) {
        printf("%d", d);
        for (int s = 0; s < 4; s++) {
            printf(" %llu %.9f",
                   (unsigned long long)survive[s][d],
                   (double)survive[s][d] / (double)trials);
        }
        printf("\n");
    }
    return 0;
}

static int bench_x16_split_stats_128(u128 p) {
    if ((u64)(p & 7) != 5) {
        printf("x16splitstats requires p ≡ 5 mod 8 for the X1(16) sampler.\n");
        return 1;
    }

    const int start_depth = 4;
    int target_depth = g_x16_split_target;
    int k = compute_k(p);
    if (target_depth < start_depth + 1) target_depth = 20;
    if (target_depth > k) target_depth = k;
    if (target_depth > 64) target_depth = 64;

    Mont128 mt; m128_init(&mt, p);
    int pbits = bitlen128(p);
    u128 rand_mask = pbits >= 128 ? (u128)0 - 1 : (((u128)1 << pbits) - 1);
    u128 sqrtm1 = powmod128_mont(2, (p - 1) >> 2, &mt);
    u64 trials = g_max_trials_override ? g_max_trials_override : 100000ULL;
    if (trials == 0) trials = 100000ULL;

    u64 split_total[3] = {0};
    u64 depth_counts[3][65] = {{0}};
    u64 survive_ge[3][65] = {{0}};

    Rng rng = {
        .s0 = 7364529176530163ULL ^ g_seed_offset,
        .s1 = 1442695040888963407ULL ^ (g_seed_offset << 1)
    };
    for (int i = 0; i < 200; i++) rng64(&rng);

    u128 pending_A = 0, pending_xP = 0;
    int have_pending_A = 0;
    double t0 = now_sec();
    for (u64 i = 0; i < trials; i++) {
        u128 A, x;
        x16_montgomery_A128(&A, &x, &pending_A, &pending_xP, &have_pending_A,
                             &rng, p, rand_mask, sqrtm1, &mt);

        u128 A2 = mulmod128_mont(A, A, &mt);
        u128 disc = submod128(A2, 4, p);
        u128 leg = powmod128_mont(disc, (p - 1) >> 1, &mt);
        int cls = (leg == 1) ? 1 : 2; /* 1=split, 2=nonsplit; A != +/-2 so no zero. */
        split_total[cls]++;

        int depth = start_depth;
        while (depth < target_depth) {
            u128 nx;
            if (!halve_once_first128(&nx, p, A, x, sqrtm1, &mt)) break;
            x = nx;
            depth++;
        }
        if (depth >= 0 && depth < 65) depth_counts[cls][depth]++;
        for (int d = start_depth; d <= depth && d <= target_depth && d < 65; d++) {
            survive_ge[cls][d]++;
        }
    }
    double elapsed = now_sec() - t0;

    printf("X1(16) split-discriminant first-branch halving stats\n");
    printf("p = "); print128(p); printf("\n");
    printf("k = %d\n", k);
    printf("samples = %llu\n", (unsigned long long)trials);
    printf("start_depth = %d  target_depth = %d\n", start_depth, target_depth);
    printf("split criterion: Legendre(A^2 - 4, p) = 1\n");
    printf("elapsed_seconds = %.6f  rate_Mps = %.6f\n",
           elapsed, (double)trials / elapsed / 1e6);
    printf("class totals: split=%llu nonsplit=%llu\n",
           (unsigned long long)split_total[1],
           (unsigned long long)split_total[2]);
    printf("depth split_survive nonsplit_survive lift_split_over_nonsplit\n");
    for (int d = start_depth; d <= target_depth; d++) {
        double sr = split_total[1] ? (double)survive_ge[1][d] / (double)split_total[1] : 0.0;
        double nr = split_total[2] ? (double)survive_ge[2][d] / (double)split_total[2] : 0.0;
        double lift = nr > 0.0 ? sr / nr : 0.0;
        printf("%d %llu %.9f %llu %.9f %.3f\n",
               d,
               (unsigned long long)survive_ge[1][d], sr,
               (unsigned long long)survive_ge[2][d], nr,
               lift);
    }
    printf("terminal_depth_histogram_nonzero:\n");
    for (int d = start_depth; d <= target_depth; d++) {
        if (depth_counts[1][d] || depth_counts[2][d]) {
            printf("  depth=%d split=%llu nonsplit=%llu\n",
                   d,
                   (unsigned long long)depth_counts[1][d],
                   (unsigned long long)depth_counts[2][d]);
        }
    }
    return 0;
}

static int first_halving_depth128(u128 p, u128 A, u128 x, int start_depth,
                                  int target_depth, u128 sqrtm1,
                                  const Mont128 *mt) {
    int depth = start_depth;
    while (depth < target_depth) {
        u128 nx;
        if (!halve_once_first128(&nx, p, A, x, sqrtm1, mt)) break;
        x = nx;
        depth++;
    }
    return depth;
}

static int translate_by_two_torsion_root128(u128 *xo, u128 p, u128 x,
                                            u128 root, u128 coeff,
                                            const Mont128 *mt) {
    u128 den = submod128(x, root, p);
    if (den == 0) return 0;
    u128 inv_den = invert128_mont(den, p, mt);
    *xo = addmod128(root, mulmod128_mont(coeff, inv_den, mt), p);
    return *xo != 0;
}

static int split_two_torsion_translates128(u128 xs[4], u128 p, u128 A,
                                           u128 x, u128 sqrtm1,
                                           const Mont128 *mt) {
    u128 disc = submod128(mulmod128_mont(A, A, mt), 4, p);
    u128 sd;
    if (!sqrtmod_p5_128(&sd, disc, p, sqrtm1, mt)) return 0;

    const u128 inv2 = (p + 1) >> 1;
    u128 alpha = mulmod128_mont(submod128(sd, A, p), inv2, mt);
    u128 beta = mulmod128_mont(submod128(submod128(0, A, p), sd, p), inv2, mt);
    u128 coeff0 = 1;
    u128 coeff_alpha = mulmod128_mont(alpha, submod128(alpha, beta, p), mt);
    u128 coeff_beta = mulmod128_mont(beta, submod128(beta, alpha, p), mt);

    xs[0] = x;
    if (!translate_by_two_torsion_root128(&xs[1], p, x, 0, coeff0, mt)) return 0;
    if (!translate_by_two_torsion_root128(&xs[2], p, x, alpha, coeff_alpha, mt)) return 0;
    if (!translate_by_two_torsion_root128(&xs[3], p, x, beta, coeff_beta, mt)) return 0;
    return 1;
}

static int bench_x16_split_translate_stats_128(u128 p) {
    if ((u64)(p & 7) != 5) {
        printf("x16splittranslate requires p ≡ 5 mod 8 for the X1(16) sampler.\n");
        return 1;
    }

    const int start_depth = 4;
    int target_depth = g_x16_split_translate_target;
    int k = compute_k(p);
    if (target_depth < start_depth + 1) target_depth = 16;
    if (target_depth > k) target_depth = k;
    if (target_depth > 64) target_depth = 64;

    Mont128 mt; m128_init(&mt, p);
    int pbits = bitlen128(p);
    u128 rand_mask = pbits >= 128 ? (u128)0 - 1 : (((u128)1 << pbits) - 1);
    u128 sqrtm1 = powmod128_mont(2, (p - 1) >> 2, &mt);
    u64 trials = g_max_trials_override ? g_max_trials_override : 100000ULL;
    if (trials == 0) trials = 100000ULL;

    const char *names[5] = {
        "original_P",
        "P_plus_T0",
        "P_plus_Talpha",
        "P_plus_Tbeta",
        "best_of_four"
    };
    u64 survive[5][65] = {{0}};
    u64 terminal[5][65] = {{0}};
    u64 attempts = 0;
    u64 split_seen = 0;
    u64 translate_fail = 0;

    Rng rng = {
        .s0 = 7364529176530163ULL ^ g_seed_offset,
        .s1 = 1442695040888963407ULL ^ (g_seed_offset << 1)
    };
    for (int i = 0; i < 200; i++) rng64(&rng);

    u128 pending_A = 0, pending_xP = 0;
    int have_pending_A = 0;
    double t0 = now_sec();
    while (split_seen < trials) {
        attempts++;
        u128 A, x16;
        x16_montgomery_A128(&A, &x16, &pending_A, &pending_xP, &have_pending_A,
                             &rng, p, rand_mask, sqrtm1, &mt);
        if (!split_discriminant128(p, A, sqrtm1, &mt)) continue;

        u128 xs[4];
        if (!split_two_torsion_translates128(xs, p, A, x16, sqrtm1, &mt)) {
            translate_fail++;
            continue;
        }

        int depths[4];
        int best_depth = start_depth;
        for (int v = 0; v < 4; v++) {
            depths[v] = first_halving_depth128(p, A, xs[v], start_depth,
                                               target_depth, sqrtm1, &mt);
            if (depths[v] > best_depth) best_depth = depths[v];
        }
        for (int v = 0; v < 4; v++) {
            terminal[v][depths[v]]++;
            for (int d = start_depth; d <= depths[v] && d <= target_depth; d++) {
                survive[v][d]++;
            }
        }
        terminal[4][best_depth]++;
        for (int d = start_depth; d <= best_depth && d <= target_depth; d++) {
            survive[4][d]++;
        }
        split_seen++;
    }
    double elapsed = now_sec() - t0;

    printf("X1(16) split 2-torsion translation first-branch stats\n");
    printf("p = "); print128(p); printf("\n");
    printf("k = %d\n", k);
    printf("split_samples = %llu\n", (unsigned long long)split_seen);
    printf("all_x1_attempts = %llu\n", (unsigned long long)attempts);
    printf("translate_fail = %llu\n", (unsigned long long)translate_fail);
    printf("start_depth = %d  target_depth = %d\n", start_depth, target_depth);
    printf("elapsed_seconds = %.6f  split_rate_Mps = %.6f  all_attempt_rate_Mps = %.6f\n",
           elapsed,
           (double)split_seen / elapsed / 1e6,
           (double)attempts / elapsed / 1e6);
    printf("depth variant survive rate\n");
    for (int d = start_depth; d <= target_depth; d++) {
        for (int v = 0; v < 5; v++) {
            printf("%d %s %llu %.9f\n",
                   d, names[v],
                   (unsigned long long)survive[v][d],
                   (double)survive[v][d] / (double)split_seen);
        }
    }
    printf("terminal_depth_histogram_nonzero:\n");
    for (int d = start_depth; d <= target_depth; d++) {
        int any = 0;
        for (int v = 0; v < 5; v++) if (terminal[v][d]) any = 1;
        if (!any) continue;
        printf("  depth=%d", d);
        for (int v = 0; v < 5; v++) {
            printf(" %s=%llu", names[v], (unsigned long long)terminal[v][d]);
        }
        printf("\n");
    }
    return 0;
}

static int bench_x16_cubic3_stats_128(u128 p) {
    if ((u64)(p & 7) != 5) {
        printf("x16cubic3stats requires p ≡ 5 mod 8 for the X1(16) sampler.\n");
        return 1;
    }
    if (p % 3 != 1) {
        printf("x16cubic3stats requires p ≡ 1 mod 3 for a nontrivial cubic character.\n");
        return 1;
    }

    const int start_depth = 4;
    int target_depth = g_x16_cubic3_target;
    int k = compute_k(p);
    if (target_depth < start_depth + 1) target_depth = 20;
    if (target_depth > k) target_depth = k;
    if (target_depth > 64) target_depth = 64;

    Mont128 mt; m128_init(&mt, p);
    int pbits = bitlen128(p);
    u128 rand_mask = pbits >= 128 ? (u128)0 - 1 : (((u128)1 << pbits) - 1);
    u128 sqrtm1 = powmod128_mont(2, (p - 1) >> 2, &mt);
    u64 trials = g_max_trials_override ? g_max_trials_override : 100000ULL;
    if (trials == 0) trials = 100000ULL;

    u64 class_total[3] = {0}; /* 1=cube, 2=noncube */
    u64 survive_ge[3][65] = {{0}};
    u64 terminal_depth[3][65] = {{0}};

    Rng rng = {
        .s0 = 7364529176530163ULL ^ g_seed_offset,
        .s1 = 1442695040888963407ULL ^ (g_seed_offset << 1)
    };
    for (int i = 0; i < 200; i++) rng64(&rng);

    u128 pending_A = 0, pending_xP = 0;
    int pending_class = 0, have_pending_A = 0;
    double t0 = now_sec();
    for (u64 i = 0; i < trials; i++) {
        u128 A, x;
        int cls = 0;
        x16_montgomery_A_nonsplit_cubic3_128(&A, &x, &cls,
                                             &pending_A, &pending_xP,
                                             &pending_class, &have_pending_A,
                                             &rng, p, rand_mask, sqrtm1, &mt);
        if (cls < 1 || cls > 2) continue;
        class_total[cls]++;

        int depth = start_depth;
        while (depth < target_depth) {
            u128 nx;
            if (!halve_once_first128(&nx, p, A, x, sqrtm1, &mt)) break;
            x = nx;
            depth++;
        }
        if (depth >= 0 && depth < 65) terminal_depth[cls][depth]++;
        for (int d = start_depth; d <= depth && d <= target_depth && d < 65; d++) {
            survive_ge[cls][d]++;
        }
    }
    double elapsed = now_sec() - t0;

    printf("X1(16) nonsplit cubic-character halving stats\n");
    printf("p = "); print128(p); printf("\n");
    printf("k = %d\n", k);
    printf("samples = %llu\n", (unsigned long long)trials);
    printf("start_depth = %d  target_depth = %d\n", start_depth, target_depth);
    printf("class: cube iff chi3(C3(y)) = 1; noncube otherwise\n");
    printf("elapsed_seconds = %.6f  rate_Mps = %.6f\n",
           elapsed, (double)trials / elapsed / 1e6);
    printf("class totals: cube=%llu noncube=%llu cube_coverage=%.6f\n",
           (unsigned long long)class_total[1],
           (unsigned long long)class_total[2],
           (double)class_total[1] / (double)trials);
    printf("depth cube_survive noncube_survive lift_cube_over_noncube\n");
    for (int d = start_depth; d <= target_depth; d++) {
        double cr = class_total[1] ? (double)survive_ge[1][d] / (double)class_total[1] : 0.0;
        double nr = class_total[2] ? (double)survive_ge[2][d] / (double)class_total[2] : 0.0;
        double lift = nr > 0.0 ? cr / nr : 0.0;
        printf("%d %llu %.9f %llu %.9f %.3f\n",
               d,
               (unsigned long long)survive_ge[1][d], cr,
               (unsigned long long)survive_ge[2][d], nr,
               lift);
    }
    printf("terminal_depth_histogram_nonzero:\n");
    for (int d = start_depth; d <= target_depth; d++) {
        if (terminal_depth[1][d] || terminal_depth[2][d]) {
            printf("  depth=%d cube=%llu noncube=%llu\n",
                   d,
                   (unsigned long long)terminal_depth[1][d],
                   (unsigned long long)terminal_depth[2][d]);
        }
    }
    return 0;
}

static int bench_x16_pair_stats_128(u128 p) {
    if ((u64)(p & 7) != 5) {
        printf("x16pairstats requires p ≡ 5 mod 8 for the X1(16) sampler.\n");
        return 1;
    }

    Mont128 mt; m128_init(&mt, p);
    int pbits = bitlen128(p);
    u128 rand_mask = pbits >= 128 ? (u128)0 - 1 : (((u128)1 << pbits) - 1);
    u128 sqrtm1 = powmod128_mont(2, (p - 1) >> 2, &mt);
    u64 target_pairs = g_max_trials_override ? g_max_trials_override : 100000ULL;
    if (target_pairs == 0) target_pairs = 100000ULL;

    u64 attempts = 0;
    u64 valid_y = 0;
    u64 valid_roots_hist[3] = {0};
    u64 root_class[2][3] = {{0}};
    u64 pair_class[3][3] = {{0}};
    const char *feature_names[] = {
        "y", "y-1", "y-2", "y+1", "y^2-y+1", "y^2-2y+2",
        "y*(y-2)", "1-y", "y*(y-1)", "(y-1)*(y-2)",
        "y^2+y+1", "y^2-3y+1", "y^2-2", "y^2-4y+2",
        "(y^2-2)*(y^2-4y+2)", "(y^2-2)/(y^2-4y+2)"
    };
    const int nfeatures = (int)(sizeof(feature_names) / sizeof(feature_names[0]));
    u64 feature_counts[16][3][3] = {{{0}}}; /* feature, class, legendre 1/2 where 2=-1 */

    Rng rng = {
        .s0 = 7364529176530163ULL ^ g_seed_offset,
        .s1 = 1442695040888963407ULL ^ (g_seed_offset << 1)
    };
    for (int i = 0; i < 200; i++) rng64(&rng);

    double t0 = now_sec();
    while (valid_y < target_pairs) {
        attempts++;
        u128 y = rand_below128(&rng, p, rand_mask);
        if (y == 0) continue;

        u128 y2 = mulmod128_mont(y, y, &mt);
        u128 y3 = mulmod128_mont(y2, y, &mt);
        u128 qa = submod128(y2, addmod128(y, y, p), p);
        if (qa == 0) continue;
        u128 qb = submod128(addmod128(y2, y2, p), y3, p);
        u128 qc = submod128(1, y, p);
        u128 D = submod128(mulmod128_mont(qb, qb, &mt),
                           mulmod128_mont(addmod128(qa, qa, p), addmod128(qc, qc, p), &mt),
                           p);
        u128 sd;
        if (!sqrtmod_p5_128(&sd, D, p, sqrtm1, &mt)) continue;

        u128 inv_2qa = invert128_mont(addmod128(qa, qa, p), p, &mt);
        u128 roots[2] = {
            mulmod128_mont(submod128(sd, qb, p), inv_2qa, &mt),
            mulmod128_mont(submod128(p - sd, qb, p), inv_2qa, &mt)
        };

        int classes[2] = {0, 0};
        int nvalid = 0;
        for (int ri = 0; ri < 2; ri++) {
            u128 x = roots[ri];
            u128 xy = mulmod128_mont(x, y, &mt);
            u128 x2 = mulmod128_mont(x, x, &mt);
            u128 denr = submod128(mulmod128_mont(x2, y, &mt), x, p);
            u128 dens = xy;
            u128 invs1[2], vals1[2] = { denr, dens };
            if (!invert_batch128(invs1, vals1, 2, p, &mt)) continue;

            u128 rnum = submod128(addmod128(submod128(mulmod128_mont(x2, y, &mt), xy, p), y, p), 1, p);
            u128 snum = addmod128(submod128(xy, y, p), 1, p);
            u128 r = mulmod128_mont(rnum, invs1[0], &mt);
            u128 s = mulmod128_mont(snum, invs1[1], &mt);
            if (r == 0 || r == 1 || s == 0) continue;

            u128 rm1 = submod128(r, 1, p);
            u128 bt = mulmod128_mont(mulmod128_mont(r, s, &mt), rm1, &mt);
            if (bt == 0) continue;
            u128 c = mulmod128_mont(s, rm1, &mt);
            u128 a = submod128(c, 1, p);
            u128 e = submod128(mulmod128_mont(a, a, &mt), addmod128(addmod128(bt, bt, p), addmod128(bt, bt, p), p), p);

            u128 rs = mulmod128_mont(r, s, &mt);
            u128 den = addmod128(submod128(rs, addmod128(r, r, p), p), 1, p);
            u128 u4 = mulmod128_mont(r, rm1, &mt);
            u128 s2 = mulmod128_mont(s, s, &mt);
            u128 term = submod128(addmod128(submod128(r, s2, p), s, p), 1, p);
            u128 denn = mulmod128_mont(den, den, &mt);

            u128 numer8 = mulmod128_mont(mulmod128_mont(u4, submod128(r, s, p), &mt), term, &mt);
            u128 three_e = addmod128(e, addmod128(e, e, p), p);
            u128 X8_num = addmod128(mulmod128_mont(36, numer8, &mt),
                                     mulmod128_mont(three_e, denn, &mt), p);
            u128 lam_num = mulmod128_mont(36, submod128(mulmod128_mont(u4, denn, &mt), numer8, p), &mt);
            if (lam_num == 0) continue;
            u128 inv_lam_num = invert128_mont(lam_num, p, &mt);
            u128 A = mulmod128_mont(addmod128(X8_num, addmod128(X8_num, X8_num, p), p), inv_lam_num, &mt);
            if (A <= 2 || A >= p - 2) continue;

            u128 disc = submod128(mulmod128_mont(A, A, &mt), 4, p);
            u128 leg = powmod128_mont(disc, (p - 1) >> 1, &mt);
            int cls = (leg == 1) ? 1 : 2;
            classes[ri] = cls;
            root_class[ri][cls]++;
            nvalid++;
        }

        if (nvalid > 0) {
            valid_y++;
            if (nvalid > 2) nvalid = 2;
            valid_roots_hist[nvalid]++;
            pair_class[classes[0]][classes[1]]++;
            int yclass = classes[0] ? classes[0] : classes[1];
            u128 ym1 = submod128(y, 1, p);
            u128 ym2 = submod128(y, 2, p);
            u128 yp1 = addmod128(y, 1, p);
            u128 features[16];
            features[0] = y;
            features[1] = ym1;
            features[2] = ym2;
            features[3] = yp1;
            features[4] = addmod128(submod128(y2, y, p), 1, p);
            features[5] = addmod128(submod128(y2, addmod128(y, y, p), p), 2, p);
            features[6] = mulmod128_mont(y, ym2, &mt);
            features[7] = submod128(1, y, p);
            features[8] = mulmod128_mont(y, ym1, &mt);
            features[9] = mulmod128_mont(ym1, ym2, &mt);
            features[10] = addmod128(addmod128(y2, y, p), 1, p);
            features[11] = addmod128(submod128(y2, addmod128(y, addmod128(y, y, p), p), p), 1, p);
            features[12] = submod128(y2, 2, p);
            features[13] = addmod128(submod128(y2, addmod128(addmod128(y, y, p), addmod128(y, y, p), p), p), 2, p);
            features[14] = mulmod128_mont(features[12], features[13], &mt);
            if (features[13] != 0) {
                features[15] = mulmod128_mont(features[12], invert128_mont(features[13], p, &mt), &mt);
            } else {
                features[15] = 0;
            }
            for (int fi = 0; fi < nfeatures; fi++) {
                if (features[fi] == 0 || yclass == 0) continue;
                u128 legf = powmod128_mont(features[fi], (p - 1) >> 1, &mt);
                int fl = (legf == 1) ? 1 : 2;
                feature_counts[fi][yclass][fl]++;
            }
        }
    }
    double elapsed = now_sec() - t0;

    printf("X1(16) quadratic-root split/nonsplit pair stats\n");
    printf("p = "); print128(p); printf("\n");
    printf("valid_y_samples = %llu\n", (unsigned long long)valid_y);
    printf("attempts = %llu\n", (unsigned long long)attempts);
    printf("elapsed_seconds = %.6f  valid_y_rate_Mps = %.6f\n",
           elapsed, (double)valid_y / elapsed / 1e6);
    printf("valid_roots_hist one=%llu two=%llu\n",
           (unsigned long long)valid_roots_hist[1],
           (unsigned long long)valid_roots_hist[2]);
    printf("root0 split=%llu nonsplit=%llu\n",
           (unsigned long long)root_class[0][1],
           (unsigned long long)root_class[0][2]);
    printf("root1 split=%llu nonsplit=%llu\n",
           (unsigned long long)root_class[1][1],
           (unsigned long long)root_class[1][2]);
    printf("pair_class rows=root0(0 missing,1 split,2 nonsplit) cols=root1(0 missing,1 split,2 nonsplit)\n");
    for (int i = 0; i < 3; i++) {
        printf("%d %llu %llu %llu\n", i,
               (unsigned long long)pair_class[i][0],
               (unsigned long long)pair_class[i][1],
               (unsigned long long)pair_class[i][2]);
    }
    printf("feature name split_plus split_minus nonsplit_plus nonsplit_minus nonsplit_if_minus_precision nonsplit_if_plus_precision\n");
    for (int fi = 0; fi < nfeatures; fi++) {
        u64 sp = feature_counts[fi][1][1];
        u64 sm = feature_counts[fi][1][2];
        u64 np = feature_counts[fi][2][1];
        u64 nm = feature_counts[fi][2][2];
        double prec_minus = (sm + nm) ? (double)nm / (double)(sm + nm) : 0.0;
        double prec_plus = (sp + np) ? (double)np / (double)(sp + np) : 0.0;
        printf("%d %s %llu %llu %llu %llu %.6f %.6f\n",
               fi, feature_names[fi],
               (unsigned long long)sp,
               (unsigned long long)sm,
               (unsigned long long)np,
               (unsigned long long)nm,
               prec_minus,
               prec_plus);
    }
    return 0;
}

static int marked_8p_is_x0_128(u128 p, u128 A, u128 xP16,
                               const Mont128 *mt, u128 inv4_m) {
    u128 Ap2_m = toM128(addmod128(A, 2, p), mt);
    u128 a24m = mm128(Ap2_m, inv4_m, mt);
    u128 X = toM128(xP16, mt);
    u128 Z = mt->one;
    for (int i = 0; i < 3; i++) {
        u128 NX, NZ;
        xDBL128(&NX, &NZ, X, Z, a24m, mt);
        X = NX;
        Z = NZ;
    }
    return X == 0 && Z != 0;
}

static int bench_x16_torsion_stats_128(u128 p) {
    if ((u64)(p & 7) != 5) {
        printf("x16torsionstats requires p ≡ 5 mod 8 for the X1(16) sampler.\n");
        return 1;
    }

    Mont128 mt; m128_init(&mt, p);
    int pbits = bitlen128(p);
    u128 rand_mask = pbits >= 128 ? (u128)0 - 1 : (((u128)1 << pbits) - 1);
    u128 sqrtm1 = powmod128_mont(2, (p - 1) >> 2, &mt);
    u64 trials = g_max_trials_override ? g_max_trials_override : 100000ULL;
    if (trials == 0) trials = 100000ULL;

    u128 inv4_m;
    { u128 four_m = toM128(4, &mt), r = mt.one, b = four_m, e = p - 2;
      while (e > 0) { if (e & 1) r = mm128(r, b, &mt); b = mm128(b, b, &mt); e >>= 1; }
      inv4_m = r; }

    u64 class_counts[3] = {0};
    u64 x8zero_counts[3][2] = {{0}};

    Rng rng = {
        .s0 = 7364529176530163ULL ^ g_seed_offset,
        .s1 = 1442695040888963407ULL ^ (g_seed_offset << 1)
    };
    for (int i = 0; i < 200; i++) rng64(&rng);

    u128 pending_A = 0, pending_xP = 0;
    int have_pending_A = 0;
    double t0 = now_sec();
    for (u64 i = 0; i < trials; i++) {
        u128 A, xP16;
        x16_montgomery_A128(&A, &xP16, &pending_A, &pending_xP, &have_pending_A,
                             &rng, p, rand_mask, sqrtm1, &mt);
        int split = split_discriminant128(p, A, sqrtm1, &mt);
        int cls = split ? 1 : 2;
        int x8zero = marked_8p_is_x0_128(p, A, xP16, &mt, inv4_m);
        class_counts[cls]++;
        x8zero_counts[cls][x8zero ? 1 : 0]++;
    }
    double elapsed = now_sec() - t0;

    printf("X1(16) marked 8P torsion classifier stats\n");
    printf("p = "); print128(p); printf("\n");
    printf("samples = %llu\n", (unsigned long long)trials);
    printf("elapsed_seconds = %.6f  rate_Mps = %.6f\n",
           elapsed, (double)trials / elapsed / 1e6);
    printf("split_total=%llu nonsplit_total=%llu\n",
           (unsigned long long)class_counts[1],
           (unsigned long long)class_counts[2]);
    printf("class x8_nonzero x8_zero\n");
    printf("split %llu %llu\n",
           (unsigned long long)x8zero_counts[1][0],
           (unsigned long long)x8zero_counts[1][1]);
    printf("nonsplit %llu %llu\n",
           (unsigned long long)x8zero_counts[2][0],
           (unsigned long long)x8zero_counts[2][1]);
    return 0;
}

/* ================================================================
 * Shared: compute_k, odd parts, Miller-Rabin (all u128-safe)
 * ================================================================ */

static int compute_k(u128 p) {
    u64 q = (u64)sqrtl((long double)p);
    while ((u128)(q+1)*(q+1)<=p) q++;
    while ((u128)q*q>p) q--;
    u64 sq = (u64)sqrtl((long double)q);
    while ((sq+1)*(sq+1)<=q) sq++;
    while (sq*sq>q) sq--;
    u64 bound = q+1+2*sq;
    int k=0; u64 v=1; while(v<=bound){k++;v<<=1;} return k;
}

static int compute_odd_parts(u128 p, int k, u64 *ms, int *max_v2s, int max_ms) {
    if (k >= 63) return 0;
    u64 twok = 1ULL << k;
    u128 pp1 = p + 1;
    u64 r = (u64)(pp1 % twok);
    u64 sqrtp = (u64)sqrtl((long double)p);
    while ((u128)(sqrtp+1)*(sqrtp+1)<=p) sqrtp++;
    while ((u128)sqrtp*sqrtp>p) sqrtp--;
    u64 two_sqrtp = sqrtp * 2 + 4;
    u64 residues[2] = { r, (twok - r) % twok };
    int count = 0;
    for (int ri = 0; ri < 2 && count < max_ms; ri++) {
        u64 res = residues[ri];
        for (int sign = -1; sign <= 1; sign += 2) {
            for (u64 j = 0; ; j++) {
                long long tv;
                if (sign > 0) tv = (long long)(res + j * twok);
                else           tv = (long long)res - (long long)((j+1) * twok);
                if (tv > (long long)two_sqrtp || tv < -(long long)two_sqrtp) break;
                if (tv == 0) continue;
                u128 N = pp1;
                if (ri == 0) { if (tv>=0) N-=(u64)tv; else N+=(u64)(-tv); }
                else         { if (tv>=0) N+=(u64)tv; else N-=(u64)(-tv); }
                if (N == 0) continue;
                int v2 = 0; u128 tmp = N;
                while (tmp % 2 == 0) { v2++; tmp /= 2; }
                if (v2 < k) continue;
                if (tmp >> 63) continue;
                u64 odd = (u64)tmp;
                if (odd == 0) continue;
                int dup = -1;
                for (int c = 0; c < count; c++) if (ms[c] == odd) { dup = c; break; }
                if (dup >= 0) {
                    if (max_v2s && v2 > max_v2s[dup]) max_v2s[dup] = v2;
                } else {
                    ms[count] = odd;
                    if (max_v2s) max_v2s[count] = v2;
                    count++;
                }
            }
        }
    }
    return count;
}

static int is_prime128(u128 n) {
    if (n < 2) return 0; if (n < 4) return 1; if (n % 2 == 0) return 0;
    u128 d = n-1; int r = 0; while (d%2==0) { d/=2; r++; }
    u64 w[] = {2,3,5,7,11,13,17,19,23,29,31,37};
    for (int i = 0; i < 12; i++) {
        u128 a = w[i]; if (a >= n) continue;
        u128 x = 1, b = a;
        for (u128 e = d; e; e >>= 1) {
            if (e & 1) x = mulmod_slow(x,b,n);
            b = mulmod_slow(b,b,n);
        }
        if (x == 1 || x == n-1) continue;
        int ok = 0;
        for (int j = 0; j < r-1; j++) { x = mulmod_slow(x,x,n); if (x==n-1){ok=1;break;} }
        if (!ok) return 0;
    }
    return 1;
}

/* ================================================================
 * Dispatch: search64 / search128
 * ================================================================ */

static int search64(u64 p) {
    int k = compute_k(p);
    u64 sqrtp = (u64)sqrtl((long double)p);
    while ((u128)(sqrtp+1)*(sqrtp+1)<=(u128)p) sqrtp++;
    while ((u128)sqrtp*sqrtp>(u128)p) sqrtp--;

    u64 ms[64]; int max_v2s[64];
    int nms = compute_odd_parts(p, k, ms, max_v2s, 64);
    printf("k = %d\n", k);
    printf("Odd parts (%d):", nms);
    for (int i=0;i<nms;i++) printf(" %llu", (unsigned long long)ms[i]);
    printf("\n");
    if (nms == 0) { printf("No valid odd parts.\n"); return 1; }

    Mont64 mt; m64_init(&mt, p);
    u64 inv4; { u64 r=1,b=4%p; for(u64 e=p-2;e;e>>=1){if(e&1)r=mulmod64(r,b,p);b=mulmod64(b,b,p);} inv4=r; }

    u64 max_trials = (u64)(20.0 * (double)sqrtp / nms);
    if (max_trials < 10000000ULL) max_trials = 10000000ULL;
    if (g_max_trials_override) max_trials = g_max_trials_override;

    int nth = 1;
#ifdef _OPENMP
    nth = omp_get_max_threads();
#endif
    printf("Trials: %llu  (heuristic: 20*sqrt(p)/%d)\n", (unsigned long long)max_trials, nms);
    printf("Threads: %d\n\n", nth);

    volatile int found = 0;
    u64 found_A = 0, found_x0 = 0;
    double t0 = now_sec();

#pragma omp parallel
    {
        int tid=0, nthr=1;
#ifdef _OPENMP
        tid = omp_get_thread_num(); nthr = omp_get_num_threads();
#endif
        Rng rng = {
            .s0=7364529176530163ULL^((u64)tid*6364136223846793005ULL)^g_seed_offset,
            .s1=1442695040888963407ULL^((u64)(tid+1)*2862933555777941757ULL)^(g_seed_offset<<1)
        };
        for (int i=0;i<200;i++) rng64(&rng);
        u64 budget = max_trials / nthr + 1, lc = 0;

        while (!found && lc < budget) {
            u64 A = rng64(&rng) % p;
            u64 x0r = rng64(&rng) % p;
            if (A<=2||A>=p-2||x0r<2) { lc++; continue; }
            u64 a24 = mulmod64(addmod64(A,2,p), inv4, p);
            u64 a24m = toM64(a24, &mt);
            u64 x0m = toM64(x0r, &mt);

            for (int mi=0; mi<nms && !found; mi++) {
                u64 QX, QZ;
                xMUL64(&QX, &QZ, x0m, ms[mi], a24m, &mt);
                if (QZ == 0) continue;
                u64 CX=QX, CZ=QZ;
                int zs = -1;
                for (int s=1; s<=k+10 && s<50; s++) {
                    xDBL64(&CX,&CZ,CX,CZ,a24m,&mt);
                    if (CZ==0) { zs=s; break; }
                }
                if (zs < k) continue;
                int target = zs - k;
                CX=QX; CZ=QZ;
                for (int s=0; s<target; s++) xDBL64(&CX,&CZ,CX,CZ,a24m,&mt);
                u64 cz = frM64(CZ, &mt);
                if (cz == 0) continue;
                u64 czinv; {u64 r2=1,b2=cz;for(u64 e=p-2;e;e>>=1){if(e&1)r2=mulmod64(r2,b2,p);b2=mulmod64(b2,b2,p);}czinv=r2;}
                u64 xR = mulmod64(frM64(CX,&mt), czinv, p);
                if (verify64(p, A, xR)) {
#pragma omp critical
                    {
                        if (!found) {
                            found=1; found_A=A; found_x0=xR;
                            double el = now_sec()-t0;
                            printf("Found after %.2fs (~%llu trials)\n\n",
                                   el, (unsigned long long)(lc * nthr));
                        }
                    }
                }
            }
            lc++;
            if (tid==0 && lc%1000000==0 && !found) {
                double el = now_sec()-t0;
                u64 est = lc * nthr;
                printf("  trials=%llu percent=%.6f elapsed=%.1f rate_Mps=%.3f\n",
                       (unsigned long long)est,
                       100.0*est/max_trials, el, est/el/1e6);
                fflush(stdout);
            }
        }
    }

    double elapsed = now_sec() - t0;
    if (found) {
        printf("%llu %llu %llu\n\n",
               (unsigned long long)p,
               (unsigned long long)found_A,
               (unsigned long long)found_x0);
        printf("Verified: %s  (%.2fs)\n", verify64(p,found_A,found_x0)?"PASS":"FAIL", elapsed);
    } else {
        printf("Not found in %.2fs. Re-run or increase budget.\n", elapsed);
    }
    return found ? 0 : 1;
}

static int search128(u128 p) {
    int k = compute_k(p);
    u64 sqrtp = (u64)sqrtl((long double)p);
    while ((u128)(sqrtp+1)*(sqrtp+1)<=p) sqrtp++;
    while ((u128)sqrtp*sqrtp>p) sqrtp--;

    u64 ms[64]; int max_v2s[64];
    int nms = compute_odd_parts(p, k, ms, max_v2s, 64);
    printf("k = %d\n", k);
    printf("Odd parts (%d):", nms);
    for (int i=0;i<nms;i++) printf(" %llu", (unsigned long long)ms[i]);
    printf("\n");
    if (nms == 0) { printf("No valid odd parts.\n"); return 1; }

    Mont128 mt; m128_init(&mt, p);
    int pbits = bitlen128(p);
    u128 rand_mask = pbits >= 128 ? (u128)0 - 1 : (((u128)1 << pbits) - 1);
    u128 sqrtm1 = 0;
    if (g_x16_mode) {
        if ((u64)(p & 7) == 5) {
            sqrtm1 = powmod128_mont(2, (p - 1) >> 2, &mt);
        } else if ((u64)(p & 3) == 3) {
            /* PATCH(p3mod4): square roots via n^((p+1)/4); sqrtm1 not needed */
            sqrtm1 = 0;
        } else {
            printf("X1(16) mode requires p ≡ 5 mod 8 or p ≡ 3 mod 4 for fast square roots (p ≡ 1 mod 8 unsupported).\n");
            return 1;
        }
    }
    /* inv4 in Montgomery form */
    u128 inv4_m;
    { u128 four_m=toM128(4,&mt), r=mt.one, b=four_m; u128 e=p-2;
      while(e>0){if(e&1)r=mm128(r,b,&mt);b=mm128(b,b,&mt);e>>=1;} inv4_m=r; }

    u64 max_trials = (u64)(20.0 * (double)sqrtp / nms);
    if (max_trials < 10000000ULL) max_trials = 10000000ULL;
    if (g_max_trials_override) max_trials = g_max_trials_override;

    int nth = 1;
#ifdef _OPENMP
    nth = omp_get_max_threads();
#endif
    printf("Trials: %llu  (heuristic: 20*sqrt(p)/%d)\n", (unsigned long long)max_trials, nms);
    printf("Threads: %d\n\n", nth);

    volatile int found = 0;
    u128 found_A = 0, found_x0 = 0;
    double t0 = now_sec();

#pragma omp parallel
    {
        int tid=0, nthr=1;
#ifdef _OPENMP
        tid = omp_get_thread_num(); nthr = omp_get_num_threads();
#endif
        Rng rng = {
            .s0=7364529176530163ULL^((u64)tid*6364136223846793005ULL)^g_seed_offset,
            .s1=1442695040888963407ULL^((u64)(tid+1)*2862933555777941757ULL)^(g_seed_offset<<1)
        };
        for (int i=0;i<200;i++) rng64(&rng);
        u64 budget = max_trials / nthr + 1, lc = 0;
        u128 pending_A = 0, pending_xP = 0;
        int have_pending_A = 0;

        while (!found && lc < budget) {
            u128 A, xP16 = 0;
            int x_start_depth = 4;
            if (g_x16_mode) {
                if (g_x16_first_d_skip_mode) {
                    x16_montgomery_A_nonsplit_dgate_skip128(&A, &xP16,
                                                            &pending_A, &pending_xP,
                                                            &have_pending_A,
                                                            &rng, p, rand_mask, sqrtm1, &mt);
                    x_start_depth = 5;
                } else if (g_x16_nonsplit_filter_mode) {
                    x16_montgomery_A_nonsplit128(&A, &xP16, &pending_A, &pending_xP, &have_pending_A,
                                                 &rng, p, rand_mask, sqrtm1, &mt);
                } else {
                    x16_montgomery_A128(&A, &xP16, &pending_A, &pending_xP, &have_pending_A,
                                         &rng, p, rand_mask, sqrtm1, &mt);
                }
            } else {
                A = rand_below128(&rng, p, rand_mask);
            }

            if (g_x16_halve_mode) {
                u128 xR;
                if (halve_chain_from_depth128(&xR, p, A, xP16, x_start_depth, k, sqrtm1, &mt)) {
#pragma omp critical
                    {
                        if (!found) {
                            found=1; found_A=A; found_x0=xR;
                            double el = now_sec()-t0;
                            printf("Found after %.2fs (~%llu X1(16) curves)\n\n",
                                   el, (unsigned long long)(lc * nthr));
                        }
                    }
                }
                lc++;
                if (tid==0 && lc%500000==0 && !found) {
                    double el = now_sec()-t0;
                    u64 est = lc * nthr;
                    printf("  trials=%llu percent=%.6f elapsed=%.1f rate_Mps=%.3f\n",
                           (unsigned long long)est,
                           100.0*est/max_trials, el, est/el/1e6);
                    fflush(stdout);
                }
                continue;
            }

            u128 x0r = rand_below128(&rng, p, rand_mask);
            if (A<=2 || A>=p-2 || x0r<2) { lc++; continue; }

            u128 Ap2_m = toM128(addmod128(A,2,p), &mt);
            u128 a24m = mm128(Ap2_m, inv4_m, &mt);
            u128 x0m = toM128(x0r, &mt);

            if (nms == 2 && ms[0] == 2*ms[1] + 1) {
                u128 QX, QZ, RX, RZ, SX, SZ, xR;

                xMULPAIR128(&QX, &QZ, &RX, &RZ, x0m, ms[1], a24m, &mt);
                if (projected_hit128(&xR, p, A, k, max_v2s[1], QX, QZ, a24m, &mt)) {
#pragma omp critical
                    {
                        if (!found) {
                            found=1; found_A=A; found_x0=xR;
                            double el = now_sec()-t0;
                            printf("Found after %.2fs (~%llu trials)\n\n",
                                   el, (unsigned long long)(lc * nthr));
                        }
                    }
                }

                xADD128(&SX, &SZ, QX, QZ, RX, RZ, x0m, &mt);
                if (!found && projected_hit128(&xR, p, A, k, max_v2s[0], SX, SZ, a24m, &mt)) {
#pragma omp critical
                    {
                        if (!found) {
                            found=1; found_A=A; found_x0=xR;
                            double el = now_sec()-t0;
                            printf("Found after %.2fs (~%llu trials)\n\n",
                                   el, (unsigned long long)(lc * nthr));
                        }
                    }
                }
            } else if (nms == 3 && ms[1] == 2*ms[0] + 1 && ms[2] == 2*ms[0] - 1) {
                u128 QX, QZ, RX, RZ, SX, SZ, xR;

                xMULPAIR128(&QX, &QZ, &RX, &RZ, x0m, ms[0], a24m, &mt);
                if (projected_hit128(&xR, p, A, k, max_v2s[0], QX, QZ, a24m, &mt)) {
#pragma omp critical
                    {
                        if (!found) {
                            found=1; found_A=A; found_x0=xR;
                            double el = now_sec()-t0;
                            printf("Found after %.2fs (~%llu trials)\n\n",
                                   el, (unsigned long long)(lc * nthr));
                        }
                    }
                }

                xADD128(&SX, &SZ, QX, QZ, RX, RZ, x0m, &mt);
                if (!found && projected_hit128(&xR, p, A, k, max_v2s[1], SX, SZ, a24m, &mt)) {
#pragma omp critical
                    {
                        if (!found) {
                            found=1; found_A=A; found_x0=xR;
                            double el = now_sec()-t0;
                            printf("Found after %.2fs (~%llu trials)\n\n",
                                   el, (unsigned long long)(lc * nthr));
                        }
                    }
                }

                xMULPAIR128(&QX, &QZ, &RX, &RZ, x0m, ms[0]-1, a24m, &mt);
                xADD128(&SX, &SZ, QX, QZ, RX, RZ, x0m, &mt);
                if (!found && projected_hit128(&xR, p, A, k, max_v2s[2], SX, SZ, a24m, &mt)) {
#pragma omp critical
                    {
                        if (!found) {
                            found=1; found_A=A; found_x0=xR;
                            double el = now_sec()-t0;
                            printf("Found after %.2fs (~%llu trials)\n\n",
                                   el, (unsigned long long)(lc * nthr));
                        }
                    }
                }
            } else {
                for (int mi=0; mi<nms && !found; mi++) {
                    u128 QX, QZ, xR;
                    xMUL128(&QX, &QZ, x0m, ms[mi], a24m, &mt);
                    if (projected_hit128(&xR, p, A, k, max_v2s[mi], QX, QZ, a24m, &mt)) {
#pragma omp critical
                        {
                            if (!found) {
                                found=1; found_A=A; found_x0=xR;
                                double el = now_sec()-t0;
                                printf("Found after %.2fs (~%llu trials)\n\n",
                                       el, (unsigned long long)(lc * nthr));
                            }
                        }
                    }
                }
            }
            lc++;
            if (tid==0 && lc%500000==0 && !found) {
                double el = now_sec()-t0;
                u64 est = lc * nthr;
                printf("  trials=%llu percent=%.6f elapsed=%.1f rate_Mps=%.3f\n",
                       (unsigned long long)est,
                       100.0*est/max_trials, el, est/el/1e6);
                fflush(stdout);
            }
        }
    }

    double elapsed = now_sec() - t0;
    if (found) {
        print128(p); printf(" "); print128(found_A); printf(" "); print128(found_x0);
        printf("\n\n");
        printf("Verified: %s  (%.2fs)\n", verify128(p,found_A,found_x0)?"PASS":"FAIL", elapsed);
    } else {
        printf("Not found in %.2fs. Re-run or increase budget.\n", elapsed);
    }
    return found ? 0 : 1;
}

/* ================================================================
 * main
 * ================================================================ */

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <prime> [seed_offset] [max_trials] [x16|x16halve|x16halvenonsplit|x16halvenonsplitdgate|x16halvenonsplitdgateskip|x16halvefull|x16ell3bench|x16ell3directybench|x16atkin5bench|x16atkin5benchnonsplit|x16atkin7bench|x16halvestats|x16branchstats|x16branchstatsnonsplit|x16branchstatsnonsplitdgate|x16gatestats [target_depth]|x16gatestatsnonsplit [target_depth]|x16lookaheadstats [lookahead_depth] [target_depth]|x16labelstats [target_depth]|x16splitstats [target_depth]|x16splitbranchstats [target_depth]|x16splittranslate [target_depth]|x16cubic3stats [target_depth]|x16quarticstats [target_depth]|x16quarticstatsnonsplit [target_depth]|x16pairstats|x16torsionstats|x1_32rootbench]\n", argv[0]);
        return 1;
    }

    u128 p = parse128(argv[1]);
    if (argc >= 3) g_seed_offset = (u64)parse128(argv[2]);
    if (argc >= 4) g_max_trials_override = (u64)parse128(argv[3]);
    if (argc >= 5 && strcmp(argv[4], "x16") == 0) g_x16_mode = 1;
    if (argc >= 5 && strcmp(argv[4], "x16halve") == 0) {
        g_x16_mode = 1;
        g_x16_halve_mode = 1;
    }
    if (argc >= 5 && strcmp(argv[4], "x16halvenonsplit") == 0) {
        g_x16_mode = 1;
        g_x16_halve_mode = 1;
        g_x16_nonsplit_filter_mode = 1;
    }
    if (argc >= 5 && strcmp(argv[4], "x16halvenonsplitdgate") == 0) {
        g_x16_mode = 1;
        g_x16_halve_mode = 1;
        g_x16_nonsplit_filter_mode = 1;
        g_x16_first_d_y_gate_mode = 1;
    }
    if (argc >= 5 && strcmp(argv[4], "x16halvenonsplitdgateskip") == 0) {
        g_x16_mode = 1;
        g_x16_halve_mode = 1;
        g_x16_nonsplit_filter_mode = 1;
        g_x16_first_d_y_gate_mode = 1;
        g_x16_first_d_skip_mode = 1;
    }
    if (argc >= 5 && strcmp(argv[4], "x16halvefull") == 0) {
        g_x16_mode = 1;
        g_x16_halve_mode = 1;
        g_x16_halve_full_mode = 1;
    }
    if (argc >= 5 && strcmp(argv[4], "x16ell3bench") == 0) {
        g_x16_mode = 1;
        g_x16_ell3_bench_mode = 1;
    }
    if (argc >= 5 && strcmp(argv[4], "x16ell3directybench") == 0) {
        g_x16_mode = 1;
        g_x16_ell3_direct_y_bench_mode = 1;
    }
    if (argc >= 5 && strcmp(argv[4], "x16atkin5bench") == 0) {
        g_x16_mode = 1;
        g_x16_atkin5_bench_mode = 1;
    }
    if (argc >= 5 && strcmp(argv[4], "x16atkin5benchnonsplit") == 0) {
        g_x16_mode = 1;
        g_x16_atkin5_bench_mode = 1;
        g_x16_nonsplit_filter_mode = 1;
    }
    if (argc >= 5 && strcmp(argv[4], "x16atkin7bench") == 0) {
        g_x16_mode = 1;
        g_x16_atkin7_bench_mode = 1;
    }
    if (argc >= 5 && strcmp(argv[4], "x16halvestats") == 0) {
        g_x16_mode = 1;
        g_x16_halve_stats_mode = 1;
    }
    if (argc >= 5 && strcmp(argv[4], "x16branchstats") == 0) {
        g_x16_mode = 1;
        g_x16_branch_stats_mode = 1;
        if (argc >= 6) g_x16_branch_target = atoi(argv[5]);
    }
    if (argc >= 5 && strcmp(argv[4], "x16branchstatsnonsplit") == 0) {
        g_x16_mode = 1;
        g_x16_branch_stats_mode = 1;
        g_x16_nonsplit_filter_mode = 1;
        if (argc >= 6) g_x16_branch_target = atoi(argv[5]);
    }
    if (argc >= 5 && strcmp(argv[4], "x16branchstatsnonsplitdgate") == 0) {
        g_x16_mode = 1;
        g_x16_branch_stats_mode = 1;
        g_x16_nonsplit_filter_mode = 1;
        g_x16_first_d_y_gate_mode = 1;
        if (argc >= 6) g_x16_branch_target = atoi(argv[5]);
    }
    if (argc >= 5 && strcmp(argv[4], "x16gatestats") == 0) {
        g_x16_mode = 1;
        g_x16_gate_stats_mode = 1;
        if (argc >= 6) g_x16_gate_target = atoi(argv[5]);
    }
    if (argc >= 5 && strcmp(argv[4], "x16gatestatsnonsplit") == 0) {
        g_x16_mode = 1;
        g_x16_gate_stats_mode = 1;
        g_x16_nonsplit_filter_mode = 1;
        if (argc >= 6) g_x16_gate_target = atoi(argv[5]);
    }
    if (argc >= 5 && strcmp(argv[4], "x16lookaheadstats") == 0) {
        g_x16_mode = 1;
        g_x16_lookahead_stats_mode = 1;
        if (argc >= 6) g_x16_lookahead_depth = atoi(argv[5]);
        if (argc >= 7) g_x16_lookahead_target = atoi(argv[6]);
    }
    if (argc >= 5 && strcmp(argv[4], "x16labelstats") == 0) {
        g_x16_mode = 1;
        g_x16_label_stats_mode = 1;
        if (argc >= 6) g_x16_label_target = atoi(argv[5]);
    }
    if (argc >= 5 && strcmp(argv[4], "x16splitstats") == 0) {
        g_x16_mode = 1;
        g_x16_split_stats_mode = 1;
        if (argc >= 6) g_x16_split_target = atoi(argv[5]);
    }
    if (argc >= 5 && strcmp(argv[4], "x16splitbranchstats") == 0) {
        g_x16_mode = 1;
        g_x16_split_branch_stats_mode = 1;
        if (argc >= 6) g_x16_split_branch_target = atoi(argv[5]);
    }
    if (argc >= 5 && strcmp(argv[4], "x16splittranslate") == 0) {
        g_x16_mode = 1;
        g_x16_split_translate_stats_mode = 1;
        if (argc >= 6) g_x16_split_translate_target = atoi(argv[5]);
    }
    if (argc >= 5 && strcmp(argv[4], "x16cubic3stats") == 0) {
        g_x16_mode = 1;
        g_x16_cubic3_stats_mode = 1;
        if (argc >= 6) g_x16_cubic3_target = atoi(argv[5]);
    }
    if (argc >= 5 && strcmp(argv[4], "x16quarticstats") == 0) {
        g_x16_mode = 1;
        g_x16_quartic_stats_mode = 1;
        if (argc >= 6) g_x16_quartic_target = atoi(argv[5]);
    }
    if (argc >= 5 && strcmp(argv[4], "x16quarticstatsnonsplit") == 0) {
        g_x16_mode = 1;
        g_x16_quartic_stats_mode = 1;
        g_x16_nonsplit_filter_mode = 1;
        if (argc >= 6) g_x16_quartic_target = atoi(argv[5]);
    }
    if (argc >= 5 && strcmp(argv[4], "x16pairstats") == 0) {
        g_x16_mode = 1;
        g_x16_pair_stats_mode = 1;
    }
    if (argc >= 5 && strcmp(argv[4], "x16torsionstats") == 0) {
        g_x16_mode = 1;
        g_x16_torsion_stats_mode = 1;
    }
    if (argc >= 5 && strcmp(argv[4], "x1_32rootbench") == 0) {
        g_x1_32_root_bench_mode = 1;
    }
    if (p < 5) { fprintf(stderr, "p must be >= 5\n"); return 1; }

    printf("Pomerance triple search\n\n");
    printf("p = "); print128(p); printf("  (%d digits)\n", digits128(p));
    if (g_seed_offset || g_max_trials_override) {
        printf("seed_offset = %llu  max_trials_override = %llu\n",
               (unsigned long long)g_seed_offset,
               (unsigned long long)g_max_trials_override);
    }
    if (g_x16_halve_full_mode) printf("curve_mode = X1(16) prescribed torsion + full successive-halving backtracking\n");
    else if (g_x16_halve_mode && g_x16_first_d_skip_mode) printf("curve_mode = X1(16) prescribed torsion + nonsplit + first-d y-gate skip-to-depth-5 halving\n");
    else if (g_x16_halve_mode && g_x16_nonsplit_filter_mode && g_x16_first_d_y_gate_mode) printf("curve_mode = X1(16) prescribed torsion + nonsplit + first-d y-gate successive halving\n");
    else if (g_x16_halve_mode && g_x16_nonsplit_filter_mode) printf("curve_mode = X1(16) prescribed torsion + nonsplit-discriminant successive halving\n");
    else if (g_x16_halve_mode) printf("curve_mode = X1(16) prescribed torsion + successive halving\n");
    else if (g_x16_halve_stats_mode) printf("curve_mode = X1(16) first-branch halving survival stats\n");
    else if (g_x16_branch_stats_mode && g_x16_nonsplit_filter_mode && g_x16_first_d_y_gate_mode) printf("curve_mode = X1(16) nonsplit + first-d y-gate bounded first-vs-all branch halving stats\n");
    else if (g_x16_branch_stats_mode && g_x16_nonsplit_filter_mode) printf("curve_mode = X1(16) nonsplit filtered bounded first-vs-all branch halving stats\n");
    else if (g_x16_branch_stats_mode) printf("curve_mode = X1(16) bounded first-vs-all branch halving stats\n");
    else if (g_x16_gate_stats_mode && g_x16_nonsplit_filter_mode) printf("curve_mode = X1(16) nonsplit filtered inverse-gate branch halving stats\n");
    else if (g_x16_gate_stats_mode) printf("curve_mode = X1(16) inverse-gate branch halving stats\n");
    else if (g_x16_lookahead_stats_mode) printf("curve_mode = X1(16) bounded lookahead branch halving stats\n");
    else if (g_x16_label_stats_mode) printf("curve_mode = X1(16) halving sign-label order stats\n");
    else if (g_x16_split_stats_mode) printf("curve_mode = X1(16) split-discriminant halving stats\n");
    else if (g_x16_split_branch_stats_mode) printf("curve_mode = X1(16) split-class bounded first-vs-all branch halving stats\n");
    else if (g_x16_split_translate_stats_mode) printf("curve_mode = X1(16) split 2-torsion translation first-branch stats\n");
    else if (g_x16_cubic3_stats_mode) printf("curve_mode = X1(16) nonsplit cubic-character halving stats\n");
    else if (g_x16_quartic_stats_mode && g_x16_nonsplit_filter_mode) printf("curve_mode = X1(16) nonsplit quartic sqrt-correction stats\n");
    else if (g_x16_quartic_stats_mode) printf("curve_mode = X1(16) quartic sqrt-correction stats\n");
    else if (g_x16_pair_stats_mode) printf("curve_mode = X1(16) quadratic-root split/nonsplit pair stats\n");
    else if (g_x16_torsion_stats_mode) printf("curve_mode = X1(16) marked 8P torsion classifier stats\n");
    else if (g_x16_ell3_bench_mode) printf("curve_mode = X1(16) ell=3 trace-filter benchmark\n");
    else if (g_x16_ell3_direct_y_bench_mode) printf("curve_mode = X1(16) direct-y ell=3 quartic benchmark\n");
    else if (g_x16_atkin5_bench_mode && g_x16_nonsplit_filter_mode) printf("curve_mode = X1(16) nonsplit Atkin/Elkies ell=5 status benchmark\n");
    else if (g_x16_atkin5_bench_mode) printf("curve_mode = X1(16) Atkin/Elkies ell=5 status benchmark\n");
    else if (g_x16_atkin7_bench_mode) printf("curve_mode = X1(16) Atkin/Elkies ell=7 status benchmark\n");
    else if (g_x1_32_root_bench_mode) printf("curve_mode = X1(32) degree-10 fiber root-existence benchmark\n");
    else if (g_x16_mode) printf("curve_mode = X1(16) prescribed torsion\n");

    if (!is_prime128(p)) {
        printf("Not prime, finding next...\n");
        if (p % 2 == 0) p++;
        while (!is_prime128(p)) p += 2;
        printf("p = "); print128(p); printf("  (%d digits)\n", digits128(p));
    }

    if (g_x16_ell3_bench_mode) {
        return bench_x16_ell3_128(p);
    }
    if (g_x16_ell3_direct_y_bench_mode) {
        return bench_x16_ell3_direct_y_128(p);
    }
    if (g_x16_atkin5_bench_mode) {
        return bench_x16_atkin5_128(p);
    }
    if (g_x16_atkin7_bench_mode) {
        return bench_x16_atkin7_128(p);
    }
    if (g_x16_halve_stats_mode) {
        return bench_x16_halve_stats_128(p);
    }
    if (g_x16_branch_stats_mode) {
        return bench_x16_branch_stats_128(p);
    }
    if (g_x16_gate_stats_mode) {
        return bench_x16_gate_stats_128(p);
    }
    if (g_x16_lookahead_stats_mode) {
        return bench_x16_lookahead_stats_128(p);
    }
    if (g_x16_label_stats_mode) {
        return bench_x16_label_stats_128(p);
    }
    if (g_x16_split_stats_mode) {
        return bench_x16_split_stats_128(p);
    }
    if (g_x16_split_branch_stats_mode) {
        return bench_x16_split_branch_stats_128(p);
    }
    if (g_x16_split_translate_stats_mode) {
        return bench_x16_split_translate_stats_128(p);
    }
    if (g_x16_cubic3_stats_mode) {
        return bench_x16_cubic3_stats_128(p);
    }
    if (g_x16_quartic_stats_mode) {
        return bench_x16_quartic_stats_128(p);
    }
    if (g_x16_pair_stats_mode) {
        return bench_x16_pair_stats_128(p);
    }
    if (g_x16_torsion_stats_mode) {
        return bench_x16_torsion_stats_128(p);
    }
    if (g_x1_32_root_bench_mode) {
        return bench_x1_32_root_128(p);
    }

    /* X1 modes live in the u128 path, even when the prime itself fits u64. */
    if (g_x16_mode) {
        printf("Using u128 path\n");
        return search128(p);
    }

    /* dispatch based on size */
    if (p < ((u128)1 << 63)) {
        printf("Using u64 fast path\n");
        return search64((u64)p);
    } else {
        printf("Using u128 path\n");
        return search128(p);
    }
}
