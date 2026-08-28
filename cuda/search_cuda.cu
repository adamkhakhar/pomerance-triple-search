/*
 * pomerance_cuda.cu -- CUDA prototype for the p23/p25/p26 X1(16) nonsplit
 * first-branch halving search path from pomerance.c.
 *
 * This is deliberately narrow: it ports the production x16halvenonsplit path
 * for primes with the fast square-root cases p == 5 mod 8 or p == 3 mod 4,
 * with p < 2^127.  The generic 2-Sylow search and diagnostic modes remain in
 * pomerance.c.
 *
 * Compile on an Ada-generation NVIDIA GPU:
 *   nvcc -O3 -std=c++17 -arch=sm_89 -o pomerance_cuda pomerance_cuda.cu
 * Add -DPOM_CUDA_DETAILED_STATS=1 to keep the per-stage diagnostic counters.
 *
 * Usage:
 *   ./pomerance_cuda <p> [seed_offset] [max_trials] [x16halvenonsplit] [chunk_trials] [blocks] [threads] [claim_batch] [auto|generic|u96]
 * The auto backend uses the specialized u96 path for p < 2^96 and the
 * generic u128 path otherwise.
 *
 * Example:
 *   ./pomerance_cuda 100000000000000000000117 0 1000000 x16halvenonsplit
 *   ./pomerance_cuda 100000000000000000000000067 121 550000000000 x16halvenonsplit 1000000000
 */

/* POM_HOST_TEST compiles the arithmetic core as ordinary C++ so the device
   math can be unit-tested on a machine with no GPU (see host_test.cpp).
   The kernels and the CUDA driver are excluded in that mode. */
#if !defined(POM_HOST_TEST) && !defined(POM_CUDA_SHIM)
#include <cuda_runtime.h>
#endif

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

using u32 = uint32_t;
using u64 = unsigned long long;
using u128 = unsigned __int128;

#ifdef POM_HOST_TEST
#define POM_NO_DRIVER 1
#define HD
#define DEV
#else
#define HD __host__ __device__
#define DEV __device__
#endif

#ifndef POM_CUDA_DETAILED_STATS
#define POM_CUDA_DETAILED_STATS 0
#endif

struct u256 {
    u128 lo;
    u128 hi;
};

struct Mont128 {
    u128 p;
    u128 ni;
    u128 R2;
    u128 one;
};

struct Rng {
    u64 s0;
    u64 s1;
};

struct SearchParams {
    u128 p;
    u128 rand_mask;
    u128 sqrtm1;
    u128 sqrtm1_m;
    u128 sqrt_exp;
    u128 inv2_m;
    u128 two_m;
    u128 four_m;
    u128 eight_m;
    u128 c24_m;
    u128 c32_m;
    u128 c48_m;
    u128 inv4_m;
    Mont128 mt;
    int k;
    int sqrt_case;
};

struct U128Parts {
    u64 lo;
    u64 hi;
};

struct GpuStats {
    u64 claimed;
    u64 processed;
    u64 y_draws;
    u64 y_nonsplit;
    u64 y_with_sqrt_D;
    u64 roots_valid;
    u64 halve_calls;
    u64 chain_survivors;
};

struct GpuResult {
    int found;
    u64 trial_index;
    U128Parts A;
    U128Parts x0;
};

HD static inline U128Parts pack128(u128 x) {
    U128Parts out;
    out.lo = (u64)x;
    out.hi = (u64)(x >> 64);
    return out;
}

static inline u128 unpack128(U128Parts x) {
    return ((u128)x.hi << 64) | (u128)x.lo;
}

HD static inline u256 wide_mul(u128 a, u128 b) {
    u64 a0 = (u64)a;
    u64 a1 = (u64)(a >> 64);
    u64 b0 = (u64)b;
    u64 b1 = (u64)(b >> 64);
    u128 ll = (u128)a0 * b0;
    u128 lh = (u128)a0 * b1;
    u128 hl = (u128)a1 * b0;
    u128 hh = (u128)a1 * b1;
    u128 mid = lh + hl;
    u128 carry_mid = (mid < lh) ? 1 : 0;
    u128 lo = ll + (mid << 64);
    u128 carry_lo = (lo < ll) ? 1 : 0;
    return u256{lo, hh + (mid >> 64) + (carry_mid << 64) + carry_lo};
}

HD static inline u256 wide_add(u256 a, u256 b) {
    u128 lo = a.lo + b.lo;
    return u256{lo, a.hi + b.hi + ((lo < a.lo) ? 1 : 0)};
}

HD static inline u128 addmod128(u128 a, u128 b, u128 p) {
    u128 s = a + b;
    return s >= p ? s - p : s;
}

HD static inline u128 submod128(u128 a, u128 b, u128 p) {
    return a >= b ? a - b : p - b + a;
}

HD static inline u128 mred128(u256 T, const Mont128 *m) {
    u128 q = T.lo * m->ni;
    u256 s = wide_add(T, wide_mul(q, m->p));
    u128 t = s.hi;
    return t >= m->p ? t - m->p : t;
}

HD static inline u128 mm128(u128 a, u128 b, const Mont128 *m) {
    return mred128(wide_mul(a, b), m);
}

HD static inline u128 toM128(u128 a, const Mont128 *m) {
    return mm128(a % m->p, m->R2, m);
}

HD static inline u128 toM128_reduced(u128 a, const Mont128 *m) {
    return mm128(a, m->R2, m);
}

HD static inline u128 frM128(u128 a, const Mont128 *m) {
    return mred128(u256{a, 0}, m);
}

HD static inline u128 mulmod128_mont(u128 a, u128 b, const Mont128 *mt) {
    return frM128(mm128(toM128(a, mt), toM128(b, mt), mt), mt);
}

HD static u128 powmod128_mont(u128 a, u128 e, const Mont128 *mt) {
    u128 r = mt->one;
    u128 b = toM128(a, mt);
    while (e > 0) {
        if (e & 1) r = mm128(r, b, mt);
        e >>= 1;
        if (e) b = mm128(b, b, mt);
    }
    return frM128(r, mt);
}

HD static u128 powmod128_mont_raw(u128 a_m, u128 e, const Mont128 *mt) {
    u128 r = mt->one;
    u128 b = a_m;
    while (e > 0) {
        if (e & 1) r = mm128(r, b, mt);
        e >>= 1;
        if (e) b = mm128(b, b, mt);
    }
    return r;
}

HD static int sqrtmod_p5_128(u128 *root, u128 n, u128 p, u128 sqrtm1,
                             const Mont128 *mt) {
    n %= p;
    if (n == 0) {
        *root = 0;
        return 1;
    }
    if ((u64)(p & 3) == 3) {
        u128 x = powmod128_mont(n, (p + 1) >> 2, mt);
        if (mulmod128_mont(x, x, mt) == n) {
            *root = x;
            return 1;
        }
        return 0;
    }
    u128 x = powmod128_mont(n, (p + 3) >> 3, mt);
    if (mulmod128_mont(x, x, mt) == n) {
        *root = x;
        return 1;
    }
    x = mulmod128_mont(x, sqrtm1, mt);
    if (mulmod128_mont(x, x, mt) == n) {
        *root = x;
        return 1;
    }
    return 0;
}

HD static int sqrtmod_p5_128_mont(u128 *root_m, u128 n_m, u128 p,
                                  u128 sqrtm1_m, const Mont128 *mt) {
    if (n_m == 0) {
        *root_m = 0;
        return 1;
    }
    u128 x_m = powmod128_mont_raw(n_m,
                                  ((u64)(p & 3) == 3) ? ((p + 1) >> 2)
                                                      : ((p + 3) >> 3),
                                  mt);
    if (mm128(x_m, x_m, mt) == n_m) {
        *root_m = x_m;
        return 1;
    }
    if ((u64)(p & 3) == 3) return 0;
    x_m = mm128(x_m, sqrtm1_m, mt);
    if (mm128(x_m, x_m, mt) == n_m) {
        *root_m = x_m;
        return 1;
    }
    return 0;
}

HD static u128 invert128_mont(u128 a, u128 p, const Mont128 *mt) {
    return powmod128_mont(a, p - 2, mt);
}

HD static u128 invert128_mont_raw(u128 a_m, u128 p, const Mont128 *mt) {
    return powmod128_mont_raw(a_m, p - 2, mt);
}

HD static int invert_batch2_128(u128 out[2], const u128 vals[2], u128 p,
                                const Mont128 *mt) {
    if (vals[0] == 0 || vals[1] == 0) return 0;
    u128 prefix1 = vals[0];
    u128 prefix2 = mulmod128_mont(vals[0], vals[1], mt);
    u128 acc = invert128_mont(prefix2, p, mt);
    out[1] = mulmod128_mont(acc, prefix1, mt);
    out[0] = mulmod128_mont(acc, vals[1], mt);
    return 1;
}

HD static int invert_batch2_128_mont(u128 out[2], const u128 vals_m[2], u128 p,
                                     const Mont128 *mt) {
    if (vals_m[0] == 0 || vals_m[1] == 0) return 0;
    u128 prefix1 = vals_m[0];
    u128 prefix2 = mm128(vals_m[0], vals_m[1], mt);
    u128 acc = invert128_mont_raw(prefix2, p, mt);
    out[1] = mm128(acc, prefix1, mt);
    out[0] = mm128(acc, vals_m[1], mt);
    return 1;
}

HD static inline void xDBL128(u128 *Xo, u128 *Zo, u128 X, u128 Z,
                              u128 a24, const Mont128 *m) {
    u128 p = m->p;
    u128 u = addmod128(X, Z, p);
    u128 v = submod128(X, Z, p);
    u = mm128(u, u, m);
    v = mm128(v, v, m);
    *Xo = mm128(u, v, m);
    u128 w = submod128(u, v, p);
    *Zo = mm128(w, addmod128(v, mm128(a24, w, m), p), m);
}

HD static int verify128_mont(u128 p, u128 A, u128 x0, int k, const Mont128 *mt) {
    if (A % p == 2 || A % p == p - 2) return 0;
    u128 inv4_m = invert128_mont(4, p, mt);
    u128 a24m = mm128(toM128(addmod128(A, 2, p), mt), toM128(inv4_m, mt), mt);
    u128 X = toM128(x0 % p, mt);
    u128 Z = mt->one;
    for (int i = 1; i <= k; i++) {
        xDBL128(&X, &Z, X, Z, a24m, mt);
        if (i < k && Z == 0) return 0;
        if (i == k && Z != 0) return 0;
    }
    return 1;
}

HD static int verify128_mont_raw(u128 A_m, u128 x0_m, const SearchParams *params) {
    const u128 p = params->p;
    if (A_m == params->two_m || A_m == submod128(0, params->two_m, p)) return 0;

    const Mont128 *mt = &params->mt;
    u128 a24m = mm128(addmod128(A_m, params->two_m, p), params->inv4_m, mt);
    u128 X = x0_m;
    u128 Z = mt->one;
    for (int i = 1; i <= params->k; i++) {
        xDBL128(&X, &Z, X, Z, a24m, mt);
        if (i < params->k && Z == 0) return 0;
        if (i == params->k && Z != 0) return 0;
    }
    return 1;
}

HD static u128 x16_A_numerator_from_y128(u128 y, u128 p, const Mont128 *mt) {
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

HD static u128 x16_A_numerator_from_y128_mont(u128 y_m, const SearchParams *params) {
    const u128 p = params->p;
    const Mont128 *mt = &params->mt;
    u128 num_m = mt->one;
    num_m = submod128(mm128(num_m, y_m, mt), params->eight_m, p);
    num_m = addmod128(mm128(num_m, y_m, mt), params->c24_m, p);
    num_m = submod128(mm128(num_m, y_m, mt), params->c32_m, p);
    num_m = addmod128(mm128(num_m, y_m, mt), params->eight_m, p);
    num_m = addmod128(mm128(num_m, y_m, mt), params->c32_m, p);
    num_m = submod128(mm128(num_m, y_m, mt), params->c48_m, p);
    num_m = addmod128(mm128(num_m, y_m, mt), params->c32_m, p);
    num_m = submod128(mm128(num_m, y_m, mt), params->eight_m, p);
    return num_m;
}

HD static int x16_y_predicts_nonsplit128(u128 p, u128 y, u128 y2,
                                         const Mont128 *mt) {
    u128 f1 = submod128(y2, 2, p);
    u128 two_y = addmod128(y, y, p);
    u128 four_y = addmod128(two_y, two_y, p);
    u128 f2 = addmod128(submod128(y2, four_y, p), 2, p);
    u128 f = mulmod128_mont(f1, f2, mt);
    if (f == 0) return 0;
    u128 leg = powmod128_mont(f, (p - 1) >> 1, mt);
    return leg != 1;
}

HD static int x16_y_predicts_nonsplit128_mont(u128 y_m, u128 y2_m,
                                              const SearchParams *params) {
    const u128 p = params->p;
    const Mont128 *mt = &params->mt;
    u128 f1_m = submod128(y2_m, params->two_m, p);
    u128 two_y_m = addmod128(y_m, y_m, p);
    u128 four_y_m = addmod128(two_y_m, two_y_m, p);
    u128 f2_m = addmod128(submod128(y2_m, four_y_m, p), params->two_m, p);
    u128 f_m = mm128(f1_m, f2_m, mt);
    if (f_m == 0) return 0;
    u128 leg_m = powmod128_mont_raw(f_m, (p - 1) >> 1, mt);
    return leg_m != mt->one;
}

HD static int x16_root_to_montgomery_A128(u128 *Ao, u128 *xPo,
                                          u128 p, u128 x, u128 y,
                                          const Mont128 *mt) {
    u128 num = x16_A_numerator_from_y128(y, p, mt);
    u128 ym1 = submod128(y, 1, p);
    u128 ym1_2 = mulmod128_mont(ym1, ym1, mt);
    u128 denA = mulmod128_mont(4, mulmod128_mont(ym1_2, ym1_2, mt), mt);
    u128 denx = submod128(x, y, p);
    u128 vals[2] = {denA, denx};
    u128 invs[2];
    if (!invert_batch2_128(invs, vals, p, mt)) return 0;

    u128 A = mulmod128_mont(num, invs[0], mt);
    u128 xP = mulmod128_mont(x, invs[1], mt);
    if (A <= 2 || A >= p - 2) return 0;
    *Ao = A;
    *xPo = xP;
    return 1;
}

HD static int x16_root_to_montgomery_A128_mont(u128 *Ao, u128 *Amo, u128 *xPmo,
                                               u128 x_m, u128 y_m,
                                               const SearchParams *params) {
    const u128 p = params->p;
    const Mont128 *mt = &params->mt;
    u128 num_m = x16_A_numerator_from_y128_mont(y_m, params);
    u128 ym1_m = submod128(y_m, mt->one, p);
    u128 ym1_2_m = mm128(ym1_m, ym1_m, mt);
    u128 denA_m = mm128(params->four_m, mm128(ym1_2_m, ym1_2_m, mt), mt);
    u128 denx_m = submod128(x_m, y_m, p);
    u128 vals_m[2] = {denA_m, denx_m};
    u128 invs_m[2];
    if (!invert_batch2_128_mont(invs_m, vals_m, p, mt)) return 0;

    u128 A_m = mm128(num_m, invs_m[0], mt);
    u128 xP_m = mm128(x_m, invs_m[1], mt);
    u128 A = frM128(A_m, mt);
    if (A <= 2 || A >= p - 2) return 0;
    *Ao = A;
    *Amo = A_m;
    *xPmo = xP_m;
    return 1;
}

HD static int halve_once_first128(u128 *xo, u128 p, u128 A, u128 x,
                                  u128 sqrtm1, const Mont128 *mt) {
    const u128 inv2 = (p + 1) >> 1;
    u128 x2 = mulmod128_mont(x, x, mt);
    u128 d = addmod128(addmod128(x2, mulmod128_mont(A, x, mt), p), 1, p);
    u128 sd;
    if (!sqrtmod_p5_128(&sd, d, p, sqrtm1, mt)) return 0;

    u128 roots_d[2] = {sd, submod128(0, sd, p)};
    for (int i = 0; i < 2; i++) {
        u128 u = addmod128(addmod128(x, x, p), addmod128(roots_d[i], roots_d[i], p), p);
        u128 w = submod128(mulmod128_mont(u, u, mt), 4, p);
        u128 sw;
        if (!sqrtmod_p5_128(&sw, w, p, sqrtm1, mt)) continue;
        u128 candidates[2] = {
            mulmod128_mont(addmod128(u, sw, p), inv2, mt),
            mulmod128_mont(submod128(u, sw, p), inv2, mt),
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

HD static int halve_once_first128_mont(u128 *xo_m, u128 A_m, u128 x_m,
                                       const SearchParams *params) {
    const u128 p = params->p;
    const Mont128 *mt = &params->mt;
    u128 x2_m = mm128(x_m, x_m, mt);
    u128 d_m = addmod128(addmod128(x2_m, mm128(A_m, x_m, mt), p), mt->one, p);
    u128 sd_m;
    if (!sqrtmod_p5_128_mont(&sd_m, d_m, p, params->sqrtm1_m, mt)) return 0;

    u128 roots_d[2] = {sd_m, submod128(0, sd_m, p)};
    for (int i = 0; i < 2; i++) {
        u128 u_m = addmod128(addmod128(x_m, x_m, p),
                             addmod128(roots_d[i], roots_d[i], p), p);
        u128 w_m = submod128(mm128(u_m, u_m, mt), params->four_m, p);
        u128 sw_m;
        if (!sqrtmod_p5_128_mont(&sw_m, w_m, p, params->sqrtm1_m, mt)) continue;
        u128 candidates[2] = {
            mm128(addmod128(u_m, sw_m, p), params->inv2_m, mt),
            mm128(submod128(u_m, sw_m, p), params->inv2_m, mt),
        };
        for (int j = 0; j < 2; j++) {
            if (candidates[j] != 0) {
                *xo_m = candidates[j];
                return 1;
            }
        }
    }
    return 0;
}

HD static int halve_chain_from_depth128(u128 *xout, u128 p, u128 A, u128 x,
                                        int depth, int k, u128 sqrtm1,
                                        const Mont128 *mt) {
    for (; depth < k; depth++) {
        if (!halve_once_first128(&x, p, A, x, sqrtm1, mt)) return 0;
    }
    if (!verify128_mont(p, A, x, k, mt)) return 0;
    *xout = x;
    return 1;
}

HD static int halve_chain_from_depth128_mont(u128 *xout_m, u128 A_m, u128 x_m,
                                             int depth, const SearchParams *params) {
    for (; depth < params->k; depth++) {
        if (!halve_once_first128_mont(&x_m, A_m, x_m, params)) return 0;
    }
    if (!verify128_mont_raw(A_m, x_m, params)) return 0;
    *xout_m = x_m;
    return 1;
}

HD static inline u64 rng64(Rng *r) {
    u64 s1 = r->s0;
    u64 s0 = r->s1;
    r->s0 = s0;
    s1 ^= s1 << 23;
    r->s1 = s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26);
    return r->s1 + s0;
}

HD static inline u128 rand_below128(Rng *rng, u128 p, u128 mask) {
    for (;;) {
        u128 v = ((u128)rng64(rng) << 64) | (u128)rng64(rng);
        v &= mask;
        if (v < p) return v;
    }
}

HD static inline u64 splitmix64_next(u64 *x) {
    u64 z = (*x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

DEV static inline int found_now(const GpuResult *result) {
    return *((volatile const int *)&result->found);
}

#ifndef POM_HOST_TEST
__global__ static void x16halvenonsplit_kernel(SearchParams params,
                                               u64 seed_offset,
                                               u64 chunk_nonce,
                                               u64 max_curves,
                                               u64 global_base,
                                               u64 claim_batch,
                                               GpuStats *__restrict__ stats,
                                               GpuResult *__restrict__ result) {
    u64 tid = (u64)blockIdx.x * (u64)blockDim.x + (u64)threadIdx.x;
    u64 seed = seed_offset ^ (chunk_nonce * 0xd1342543de82ef95ULL) ^
               (tid * 0x9e3779b97f4a7c15ULL) ^ 0x7364529176530163ULL;
    Rng rng;
    rng.s0 = splitmix64_next(&seed);
    rng.s1 = splitmix64_next(&seed);
    if ((rng.s0 | rng.s1) == 0) rng.s1 = 1442695040888963407ULL;
    for (int i = 0; i < 200; i++) (void)rng64(&rng);

    const u128 p = params.p;
    const Mont128 *mt = &params.mt;
    u64 local_processed = 0;
#if POM_CUDA_DETAILED_STATS
    u64 local_y_draws = 0;
    u64 local_y_nonsplit = 0;
    u64 local_y_with_sqrt_D = 0;
    u64 local_roots_valid = 0;
    u64 local_halve_calls = 0;
    u64 local_chain_survivors = 0;
#endif
    u64 next_claim = 0;
    u64 claim_limit = 0;

    while (!found_now(result)) {
        u128 y = rand_below128(&rng, p, params.rand_mask);
#if POM_CUDA_DETAILED_STATS
        local_y_draws++;
#endif
        if (y == 0) continue;

        u128 y_m = toM128_reduced(y, mt);
        u128 y2_m = mm128(y_m, y_m, mt);
        if (!x16_y_predicts_nonsplit128_mont(y_m, y2_m, &params)) continue;
#if POM_CUDA_DETAILED_STATS
        local_y_nonsplit++;
#endif

        u128 y3_m = mm128(y2_m, y_m, mt);
        u128 two_y_m = addmod128(y_m, y_m, p);
        u128 qa_m = submod128(y2_m, two_y_m, p);
        if (qa_m == 0) continue;
        u128 qb_m = submod128(addmod128(y2_m, y2_m, p), y3_m, p);
        u128 qc_m = submod128(mt->one, y_m, p);
        u128 D_m = submod128(
            mm128(qb_m, qb_m, mt),
            mm128(addmod128(qa_m, qa_m, p), addmod128(qc_m, qc_m, p), mt),
            p);
        u128 sd_m;
        if (!sqrtmod_p5_128_mont(&sd_m, D_m, p, params.sqrtm1_m, mt)) continue;
#if POM_CUDA_DETAILED_STATS
        local_y_with_sqrt_D++;
#endif

        u128 inv_2qa_m = invert128_mont_raw(addmod128(qa_m, qa_m, p), p, mt);
        u128 roots_m[2] = {
            mm128(submod128(sd_m, qb_m, p), inv_2qa_m, mt),
            mm128(submod128(submod128(0, sd_m, p), qb_m, p), inv_2qa_m, mt),
        };

        int stop = 0;
        for (int ri = 0; ri < 2; ri++) {
            u128 A;
            u128 A_m;
            u128 xP16_m;
            if (!x16_root_to_montgomery_A128_mont(&A, &A_m, &xP16_m, roots_m[ri],
                                                  y_m, &params)) continue;
#if POM_CUDA_DETAILED_STATS
            local_roots_valid++;
#endif

            if (next_claim >= claim_limit) {
                next_claim = atomicAdd(&stats->claimed, claim_batch);
                claim_limit = next_claim + claim_batch;
            }
            u64 claim = next_claim++;
            if (claim >= max_curves || found_now(result)) {
                stop = 1;
                break;
            }

            local_processed++;
#if POM_CUDA_DETAILED_STATS
            local_halve_calls++;
#endif
            u128 xR_m;
            if (halve_chain_from_depth128_mont(&xR_m, A_m, xP16_m, 4, &params)) {
#if POM_CUDA_DETAILED_STATS
                local_chain_survivors++;
#endif
                if (atomicCAS(&result->found, 0, 1) == 0) {
                    result->trial_index = global_base + claim;
                    result->A = pack128(A);
                    result->x0 = pack128(frM128(xR_m, mt));
                }
                stop = 1;
                break;
            }
        }
        if (stop) break;
    }

    atomicAdd(&stats->processed, local_processed);
#if POM_CUDA_DETAILED_STATS
    atomicAdd(&stats->y_draws, local_y_draws);
    atomicAdd(&stats->y_nonsplit, local_y_nonsplit);
    atomicAdd(&stats->y_with_sqrt_D, local_y_with_sqrt_D);
    atomicAdd(&stats->roots_valid, local_roots_valid);
    atomicAdd(&stats->halve_calls, local_halve_calls);
    atomicAdd(&stats->chain_survivors, local_chain_survivors);
#endif
}
#endif  /* POM_HOST_TEST */

struct U96 {
    u32 v[3];
};

struct Field96 {
    U96 p;
    U96 r2;
    U96 one;
    u32 nprime;
};

struct SearchParams96 {
    Field96 f;
    U96 rand_mask;
    U96 sqrtm1_m;
    U96 inv2_m;
    U96 two_m;
    U96 four_m;
    U96 eight_m;
    U96 c24_m;
    U96 c32_m;
    U96 c48_m;
    U96 inv4_m;
    u128 leg_exp;
    u128 sqrt_exp;
    u128 inv_exp;
    int k;
    int sqrt_case;
};

HD static inline U96 u96_from_u64(u64 x) {
    U96 out{{(u32)x, (u32)(x >> 32), 0}};
    return out;
}

HD static inline U96 u96_from_u128(u128 x) {
    U96 out{{(u32)x, (u32)(x >> 32), (u32)(x >> 64)}};
    return out;
}

HD static inline u128 u96_to_u128(U96 x) {
    return (u128)x.v[0] | ((u128)x.v[1] << 32) | ((u128)x.v[2] << 64);
}

HD static inline U128Parts pack96(U96 x) {
    return pack128(u96_to_u128(x));
}

HD static inline int cmp96(U96 a, U96 b) {
    if (a.v[2] < b.v[2]) return -1;
    if (a.v[2] > b.v[2]) return 1;
    if (a.v[1] < b.v[1]) return -1;
    if (a.v[1] > b.v[1]) return 1;
    if (a.v[0] < b.v[0]) return -1;
    if (a.v[0] > b.v[0]) return 1;
    return 0;
}

HD static inline int is_zero96(U96 a) {
    return (a.v[0] | a.v[1] | a.v[2]) == 0;
}

HD static inline int eq96(U96 a, U96 b) {
    return ((a.v[0] ^ b.v[0]) | (a.v[1] ^ b.v[1]) | (a.v[2] ^ b.v[2])) == 0;
}

HD static inline U96 and96(U96 a, U96 b) {
    U96 out{{a.v[0] & b.v[0], a.v[1] & b.v[1], a.v[2] & b.v[2]}};
    return out;
}

HD static U96 sub96_raw(U96 a, U96 b, u32 *borrow_out = nullptr) {
    U96 out;
    u64 bi = b.v[0];
    out.v[0] = (u32)((u64)a.v[0] - bi);
    u64 borrow = ((u64)a.v[0] < bi) ? 1 : 0;
    bi = (u64)b.v[1] + borrow;
    out.v[1] = (u32)((u64)a.v[1] - bi);
    borrow = ((u64)a.v[1] < bi) ? 1 : 0;
    bi = (u64)b.v[2] + borrow;
    out.v[2] = (u32)((u64)a.v[2] - bi);
    borrow = ((u64)a.v[2] < bi) ? 1 : 0;
    if (borrow_out) *borrow_out = (u32)borrow;
    return out;
}

HD static U96 add96_raw(U96 a, U96 b, u32 *carry_out = nullptr) {
    U96 out;
    u64 s = (u64)a.v[0] + b.v[0];
    out.v[0] = (u32)s;
    u64 carry = s >> 32;
    s = (u64)a.v[1] + b.v[1] + carry;
    out.v[1] = (u32)s;
    carry = s >> 32;
    s = (u64)a.v[2] + b.v[2] + carry;
    out.v[2] = (u32)s;
    carry = s >> 32;
    if (carry_out) *carry_out = (u32)carry;
    return out;
}

HD static U96 addmod96(U96 a, U96 b, U96 p) {
    u32 carry = 0;
    U96 s = add96_raw(a, b, &carry);
    if (carry || cmp96(s, p) >= 0) s = sub96_raw(s, p);
    return s;
}

HD static U96 submod96(U96 a, U96 b, U96 p) {
    u32 borrow = 0;
    U96 d = sub96_raw(a, b, &borrow);
    if (borrow) d = add96_raw(d, p);
    return d;
}

HD static u32 inv32_odd(u32 p0) {
    u32 x = 1;
    for (int i = 0; i < 5; i++) x *= 2u - p0 * x;
    return 0u - x;
}

HD static U96 mont_red96(const u32 in[6], const Field96 *f) {
    u32 t[7] = {in[0], in[1], in[2], in[3], in[4], in[5], 0};
    for (int i = 0; i < 3; i++) {
        u32 m = t[i] * f->nprime;
        u64 carry = 0;
        for (int j = 0; j < 3; j++) {
            u64 uv = (u64)t[i + j] + (u64)m * f->p.v[j] + carry;
            t[i + j] = (u32)uv;
            carry = uv >> 32;
        }
        int k = i + 3;
        while (carry) {
            u64 uv = (u64)t[k] + carry;
            t[k] = (u32)uv;
            carry = uv >> 32;
            k++;
        }
    }

    U96 r{{t[3], t[4], t[5]}};
    u32 high = t[6];
    if (high || cmp96(r, f->p) >= 0) {
        u32 borrow = 0;
        U96 s = sub96_raw(r, f->p, &borrow);
        high -= borrow;
        r = s;
    }
    return r;
}

HD static U96 mont_mul96(U96 a, U96 b, const Field96 *f) {
    const u32 a0 = a.v[0], a1 = a.v[1], a2 = a.v[2];
    const u32 b0 = b.v[0], b1 = b.v[1], b2 = b.v[2];

    u64 uv = (u64)a0 * b0;
    u32 t0 = (u32)uv;
    u64 carry = uv >> 32;
    uv = (u64)a0 * b1 + carry;
    u32 t1 = (u32)uv;
    carry = uv >> 32;
    uv = (u64)a0 * b2 + carry;
    u32 t2 = (u32)uv;
    u32 t3 = (u32)(uv >> 32);

    uv = (u64)t1 + (u64)a1 * b0;
    t1 = (u32)uv;
    carry = uv >> 32;
    uv = (u64)t2 + (u64)a1 * b1 + carry;
    t2 = (u32)uv;
    carry = uv >> 32;
    uv = (u64)t3 + (u64)a1 * b2 + carry;
    t3 = (u32)uv;
    u32 t4 = (u32)(uv >> 32);

    uv = (u64)t2 + (u64)a2 * b0;
    t2 = (u32)uv;
    carry = uv >> 32;
    uv = (u64)t3 + (u64)a2 * b1 + carry;
    t3 = (u32)uv;
    carry = uv >> 32;
    uv = (u64)t4 + (u64)a2 * b2 + carry;
    t4 = (u32)uv;
    u32 t5 = (u32)(uv >> 32);
    u32 t6 = 0;

    const u32 p0 = f->p.v[0], p1 = f->p.v[1], p2 = f->p.v[2];
    const u32 nprime = f->nprime;

    u32 m = t0 * nprime;
    uv = (u64)t0 + (u64)m * p0;
    carry = uv >> 32;
    uv = (u64)t1 + (u64)m * p1 + carry;
    t1 = (u32)uv;
    carry = uv >> 32;
    uv = (u64)t2 + (u64)m * p2 + carry;
    t2 = (u32)uv;
    carry = uv >> 32;
    uv = (u64)t3 + carry;
    t3 = (u32)uv;
    carry = uv >> 32;
    uv = (u64)t4 + carry;
    t4 = (u32)uv;
    carry = uv >> 32;
    uv = (u64)t5 + carry;
    t5 = (u32)uv;
    t6 = (u32)(uv >> 32);

    m = t1 * nprime;
    uv = (u64)t1 + (u64)m * p0;
    carry = uv >> 32;
    uv = (u64)t2 + (u64)m * p1 + carry;
    t2 = (u32)uv;
    carry = uv >> 32;
    uv = (u64)t3 + (u64)m * p2 + carry;
    t3 = (u32)uv;
    carry = uv >> 32;
    uv = (u64)t4 + carry;
    t4 = (u32)uv;
    carry = uv >> 32;
    uv = (u64)t5 + carry;
    t5 = (u32)uv;
    carry = uv >> 32;
    uv = (u64)t6 + carry;
    t6 = (u32)uv;

    m = t2 * nprime;
    uv = (u64)t2 + (u64)m * p0;
    carry = uv >> 32;
    uv = (u64)t3 + (u64)m * p1 + carry;
    t3 = (u32)uv;
    carry = uv >> 32;
    uv = (u64)t4 + (u64)m * p2 + carry;
    t4 = (u32)uv;
    carry = uv >> 32;
    uv = (u64)t5 + carry;
    t5 = (u32)uv;
    carry = uv >> 32;
    uv = (u64)t6 + carry;
    t6 = (u32)uv;

    U96 r{{t3, t4, t5}};
    if (t6 || cmp96(r, f->p) >= 0) r = sub96_raw(r, f->p);
    return r;
}

HD static U96 to_mont96(U96 a, const Field96 *f) {
    return mont_mul96(a, f->r2, f);
}

HD static U96 from_mont96(U96 a, const Field96 *f) {
    return mont_mul96(a, u96_from_u64(1), f);
}

HD static U96 mulmod96(U96 a, U96 b, const Field96 *f) {
    return from_mont96(mont_mul96(to_mont96(a, f), to_mont96(b, f), f), f);
}

HD static U96 mulmod96_to_mont(U96 a, U96 b, const Field96 *f) {
    return mont_mul96(to_mont96(a, f), to_mont96(b, f), f);
}

HD static U96 powmod96_mont(U96 a_m, u128 e, const Field96 *f) {
    U96 r = f->one;
    U96 b = a_m;
    while (e > 0) {
        if (e & 1) r = mont_mul96(r, b, f);
        e >>= 1;
        if (e) b = mont_mul96(b, b, f);
    }
    return r;
}

HD static int sqrtmod_p5_96_mont(U96 *root_m, U96 n_m, const SearchParams96 *params) {
    if (is_zero96(n_m)) {
        *root_m = u96_from_u64(0);
        return 1;
    }
    U96 x_m = powmod96_mont(n_m, params->sqrt_exp, &params->f);
    if (eq96(mont_mul96(x_m, x_m, &params->f), n_m)) {
        *root_m = x_m;
        return 1;
    }
    if (params->sqrt_case == 3) return 0;
    x_m = mont_mul96(x_m, params->sqrtm1_m, &params->f);
    if (eq96(mont_mul96(x_m, x_m, &params->f), n_m)) {
        *root_m = x_m;
        return 1;
    }
    return 0;
}

HD static U96 invert96_mont(U96 a_m, const SearchParams96 *params) {
    return powmod96_mont(a_m, params->inv_exp, &params->f);
}

HD static int invert_batch2_96_mont(U96 out[2], const U96 vals_m[2],
                                    const SearchParams96 *params) {
    if (is_zero96(vals_m[0]) || is_zero96(vals_m[1])) return 0;
    U96 prefix1 = vals_m[0];
    U96 prefix2 = mont_mul96(vals_m[0], vals_m[1], &params->f);
    U96 acc = invert96_mont(prefix2, params);
    out[1] = mont_mul96(acc, prefix1, &params->f);
    out[0] = mont_mul96(acc, vals_m[1], &params->f);
    return 1;
}

HD static inline void xDBL96(U96 *Xo, U96 *Zo, U96 X, U96 Z, U96 a24,
                             const Field96 *f) {
    U96 u = addmod96(X, Z, f->p);
    U96 v = submod96(X, Z, f->p);
    u = mont_mul96(u, u, f);
    v = mont_mul96(v, v, f);
    *Xo = mont_mul96(u, v, f);
    U96 w = submod96(u, v, f->p);
    *Zo = mont_mul96(w, addmod96(v, mont_mul96(a24, w, f), f->p), f);
}

HD static int verify96_mont(U96 A_m, U96 x0_m, const SearchParams96 *params) {
    U96 pm2_m = submod96(u96_from_u64(0), params->two_m, params->f.p);
    if (eq96(A_m, params->two_m) || eq96(A_m, pm2_m)) return 0;

    U96 a24m = mont_mul96(addmod96(A_m, params->two_m, params->f.p),
                          params->inv4_m, &params->f);
    U96 X = x0_m;
    U96 Z = params->f.one;
    for (int i = 1; i <= params->k; i++) {
        xDBL96(&X, &Z, X, Z, a24m, &params->f);
        if (i < params->k && is_zero96(Z)) return 0;
        if (i == params->k && !is_zero96(Z)) return 0;
    }
    return 1;
}

HD static U96 x16_A_numerator_from_y96_mont(U96 y_m, const SearchParams96 *params) {
    U96 num_m = params->f.one;
    num_m = submod96(mont_mul96(num_m, y_m, &params->f), params->eight_m, params->f.p);
    num_m = addmod96(mont_mul96(num_m, y_m, &params->f), params->c24_m, params->f.p);
    num_m = submod96(mont_mul96(num_m, y_m, &params->f), params->c32_m, params->f.p);
    num_m = addmod96(mont_mul96(num_m, y_m, &params->f), params->eight_m, params->f.p);
    num_m = addmod96(mont_mul96(num_m, y_m, &params->f), params->c32_m, params->f.p);
    num_m = submod96(mont_mul96(num_m, y_m, &params->f), params->c48_m, params->f.p);
    num_m = addmod96(mont_mul96(num_m, y_m, &params->f), params->c32_m, params->f.p);
    num_m = submod96(mont_mul96(num_m, y_m, &params->f), params->eight_m, params->f.p);
    return num_m;
}

HD static int x16_y_predicts_nonsplit96(U96 y, U96 y2, const SearchParams96 *params) {
    U96 f1 = submod96(y2, u96_from_u64(2), params->f.p);
    U96 two_y = addmod96(y, y, params->f.p);
    U96 four_y = addmod96(two_y, two_y, params->f.p);
    U96 f2 = addmod96(submod96(y2, four_y, params->f.p), u96_from_u64(2), params->f.p);
    U96 f_m = mulmod96_to_mont(f1, f2, &params->f);
    if (is_zero96(f_m)) return 0;
    U96 leg_m = powmod96_mont(f_m, params->leg_exp, &params->f);
    return !eq96(leg_m, params->f.one);
}

HD static int x16_y_predicts_nonsplit96_mont(U96 y_m, U96 y2_m,
                                             const SearchParams96 *params) {
    U96 f1_m = submod96(y2_m, params->two_m, params->f.p);
    U96 two_y_m = addmod96(y_m, y_m, params->f.p);
    U96 four_y_m = addmod96(two_y_m, two_y_m, params->f.p);
    U96 f2_m = addmod96(submod96(y2_m, four_y_m, params->f.p),
                        params->two_m, params->f.p);
    U96 f_m = mont_mul96(f1_m, f2_m, &params->f);
    if (is_zero96(f_m)) return 0;
    U96 leg_m = powmod96_mont(f_m, params->leg_exp, &params->f);
    return !eq96(leg_m, params->f.one);
}

HD static int x16_root_to_montgomery_A96_mont(U96 *Ao, U96 *Amo, U96 *xPmo,
                                              U96 x_m, U96 y_m,
                                              const SearchParams96 *params) {
    U96 num_m = x16_A_numerator_from_y96_mont(y_m, params);
    U96 ym1_m = submod96(y_m, params->f.one, params->f.p);
    U96 ym1_2_m = mont_mul96(ym1_m, ym1_m, &params->f);
    U96 denA_m = mont_mul96(params->four_m,
                            mont_mul96(ym1_2_m, ym1_2_m, &params->f),
                            &params->f);
    U96 denx_m = submod96(x_m, y_m, params->f.p);
    U96 vals_m[2] = {denA_m, denx_m};
    U96 invs_m[2];
    if (!invert_batch2_96_mont(invs_m, vals_m, params)) return 0;

    U96 A_m = mont_mul96(num_m, invs_m[0], &params->f);
    U96 xP_m = mont_mul96(x_m, invs_m[1], &params->f);
    U96 A = from_mont96(A_m, &params->f);
    U96 two = u96_from_u64(2);
    U96 pm2 = submod96(params->f.p, two, params->f.p);
    if (cmp96(A, two) <= 0 || cmp96(A, pm2) >= 0) return 0;
    *Ao = A;
    *Amo = A_m;
    *xPmo = xP_m;
    return 1;
}

HD static int x16_root_to_montgomery_xP96_mont(U96 *xPmo, U96 x_m, U96 y_m,
                                               const SearchParams96 *params) {
    U96 denx_m = submod96(x_m, y_m, params->f.p);
    if (is_zero96(denx_m)) return 0;
    U96 inv_denx_m = invert96_mont(denx_m, params);
    *xPmo = mont_mul96(x_m, inv_denx_m, &params->f);
    return 1;
}

HD static int halve_once_first96_mont(U96 *xo_m, U96 A_m, U96 x_m,
                                      const SearchParams96 *params) {
    U96 x2_m = mont_mul96(x_m, x_m, &params->f);
    U96 d_m = addmod96(addmod96(x2_m, mont_mul96(A_m, x_m, &params->f),
                                params->f.p),
                       params->f.one, params->f.p);
    U96 sd_m;
    if (!sqrtmod_p5_96_mont(&sd_m, d_m, params)) return 0;

    U96 roots_d[2] = {sd_m, submod96(u96_from_u64(0), sd_m, params->f.p)};
    for (int i = 0; i < 2; i++) {
        U96 u_m = addmod96(addmod96(x_m, x_m, params->f.p),
                           addmod96(roots_d[i], roots_d[i], params->f.p),
                           params->f.p);
        U96 w_m = submod96(mont_mul96(u_m, u_m, &params->f), params->four_m,
                           params->f.p);
        U96 sw_m;
        if (!sqrtmod_p5_96_mont(&sw_m, w_m, params)) continue;
        U96 candidates[2] = {
            mont_mul96(addmod96(u_m, sw_m, params->f.p), params->inv2_m, &params->f),
            mont_mul96(submod96(u_m, sw_m, params->f.p), params->inv2_m, &params->f),
        };
        for (int j = 0; j < 2; j++) {
            if (!is_zero96(candidates[j])) {
                *xo_m = candidates[j];
                return 1;
            }
        }
    }
    return 0;
}

HD static int halve_chain_from_depth96_mont(U96 *xout_m, U96 A_m, U96 x_m,
                                            int depth, const SearchParams96 *params) {
    for (; depth < params->k; depth++) {
        if (!halve_once_first96_mont(&x_m, A_m, x_m, params)) return 0;
    }
    if (!verify96_mont(A_m, x_m, params)) return 0;
    *xout_m = x_m;
    return 1;
}

DEV static U96 rand_below96(Rng *rng, const SearchParams96 *params) {
    for (;;) {
        u64 a = rng64(rng);
        u64 b = rng64(rng);
        U96 v{{(u32)a, (u32)(a >> 32), (u32)b}};
        v = and96(v, params->rand_mask);
        if (cmp96(v, params->f.p) < 0) return v;
    }
}

/* ================================================================
 * Optimized source (added 2026-08-28)
 *
 * Two exact layers on top of the record-stack kernel above.
 *
 * (1) Reciprocal-mark deduplication.  The X1(16) auxiliary quadratic emits
 *     two roots per curve; they give the same Montgomery A and reciprocal
 *     marked x-coordinates.  On a nonsplit curve those two marks differ by
 *     the rational 2-torsion translate x -> 1/x, which preserves halving
 *     depth, so they always return the same verdict.  The kernel above
 *     tests both (its `for (ri...)` loop); the kernel below tests one.
 *     Exact 2x, no distributional change.
 *
 * (2) Genus-one first-lift cover source.  Writing G = (y^2-2)(y^2-2y+2),
 *     the X1(16) fiber discriminant D = y(y-2)G and the first halving
 *     obstruction H = (y-1)G satisfy the polynomial identity
 *
 *         D * H = y(y-1)(y-2) * G^2.
 *
 *     With X = y-1 that first factor is X^3 - X.  So on the fixed elliptic
 *     curve E0: W^2 = X^3 - X, every point gives H = (W G)^2 / D: once D is
 *     a square, H is a square automatically.  Sampling y from E0 instead of
 *     uniformly therefore emits curves whose first halving gate is already
 *     passed - exactly 2x the hazard per emitted candidate - and the root
 *     the halving step needs is transported algebraically rather than
 *     extracted:
 *
 *         sqrt(d1) = y * sD * W / (dx * (y-1)^2),   dx = r0 - 2*qa*y
 *
 *     with r0 = sD - qb the retained quadratic numerator (so that the
 *     marked point is xP = r0/dx).  E0 is walked with a fixed step, and
 *     POM_COVER_LANES independent walks share one modular inversion by
 *     Montgomery's trick, so a cover point costs a few field multiplies
 *     plus 1/LANES of an inversion.
 *
 * The fused conversion below also removes an inversion the original kernel
 * paid: it never forms the affine quadratic root x (which needed 1/(2*qa)),
 * working directly with r0/dx, so one batch-2 inversion covers both the
 * curve coefficient and the marked point.
 * ================================================================ */

#ifndef POM_COVER_LANES
#define POM_COVER_LANES 4
#endif
#define POM_STR2(x) #x
#define POM_STR(x) POM_STR2(x)

struct CoverPt96 {
    U96 x;
    U96 w;
};

/* Montgomery's trick: invert n field elements with one exponentiation. */
HD static int batch_invert96(U96 *out, const U96 *in, int n,
                             const SearchParams96 *params) {
    U96 prefix[POM_COVER_LANES];
    U96 acc = params->f.one;
    for (int i = 0; i < n; i++) {
        if (is_zero96(in[i])) return 0;
        prefix[i] = acc;
        acc = mont_mul96(acc, in[i], &params->f);
    }
    U96 running = invert96_mont(acc, params);
    for (int i = n - 1; i >= 0; i--) {
        out[i] = mont_mul96(running, prefix[i], &params->f);
        running = mont_mul96(running, in[i], &params->f);
    }
    return 1;
}

/* w^2 == x^3 - x ? (Montgomery form.)  Used by the self-test. */
HD static int cover_on_curve96(U96 x_m, U96 w_m, const SearchParams96 *params) {
    U96 x3_m = mont_mul96(mont_mul96(x_m, x_m, &params->f), x_m, &params->f);
    U96 rhs_m = submod96(x3_m, x_m, params->f.p);
    return eq96(mont_mul96(w_m, w_m, &params->f), rhs_m);
}

/* Advance n independent walks on E0 by one fixed step (sx, sw), sharing a
   single inversion.  Returns 0 if any lane hits the doubling/infinity case
   (probability ~n/p; the caller reseeds). */
HD static int cover_step96(CoverPt96 *pts, int n, U96 sx_m, U96 sw_m,
                           const SearchParams96 *params) {
    U96 den[POM_COVER_LANES], inv[POM_COVER_LANES];
    for (int i = 0; i < n; i++)
        den[i] = submod96(pts[i].x, sx_m, params->f.p);
    if (!batch_invert96(inv, den, n, params)) return 0;
    for (int i = 0; i < n; i++) {
        U96 lam = mont_mul96(submod96(pts[i].w, sw_m, params->f.p), inv[i],
                             &params->f);
        U96 nx = submod96(submod96(mont_mul96(lam, lam, &params->f),
                                   pts[i].x, params->f.p),
                          sx_m, params->f.p);
        U96 nw = submod96(mont_mul96(lam,
                                     submod96(pts[i].x, nx, params->f.p),
                                     &params->f),
                          pts[i].w, params->f.p);
        pts[i].x = nx;
        pts[i].w = nw;
    }
    return 1;
}

/* Halve a marked point whose halving discriminant root is already known.
   Mirrors halve_once_first96_mont with the sqrt(d) step removed. */
HD static int halve_once_known_d96_mont(U96 *xo_m, U96 x_m, U96 sd_m,
                                        const SearchParams96 *params) {
    U96 roots_d[2] = {sd_m, submod96(u96_from_u64(0), sd_m, params->f.p)};
    for (int i = 0; i < 2; i++) {
        U96 u_m = addmod96(addmod96(x_m, x_m, params->f.p),
                           addmod96(roots_d[i], roots_d[i], params->f.p),
                           params->f.p);
        U96 w_m = submod96(mont_mul96(u_m, u_m, &params->f), params->four_m,
                           params->f.p);
        U96 sw_m;
        if (!sqrtmod_p5_96_mont(&sw_m, w_m, params)) continue;
        U96 candidates[2] = {
            mont_mul96(addmod96(u_m, sw_m, params->f.p), params->inv2_m,
                       &params->f),
            mont_mul96(submod96(u_m, sw_m, params->f.p), params->inv2_m,
                       &params->f),
        };
        for (int j = 0; j < 2; j++)
            if (!is_zero96(candidates[j])) { *xo_m = candidates[j]; return 1; }
    }
    return 0;
}

/* One cover point -> curve A plus a marked point of exact order 32.
   Returns 0 when this cover point yields no usable curve (nonsplit test
   fails, D is a nonsquare, or a degenerate denominator appears).
   Costs: 1 Legendre power (nonsplit), 1 sqrt (D), 1 batch-2 inversion -
   and no square root for the first halving level, which the cover has
   already guaranteed. */
HD static int x16_cover_to_depth5_96_mont(U96 *Ao, U96 *Amo, U96 *x32mo,
                                          U96 X_m, U96 W_m,
                                          const SearchParams96 *params) {
    const U96 one = params->f.one;
    const U96 p = params->f.p;
    U96 y_m = addmod96(X_m, one, p);                 /* y = X + 1  */
    U96 y2_m = mont_mul96(y_m, y_m, &params->f);
    if (!x16_y_predicts_nonsplit96_mont(y_m, y2_m, params)) return 0;

    U96 y3_m = mont_mul96(y2_m, y_m, &params->f);
    U96 two_y_m = addmod96(y_m, y_m, p);
    U96 qa_m = submod96(y2_m, two_y_m, p);           /* qa = y^2 - 2y */
    if (is_zero96(qa_m)) return 0;
    U96 qb_m = submod96(addmod96(y2_m, y2_m, p), y3_m, p);
    U96 qc_m = submod96(one, y_m, p);
    U96 D_m = submod96(mont_mul96(qb_m, qb_m, &params->f),
                       mont_mul96(addmod96(qa_m, qa_m, p),
                                  addmod96(qc_m, qc_m, p), &params->f), p);
    U96 sd_m;
    if (!sqrtmod_p5_96_mont(&sd_m, D_m, params)) return 0;

    /* Marked point as r0/dx: avoids inverting 2*qa separately. */
    U96 r0_m = submod96(sd_m, qb_m, p);
    U96 dx_m = submod96(r0_m, mont_mul96(addmod96(qa_m, qa_m, p), y_m,
                                         &params->f), p);
    if (is_zero96(dx_m)) return 0;

    /* y - 1 == X, so the A denominator is 4*X^4. */
    U96 X2_m = mont_mul96(X_m, X_m, &params->f);
    U96 denA_m = mont_mul96(params->four_m,
                            mont_mul96(X2_m, X2_m, &params->f), &params->f);
    if (is_zero96(denA_m)) return 0;

    U96 vals_m[2] = {dx_m, denA_m};
    U96 invs_m[2];
    if (!batch_invert96(invs_m, vals_m, 2, params)) return 0;
    U96 inv_dx_m = invs_m[0], inv_denA_m = invs_m[1];

    U96 A_m = mont_mul96(x16_A_numerator_from_y96_mont(y_m, params),
                         inv_denA_m, &params->f);
    U96 A = from_mont96(A_m, &params->f);
    U96 two = u96_from_u64(2);
    U96 pm2 = submod96(params->f.p, two, params->f.p);
    if (cmp96(A, two) <= 0 || cmp96(A, pm2) >= 0) return 0;

    U96 xP16_m = mont_mul96(r0_m, inv_dx_m, &params->f);

    /* 1/(y-1)^2 = 4*X^2 * 1/(4*X^4) */
    U96 inv_X2_m = mont_mul96(mont_mul96(params->four_m, X2_m, &params->f),
                              inv_denA_m, &params->f);
    /* sqrt(d1) = y * sD * W / (dx * (y-1)^2) */
    U96 sd1_m = mont_mul96(
        mont_mul96(mont_mul96(y_m, sd_m, &params->f), W_m, &params->f),
        mont_mul96(inv_dx_m, inv_X2_m, &params->f), &params->f);

    U96 x32_m;
    if (!halve_once_known_d96_mont(&x32_m, xP16_m, sd1_m, params)) return 0;

    *Ao = A;
    *Amo = A_m;
    *x32mo = x32_m;
    return 1;
}

/* Verify the transported root against a direct square root: sd1^2 must be
   the halving discriminant of the marked point.  Self-test only. */
HD static int cover_transport_selfcheck96(U96 X_m, U96 W_m,
                                          const SearchParams96 *params) {
    const U96 one = params->f.one, p = params->f.p;
    U96 y_m = addmod96(X_m, one, p);
    U96 y2_m = mont_mul96(y_m, y_m, &params->f);
    U96 y3_m = mont_mul96(y2_m, y_m, &params->f);
    U96 qa_m = submod96(y2_m, addmod96(y_m, y_m, p), p);
    if (is_zero96(qa_m)) return -1;
    U96 qb_m = submod96(addmod96(y2_m, y2_m, p), y3_m, p);
    U96 qc_m = submod96(one, y_m, p);
    U96 D_m = submod96(mont_mul96(qb_m, qb_m, &params->f),
                       mont_mul96(addmod96(qa_m, qa_m, p),
                                  addmod96(qc_m, qc_m, p), &params->f), p);
    U96 sd_m;
    if (!sqrtmod_p5_96_mont(&sd_m, D_m, params)) return -1;
    U96 r0_m = submod96(sd_m, qb_m, p);
    U96 dx_m = submod96(r0_m, mont_mul96(addmod96(qa_m, qa_m, p), y_m,
                                         &params->f), p);
    if (is_zero96(dx_m)) return -1;
    U96 X2_m = mont_mul96(X_m, X_m, &params->f);
    U96 denA_m = mont_mul96(params->four_m,
                            mont_mul96(X2_m, X2_m, &params->f), &params->f);
    if (is_zero96(denA_m)) return -1;
    U96 vals_m[2] = {dx_m, denA_m}, invs_m[2];
    if (!batch_invert96(invs_m, vals_m, 2, params)) return -1;
    U96 A_m = mont_mul96(x16_A_numerator_from_y96_mont(y_m, params),
                         invs_m[1], &params->f);
    U96 xP_m = mont_mul96(r0_m, invs_m[0], &params->f);
    U96 inv_X2_m = mont_mul96(mont_mul96(params->four_m, X2_m, &params->f),
                              invs_m[1], &params->f);
    U96 sd1_m = mont_mul96(
        mont_mul96(mont_mul96(y_m, sd_m, &params->f), W_m, &params->f),
        mont_mul96(invs_m[0], inv_X2_m, &params->f), &params->f);
    /* d = xP^2 + A*xP + 1 */
    U96 d_m = addmod96(addmod96(mont_mul96(xP_m, xP_m, &params->f),
                                mont_mul96(A_m, xP_m, &params->f), p),
                       one, p);
    return eq96(mont_mul96(sd1_m, sd1_m, &params->f), d_m) ? 1 : 0;
}

/* Draw a random point on E0: W^2 = X^3 - X (about half of all X work). */
DEV static int find_cover_point96(Rng *rng, const SearchParams96 *params,
                                  CoverPt96 *out) {
    for (int t = 0; t < 128; t++) {
        U96 x = rand_below96(rng, params);
        if (is_zero96(x)) continue;
        U96 xm = to_mont96(x, &params->f);
        U96 rhs = submod96(mont_mul96(mont_mul96(xm, xm, &params->f), xm,
                                      &params->f), xm, params->f.p);
        if (is_zero96(rhs)) continue;
        U96 w;
        if (sqrtmod_p5_96_mont(&w, rhs, params)) {
            out->x = xm;
            out->w = w;
            return 1;
        }
    }
    return 0;
}

#ifndef POM_HOST_TEST
/* Cover-source production kernel: deduplicated marks, first halving gate
   sourced rather than tested.  Each thread walks POM_COVER_LANES independent
   points on E0 with one shared inversion per step. */
__global__ static void x16cover_kernel96(SearchParams96 params,
                                         u64 seed_offset,
                                         u64 chunk_nonce,
                                         u64 max_curves,
                                         u64 global_base,
                                         u64 claim_batch,
                                         GpuStats *__restrict__ stats,
                                         GpuResult *__restrict__ result) {
    u64 tid = (u64)blockIdx.x * (u64)blockDim.x + (u64)threadIdx.x;
    u64 seed = seed_offset ^ (chunk_nonce * 0xd1342543de82ef95ULL) ^
               (tid * 0x9e3779b97f4a7c15ULL) ^ 0x53c5ca59b9f7a1e3ULL;
    Rng rng;
    rng.s0 = splitmix64_next(&seed);
    rng.s1 = splitmix64_next(&seed);
    if ((rng.s0 | rng.s1) == 0) rng.s1 = 1442695040888963407ULL;
    for (int i = 0; i < 200; i++) (void)rng64(&rng);

    CoverPt96 pts[POM_COVER_LANES];
    CoverPt96 step;
    if (!find_cover_point96(&rng, &params, &step)) return;
    for (int i = 0; i < POM_COVER_LANES; i++)
        if (!find_cover_point96(&rng, &params, &pts[i])) return;

    u64 local_processed = 0;
    u64 next_claim = 0;
    u64 claim_limit = 0;

    while (!found_now(result)) {
        if (!cover_step96(pts, POM_COVER_LANES, step.x, step.w, &params)) {
            /* Degenerate step for some lane (probability ~LANES/p): reseed. */
            for (int i = 0; i < POM_COVER_LANES; i++)
                if (!find_cover_point96(&rng, &params, &pts[i])) return;
            continue;
        }
        int stop = 0;
        for (int i = 0; i < POM_COVER_LANES; i++) {
            U96 A, A_m, x32_m;
            if (!x16_cover_to_depth5_96_mont(&A, &A_m, &x32_m, pts[i].x,
                                             pts[i].w, &params))
                continue;

            if (next_claim >= claim_limit) {
                next_claim = atomicAdd(&stats->claimed, claim_batch);
                claim_limit = next_claim + claim_batch;
            }
            u64 claim = next_claim++;
            if (claim >= max_curves || found_now(result)) { stop = 1; break; }
            local_processed++;

            U96 xR_m;
            if (halve_chain_from_depth96_mont(&xR_m, A_m, x32_m, 5, &params)) {
                if (atomicCAS(&result->found, 0, 1) == 0) {
                    result->trial_index = global_base + claim;
                    result->A = pack96(A);
                    result->x0 = pack96(from_mont96(xR_m, &params.f));
                }
                stop = 1;
                break;
            }
        }
        if (stop) break;
    }
    atomicAdd(&stats->processed, local_processed);
}

/* Device-side proof that the transported root and the emitted torsion order
   are right on this GPU, this prime, this compiler.  Run before production. */
__global__ static void cover_selftest_kernel96(SearchParams96 params,
                                               u64 seed_offset,
                                               int per_thread,
                                               u64 *__restrict__ checked,
                                               u64 *__restrict__ transport_bad,
                                               u64 *__restrict__ emitted,
                                               u64 *__restrict__ order_bad,
                                               u64 *__restrict__ curve_bad) {
    u64 tid = (u64)blockIdx.x * (u64)blockDim.x + (u64)threadIdx.x;
    u64 seed = seed_offset ^ (tid * 0x9e3779b97f4a7c15ULL) ^ 0x1234567deadbeefULL;
    Rng rng;
    rng.s0 = splitmix64_next(&seed);
    rng.s1 = splitmix64_next(&seed);
    if ((rng.s0 | rng.s1) == 0) rng.s1 = 1442695040888963407ULL;
    for (int i = 0; i < 200; i++) (void)rng64(&rng);

    CoverPt96 pts[POM_COVER_LANES], step;
    if (!find_cover_point96(&rng, &params, &step)) return;
    for (int i = 0; i < POM_COVER_LANES; i++)
        if (!find_cover_point96(&rng, &params, &pts[i])) return;

    u64 l_checked = 0, l_tbad = 0, l_emit = 0, l_obad = 0, l_cbad = 0;
    for (int it = 0; it < per_thread; it++) {
        if (!cover_step96(pts, POM_COVER_LANES, step.x, step.w, &params)) continue;
        for (int i = 0; i < POM_COVER_LANES; i++) {
            if (!cover_on_curve96(pts[i].x, pts[i].w, &params)) l_cbad++;
            int r = cover_transport_selfcheck96(pts[i].x, pts[i].w, &params);
            if (r < 0) continue;
            l_checked++;
            if (r == 0) l_tbad++;

            U96 A, A_m, x32_m;
            if (!x16_cover_to_depth5_96_mont(&A, &A_m, &x32_m, pts[i].x,
                                             pts[i].w, &params))
                continue;
            l_emit++;
            U96 a24 = mont_mul96(addmod96(A_m, params.two_m, params.f.p),
                                 params.inv4_m, &params.f);
            U96 X = x32_m, Z = params.f.one;
            int ok = 1;
            for (int j = 0; j < 4; j++) {
                xDBL96(&X, &Z, X, Z, a24, &params.f);
                if (is_zero96(Z)) { ok = 0; break; }
            }
            if (ok) {
                xDBL96(&X, &Z, X, Z, a24, &params.f);
                ok = is_zero96(Z);
            }
            if (!ok) l_obad++;
        }
    }
    atomicAdd(checked, l_checked);
    atomicAdd(transport_bad, l_tbad);
    atomicAdd(emitted, l_emit);
    atomicAdd(order_bad, l_obad);
    atomicAdd(curve_bad, l_cbad);
}

__global__ static void x16halvenonsplit_kernel96(SearchParams96 params,
                                                 u64 seed_offset,
                                                 u64 chunk_nonce,
                                                 u64 max_curves,
                                                 u64 global_base,
                                                 u64 claim_batch,
                                                 GpuStats *__restrict__ stats,
                                                 GpuResult *__restrict__ result) {
    u64 tid = (u64)blockIdx.x * (u64)blockDim.x + (u64)threadIdx.x;
    u64 seed = seed_offset ^ (chunk_nonce * 0xd1342543de82ef95ULL) ^
               (tid * 0x9e3779b97f4a7c15ULL) ^ 0x7364529176530163ULL;
    Rng rng;
    rng.s0 = splitmix64_next(&seed);
    rng.s1 = splitmix64_next(&seed);
    if ((rng.s0 | rng.s1) == 0) rng.s1 = 1442695040888963407ULL;
    for (int i = 0; i < 200; i++) (void)rng64(&rng);

    u64 local_processed = 0;
#if POM_CUDA_DETAILED_STATS
    u64 local_y_draws = 0;
    u64 local_y_nonsplit = 0;
    u64 local_y_with_sqrt_D = 0;
    u64 local_roots_valid = 0;
    u64 local_halve_calls = 0;
    u64 local_chain_survivors = 0;
#endif
    u64 next_claim = 0;
    u64 claim_limit = 0;

    while (!found_now(result)) {
        U96 y = rand_below96(&rng, &params);
#if POM_CUDA_DETAILED_STATS
        local_y_draws++;
#endif
        if (is_zero96(y)) continue;

        U96 y_m = to_mont96(y, &params.f);
        U96 y2_m = mont_mul96(y_m, y_m, &params.f);
        if (!x16_y_predicts_nonsplit96_mont(y_m, y2_m, &params)) continue;
#if POM_CUDA_DETAILED_STATS
        local_y_nonsplit++;
#endif

        U96 y3_m = mont_mul96(y2_m, y_m, &params.f);
        U96 two_y_m = addmod96(y_m, y_m, params.f.p);
        U96 qa_m = submod96(y2_m, two_y_m, params.f.p);
        if (is_zero96(qa_m)) continue;
        U96 qb_m = submod96(addmod96(y2_m, y2_m, params.f.p), y3_m, params.f.p);
        U96 qc_m = submod96(params.f.one, y_m, params.f.p);
        U96 D_m = submod96(
            mont_mul96(qb_m, qb_m, &params.f),
            mont_mul96(addmod96(qa_m, qa_m, params.f.p),
                       addmod96(qc_m, qc_m, params.f.p), &params.f),
            params.f.p);
        U96 sd_m;
        if (!sqrtmod_p5_96_mont(&sd_m, D_m, &params)) continue;
#if POM_CUDA_DETAILED_STATS
        local_y_with_sqrt_D++;
#endif

        U96 inv_2qa_m = invert96_mont(addmod96(qa_m, qa_m, params.f.p), &params);
        U96 roots_m[2] = {
            mont_mul96(submod96(sd_m, qb_m, params.f.p), inv_2qa_m, &params.f),
            mont_mul96(submod96(submod96(u96_from_u64(0), sd_m, params.f.p), qb_m,
                                params.f.p),
                       inv_2qa_m, &params.f),
        };

        U96 A;
        U96 A_m;
        U96 xP16_m[2];
        int root_valid[2] = {0, 0};
        root_valid[0] = x16_root_to_montgomery_A96_mont(&A, &A_m, &xP16_m[0],
                                                        roots_m[0], y_m, &params);
        if (root_valid[0]) {
            root_valid[1] = x16_root_to_montgomery_xP96_mont(&xP16_m[1],
                                                             roots_m[1], y_m,
                                                             &params);
        } else {
            root_valid[1] = x16_root_to_montgomery_A96_mont(&A, &A_m, &xP16_m[1],
                                                            roots_m[1], y_m,
                                                            &params);
        }

        int stop = 0;
        for (int ri = 0; ri < 2; ri++) {
            if (!root_valid[ri]) continue;
#if POM_CUDA_DETAILED_STATS
            local_roots_valid++;
#endif

            if (next_claim >= claim_limit) {
                next_claim = atomicAdd(&stats->claimed, claim_batch);
                claim_limit = next_claim + claim_batch;
            }
            u64 claim = next_claim++;
            if (claim >= max_curves || found_now(result)) {
                stop = 1;
                break;
            }

            local_processed++;
#if POM_CUDA_DETAILED_STATS
            local_halve_calls++;
#endif
            U96 xR_m;
            if (halve_chain_from_depth96_mont(&xR_m, A_m, xP16_m[ri], 4, &params)) {
#if POM_CUDA_DETAILED_STATS
                local_chain_survivors++;
#endif
                if (atomicCAS(&result->found, 0, 1) == 0) {
                    result->trial_index = global_base + claim;
                    result->A = pack96(A);
                    result->x0 = pack96(from_mont96(xR_m, &params.f));
                }
                stop = 1;
                break;
            }
        }
        if (stop) break;
    }

    atomicAdd(&stats->processed, local_processed);
#if POM_CUDA_DETAILED_STATS
    atomicAdd(&stats->y_draws, local_y_draws);
    atomicAdd(&stats->y_nonsplit, local_y_nonsplit);
    atomicAdd(&stats->y_with_sqrt_D, local_y_with_sqrt_D);
    atomicAdd(&stats->roots_valid, local_roots_valid);
    atomicAdd(&stats->halve_calls, local_halve_calls);
    atomicAdd(&stats->chain_survivors, local_chain_survivors);
#endif
}

static void die_cuda(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return;
    std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(err));
    std::exit(1);
}
#endif  /* POM_HOST_TEST */

static u128 parse128(const char *s) {
    u128 v = 0;
    while (*s >= '0' && *s <= '9') {
        v = v * 10 + (unsigned)(*s - '0');
        s++;
    }
    return v;
}

static std::string sprint128(u128 v) {
    if (v == 0) return "0";
    char tmp[50];
    int i = 49;
    tmp[i] = '\0';
    while (v > 0) {
        tmp[--i] = char('0' + (int)(v % 10));
        v /= 10;
    }
    return std::string(tmp + i);
}

static int bitlen128(u128 x) {
    u64 hi = (u64)(x >> 64);
    if (hi) return 64 + (64 - __builtin_clzll(hi));
    u64 lo = (u64)x;
    return lo ? 64 - __builtin_clzll(lo) : 0;
}

static void m128_init(Mont128 *m, u128 p) {
    m->p = p;
    u128 x = 1;
    for (int i = 0; i < 7; i++) x *= 2 - p * x;
    m->ni = (u128)0 - x;
    u128 r = 1;
    for (int i = 0; i < 128; i++) {
        r <<= 1;
        if (r >= p) r -= p;
    }
    m->one = r;
    for (int i = 0; i < 128; i++) {
        r <<= 1;
        if (r >= p) r -= p;
    }
    m->R2 = r;
}

static U96 u96_mask_bits(int bits) {
    U96 mask{{0, 0, 0}};
    if (bits >= 96) {
        mask.v[0] = mask.v[1] = mask.v[2] = 0xffffffffu;
        return mask;
    }
    for (int i = 0; i < 3; i++) {
        int limb_bits = bits - 32 * i;
        if (limb_bits >= 32) mask.v[i] = 0xffffffffu;
        else if (limb_bits > 0) mask.v[i] = (1u << limb_bits) - 1u;
        else mask.v[i] = 0;
    }
    return mask;
}

static Field96 field96_init(u128 p128) {
    Field96 f{};
    f.p = u96_from_u128(p128);
    f.nprime = inv32_odd(f.p.v[0]);

    U96 r = u96_from_u64(1);
    for (int i = 0; i < 96; i++) r = addmod96(r, r, f.p);
    f.one = r;
    for (int i = 0; i < 96; i++) r = addmod96(r, r, f.p);
    f.r2 = r;
    return f;
}

static SearchParams96 search_params96_init(u128 p, int k, int pbits) {
    SearchParams96 params{};
    params.f = field96_init(p);
    params.rand_mask = u96_mask_bits(pbits);
    params.leg_exp = (p - 1) >> 1;
    params.sqrt_case = ((u64)(p & 3) == 3) ? 3 : 5;
    params.sqrt_exp = params.sqrt_case == 3 ? ((p + 1) >> 2) : ((p + 3) >> 3);
    params.inv_exp = p - 2;
    params.k = k;
    params.two_m = to_mont96(u96_from_u64(2), &params.f);
    params.four_m = to_mont96(u96_from_u64(4), &params.f);
    params.eight_m = to_mont96(u96_from_u64(8), &params.f);
    params.c24_m = to_mont96(u96_from_u64(24), &params.f);
    params.c32_m = to_mont96(u96_from_u64(32), &params.f);
    params.c48_m = to_mont96(u96_from_u64(48), &params.f);
    params.sqrtm1_m = params.sqrt_case == 5
        ? powmod96_mont(params.two_m, (p - 1) >> 2, &params.f)
        : u96_from_u64(0);
    params.inv2_m = to_mont96(u96_from_u128((p + 1) >> 1), &params.f);
    params.inv4_m = invert96_mont(params.four_m, &params);
    return params;
}

static int compute_k(u128 p) {
    u64 q = (u64)std::sqrt((long double)p);
    while ((u128)(q + 1) * (q + 1) <= p) q++;
    while ((u128)q * q > p) q--;
    u64 sq = (u64)std::sqrt((long double)q);
    while ((sq + 1) * (sq + 1) <= q) sq++;
    while (sq * sq > q) sq--;
    u64 bound = q + 1 + 2 * sq;
    int k = 0;
    u64 v = 1;
    while (v <= bound) {
        k++;
        v <<= 1;
    }
    return k;
}

#ifndef POM_NO_DRIVER
static u64 parse_u64_arg(const char *s, u64 fallback) {
    if (!s || !*s) return fallback;
    return std::strtoull(s, nullptr, 10);
}

static void usage(const char *argv0) {
    std::fprintf(stderr,
                 "Usage: %s <p> [seed_offset] [max_trials] [x16halvenonsplit] "
                 "[chunk_trials] [blocks] [threads] [claim_batch] [auto|generic|u96]\n",
                 argv0);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    const char *mode = argc >= 5 ? argv[4] : "x16halvenonsplit";
    if (std::strcmp(mode, "x16halvenonsplit") != 0) {
        std::fprintf(stderr, "Only x16halvenonsplit is implemented in this CUDA prototype.\n");
        return 1;
    }

    u128 p = parse128(argv[1]);
    u64 seed_offset = argc >= 3 ? parse_u64_arg(argv[2], 0) : 0;
    u64 max_trials = argc >= 4 ? parse_u64_arg(argv[3], 1000000ULL) : 1000000ULL;
    u64 default_chunk = max_trials < 1000000000ULL ? max_trials : 1000000000ULL;
    u64 chunk_trials = argc >= 6 ? parse_u64_arg(argv[5], default_chunk) : default_chunk;
    if (max_trials == 0) max_trials = 1000000ULL;
    if (chunk_trials == 0) chunk_trials = max_trials;

    u64 p_mod8 = (u64)(p & 7);
    if (p_mod8 != 5 && (p_mod8 & 3) != 3) {
        std::fprintf(stderr,
                     "x16halvenonsplit requires p == 5 mod 8 or p == 3 mod 4 "
                     "for fast square roots.\n");
        return 1;
    }

    int device = 0;
    die_cuda(cudaSetDevice(device), "cudaSetDevice");
    cudaDeviceProp prop{};
    die_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

    int blocks = argc >= 7 ? (int)parse_u64_arg(argv[6], 0) : 0;
    int threads = argc >= 8 ? (int)parse_u64_arg(argv[7], 0) : 128;
    if (threads <= 0) threads = 128;
    u64 claim_batch = argc >= 9 ? parse_u64_arg(argv[8], 1ULL) : 1ULL;
    if (claim_batch == 0) claim_batch = 1ULL;
    const char *backend_arg = argc >= 10 ? argv[9] : "auto";

    SearchParams params{};
    params.p = p;
    m128_init(&params.mt, p);
    int pbits = bitlen128(p);
    params.rand_mask = pbits >= 128 ? (u128)0 - 1 : (((u128)1 << pbits) - 1);
    params.sqrt_case = ((u64)(p & 3) == 3) ? 3 : 5;
    params.sqrt_exp = params.sqrt_case == 3 ? ((p + 1) >> 2) : ((p + 3) >> 3);
    params.sqrtm1 = params.sqrt_case == 5
        ? powmod128_mont(2, (p - 1) >> 2, &params.mt)
        : 0;
    params.two_m = toM128(2, &params.mt);
    params.four_m = toM128(4, &params.mt);
    params.eight_m = toM128(8, &params.mt);
    params.c24_m = toM128(24, &params.mt);
    params.c32_m = toM128(32, &params.mt);
    params.c48_m = toM128(48, &params.mt);
    params.sqrtm1_m = params.sqrt_case == 5
        ? powmod128_mont_raw(params.two_m, (p - 1) >> 2, &params.mt)
        : 0;
    params.inv2_m = toM128((p + 1) >> 1, &params.mt);
    params.inv4_m = invert128_mont_raw(params.four_m, p, &params.mt);
    params.k = compute_k(p);

    bool use_u96 = false;
    if (std::strcmp(backend_arg, "auto") == 0) {
        use_u96 = pbits <= 96;
    } else if (std::strcmp(backend_arg, "u96") == 0) {
        if (pbits > 96) {
            std::fprintf(stderr, "u96 backend requires p < 2^96.\n");
            return 1;
        }
        use_u96 = true;
    } else if (std::strcmp(backend_arg, "generic") == 0) {
        use_u96 = false;
    } else {
        std::fprintf(stderr, "Unknown backend: %s\n", backend_arg);
        return 1;
    }
    if (blocks <= 0) blocks = prop.multiProcessorCount * (use_u96 ? 8 : 4);
    SearchParams96 params96{};
    if (use_u96) params96 = search_params96_init(p, params.k, pbits);

    std::printf("Pomerance CUDA x16halvenonsplit prototype\n\n");
    std::printf("p = %s\n", sprint128(p).c_str());
    std::printf("seed_offset = %llu\n", seed_offset);
    std::printf("max_trials = %llu\n", max_trials);
    std::printf("chunk_trials = %llu\n", chunk_trials);
    std::printf("k = %d\n", params.k);
    /* POM_SOURCE=legacy restores the record-stack kernel (both reciprocal
       marks, uniform y) for same-seed A/B measurement.  The cover source is
       the default and is only available on the u96 backend. */
    const char *source_env = getenv("POM_SOURCE");
    bool use_cover = use_u96 && !(source_env && std::strcmp(source_env, "legacy") == 0);
    std::printf("backend = %s\n", use_u96 ? "u96" : "generic-u128");
    std::printf("source  = %s\n",
                use_cover ? "cover (dedup + genus-one first-lift, LANES="
                            POM_STR(POM_COVER_LANES) ")"
                          : "legacy record stack (both marks, uniform y)");
    std::printf("GPU = %s  SMs=%d  blocks=%d  threads=%d  claim_batch=%llu\n\n",
                prop.name, prop.multiProcessorCount, blocks, threads, claim_batch);

    /* On-device correctness gate.  Runs by default before any production
       search on the cover source (POM_SELFTEST=0 skips, =only runs and
       exits).  It proves, on this GPU and this prime, that every cover
       point lies on E0, that the transported first-halving root squares to
       the marked point's halving discriminant, and that every emitted point
       has exact order 32. */
    const char *st_env = getenv("POM_SELFTEST");
    bool st_skip = st_env && std::strcmp(st_env, "0") == 0;
    bool st_only = st_env && std::strcmp(st_env, "only") == 0;
    if (use_cover && !st_skip) {
        u64 *d_ctr = nullptr;
        die_cuda(cudaMalloc(&d_ctr, 5 * sizeof(u64)), "cudaMalloc selftest");
        die_cuda(cudaMemset(d_ctr, 0, 5 * sizeof(u64)), "cudaMemset selftest");
        int st_blocks = blocks < 64 ? blocks : 64;
        cover_selftest_kernel96<<<st_blocks, threads>>>(params96, seed_offset ^ 0x5e1f7e57,
                                                        64, d_ctr, d_ctr + 1,
                                                        d_ctr + 2, d_ctr + 3, d_ctr + 4);
        die_cuda(cudaGetLastError(), "selftest launch");
        die_cuda(cudaDeviceSynchronize(), "selftest sync");
        u64 h[5];
        die_cuda(cudaMemcpy(h, d_ctr, sizeof h, cudaMemcpyDeviceToHost), "selftest copy");
        cudaFree(d_ctr);
        std::printf("selftest: checked=%llu transport_bad=%llu emitted=%llu "
                    "order_bad=%llu off_curve=%llu\n",
                    h[0], h[1], h[2], h[3], h[4]);
        if (h[0] < 1000 || h[1] || h[3] || h[4]) {
            std::fprintf(stderr, "SELFTEST FAILED - refusing to run production search\n");
            return 3;
        }
        std::printf("selftest: PASS\n\n");
        if (st_only) return 0;
    }

    GpuStats *d_stats = nullptr;
    GpuResult *d_result = nullptr;
    die_cuda(cudaMalloc(&d_stats, sizeof(GpuStats)), "cudaMalloc stats");
    die_cuda(cudaMalloc(&d_result, sizeof(GpuResult)), "cudaMalloc result");

    u64 total = 0;
    u64 chunk_nonce = 0;
    GpuResult final_result{};
    auto t0 = std::chrono::steady_clock::now();

    while (total < max_trials && !final_result.found) {
        u64 remaining = max_trials - total;
        u64 this_chunk = remaining < chunk_trials ? remaining : chunk_trials;
        die_cuda(cudaMemset(d_stats, 0, sizeof(GpuStats)), "cudaMemset stats");
        die_cuda(cudaMemset(d_result, 0, sizeof(GpuResult)), "cudaMemset result");

        if (use_cover) {
            x16cover_kernel96<<<blocks, threads>>>(params96, seed_offset, chunk_nonce,
                                                   this_chunk, total, claim_batch,
                                                   d_stats, d_result);
        } else if (use_u96) {
            x16halvenonsplit_kernel96<<<blocks, threads>>>(params96, seed_offset, chunk_nonce,
                                                           this_chunk, total, claim_batch,
                                                           d_stats, d_result);
        } else {
            x16halvenonsplit_kernel<<<blocks, threads>>>(params, seed_offset, chunk_nonce,
                                                         this_chunk, total, claim_batch,
                                                         d_stats, d_result);
        }
        die_cuda(cudaGetLastError(), "kernel launch");
        die_cuda(cudaDeviceSynchronize(), "kernel synchronize");

        GpuStats stats{};
        GpuResult result{};
        die_cuda(cudaMemcpy(&stats, d_stats, sizeof(stats), cudaMemcpyDeviceToHost),
                 "copy stats");
        die_cuda(cudaMemcpy(&result, d_result, sizeof(result), cudaMemcpyDeviceToHost),
                 "copy result");

        u64 done = stats.processed;
        if (done > this_chunk) done = this_chunk;
        total += done;
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - t0).count();
        double rate = elapsed > 0.0 ? (double)total / elapsed / 1e6 : 0.0;
#if POM_CUDA_DETAILED_STATS
        std::printf("  trials=%llu elapsed=%.3f rate_Mps=%.6f y_draws=%llu "
                    "nonsplit_y=%llu sqrtD_y=%llu roots=%llu\n",
                    total, elapsed, rate,
                    stats.y_draws, stats.y_nonsplit, stats.y_with_sqrt_D,
                    stats.roots_valid);
#else
        std::printf("  trials=%llu elapsed=%.3f rate_Mps=%.6f claimed=%llu\n",
                    total, elapsed, rate, stats.claimed);
#endif
        std::fflush(stdout);

        if (result.found) {
            final_result = result;
            total = result.trial_index + 1;
            break;
        }
        if (done == 0) {
            std::fprintf(stderr, "Kernel made no candidate progress; stopping.\n");
            break;
        }
        chunk_nonce++;
    }

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::printf("\n");
    if (final_result.found) {
        u128 A = unpack128(final_result.A);
        u128 x0 = unpack128(final_result.x0);
        std::printf("Found after %.2fs (~%llu X1(16) curves)\n\n",
                    elapsed, final_result.trial_index + 1);
        std::printf("%s %s %s\n\n",
                    sprint128(p).c_str(), sprint128(A).c_str(), sprint128(x0).c_str());
        std::printf("Verified: %s  (%.2fs)\n",
                    verify128_mont(p, A, x0, params.k, &params.mt) ? "PASS" : "FAIL",
                    elapsed);
    } else {
        double rate = elapsed > 0.0 ? (double)total / elapsed / 1e6 : 0.0;
        std::printf("Not found in %.2fs. trials=%llu rate_Mps=%.6f\n",
                    elapsed, total, rate);
    }

    cudaFree(d_stats);
    cudaFree(d_result);
    return final_result.found ? 0 : 1;
}
#endif  /* POM_NO_DRIVER */
