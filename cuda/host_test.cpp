/*
 * host_test.cpp -- validate the CUDA engine's device math on a machine with
 * no GPU.  Every numeric routine in search_cuda.cu is marked HD (host and
 * device); compiling with -DPOM_HOST_TEST drops the CUDA includes, kernels
 * and driver, leaving exactly the arithmetic the GPU will execute.
 *
 *   c++ -O2 -std=c++17 -DPOM_HOST_TEST -o host_test host_test.cpp
 *   ./host_test
 *
 * Checks, in order:
 *   1. Montgomery arithmetic against __int128 reference arithmetic.
 *   2. Cover points stay on E0: W^2 = X^3 - X, across a long walk.
 *   3. The batch inversion agrees with individual inversions.
 *   4. THE TRANSPORT IDENTITY: sqrt(d1) = y*sD*W/(dx*(y-1)^2) really is a
 *      square root of the marked point's halving discriminant.  This is the
 *      claim the whole cover source rests on.
 *   5. Emitted points have exact order 32 (nonzero through 4 doublings,
 *      zero on the 5th) -- i.e. the cover really starts the chain at depth 5.
 *   6. Hazard: cover-sourced candidates reach depth 12 about twice as often
 *      as candidates from the record-stack source, on the same prime.
 */

#include "search_cuda.cu"

#include <cinttypes>
#include <cstdio>
#include <random>

static u128 parse_u128(const char *s) {
    u128 v = 0;
    while (*s >= '0' && *s <= '9') v = v * 10 + (u32)(*s++ - '0');
    return v;
}

static void print_u128(u128 v) {
    char buf[44];
    int i = 43;
    buf[i] = 0;
    if (!v) buf[--i] = '0';
    while (v) { buf[--i] = char('0' + (int)(v % 10)); v /= 10; }
    std::printf("%s", buf + i);
}

static u128 mulmod_ref(u128 a, u128 b, u128 p) {   /* schoolbook, 96-bit safe */
    u128 r = 0;
    a %= p;
    while (b) {
        if (b & 1) { r += a; if (r >= p) r -= p; }
        a <<= 1; if (a >= p) a -= p;
        b >>= 1;
    }
    return r;
}

static u128 powmod_ref(u128 a, u128 e, u128 p) {
    u128 r = 1;
    a %= p;
    while (e) {
        if (e & 1) r = mulmod_ref(r, a, p);
        a = mulmod_ref(a, a, p);
        e >>= 1;
    }
    return r;
}

struct Ctx {
    u128 p;
    SearchParams96 params;
};

static Ctx make_ctx(u128 p) {
    Ctx c;
    c.p = p;
    int bits = 0;
    for (u128 t = p; t; t >>= 1) bits++;
    c.params = search_params96_init(p, compute_k(p), bits);
    return c;
}

static U96 rnd_field(std::mt19937_64 &rng, const Ctx &c) {
    for (;;) {
        u128 v = ((u128)rng() << 64) | rng();
        v %= c.p;
        if (v > 2) return u96_from_u128(v);
    }
}

/* Find one point on E0: W^2 = X^3 - X. */
static bool find_cover_point(std::mt19937_64 &rng, const Ctx &c, CoverPt96 *out) {
    for (int tries = 0; tries < 200; tries++) {
        U96 x = rnd_field(rng, c);
        U96 xm = to_mont96(x, &c.params.f);
        U96 rhs = submod96(mont_mul96(mont_mul96(xm, xm, &c.params.f), xm,
                                      &c.params.f), xm, c.params.f.p);
        U96 w;
        if (is_zero96(rhs)) continue;
        if (sqrtmod_p5_96_mont(&w, rhs, &c.params)) { out->x = xm; out->w = w; return true; }
    }
    return false;
}

static int fails = 0;
static void report(const char *name, bool ok, const char *detail = "") {
    std::printf("  [%s] %-44s %s\n", ok ? "PASS" : "FAIL", name, detail);
    if (!ok) fails++;
}

/* ---- 1. Montgomery arithmetic vs reference ---- */
static void test_arith(const Ctx &c, std::mt19937_64 &rng) {
    int bad = 0;
    for (int i = 0; i < 20000; i++) {
        U96 a = rnd_field(rng, c), b = rnd_field(rng, c);
        u128 av = u96_to_u128(a), bv = u96_to_u128(b);
        U96 prod = mont_mul96(to_mont96(a, &c.params.f), to_mont96(b, &c.params.f),
                              &c.params.f);
        if (u96_to_u128(from_mont96(prod, &c.params.f)) != mulmod_ref(av, bv, c.p)) bad++;
        U96 s = addmod96(a, b, c.params.f.p);
        if (u96_to_u128(s) != (av + bv) % c.p) bad++;
    }
    char d[64]; std::snprintf(d, sizeof d, "20000 products/sums, %d bad", bad);
    report("montgomery arithmetic vs __int128 reference", bad == 0, d);
}

/* ---- 2/3. cover walk stays on the curve; batch inversion is correct ---- */
static void test_walk(const Ctx &c, std::mt19937_64 &rng) {
    CoverPt96 pts[POM_COVER_LANES];
    CoverPt96 step;
    for (int i = 0; i < POM_COVER_LANES; i++)
        if (!find_cover_point(rng, c, &pts[i])) { report("cover point setup", false); return; }
    if (!find_cover_point(rng, c, &step)) { report("cover step setup", false); return; }

    int off_curve = 0, steps = 0;
    for (int it = 0; it < 4000; it++) {
        if (!cover_step96(pts, POM_COVER_LANES, step.x, step.w, &c.params)) continue;
        steps++;
        for (int i = 0; i < POM_COVER_LANES; i++)
            if (!cover_on_curve96(pts[i].x, pts[i].w, &c.params)) off_curve++;
    }
    char d[96];
    std::snprintf(d, sizeof d, "%d steps x %d lanes, %d off-curve", steps,
                  POM_COVER_LANES, off_curve);
    report("cover walk stays on W^2 = X^3 - X", off_curve == 0 && steps > 3000, d);

    int bad = 0;
    for (int it = 0; it < 2000; it++) {
        U96 in[POM_COVER_LANES], out[POM_COVER_LANES];
        for (int i = 0; i < POM_COVER_LANES; i++)
            in[i] = to_mont96(rnd_field(rng, c), &c.params.f);
        if (!batch_invert96(out, in, POM_COVER_LANES, &c.params)) { bad++; continue; }
        for (int i = 0; i < POM_COVER_LANES; i++)
            if (!eq96(mont_mul96(in[i], out[i], &c.params.f), c.params.f.one)) bad++;
    }
    std::snprintf(d, sizeof d, "2000 batches of %d, %d bad", POM_COVER_LANES, bad);
    report("batch inversion (Montgomery's trick)", bad == 0, d);
}

/* ---- 4/5. the transport identity, and exact order 32 ---- */
static void test_transport(const Ctx &c, std::mt19937_64 &rng) {
    CoverPt96 pts[POM_COVER_LANES], step;
    for (int i = 0; i < POM_COVER_LANES; i++)
        if (!find_cover_point(rng, c, &pts[i])) { report("transport setup", false); return; }
    if (!find_cover_point(rng, c, &step)) { report("transport setup", false); return; }

    long checked = 0, good = 0, emitted = 0, order_bad = 0;
    for (int it = 0; it < 60000 && checked < 4000; it++) {
        if (!cover_step96(pts, POM_COVER_LANES, step.x, step.w, &c.params)) continue;
        for (int i = 0; i < POM_COVER_LANES; i++) {
            int r = cover_transport_selfcheck96(pts[i].x, pts[i].w, &c.params);
            if (r < 0) continue;
            checked++;
            good += (r == 1);

            U96 A, Am, x32;
            if (x16_cover_to_depth5_96_mont(&A, &Am, &x32, pts[i].x, pts[i].w,
                                            &c.params)) {
                emitted++;
                /* exact order 32: 4 doublings nonzero, 5th zero */
                U96 a24 = mont_mul96(addmod96(Am, c.params.two_m, c.params.f.p),
                                     c.params.inv4_m, &c.params.f);
                U96 X = x32, Z = c.params.f.one;
                bool ok = true;
                for (int j = 0; j < 4; j++) {
                    xDBL96(&X, &Z, X, Z, a24, &c.params.f);
                    if (is_zero96(Z)) { ok = false; break; }
                }
                if (ok) { xDBL96(&X, &Z, X, Z, a24, &c.params.f); ok = is_zero96(Z); }
                if (!ok) order_bad++;
            }
        }
    }
    char d[128];
    std::snprintf(d, sizeof d, "%ld/%ld transported roots correct", good, checked);
    report("TRANSPORT: sqrt(d1) = y*sD*W/(dx*(y-1)^2)", checked > 500 && good == checked, d);
    std::snprintf(d, sizeof d, "%ld emitted, %ld wrong order", emitted, order_bad);
    report("emitted marked points have exact order 32", emitted > 100 && order_bad == 0, d);
}

/* ---- 6. hazard: cover doubles the depth-12 survival rate ---- */
static void test_hazard(const Ctx &c, std::mt19937_64 &rng) {
    const int DEPTH = 12;
    long plain_tested = 0, plain_surv = 0, cover_tested = 0, cover_surv = 0;

    /* record-stack source: uniform y, one mark per curve (dedup'd) */
    for (int it = 0; it < 400000 && plain_tested < 120000; it++) {
        U96 y = rnd_field(rng, c);
        U96 ym = to_mont96(y, &c.params.f);
        U96 y2m = mont_mul96(ym, ym, &c.params.f);
        if (!x16_y_predicts_nonsplit96_mont(ym, y2m, &c.params)) continue;
        U96 y3m = mont_mul96(y2m, ym, &c.params.f);
        U96 qa = submod96(y2m, addmod96(ym, ym, c.params.f.p), c.params.f.p);
        if (is_zero96(qa)) continue;
        U96 qb = submod96(addmod96(y2m, y2m, c.params.f.p), y3m, c.params.f.p);
        U96 qc = submod96(c.params.f.one, ym, c.params.f.p);
        U96 D = submod96(mont_mul96(qb, qb, &c.params.f),
                         mont_mul96(addmod96(qa, qa, c.params.f.p),
                                    addmod96(qc, qc, c.params.f.p), &c.params.f),
                         c.params.f.p);
        U96 sd;
        if (!sqrtmod_p5_96_mont(&sd, D, &c.params)) continue;
        U96 inv2qa = invert96_mont(addmod96(qa, qa, c.params.f.p), &c.params);
        U96 xq = mont_mul96(submod96(sd, qb, c.params.f.p), inv2qa, &c.params.f);
        U96 A, Am, xP;
        if (!x16_root_to_montgomery_A96_mont(&A, &Am, &xP, xq, ym, &c.params)) continue;
        plain_tested++;
        U96 x = xP;
        int depth = 4;
        while (depth < DEPTH && halve_once_first96_mont(&x, Am, x, &c.params)) depth++;
        if (depth >= DEPTH) plain_surv++;
    }

    /* cover source: same prime, one candidate per emitted cover curve */
    CoverPt96 pts[POM_COVER_LANES], step;
    for (int i = 0; i < POM_COVER_LANES; i++)
        if (!find_cover_point(rng, c, &pts[i])) { report("hazard setup", false); return; }
    if (!find_cover_point(rng, c, &step)) { report("hazard setup", false); return; }
    for (int it = 0; it < 400000 && cover_tested < 120000; it++) {
        if (!cover_step96(pts, POM_COVER_LANES, step.x, step.w, &c.params)) continue;
        for (int i = 0; i < POM_COVER_LANES; i++) {
            U96 A, Am, x32;
            if (!x16_cover_to_depth5_96_mont(&A, &Am, &x32, pts[i].x, pts[i].w,
                                             &c.params)) continue;
            cover_tested++;
            U96 x = x32;
            int depth = 5;
            while (depth < DEPTH && halve_once_first96_mont(&x, Am, x, &c.params)) depth++;
            if (depth >= DEPTH) cover_surv++;
        }
    }

    double pr = plain_tested ? (double)plain_surv / plain_tested : 0;
    double cr = cover_tested ? (double)cover_surv / cover_tested : 0;
    double ratio = pr > 0 ? cr / pr : 0;
    char d[160];
    std::snprintf(d, sizeof d,
                  "plain %ld/%ld  cover %ld/%ld  ratio %.3fx (expect ~2)",
                  plain_surv, plain_tested, cover_surv, cover_tested, ratio);
    report("cover doubles depth-12 hazard", ratio > 1.75 && ratio < 2.30, d);
}

/* ---- 7. end to end: the cover source finds a real, verifiable triple ---- */
static void test_end_to_end(const Ctx &c, std::mt19937_64 &rng) {
    CoverPt96 pts[POM_COVER_LANES], step;
    for (int i = 0; i < POM_COVER_LANES; i++)
        if (!find_cover_point(rng, c, &pts[i])) { report("e2e setup", false); return; }
    if (!find_cover_point(rng, c, &step)) { report("e2e setup", false); return; }

    long tested = 0;
    for (int it = 0; it < 40000000; it++) {
        if (!cover_step96(pts, POM_COVER_LANES, step.x, step.w, &c.params)) continue;
        for (int i = 0; i < POM_COVER_LANES; i++) {
            U96 A, Am, x32;
            if (!x16_cover_to_depth5_96_mont(&A, &Am, &x32, pts[i].x, pts[i].w,
                                             &c.params)) continue;
            tested++;
            U96 xR;
            if (!halve_chain_from_depth96_mont(&xR, Am, x32, 5, &c.params)) continue;
            /* halve_chain_from_depth96_mont already ran verify96_mont */
            U96 x0 = from_mont96(xR, &c.params.f);
            char d[160];
            std::snprintf(d, sizeof d, "%ld candidates", tested);
            report("end-to-end: cover source finds a verified triple", true, d);
            std::printf("        TRIPLE  p="); print_u128(c.p);
            std::printf("  A="); print_u128(u96_to_u128(A));
            std::printf("  x0="); print_u128(u96_to_u128(x0));
            std::printf("\n        (check: python3 ../tools/vpp.py ");
            print_u128(c.p); std::printf(" "); print_u128(u96_to_u128(A));
            std::printf(" "); print_u128(u96_to_u128(x0)); std::printf(")\n");
            return;
        }
    }
    report("end-to-end: cover source finds a verified triple", false, "no hit in budget");
}

int main(int argc, char **argv) {
    /* Default: a 90-bit p = 3 mod 4 prime (the frontier's residue class and
       size), plus a 5 mod 8 prime to exercise the other square-root branch. */
    const char *primes[] = {
        "1000000000000000000000000103",   /* 10^27+103: the actual target */
        "1237940039285380274899124357",   /* 90-bit, p = 5 mod 8            */
        "1000000000039",                  /* small: full end-to-end solve   */
        "1000000000061",                  /* small, p = 5 mod 8             */
    };
    int n = (int)(sizeof primes / sizeof primes[0]);
    if (argc > 1) { primes[0] = argv[1]; n = 1; }

    for (int i = 0; i < n; i++) {
        u128 p = parse_u128(primes[i]);
        Ctx c = make_ctx(p);
        std::printf("\np = "); print_u128(p);
        std::printf("   (p mod 8 = %d, k = %d, sqrt_case = %d)\n",
                    (int)(p & 7), c.params.k, c.params.sqrt_case);
        if (powmod_ref(2, p - 1, p) != 1) {
            std::printf("  [SKIP] not prime by Fermat base 2\n");
            continue;
        }
        std::mt19937_64 rng(0xC0FFEE ^ (u64)i);
        test_arith(c, rng);
        test_walk(c, rng);
        test_transport(c, rng);
        if (p < ((u128)1 << 50)) {
            test_end_to_end(c, rng);      /* cheap enough to solve outright */
        } else {
            test_hazard(c, rng);          /* statistical check at real size */
        }
    }
    std::printf("\n%s (%d failures)\n", fails ? "FAILED" : "ALL CHECKS PASSED", fails);
    return fails ? 1 : 0;
}
