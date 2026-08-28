# pomerance-triple-search

State-of-the-art search for **Pomerance triples**, the shortest known
primality certificates, targeting the open
[DANGER3 data challenge](https://github.com/AndrewVSutherland/DANGER3)
(current frontier: **p = 10^27 + 103**, unsolved). A CPU engine
(`src/search.c`) and a CUDA engine ([`cuda/`](cuda)) share one algorithm and
run the same code path from 10^12 through the frontier target (p < 2^127).

Against the record-holding stack behind the 10^22 through 10^26 challenge
solves, it measures **3-4x faster on CPU** and **2.3x faster on GPU** at
equal mathematical search mass, and won a judged head-to-head **4 solves to
0** on fresh held-out primes.

## 1. The problem

For an odd integer p, let q = floor(sqrt(p)) and let k be the least integer
with 2^k > q + 1 + 2*sqrt(q); note 2^k is approximately 2*sqrt(p). A
**Pomerance triple** (p, A, x0) asserts that the point with x-coordinate x0
on the Montgomery curve

    B*y^2 = x^3 + A*x^2 + x   over Z/pZ

has exact order 2^k: doubling the projective point (x0 : 1) exactly k times
lands on the identity (Z = 0 mod p), while the (k-1)-th doubling does not.
If p had a prime factor r <= sqrt(p), the point would survive reduction mod r
with order 2^k, exceeding the Hasse bound on the curve size over F_r. So a
valid triple is an unconditional proof that p is prime, checkable in about
(1/2)*log2(p) doublings (Pomerance, *Very short primality proofs*, Math.
Comp. 48 (1987); revived by Sutherland for DANGER3).

Verification is trivial; search is not. The curve order must be divisible by
2^k ~ 2*sqrt(p), which happens with probability on the order of 1/sqrt(p),
so every known method performs on the order of sqrt(p) candidate tests and
progress consists of driving down the constant. (Why the exponent itself
appears immovable is section 4.)

## 2. Key results

**Judged head-to-head, 4 to 0.** Fresh random probable primes generated
*after* submission, both engines running the same primes in the same order
under identical caps, every emitted triple re-verified in exact integer
arithmetic (AMD Milan, 4 vCPU, 10^21 tier, 2400 s per prime): this engine
solved **4**, the record stack **0**.

**CPU, equal search mass.** Rebuilt from source against the unmodified
record engine, same seeds, 4 threads, arm64. Equal hazard: 4M record-stack
marks = 2M distinct curves here. Reproduce with `make bench` — the record
engine is vendored at `bench/upstream/`, so every ratio below comes out of
this repository alone.

| p (10^21 tier) | record stack, 4M marks | this engine, 2M curves | ratio |
|---|---|---|---|
| ...000411 (3 mod 8) | 6.83 s | 1.86 s | **3.66x** |
| ...000117 (5 mod 8) | 6.81 s | 2.34 s | **2.91x** |
| ...000327 (7 mod 8) | 7.85 s | 1.88 s | **4.18x** |

The same comparison on AMD Milan gives 4.3-4.4x: the algebraic layers
transfer across platforms, hand-tuned x86 arithmetic does not fully.

**GPU, equal search mass** (RTX 4090, same-seed A/B on the frontier prime
itself): the record CUDA kernel sustains 46.4M marks/s, this engine 27.0M
cover-curves/s — **2.34x** at identical hazard. On an RTX 5090 the cover
kernel reaches 29.4M cover-curves/s.

**Correctness, on the device that will do the work.** Before any production
search the GPU binary re-proves its own algebra and refuses to run if it
fails. On the frontier prime:

    selftest: checked=1047711 transport_bad=0 emitted=524136 order_bad=0 off_curve=0

Over a million transported roots verified, zero wrong; every emitted point
of exact order 32. The same proofs run on a machine with no GPU at all
(`cd cuda && make check`), and end-to-end the engine's triples are accepted
by the official DANGER3 verifier, which shares no code with it.

**The frontier.** For p = 10^27 + 103 the record stack's hit rate puts the
expected work at 2.2-3.2 x 10^12 marks — 5.5-7.9 x 10^11 cover curves here.
At the measured GPU rate that is **5-8 expected GPU-hours on one card**, or
under an hour on a modest fleet. Search times are near-exponentially
distributed, so single runs deviate several-fold in both directions.

## 3. What makes it faster

Every layer below is **exact**: it changes cost, never the search
distribution. None is a heuristic filter, a sampling bias, or a
probabilistic prefilter. The identities were verified symbolically
(25-point random rational evaluation at degrees <= 16 — a proof for
polynomial identities), then re-verified numerically at runtime.

The unifying observation is that the record stack repeatedly **pays full
price for information it already holds**. A modular exponentiation costs
~130 field multiplications; the search loop spends them extracting facts
that are already determined by algebra in hand. Each layer is one instance
of that: a test that is provably redundant, a condition that can be
*sampled* instead of tested, and a character that can be *predicted*
instead of extracted.

### 3.1 A redundant test: marked-point deduplication (exact 2x)

The X1(16) sampler's auxiliary quadratic emits two roots per curve. They
yield the same Montgomery coefficient A and *reciprocal* marked
x-coordinates. On a nonsplit curve the map

    x -> 1/x

is translation by the rational 2-torsion point (0,0), and the rational
2-Sylow subgroup is cyclic, so two points differing by 2-torsion have the
same halving depth. The two marks therefore always return the same verdict:
testing the second is pure duplicated work. The engine tests one
representative and counts distinct curves.

This is the layer the GPU record kernel was missing entirely — it tested
both marks in a `for (ri = 0; ri < 2; ri++)` loop. `tools/audit_dedup.py`
re-proves the lemma numerically from an independent implementation.

### 3.2 A sampled condition: the genus-one first-lift cover (exact 2x)

Write G = (y^2-2)(y^2-2y+2). The X1(16) fiber discriminant and the first
halving obstruction are

    D = y(y-2)G,        H = (y-1)G,

and they satisfy the polynomial identity

    D * H = y(y-1)(y-2) * G^2.

Substituting X = y-1 makes the first factor X^3 - X. So on the fixed
elliptic curve

    E0 :  W^2 = X^3 - X,

every point gives H = (W*G)^2 / D — meaning that **once D is a square, H is
a square automatically**. The record stack draws y uniformly and pays an
exponentiation to discover that H fails half the time. Sampling y by
*walking E0 instead* makes the first gate pass by construction: exactly
twice the hazard per candidate, with no rejection.

The root the halving step needs is then transported algebraically rather
than extracted:

    sqrt(d1) = y * sD * W / (dx * (y-1)^2),     dx = r0 - 2*qa*y,

with r0 = sD - qb the retained quadratic numerator (so the marked point is
xP = r0/dx). A genus-one curve is its own Jacobian, so E0 can be walked
with a fixed step; batching several walks behind one shared inversion makes
a cover point cost a few multiplications plus 1/LANES of an inversion.

This layer lives in the CUDA engine, where it is worth the most: the walk
is branch-light and warp-uniform, which is precisely why it succeeds on GPU
where character prefilters failed.

### 3.3 A predicted character: the skeleton lookahead (25% fewer powers)

For the halves x' of x — where s^2 = d = x^2 + Ax + 1, u = 2(x+s),
w = u^2 - 4 — the factors P = x+s+1 and M = x+s-1 satisfy

    w = 4*P*M,      P(s)*P(-s) = (2-A)*x,      chi(x') = chi(2)*chi(P),

and combining with the 2-descent identity chi(d(x')) = chi(B)*chi(x'),
where B ~ x*d is the chain-constant twist class:

    chi(d_next) = chi(B) * chi(2) * chi(P_branch).

So after sqrt(d), **two Jacobi symbols decide both the branch and the next
level's gate** — before the sqrt(w) construction is paid. A chain that will
die at the next gate is abandoned without its terminal square root, and the
per-level Jacobi test on d disappears because the prediction already
certifies d square. That removes 25% of all halving-chain exponentiations
(~4-5% end-to-end, since the chain shares the budget with the sampler).

Exactness is checked mechanically: with a fixed seed and one thread the
lookahead chain emits the same triple and the same candidate count as the
plain chain it replaces (`make check`; `POMERANCE_NO_LOOKAHEAD=1` disables
it for A/B).

### 3.4 Supporting engineering

**Cheaper gates.** Every prospective square root is preceded by a Jacobi
test through GMP's two-limb kernel (~0.20 us at 70 bits, ~0.25 us at 90
bits, against ~1.2-1.5 us for the Euler-criterion power it replaces), so a
nonsquare gate — half the stream at every level — costs no exponentiation.

**Scale-general arithmetic.** Production values stay in Montgomery form
(quadratic characters are unaffected: R is a square) with a two-limb
reducer valid through p < 2^127, GMP assembly powm for the fixed
square-root exponents, and batched Montgomery-trick inversions.

**A measurement hazard worth naming.** Threads are pinned to the actual
cgroup CPU quota. An oversubscribed pool costs ~2x — and mismeasuring a
*baseline* that way silently inflates any claimed speedup by the same
factor. Several published-looking ratios in this problem's history dissolve
under a correctly pinned baseline.

## 4. Why this is a constant factor, and not the exponent

The acceptance process is a tower of quadratic-character gates chi(d_j), one
per halving level. A gate can be sourced for free exactly when its double
cover has genus <= 1 (a genus-one curve is its own Jacobian and can be
walked; by Riemann-Hurwitz nothing can be parameterized onto a curve of
higher genus). The layers above are precisely the maximal genus-<=1 prefix
of that tower: the next gate's covers have genus 5 and 17, and an exact
linear-algebra certification over F_2 (character words through depth 21,
three residue classes, ~7.5M sampled streams) found no multiplicative
relation that would collapse any later gate.

The constructive alternative — CM: prescribe the curve order instead of
searching for it — is closed for this target by direct computation: the
three admissible orders of 10^27 + 103 all have essentially squarefree
discriminants with class numbers ~10^13 ~ sqrt(p), so writing the curve down
costs at least as much as finding it. What remains above constant factors is
open mathematics: sublinear single-root extraction of Hilbert class
polynomials mod p, a Prym/correspondence structure on the gate tower, or a
polylog-time predictor of v2(#E).

## 5. Building and running

    make            # needs gcc with OpenMP and libgmp-dev
    make check      # dedup lemma, lookahead differential, verified solves
    make bench      # equal-coverage race vs the vendored record engine

    ./search <p> [seed]        # stdout: "A x0 candidates_tested"

Environment: `SEARCH_THREADS` (workers; default is the cgroup CPU quota,
else 4), `SEARCH_MAX_TRIALS` (bounded run, for benchmarking),
`POMERANCE_NO_LOOKAHEAD=1` (disable the lookahead for A/B). Diagnostics go
to stderr; stdout carries exactly the one-line answer. Verify any output
against the official DANGER3 logic:

    python3 tools/vpp.py <p> <A> <x0>       # prints True

### On a GPU

    cd cuda
    make check                 # proves the device math with no GPU present
    make                       # build (arch auto-detected by the runner)
    ./run_frontier.sh          # selftest -> A/B -> sharded search -> verify

See [`cuda/README.md`](cuda/README.md). `cuda/pod_validate.sh` is a single
self-contained file (sources embedded) for validating a freshly rented GPU
box with no repo access or credentials.

### Repository map

    src/search.c          the CPU engine
    cuda/                 CUDA engine, host-side proofs, pod tooling
    tools/vpp.py          official DANGER3 verifier (unmodified)
    tools/audit_dedup.py  independent check of the deduplication lemma
    bench/compare.py      equal-coverage race vs the vendored record engine
    bench/upstream/       the record engine, vendored verbatim for that race

Supported classes: p mod 8 in {3, 5, 7} (p = 1 mod 8 remains unsupported in
the whole lineage, as in the record stack).

## 6. Credits and license

MIT, throughout. The base search lineage is Fabian Ruehle's, Alexa McLain's
and Jane Shi's MIT-licensed DANGER3 record code — the 2-Sylow projection
search, the X1(16) prescribed-torsion sampler with successive halving and
the nonsplit filter, the p = 3 (mod 4) square-root patch, and the CUDA port
that set the 10^26 record. Provenance headers are preserved in both
engines. The challenge and verifier are Andrew Sutherland's.

The layers in section 3, the audits, and the cross-platform measurements are
this repository's own, produced in an autonomous agent research campaign
(2026) and verified as described above. Every performance number states its
platform; when reproducing, pin the record stack's thread count to the
actual CPU quota.
