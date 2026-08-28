# reciprocal-mark-cut

A fast search for **Pomerance triples**, the shortest known primality
certificates, targeting the open [DANGER3 data challenge](https://github.com/AndrewVSutherland/DANGER3)
(current frontier: **p = 10^27 + 103**, unsolved). One engine, `src/search.c`:
the record-holding search stack (the code lineage behind the 10^22 through
10^26 challenge solves) plus four further exact layers, measured **~3-4x
faster than the record stack at equal mathematical search mass** and
confirmed in a judged paired race on fresh held-out primes: **4 solved
versus 0** for the record stack at the 10^21 tier under identical caps. The
engine is scale-general - the same code path runs from 10^12 through the
frontier target and beyond (p < 2^127).

The name refers to the first of those layers: the proof that the record
method spent exactly half of its work on candidates that could never change
any outcome.

## 1. What is being searched

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
appears immovable is summarized in section 4.)

## 2. The algorithm

The base is the record stack (`x16halvenonsplit`; 2-Sylow base by Fabian
Ruehle, X1(16) prescribed torsion, successive halving and nonsplit filter by
Alexa McLain, p = 3 mod 4 square roots by Jane Shi; MIT licensed): sample a
parameter y of the X1(16) Tate normal form so every curve has rational
16-torsion (~16x hit rate), keep the nonsplit branch chi(A^2 - 4) = -1 where
the rational 2-Sylow subgroup is cyclic, and climb one halving per level -
marked-point depth then equals v2(#E), and an early abort costs about two
square-root exponentiations per candidate on average.

On that base this engine adds four exact layers. The identities were
verified symbolically (25-point random rational evaluation at degrees <= 16,
which is a proof for polynomial identities), and each layer preserves the
search distribution exactly - these are not heuristic filters.

**Layer 1 - the reciprocal-mark cut (exact 2x).** The X1(16) sampler's
auxiliary quadratic emits two roots per curve. They yield the same
Montgomery A and *reciprocal* marked x-coordinates, and on a nonsplit curve
the two marks differ by the rational 2-torsion translate x -> 1/x, which
preserves divisibility depth. Testing both can never change an outcome; the
engine tests one representative per curve and counts distinct curves.
(`tools/audit_reciprocal.py` re-proves this numerically from an independent
implementation.)

**Layer 2 - Jacobi-before-root gating (~6x cheaper gates).** Every
prospective square root is preceded by a Jacobi-symbol test through GMP's
two-limb kernel (~0.20 us at 70 bits, ~0.25 us at 90 bits, versus
~1.2-1.5 us for the Euler-criterion power it replaces). A nonsquare gate -
half the stream at every halving level - returns without a modular
exponentiation.

**Layer 3 - the skeleton lookahead (this repository's contribution).** For
the halves x' of x (where s^2 = d = x^2 + Ax + 1, u = 2(x+s), w = u^2 - 4),
the factors P = x+s+1 and M = x+s-1 satisfy

    w = 4*P*M,     P(s)*P(-s) = (2-A)*x,     chi(x') = chi(2)*chi(P),

and combining with the 2-descent identity chi(d(x')) = chi(B)*chi(x')
(B ~ x*d is the chain-constant twist class):

    chi(d_next) = chi(B) * chi(2) * chi(P_branch).

So after sqrt(d), two Jacobi symbols decide both the w-branch (chi(w+) =
chi(P+)chi(M+)) and the character of the *next* level's gate - before the
sqrt(w) construction is paid. A chain that will die at the next gate is
abandoned without its terminal square root, and the per-level jacobi(d)
after the first is dropped because the prediction already certifies d
square. This removes 25% of all halving-chain exponentiations; end-to-end
it measures ~4-5% because the chain shares the exponentiation budget with
the amortized sampler. The identities are exact: with a fixed seed and one
thread the lookahead chain emits the same triple and the same candidate
count as the plain chain it replaces (`make check` runs this differential;
`POMERANCE_NO_LOOKAHEAD=1` disables the layer for A/B).

**Layer 4 - scale-general arithmetic.** All production values stay in
Montgomery form (quadratic characters are unaffected: R is a square), with
a two-limb reducer valid through p < 2^127, GMP's assembly powm for the
fixed square-root exponents, batched Montgomery-trick inversions in the
sampler, and threads pinned to the actual cgroup CPU quota - an
oversubscribed thread pool silently costs ~2x, and mismeasuring the
baseline the same way silently *inflates* any claimed ratio by ~2x.

**Verified structure not yet merged.** The same research campaign that
produced this engine proved two more exact identities on sibling engines:
the conjugate-fiber rescue (D(u)D(-u) = (u^4-1)^2 f(u) with A(u) = A(-u):
exactly one of two conjugate fibers always lifts on the nonsplit branch, a
failed root algebraically constructing the partner's) and the genus-one
first-lift cover (D*H = y(y-1)(y-2)*G^2: walking w^2 = X^3 - X emits curves
that pre-pass the first halving gate at exactly 2x hazard - independently
re-measured at 2.01-2.11x survivor density). Merging them into this engine
is the highest-value open task here; they are documented so the next edit
starts from the identities, not the archaeology.

## 3. Results

**Judged paired race** (fresh random probable primes generated after
submission, both sides running the same primes in the same order under
identical caps, every triple re-verified in exact integer arithmetic against
the official DANGER3 `vpp.py` logic; AMD Milan, 4 vCPU, 10^21 tier, 2400 s
per-prime cap): this engine's lineage solved **4**, the record stack **0**.
Across the ten-trajectory campaign it came from, the pooled score on
identical prime streams was 29 solves for the new engines versus 8 for the
record stack.

**Equal-mass benchmark, re-measured from this repository** (engine rebuilt
from source against the unmodified record engine, same seeds, 4 threads,
arm64; equal hazard: 4M record-stack marks = 2M distinct curves by layer 1;
reproduce with `make bench`):

| p (10^21 tier) | record stack, 4M marks | this engine, 2M curves | ratio |
|---|---|---|---|
| ...000411 (3 mod 8) | 6.83 s | 1.86 s | 3.66x |
| ...000117 (5 mod 8) | 6.81 s | 2.34 s | 2.91x |
| ...000327 (7 mod 8) | 7.85 s | 1.88 s | 4.18x |

The campaign's own AMD Milan measurements of the same comparison were
4.3-4.4x: the algebraic layers transfer across platforms; hand-tuned x86
arithmetic does not fully.

**Skeleton lookahead A/B** (engine vs itself with
`POMERANCE_NO_LOOKAHEAD=1`, 3 repetitions, 4 threads, arm64):

| workload | lookahead | plain chain | gain |
|---|---|---|---|
| 10^21, 2M curves x 3 classes | 15.38 s | 16.05 s | +4.3% |
| **10^27 + 103**, 1M curves x 3 | 3.31 s | 3.45 s | +4.2% |

The frontier row is a bounded run on the actual open target - the same
binary, no special casing.

**The frontier.** For p = 10^27 + 103 (7 mod 8), the record stack's hit
rate puts the expected work at 2.2-3.2 x 10^12 marks, i.e. 1.1-1.6 x 10^12
distinct curves for this engine. Bounded runs measure ~0.9M distinct
curves/s on 4 arm64 cores - at both the 10^21 tier and on the frontier
prime itself, the two-limb path barely notices the extra 20 bits. That is
still weeks on a workstation and near a day on a 64-core box; the realistic
path to the frontier is porting layers 1 and 3 (plus the unmerged cover)
into the existing CUDA engine - the 10^26 record was 45 minutes on one
RTX 6000, and the verified layers project the expected 10^27 run into
single-digit GPU-hours. Search times are close to exponentially
distributed: single runs routinely deviate several-fold from expectation in
both directions.

## 4. Why these are constant factors, and not the exponent

The acceptance process is a tower of quadratic-character gates chi(d_j), one
per halving level. A gate can be sourced for free exactly when its double
cover has genus <= 1 (a genus-one curve is its own Jacobian and can be
walked; by Riemann-Hurwitz nothing can be parameterized onto a curve of
higher genus). The verified layers are precisely the maximal genus-<=1
prefix of that tower: the next gate's covers have genus 5 and 17, and an
exact linear-algebra certification over F_2 (character words through depth
21, three residue classes, ~7.5M sampled streams) found no multiplicative
relation that would collapse any later gate. The constructive alternative
(CM: prescribe the curve order instead of searching for it) is closed for
this target: the three admissible orders of 10^27 + 103 all have essentially
squarefree discriminants with class numbers ~10^13 ~ sqrt(p). What remains
above constant factors is open mathematics - sublinear single-root
extraction of Hilbert class polynomials mod p, a Prym/correspondence
structure on the gate tower, or a polylog predictor of v2(#E).

## 5. Building and running

    make            # needs gcc with OpenMP and libgmp-dev
    make check      # lemma audit, lookahead differential, verified solves
    make bench      # equal-coverage race vs the vendored record engine

    ./search <p> [seed]        # stdout: "A x0 candidates_tested"

Environment variables: `SEARCH_THREADS` (worker count; default is the
cgroup CPU quota, else 4), `SEARCH_MAX_TRIALS` (bounded run, for
benchmarking), `POMERANCE_NO_LOOKAHEAD=1` (disable layer 3 for A/B).
Diagnostics go to stderr; stdout carries exactly the one-line answer.
Verify any output against the official DANGER3 logic:

    python3 tools/vpp.py <p> <A> <x0>       # prints True

Supported classes: p mod 8 in {3, 5, 7} (p = 1 mod 8 remains unsupported in
the whole lineage, as in the record stack). `bench/upstream/pomerance.c` is
the record engine vendored verbatim so every ratio in this README is
reproducible from this repository alone.

## 6. Provenance and license

MIT, throughout. The base search lineage is Fabian Ruehle's, Alexa McLain's
and Jane Shi's MIT-licensed DANGER3 record code; their headers are preserved
in the source. The engine was generated by an autonomous research agent
(GPT-5.6-sol under the codex harness) inside the pomerance-bench evaluation
environment on 2026-08-28 and independently verified as described above; the
skeleton-lookahead layer, the reciprocal-mark lemma audit, the
cross-platform measurements and this document are the repository's own
contributions. Every performance number states its platform; when
reproducing, pin the record stack's thread count to the actual CPU quota.
