# reciprocal-mark-cut

A fast search for **Pomerance triples**, the shortest known primality
certificates, targeting the open [DANGER3 data challenge](https://github.com/AndrewVSutherland/DANGER3)
(current frontier: **p = 10^27 + 103**, unsolved). This repository carries the
state of the art as of 2026-08-28: the record-holding search stack (the code
lineage behind the 10^22 through 10^26 challenge solves) plus five further
verified algorithmic layers, ending in engines measured **~3-4x faster than
the record stack at equal mathematical search mass** and confirmed by paired
fresh-prime races (best verdict: 4 solved vs 0 for the record stack at the
10^21 tier).

The name refers to the first of those layers - the proof that the record
method spent exactly half of its work on candidates that could never change
any outcome - and is kept for the whole stack.

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
appears immovable is summarized in section 5.)

## 2. The algorithm, layer by layer

The base is the record stack (`x16halvenonsplit`; 2-Sylow base by Fabian
Ruehle, X1(16) prescribed torsion, successive halving and nonsplit filter by
Alexa McLain, p = 3 mod 4 square roots by Jane Shi; MIT licensed): sample a
parameter y of the X1(16) Tate normal form so every curve has rational
16-torsion (~16x hit rate), keep the nonsplit branch chi(A^2-4) = -1 where
the rational 2-Sylow subgroup is cyclic, and climb one halving per level -
marked-point depth then equals v2(#E), so each candidate costs about two
square-root exponentiations on average before an early abort.

On top of that base, this repository's engines carry five exact layers. Each
identity below was verified symbolically (25-point random rational
evaluation, degrees <= 16, which is a proof for polynomial identities).

**Layer 1 - the reciprocal-mark cut (exact 2x).** The X1(16) sampler's
auxiliary quadratic emits two roots per curve. They yield the same Montgomery
A and *reciprocal* marked x-coordinates, and on a nonsplit curve the two
marks differ by the rational 2-torsion translate x -> 1/x, which preserves
divisibility depth. Testing both can never change an outcome; every engine
here tests one representative and counts distinct curves.

**Layer 2 - the conjugate-fiber rescue (~1.4x).** With u = y - 1, the fiber
discriminant satisfies

    D(u) * D(-u) = (u^4 - 1)^2 * f(u),      A(u) = A(-u),

where f = u^4 - 6u^2 + 1 is the nonsplit quartic. On the nonsplit branch
chi(f) = -1, so *exactly one* of the two conjugate fibers lifts - a failed
square root is not a discard but a witness that algebraically constructs the
partner fiber's root (for p = 3 mod 4, sqrt(D(-u)) = r_f * (u^4-1) / r_D
from the failed powers r_f, r_D). No fiber is ever thrown away.

**Layer 3 - genus-one sources.** Two of the stack's rejection tests are
governed by curves of genus one, and a genus-one curve is its own Jacobian:
instead of paying a character test that rejects half the stream, one *walks*
the curve with batched group additions (Montgomery-trick inversions, a few
field products per point).

  - First halving gate: D * H = y(y-1)(y-2) * G^2 with G = (y^2-2)(y^2-2y+2),
    so given the fiber root, the first halving obstruction H is controlled by
    the fixed elliptic curve w^2 = X^3 - X (X = y-1). Walking it emits curves
    that start at depth 5 with exactly 2x the success hazard, and the root
    the halving step needs is transported algebraically: sqrt(H) = wG/sqrt(D).
  - Nonsplit gate: the twist v^2 = eta * (u^4 - 6u^2 + 1) (eta a fixed
    nonsquare) is also genus one; every emitted u has chi(f) = -1 by
    construction.

**Layer 4 - Jacobi-before-root gating (~6x cheaper gates).** Every
prospective square root is preceded by a Jacobi-symbol test through GMP's
two-limb kernel (~0.20 us at 70 bits, ~0.25 us at 90 bits, versus ~1.2-1.5 us
for the Euler-criterion power it replaces). Nonsquare gates - half the
stream at every level - return without a modular exponentiation.

**Layer 5 - the skeleton lookahead (new in this repository).** For the
halves x' of x (s^2 = d = x^2+Ax+1, u = 2(x+s), w = u^2-4), the factors
P = x+s+1 and M = x+s-1 satisfy

    w = 4*P*M,     P(s)*P(-s) = (2-A)*x,     chi(x') = chi(2)*chi(P),

and combining with the 2-descent identity chi(d(x')) = chi(B)*chi(x') (B ~
x*d is the chain-constant twist class):

    chi(d_next) = chi(B) * chi(2) * chi(P_branch).

So after sqrt(d), two Jacobi symbols decide both the w-branch (chi(w+) =
chi(P+)chi(M+)) and the character of the *next* level's gate - before the
sqrt(w) construction is paid. A chain that will die at the next gate is
abandoned without its terminal square root, and the per-level jacobi(d)
after the first is dropped because the prediction already certifies d
square. This removes 25% of all halving-chain exponentiations; measured
end-to-end it is worth ~4-5% in the heavily amortized production engine
(the chain is one of several exponentiation consumers). The identities are
exact: with a fixed seed and one thread, the lookahead chain emits the same
triple and the same candidate count as the plain chain (`make check` runs
this differential).

## 3. Engines

| file | layers | dependencies | scope |
|---|---|---|---|
| `src/search.c` | 1, 4, 5 + scale-general two-limb arithmetic | GMP | p < 2^127 |
| `src/search_conjugate.c` | 1, 2, 3 (nonsplit walk) | none | p < 2^126 |
| `src/legacy/fast65.c` | 1 + centered-Barrett 65-bit field | none | 2^64 < p < 2^65 |

`search` is the recommended engine: it is the one whose speed reproduces on
every platform tested, and the only one carrying the lookahead. The two
production engines were written by autonomous research agents in the
pomerance-bench batch-2 campaign (2026-08-28) and are preserved here exactly
as judged, except that `search.c` additionally carries the layer-5 lookahead
(added in this repository, differential-tested against the judged chain).
No single engine yet carries all five layers - the first-lift cover walk
(layer 3) has not been merged into the Jacobi/lookahead engine; that merge
is the highest-value open engineering task in this repository.

## 4. Results

**Judged paired races** (harness: fresh random probable primes generated
after submission, both sides run the same primes in the same order under
identical caps, every triple re-verified in exact integer arithmetic against
the official DANGER3 `vpp.py` logic; AMD Milan, 4 vCPU):

| engine | tier | solves (engine v record stack) |
|---|---|---|
| `search` lineage | 10^21, 6 primes, 2400 s cap | **4 v 0** |
| `search_conjugate` lineage | 10^21, 6 primes, 2400 s cap | **4 v 0** |

Across the full ten-trajectory campaign that produced these engines, the
pooled score was 29 solves for the new engines versus 8 for the record stack
on identical prime streams.

**Equal-mass benchmarks, independently re-measured** (this repository's
engines rebuilt from source against the unmodified record engine, same
seeds, 4 threads, arm64; equal hazard: 4M record-stack marks = 2M distinct
curves by layer 1):

| p (10^21 tier) | record stack, 4M marks | `search`, 2M curves | ratio |
|---|---|---|---|
| ...000411 (3 mod 8) | 6.83 s | 1.86 s | 3.66x |
| ...000117 (5 mod 8) | 6.81 s | 2.34 s | 2.91x |
| ...000327 (7 mod 8) | 7.85 s | 1.88 s | 4.18x |

The agents' own Milan measurements for the same comparisons were 4.3-4.4x;
the algebraic layers transfer across platforms, the hand-tuned x86 arithmetic
does not fully.

**Skeleton lookahead A/B** (`search` vs itself with
`POMERANCE_NO_LOOKAHEAD=1`, 3 repetitions, 4 threads, arm64):

| workload | lookahead | plain chain | gain |
|---|---|---|---|
| 10^21, 2M curves x 3 classes | 15.38 s | 16.05 s | +4.3% |
| **10^27 + 103**, 1M curves x 3 | 3.31 s | 3.45 s | +4.2% |

**Caveat on `search_conjugate`**: its judged 4 v 0 and its Milan self-
benchmark are real, but on arm64 this engine measures *below* the record
stack (0.70-0.88x here) - its custom two-limb reducer appears strongly
x86-tuned, and its own 4.38x table used a baseline measurement consistent
with an unthrottled (oversubscribed) record-stack run. It is preserved as
judged; prefer `search` unless you are on the judged platform class.

**Density audit for layer 3** (independent rerun of the cover-conditioning
claim, 320,000 curves per arm per class at the 2x10^19 tier): depth-12
survivors 2530/2556/2518 on the cover stream versus 1254/1270/1191 plain -
2.02x, 2.01x, 2.11x against a predicted exact 2x, zero curve-construction or
torsion-order failures.

**The frontier.** For p = 10^27 + 103 (7 mod 8), the expected work at the
record stack's hit rate is 2.2-3.2 x 10^12 marks, i.e. 5.5-8 x 10^11
distinct conditioned curves - a projection confirmed by a direct bounded run
of 10M curves on the target prime itself (189K curves/s on 4 Milan vCPU).
On CPUs that is ~2-3 days on 64 cores; the realistic path is porting layers
1, 3 and 5 into the existing CUDA engine (the 10^26 record was 45 minutes on
one RTX 6000 at 5.2 x 10^7 marks/s; the verified layers project the expected
10^27 run into the ~3-8 GPU-hour range, with the usual exponential-draw
spread). Note the search-time distribution is close to exponential: single
runs routinely deviate several-fold from expectation in both directions.

## 5. Why these are constant factors, and not the exponent

The acceptance process behind every engine here is a tower of quadratic-
character gates chi(d_j), one per halving level. A gate can be sourced for
free exactly when its double cover has genus <= 1 (a genus-one curve can be
walked; by Riemann-Hurwitz nothing of higher genus can be parameterized or
walked onto). The layers above are precisely the maximal genus-<=1 prefix of
that tower: the next gate's covers have genus 5 and 17, and an exact-
linear-algebra certification over F_2 (all character words through depth 21,
three residue classes, ~7.5M sampled streams) found no multiplicative
relation that would collapse any later gate. The constructive alternative
(CM: prescribe the curve order) is also closed for this target: the three
admissible orders of 10^27 + 103 all have essentially squarefree
discriminants with class numbers ~10^13 ~ sqrt(p). Every remaining route to
a better exponent is an open mathematics problem (sublinear single-root
extraction of Hilbert class polynomials mod p; a Prym/correspondence
structure on the gate tower; a polylog predictor of v2(#E)).

## 6. Building and running

    make            # search (needs libgmp-dev) + search_conjugate
    make check      # lemma audit, lookahead differential, verified solves

    ./search <p> [seed]                 # prints: A x0 candidates_tested
    ./search_conjugate <p> [seed]

Environment variables:

| variable | engine | effect |
|---|---|---|
| `SEARCH_THREADS` | search | worker count (default: cgroup quota, else 4) |
| `SEARCH_MAX_TRIALS` | search | bounded run, for benchmarking |
| `POMERANCE_NO_LOOKAHEAD=1` | search | disable layer 5 (A/B and differential) |
| `POM_THREADS` | search_conjugate | worker count |

Third positional argument of `search_conjugate` is a bounded trial budget.
Diagnostics go to stderr; stdout carries exactly the one-line answer. Verify
any output against the official DANGER3 logic:

    python3 tools/vpp.py <p> <A> <x0>       # prints True

Both engines support p mod 8 in {3, 5, 7} (p = 1 mod 8 remains unsupported
in the whole lineage, as in the record stack). OpenMP is required for
multithreading; without it the engines build single-threaded.

## 7. Provenance and license

MIT, throughout. The base search lineage is Fabian Ruehle's, Alexa McLain's
and Jane Shi's MIT-licensed DANGER3 record code; their headers are preserved
in the sources. The production engines were generated by autonomous research
agents (GPT-5.6-sol under the codex harness) inside the pomerance-bench
evaluation environment on 2026-08-28, independently verified as described
above, and are preserved as judged. The reciprocal-mark lemma audit, the
skeleton-lookahead layer, the cross-platform measurements and this document
are the repository's own contributions. Every performance number in this
README states its platform; when reproducing, measure the record stack with
its thread count pinned to the actual CPU quota - an oversubscribed baseline
silently inflates any ratio by ~2x.
