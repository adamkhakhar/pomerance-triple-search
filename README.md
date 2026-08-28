# reciprocal-mark-cut

A fast search for **Pomerance triples**, the shortest known primality
certificates, targeting the open [DANGER3 data challenge](https://github.com/AndrewVSutherland/DANGER3).
This implementation improves on the record-holding search method (the code
lineage behind the 10^22 through 10^26 challenge solves) by a measured
**1.7x to 1.8x** under matched conditions. The core of the improvement is a
proof that the prior state of the art spent exactly half of its main work on
candidates that could never change any outcome, plus supporting field
arithmetic specialized to the search regime.

## Background: what is being searched for

For an odd integer p, let q = floor(sqrt(p)) and let k be the least integer
with 2^k > q + 1 + 2*sqrt(q); note 2^k is approximately 2*sqrt(p). A
**Pomerance triple** (p, A, x0) asserts that the point with x-coordinate x0
on the Montgomery curve

    B*y^2 = x^3 + A*x^2 + x   over Z/pZ

has exact order 2^k: doubling the projective point (x0 : 1) exactly k times
lands on the identity (Z = 0 mod p), while the (k-1)-th doubling does not.
If p had a prime factor r <= sqrt(p), the point would survive reduction mod r
with order 2^k, exceeding the Hasse bound on the curve's size over F_r. So a
valid triple is an unconditional proof that p is prime, checkable in about
(1/2)*log2(p) doublings (Pomerance, *Very short primality proofs*, Math.
Comp. 48 (1987); refined by Sutherland for DANGER3).

Verification is trivial. Search is not: a random curve's order is divisible
by 2^k with probability on the order of 1/sqrt(p), so every known method
performs on the order of sqrt(p) candidate tests, and progress consists of
driving down the constant. Human and AI teams walked the challenge from
10^19 to 10^26 during 2026; p = 10^27 + 103 is open at the time of writing.

## The prior state of the art

The record method (`x16halvenonsplit`; base by Fabian Ruehle, X1(16) sampler,
successive halving and nonsplit filter by Alexa McLain, p = 3 mod 4 square
roots by Jane Shi; all MIT licensed) improves blind random search in three
ways:

1. **Prescribed 16-torsion.** Curves are drawn from the one-parameter X1(16)
   family (Tate normal form of the modular curve X1(16)): a random field
   element y determines a curve A(y) together with a rational point of order
   16 on it. This forces 16 to divide the curve order, a roughly 16x hit
   rate boost, and every candidate test starts from a known point of
   order 16.
2. **Successive halving.** A curve is tested by repeatedly halving that
   point: each step solves [2]Q' = Q on the x-line at the cost of one or two
   modular square roots. A successful halving doubles the point's 2-power
   order; a failed one proves the chain can never reach 2^k and the curve is
   abandoned. Halving fails with probability about one half per level, so
   almost all curves die after a couple of cheap steps.
3. **Nonsplit filter.** A quadratic character test on (y^2 - 2)(y^2 - 4y + 2)
   rejects, before any halving, the samples whose rational 2-power subgroup
   splits; the surviving (cyclic) family concentrates the deep halving
   chains.

Measured expectation for this stack: roughly 0.07 to 0.1 times sqrt(p)
candidates per certificate.

## What this implementation changes

### 1. The reciprocal-mark cut (main result)

To construct each family curve, the sampler solves the quadratic

    qa*x^2 + qb*x + qc = 0,
    qa = y^2 - 2y,  qb = 2y^2 - y^3,  qc = 1 - y

and the prior implementation emitted **both** roots as two marked points on
the curve, running the full halving test on each.

**Lemma.** The two marked points have reciprocal x-coordinates, x and 1/x.
On B*y^2 = x^3 + A*x^2 + x, the map x -> 1/x is translation by the rational
2-torsion point (0,0). On the nonsplit family the rational 2-power subgroup
is cyclic, so two points that differ by 2-torsion have the same halving
depth. Hence the two marks always return the same verdict: both reach order
2^k, or neither does.

The second halving chain therefore contributes zero additional hit
probability at full cost. This implementation keeps one mark per curve: an
exact halving of the dominant work term, with no change to the distribution
of curves searched.

The lemma was validated empirically twice, with independent code: 15,000
sampled emission pairs across residue classes p mod 8 = 3, 5, 7 (every pair
shared its curve, was reciprocal, and had equal halving depth), and a
separate from-scratch depth check on 3,200 random (p, A, x) draws across
primes from 10^6 to 2*10^19, with zero mismatches.

### 2. Centered 65-bit Barrett arithmetic

For 2^64 < p < 2^65, a residue centered into (-p/2, p/2] has a 64-bit
magnitude, and a product of two such magnitudes fits in 128 bits. Every
modular multiplication in the hot path is then one 64x64 multiply plus a
one-step Barrett correction with the precomputed constant
mu = floor(2^128 / p). This removes all Montgomery-domain machinery and
conversions; the entire program runs on plain residues. Equivalence with a
reference multiplier was checked on 6 million random products across the
three residue classes.

### 3. Batched inversions

The marked point is formed projectively, x/(x - y) = r/(r - 2*qa*y), which
avoids inverting the quadratic denominator; the two remaining inversions per
curve (coefficient denominator and marked-point denominator) are batched
across 32 curves with Montgomery's product trick. Net cost: about one field
inversion per 32 curves, versus three inversions per emitted pair before.

## Measured results

All measurements in a fixed container on the same machine, both sides at
identical thread counts, identical seeds, primes covering the three residue
classes near 2*10^19.

**Equal-coverage benchmark.** The prior stack tests two marked points per
curve, so N curves of coverage means 2N marked points for it and N for this
implementation. Coverage of 300,000 curves per prime:

| p (near 2*10^19) | prior stack | this implementation | speedup |
|---|---:|---:|---:|
| ...0011 (p mod 8 = 3) | 0.64 s | 0.35 s | 1.83x |
| ...0173 (p mod 8 = 5) | 0.65 s | 0.36 s | 1.81x |
| ...0447 (p mod 8 = 7) | 0.65 s | 0.36 s | 1.81x |

The original research prototype of this algorithm measured the same
comparison at 3.59x to 3.74x; roughly 2.05x of that came from a thread
misconfiguration in how the baseline was deployed (48 OpenMP workers on a
4-CPU cgroup quota), not from the algorithm, and is excluded here. The
honest algorithmic figure is **1.7x to 1.8x**.

**End-to-end certificates.** Full searches with this implementation (6
threads, seed 7) across all three supported residue classes, every triple
accepted by the official DANGER3 verifier (`tools/vpp.py`, exact integer
arithmetic); the search driver additionally re-verifies every candidate with
an independent schoolbook-arithmetic checker before printing it:

| p | curves tested | time | A | x0 |
|---|---:|---:|---|---|
| 20000000000000000011 | 632,413,749 | 757 s | 9259440997486396871 | 8427535570484589738 |
| 20000000000000000173 | 778,988,441 | 950 s | 19476394758607495157 | 4305480688130056071 |
| 20000000000000000447 | 804,917,759 | 970 s | 19660646159258000845 | 5142638602974011965 |

(As a fidelity check, the second solve reproduces the research prototype's
logged triple for the same prime and seed exactly, so the restructuring
preserves the original search trajectory bit for bit. Search times here are
tail draws above the mean; that is normal for an exponential distribution
and irrelevant to the equal-coverage throughput comparison above.)

**Why this is state of the art.** The baseline in these measurements is the
exact code lineage that set the 10^22, 10^23, 10^24, 10^25, and 10^26
DANGER3 records, run in its production mode (`x16halvenonsplit`) at matched
threads on the same machine and primes. A 1.7x to 1.8x reduction in
wall-clock per unit of search coverage, with the sampled curve distribution
unchanged, is a direct improvement on that record method in its own regime.

**What is not claimed.** Asymptotic complexity is unchanged: expected cost
is still on the order of sqrt(p) candidates; the hit probability per curve
is untouched. Search time per prime remains approximately exponentially
distributed, so single-solve comparisons are meaningless; all speedup claims
here are equal-coverage throughput measurements, which are deterministic up
to timer noise.

## Scope and limitations

- **Size**: requires 2^64 < p < 2^65. This is where the centered multiplier
  applies and where the speedup was measured. Note that the research
  prototype fell back to a slow conversion-heavy path outside this window
  and could run far slower than the prior stack there; this implementation
  refuses out-of-range input instead. Extending the centered-residue idea to
  two-limb operands (needed for the 10^27 + 103 frontier target, a 90-bit
  prime) is the natural next step and is expected to yield a smaller
  constant.
- **Residues**: requires p mod 8 in {3, 5, 7}, inherited from the square
  root formulas of the X1(16) method (n^((p+1)/4) for p = 3 mod 4;
  n^((p+3)/8) with a sqrt(-1) correction for p = 5 mod 8). The frontier
  target 10^27 + 103 is 7 mod 8. A fast p = 1 mod 8 route is an open gap in
  the entire method family.
- The reciprocal-mark cut and the batched inversions are structural and
  carry to any size, including GPU implementations.

## Build and run

```sh
make            # gcc -O3 -fopenmp
./pomerance_search <p> [seed] [max_curves]
```

Output on success is one line on stdout, `A x0 curves_tested`, where
`curves_tested` counts distinct curves (the prior stack counted marked
points, which reads 2x higher at identical work). Progress goes to stderr.
Verify any result against the official checker:

```sh
python3 tools/vpp.py <p> <A> <x0>    # prints True
```

If `OMP_NUM_THREADS` is unset, the thread count is taken from the cgroup
CPU quota rather than `nproc`, which avoids a roughly 2x oversubscription
penalty on shared machines that expose more logical cores than the quota
allows.

## Provenance and license

MIT. This code derives from the MIT-licensed DANGER3 record implementations:
2-Sylow projection search base by Fabian Ruehle (with Claude Opus 4.6),
X1(16) prescribed-torsion sampler, successive-halving descent, and nonsplit
discriminant filter by Alexa McLain (with GPT 5.5 Codex), and p = 3 mod 4
square-root support by Jane Shi. The reciprocal-mark cut, the centered
65-bit Barrett arithmetic, and the batched-inversion sampler were produced
by an autonomous GPT-5.6 research session inside the pomerance-bench agent
environment (2026) and independently verified as described above; this
repository is a clean single-file restructuring of that work restricted to
its validated scope. The verifier `tools/vpp.py` is by Andrew V. Sutherland
(DANGER3).
