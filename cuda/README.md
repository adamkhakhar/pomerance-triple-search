# CUDA engine

GPU search for the frontier target **p = 10^27 + 103**. Built on McLain's
CUDA port of the record stack (the engine that set the 10^26 record: 52M
candidates/s on one RTX 6000 Ada), with the two exact layers from
`../src/search.c` added to the u96 path.

p = 10^27 + 103 is 90 bits and p = 3 mod 4, so it uses the fast `u96`
backend and the `n^((p+1)/4)` square-root branch — the same branch the 10^26
record run exercised. No new arithmetic is needed at this size.

## What was added

**1. Reciprocal-mark deduplication (exact 2x).** The record kernel tests
both roots of the X1(16) auxiliary quadratic (its `for (ri = 0; ri < 2; ri++)`
loop). Those two marks are the same curve: they have the same Montgomery A
and reciprocal x-coordinates, and on a nonsplit curve they differ by the
rational 2-torsion translate x -> 1/x, which preserves halving depth. They
always return the same verdict, so the cover kernel tests one. This layer
was in the CPU engines but had never been ported to the GPU.

**2. Genus-one first-lift cover source (exact 2x hazard).** With
G = (y^2-2)(y^2-2y+2), the X1(16) fiber discriminant D = y(y-2)G and the
first halving obstruction H = (y-1)G satisfy

    D * H = y(y-1)(y-2) * G^2,

and with X = y-1 that first factor is X^3 - X. So on the fixed elliptic
curve **E0: W^2 = X^3 - X**, every point gives H = (W G)^2 / D: once D is a
square, H is a square automatically. Sampling y by walking E0 instead of
drawing it uniformly emits curves whose first halving gate is already
passed — exactly twice the hazard per candidate — and the root the halving
step needs is transported algebraically instead of extracted:

    sqrt(d1) = y * sD * W / (dx * (y-1)^2),    dx = r0 - 2*qa*y

with r0 = sD - qb, so the marked point is xP = r0/dx. Each thread walks
`POM_COVER_LANES` (default 4) independent points on E0 with a fixed step,
sharing one modular inversion per step by Montgomery's trick, so a cover
point costs a few field multiplies plus 1/LANES of an inversion. The walk is
branch-light and uniform across a warp — the property that made character
prefilters fail on GPU at p26 and makes this source work.

**3. One inversion less per curve.** The fused conversion never forms the
affine quadratic root (which needed 1/(2*qa)); working directly with r0/dx
lets a single batch-2 inversion cover both the curve coefficient and the
marked point. The record kernel paid two inversions per curve.

E0 is supersingular for p = 3 mod 4 (order p+1). That is irrelevant to the
certificate: E0 is only a parameter source for y — the curve that gets
certified is the Montgomery curve with coefficient A, found by the ordinary
search. Nothing supersingular or CM-derived enters the triple.

## Correctness, proven before spending GPU time

Every numeric routine is marked `HD` (host and device), so the exact code the
GPU runs can be compiled and tested on an ordinary machine:

    make check          # no GPU required

which runs, on 10^27+103 itself, on a 90-bit p = 5 mod 8 prime, and on small
primes of each supported class:

| check | result |
|---|---|
| Montgomery arithmetic vs `__int128` reference | 20000 products/sums, 0 bad |
| cover walk stays on W^2 = X^3 - X | 4000 steps x 4 lanes, 0 off-curve |
| batch inversion vs individual inversions | 2000 batches, 0 bad |
| **transport identity `sqrt(d1)^2 == d`** | **4001/4001 and 4000/4000 correct** |
| emitted points have exact order 32 | 2032 emitted, 0 wrong order |
| cover doubles depth-12 hazard | 1.90x and 2.16x (expect 2) |
| end-to-end: finds a real triple | verified by `tools/vpp.py` |

The end-to-end triples the cover source found on small primes —
`1000000000039 737021749285 253716261169` and
`1000000000061 881352043958 712647305595` — are accepted by the official
DANGER3 verifier, which shares no code with this engine.

`make kernel-syntax` additionally compiles the `__global__` kernels against a
small CUDA shim, so kernel typos surface on a GPU-less box.

On the GPU, the binary runs the same proofs on-device before any production
search (`selftest: PASS`), and refuses to search if they fail.

## Running

    make                      # ARCH defaults to sm_89 (Ada); A100 sm_80, H100 sm_90
    ./run_frontier.sh         # build -> selftest -> A/B -> production -> verify
    ./run_frontier.sh --ab-only     # stop after the measured A/B

`run_frontier.sh` shards across every visible GPU on disjoint seed ranges and
stops all workers as soon as one reports a verified hit.

Environment: `POM_SOURCE=legacy` restores the record-stack kernel for
same-seed A/B; `POM_SELFTEST=0` skips the gate, `=only` runs it and exits;
`LANES=n` sets walks per thread (4 is the register-pressure sweet spot; 8 is
worth an A/B on large-register parts).

**Budget units differ between sources.** Legacy counts marks; the cover
kernel counts cover curves, each worth 2 distinct curves = 4 legacy marks of
hazard. Expected work for 10^27+103 is 2.2-3.2e12 marks, i.e.
**5.5-8e11 cover curves**; `run_frontier.sh` defaults to 2.4e12, about 3x
expectation (the ~95th percentile of an exponential draw).

## Validating on a rented GPU

`pod_validate.sh` is a single self-contained file: the sources are embedded,
so a fresh pod needs no repo access, no tokens and no file transfer tooling.
Copy it to any NVIDIA pod with a CUDA **devel** image (nvcc required) and run:

    bash pod_validate.sh              # build + host proof + device selftest + A/B
    bash pod_validate.sh --search     # ...and continue into the production search

It detects the GPU architecture, proves the device math on the host, builds,
runs the on-device selftest, then times the legacy record kernel against the
cover kernel at identical hazard on the frontier prime and projects the
expected time-to-triple from the measured rate. Any failure stops the script
with a non-zero status.
