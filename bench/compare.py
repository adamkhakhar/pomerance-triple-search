#!/usr/bin/env python3
"""
compare.py: reproduce the equal-coverage benchmark against the prior state
of the art, from sources contained entirely in this repository.

Protocol. The prior stack (bench/upstream/pomerance.c, vendored verbatim
from the record implementation) tests two marked points per curve; this
implementation tests one. Equal search coverage of N curves therefore means
a budget of 2N candidates for the upstream binary and N for ours. Both sides
run in their production mode on the same primes with the same seed and the
same OpenMP thread count, and neither is expected to find a certificate at
these small budgets, so the measurement is deterministic throughput, not a
solve-time lottery.

Usage:
    python3 bench/compare.py [--curves N] [--threads T] [--seed S]

Requires gcc with OpenMP. Builds both binaries into /tmp.
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# One prime per supported residue class, just above 2 * 10^19.
PRIMES = [
    ("p mod 8 = 3", "20000000000000000011"),
    ("p mod 8 = 5", "20000000000000000173"),
    ("p mod 8 = 7", "20000000000000000447"),
]


def build(src, out, extra=()):
    cmd = ["gcc", "-O3", "-fopenmp", "-DNDEBUG", "-o", out, src, "-lm", *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"build failed: {' '.join(cmd)}\n{r.stderr[-800:]}")


def timed(cmd, env):
    t0 = time.monotonic()
    subprocess.run(cmd, capture_output=True, env=env)
    return time.monotonic() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", type=int, default=300000,
                    help="coverage per prime, in distinct curves (default 300000)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", default="9001")
    args = ap.parse_args()

    ours = "/tmp/pts_bench_ours"
    theirs = "/tmp/pts_bench_upstream"
    build(os.path.join(ROOT, "src", "search.c"), ours, ("-lgmp",))
    build(os.path.join(HERE, "upstream", "pomerance.c"), theirs)

    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads),
               SEARCH_THREADS=str(args.threads),
               SEARCH_MAX_TRIALS=str(args.curves))
    print(f"coverage per prime: {args.curves} curves "
          f"(= {2 * args.curves} upstream candidates), "
          f"threads: {args.threads}, seed: {args.seed}\n")
    print(f"{'class':14s} {'upstream':>10s} {'this repo':>10s} {'speedup':>9s}")

    tot_up = tot_us = 0.0
    for label, p in PRIMES:
        t_up = timed([theirs, p, args.seed, str(2 * args.curves),
                      "x16halvenonsplit"], env)
        t_us = timed([ours, p, args.seed], env)
        tot_up += t_up
        tot_us += t_us
        print(f"{label:14s} {t_up:9.2f}s {t_us:9.2f}s {t_up / t_us:8.2f}x")

    print(f"{'total':14s} {tot_up:9.2f}s {tot_us:9.2f}s {tot_up / tot_us:8.2f}x")


if __name__ == "__main__":
    main()
