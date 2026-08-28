#!/usr/bin/env python3
"""
audit_dedup.py: reproduce the empirical validation of the marked-point
deduplication lemma behind the search engine's exact 2x layer.

Lemma. On the Montgomery curve B*y^2 = x^3 + A*x^2 + x over Z/pZ, the map
x -> 1/x is translation by the rational 2-torsion point (0,0). On the
nonsplit X1(16) family the rational 2-power subgroup is cyclic, so two
points differing by 2-torsion have the same halving depth. Consequently the
two marked points emitted per curve by the record-stack sampler
(which have reciprocal x-coordinates) always return the same search verdict,
and testing the second one is pure double work.

This script checks the observable consequence directly, with code that is
independent of src/search.c: for random (p, A, x), the doubling
chains started at x and at 1/x mod p must reach Z = 0 at the same step
(or both fail to within the depth bound). Any mismatch would break the
lemma and the deduplication.

Usage:
    python3 tools/audit_dedup.py [--samples N] [--seed S]

Exit code 0 and a summary line if no mismatches are found; exit 1 otherwise.
"""

import argparse
import random
import sys


def death_depth(p, A, x, kmax=80):
    """Step at which the doubling chain from (x : 1) first hits Z = 0 mod p,
    or None if it survives kmax steps. Uses the (X+Z)/(X-Z) doubling form of
    the official DANGER3 verifier."""
    inv4 = (p + 1) // 4 if p % 4 == 3 else (3 * p + 1) // 4
    C = ((A + 2) * inv4) % p
    X, Z = x % p, 1
    for s in range(1, kmax + 1):
        U, V = (X + Z) * (X + Z) % p, (X - Z) * (X - Z) % p
        W = U - V
        X, Z = U * V % p, W * (V + C * W) % p
        if Z % p == 0:
            return s
    return None


def is_probable_prime(n, rounds=40, rng=random):
    if n < 2:
        return False
    for sp in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % sp == 0:
            return n == sp
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for _ in range(rounds):
        a = rng.randrange(2, n - 1)
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def next_prime(n, rng):
    n |= 1
    while not is_probable_prime(n, rng=rng):
        n += 2
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=800,
                    help="random (A, x) draws per prime (default 800)")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    # Primes across scales, ending inside this implementation's 65-bit window.
    bases = [10**6 + 3, 10**9 + 7, 10**12 + 39, 2 * 10**19 + 11]
    primes = [next_prime(b, rng) for b in bases]

    total = mismatches = 0
    for p in primes:
        for _ in range(args.samples):
            A = rng.randrange(3, p - 3)
            if (A * A - 4) % p == 0:
                continue
            x = rng.randrange(2, p - 1)
            xi = pow(x, p - 2, p)
            d1 = death_depth(p, A, x)
            d2 = death_depth(p, A, xi)
            total += 1
            if d1 != d2:
                mismatches += 1
                print(f"MISMATCH p={p} A={A} x={x} depth(x)={d1} depth(1/x)={d2}")

    print(f"samples={total} primes={len(primes)} depth_mismatches={mismatches}")
    if mismatches:
        print("LEMMA VIOLATED: do not trust the deduplication.")
        return 1
    print("ok: reciprocal points always share their doubling death depth")
    return 0


if __name__ == "__main__":
    sys.exit(main())
