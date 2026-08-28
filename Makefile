CC ?= gcc
CFLAGS ?= -O3 -fopenmp -flto -DNDEBUG

# search           - recommended engine: Jacobi-gated halving + skeleton
#                    lookahead, scale-general (p < 2^127).  Needs GMP
#                    (libgmp-dev) for the two-limb Jacobi/powm kernels.
# search_conjugate - conjugate-lift quotient engine, dependency-free.
# legacy_fast65    - the original reciprocal-mark-cut engine (2^64 < p < 2^65
#                    window only), kept for provenance.

all: search search_conjugate

search: src/search.c
	$(CC) $(CFLAGS) -o $@ $< -lgmp -lm

search_conjugate: src/search_conjugate.c
	$(CC) $(CFLAGS) -o $@ $< -lm

legacy_fast65: src/legacy/fast65.c
	$(CC) $(CFLAGS) -o $@ $< -lm

# Fast confidence check:
#   1. the reciprocal-mark lemma audit (independent Python implementation);
#   2. the skeleton-lookahead differential: with a fixed seed and one thread,
#      the lookahead chain must emit the exact same triple and candidate
#      count as the plain chain it replaces;
#   3. both engines solve small primes in every supported residue class and
#      the official DANGER3 verifier accepts each emitted certificate.
.PHONY: check
check: search search_conjugate
	python3 tools/audit_reciprocal.py --samples 200
	@a=$$(SEARCH_THREADS=1 ./search 1000000000039 7); \
	b=$$(SEARCH_THREADS=1 POMERANCE_NO_LOOKAHEAD=1 ./search 1000000000039 7); \
	if [ "$$a" = "$$b" ]; then echo "lookahead differential: MATCH  [$$a]"; \
	else echo "lookahead differential: MISMATCH  [$$a] vs [$$b]"; exit 1; fi
	@for P in 1000000000039 1000000000061 1000000000091; do \
	  for eng in ./search ./search_conjugate; do \
	    out=$$($$eng $$P 3); set -- $$out; \
	    if python3 tools/vpp.py $$P $$1 $$2 | grep -qx True; then \
	      echo "$$eng p=$$P: verified  [$$out]"; \
	    else echo "$$eng p=$$P: VERIFIER REJECTED  [$$out]"; exit 1; fi; \
	  done; \
	done
	@echo "check: all passed"

.PHONY: clean
clean:
	rm -f search search_conjugate legacy_fast65 pomerance_search
