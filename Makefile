CC ?= gcc
CFLAGS ?= -O3 -fopenmp -flto -DNDEBUG

# The engine is scale-general (p < 2^127) and needs GMP (libgmp-dev) for its
# two-limb Jacobi and powm kernels.
search: src/search.c
	$(CC) $(CFLAGS) -o $@ $< -lgmp -lm

# Fast confidence check:
#   1. the reciprocal-mark lemma audit (independent Python implementation);
#   2. the skeleton-lookahead differential: with a fixed seed and one thread,
#      the lookahead chain must emit the exact same triple and candidate
#      count as the plain chain it replaces;
#   3. solves in every supported residue class, each certificate accepted by
#      the official DANGER3 verifier logic.
.PHONY: check
check: search
	python3 tools/audit_reciprocal.py --samples 200
	@a=$$(SEARCH_THREADS=1 ./search 1000000000039 7); \
	b=$$(SEARCH_THREADS=1 POMERANCE_NO_LOOKAHEAD=1 ./search 1000000000039 7); \
	if [ "$$a" = "$$b" ]; then echo "lookahead differential: MATCH  [$$a]"; \
	else echo "lookahead differential: MISMATCH  [$$a] vs [$$b]"; exit 1; fi
	@for P in 1000000000039 1000000000061 1000000000091; do \
	  out=$$(./search $$P 3); set -- $$out; \
	  if python3 tools/vpp.py $$P $$1 $$2 | grep -qx True; then \
	    echo "p=$$P: verified  [$$out]"; \
	  else echo "p=$$P: VERIFIER REJECTED  [$$out]"; exit 1; fi; \
	done
	@echo "check: all passed"

# Equal-coverage benchmark against the vendored record engine
# (bench/upstream/pomerance.c): 2N upstream marks vs N distinct curves.
.PHONY: bench
bench:
	python3 bench/compare.py

.PHONY: clean
clean:
	rm -f search
