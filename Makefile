CC ?= gcc
CFLAGS ?= -O3 -fopenmp -Wall -Wextra
LDLIBS ?= -lm

pomerance_search: src/pomerance_search.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

# Fast confidence check:
#   1. the reciprocal-mark lemma audit (independent Python implementation);
#   2. a bounded search smoke run (budget exhaustion, exit code 1, is fine);
#   3. the official verifier on a known certificate produced by this code.
.PHONY: check
check: pomerance_search
	python3 tools/audit_reciprocal.py --samples 200
	./pomerance_search 20000000000000000173 7 200000 || test $$? -eq 1
	python3 tools/vpp.py 20000000000000000173 19476394758607495157 4305480688130056071 | grep -qx True
	@echo "check: all passed"

# Equal-coverage benchmark against the vendored prior state of the art.
.PHONY: bench
bench:
	python3 bench/compare.py

.PHONY: clean
clean:
	rm -f pomerance_search
