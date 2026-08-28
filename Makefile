CC ?= gcc
CFLAGS ?= -O3 -fopenmp -Wall -Wextra
LDLIBS ?= -lm

pomerance_search: src/pomerance_search.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

.PHONY: clean
clean:
	rm -f pomerance_search
