# Akita Stage 6 Booleanity fusion experiment

Date: 2026-07-30 EDT

## Question

Can the first three delayed-binding rounds of lattice Booleanity exploit
repeated zero and equal lane patterns across the base RA and fused-increment
columns, reducing the `T = 2^28` Stage 6 cost without changing the proof or
adding trace-scaled storage?

The accepted D128 trace spends 7.91 seconds in Booleanity messages and 2.63
seconds ingesting their challenges. The nine fused-increment columns use the
same address table and cycle schedule as the base RA columns but currently
live in a separate polynomial representation.

## Search

1. Measure exact source-pattern multiplicities for delayed-binding widths
   two, four, and eight.
2. Reject arithmetic grouping if the resulting product-count ceiling is
   below 5% of Booleanity's affected work.
3. Otherwise implement the smallest source-aware kernel that preserves every
   round polynomial and final opening.
4. Promote from `2^22` to `2^26`, then `2^28`, only after an adjacent focused
   win.

The first candidate may skip provably absent RAM groups and combine columns
whose complete source-lane tuples are equal. It must not infer equality from
the honest witness in the protocol: grouping is only a prover evaluation
strategy for the same polynomial.

## Acceptance

- proof verification succeeds;
- Booleanity message plus bind time improves by at least 5%;
- the full prover does not regress beyond the adjacent noise band;
- no trace-scaled allocation is introduced;
- peak RSS does not increase materially and the `2^28` run has zero swaps;
- focused equivalence tests compare every delayed round and final claim with
  the existing implementation.

The transcript, claims, verifier, packed commitment, D128 schedule, K256
configuration, workload, and benchmark parser are frozen.
