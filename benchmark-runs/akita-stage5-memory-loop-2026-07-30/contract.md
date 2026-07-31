# Akita Stage 5 memory-traffic loop

Date: 2026-07-30 EDT

## Question

Can the high-`T` PIOP remove trace-sized memory traffic without changing the
protocol, slowing the affected stage, or increasing its analytical live set?

The fixed workload is SHA-256 chain with K256 physical packing and 32-bit
virtual instruction chunks. Screens use `T = 2^22`; viable candidates advance
to `2^26`, and accepted structural changes receive an exact `2^28` run with
peak-RSS and swap counters.

## Candidates

1. Collapse source-identifiable zero lanes in the first three lattice
   Booleanity rounds.
2. Fuse read-RAF `u_evals` condensation into the existing RAF-Q scan.
3. Store per-table read-RAF cycle buckets as `u32` instead of `usize`.

Each candidate changes prover evaluation or storage only. Sumcheck messages,
claims, transcript order, verifier behavior, K256, D128, and the Akita
commitment/opening schedule remain fixed.

## Acceptance

- the proof verifies;
- the directly affected kernel or stage does not regress;
- a memory candidate removes a statically countable live owner and adds no
  pass over the trace;
- the exact `2^28` run remains below 90 GiB maximum RSS with zero process
  swaps;
- standard, ZK, and Akita muldiv suites and all required clippy modes pass.

The evaluator is:

```text
PERF_LOG_T={22|26|28}
PERF_LOG_K_CHUNK=8
PERF_LOOKUPS_RA_VIRTUAL_LOG_K_CHUNK=32
PERF_TRACE=1
cargo nextest run --release -p jolt-prover-legacy --features akita \
  -E 'test(sha2_chain_akita_perf)' --run-ignored all --no-capture --cargo-quiet
```
