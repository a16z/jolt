# Akita D128 commitment loop

Date: 2026-07-30 EDT

## Question

Can the D128/K256 root commitment at exactly `T = 2^28` be reduced by at
least 3% without increasing analytical live memory, peak RSS, or any
unaffected prover span?

The accepted parent is Jolt `b8dec4136` with Akita `67c3c88d`. Its retained
target measurement is:

| Metric | Parent |
|---|---:|
| Root accumulation | 87.61 s |
| Commitment | 87.93 s |
| Evaluation proof | 33.32 s |
| Whole prover | 199.25 s |
| Maximum RSS | 89,992,118,272 B |
| Process swaps | 0 |

The D128 policy is selected only for the K256 packed shape with at least 41
variables, so experiments in this run do not change the `2^26` policy.

## Evaluator and guards

The immutable target is the ignored forced-K256 `sha2_chain_akita_perf` test:

```text
PERF_LOG_T=28 PERF_TRACE=1 /usr/bin/time -l \
  cargo nextest run --release -p jolt-prover-legacy --features akita \
  -E 'test(sha2_chain_akita_perf)' --run-ignored all --no-capture --cargo-quiet
```

The primary metric is the inclusive `trace_onehot_commit_accumulate` span.
A candidate advances only if it improves that span by at least 3%, verifies
the proof, introduces no analytical live-memory increase, remains below 90
GiB RSS, reports zero process swaps, and does not reproducibly regress an
unaffected span by more than 3%.

Small synthetic or reduced-size measurements are directional screens only.
Any accepted source change requires an exact `2^28` run and an adjacent
parent/candidate recheck when the effect is close to the threshold.

## Candidate order

1. Replace D128's wide rank accumulator with canonical Fp128 coefficients.
   This halves the active accumulator from 116 KiB to 58 KiB while replacing
   carry-free NEON additions with modular additions.
2. If the first ablation shows that footprint matters but modular arithmetic
   costs too much, test a bounded 128-bit representation with periodic
   Solinas normalization.
3. Batch aligned trace blocks so the same A position can serve several
   blocks before leaving cache. Batch size must be small enough that
   destination accumulators do not lose locality.
4. Only then consider a new D128 shift layout. The earlier global
   coefficient-major D64 experiment regressed 63%, so it is not repeated
   without D128-specific evidence.

Each attempt changes one mechanism. Rejected code is restored before the
next attempt. Accepted implementation commits are separate from benchmark
documentation commits.

## Budget

- four exploratory trials and two validation trials;
- two hours of local M4 Max compute;
- no external services;
- stop after two consecutive low-signal candidate regions.
