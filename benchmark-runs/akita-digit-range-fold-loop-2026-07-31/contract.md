# Akita high-basis digit-range fold experiment

Date: 2026-07-31 EDT

## Question

Can the large D128/basis-64 evaluation proof parallelize its exact-prefix
table folds without allocating a second table or changing the proof?

The accepted `2^28` parent spends 5.800 seconds in 264
`digit_range_fold_lanes` spans. The dominant root instance starts with
629,162,496 live digits and materializes 157,290,624 eight-lane rows after
the first two rounds.

## Candidate

Keep the existing in-place table and fold a small prefix serially. Then process
geometric output ranges `[a, 2a)` in parallel. Pair `i` writes row `i` and
reads rows `2i` and `2i + 1`; after `[0, a)` has finished, the next range's
outputs can no longer overwrite an unread input.

The candidate may change scheduling only. It may not change field arithmetic,
round messages, transcript order, table contents, or the protocol.

## Evaluator

- Focused screen: basis 64, domain `2^26`, three-quarter live prefix,
  `Prime128Offset275`.
- Target: exact forced-K256 SHA2-chain workload at `T = 2^28`, D128.
- Target command:

  ```bash
  PERF_LOG_T=28 PERF_LOG_K_CHUNK=8 \
  PERF_LOOKUPS_RA_VIRTUAL_LOG_K_CHUNK=32 PERF_TRACE=1 \
  /usr/bin/time -l cargo nextest run --release \
    -p jolt-prover-legacy --features akita \
    -E 'test(sha2_chain_akita_perf)' \
    --run-ignored all --no-capture --cargo-quiet
  ```

## Acceptance

- at least 25% less time in `digit_range_fold_lanes`;
- at least 2 seconds less target-scale evaluation-proof time;
- exact proof verification;
- no new table-sized allocation or higher analytical peak;
- peak RSS below 90 GiB and zero swaps;
- parallel and sequential Akita tests and warning-denying Clippy pass;
- standard, ZK, and Akita Jolt validation pass.
