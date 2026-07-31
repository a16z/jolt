# Packed increment-cache reuse experiment

Date: 2026-07-31 EDT

## Question

Can Stage 6 and Stage 7 consume the nine fused-increment one-hot lanes already
stored in the packed `OneHotTrace` row cache, instead of materializing nine
additional trace-length byte columns?

The candidate keeps the committed polynomial, protocol, transcript, and
claims unchanged. It adds a strided dense view to `RaPolynomial` and lets the
Booleanity and hamming-weight paths read the existing row-major lanes
directly.

## Fixed configuration

- Machine: Apple M4 Max
- Control revision: `6536b7af906c37345c1e4c25f8fc1306aa251ab4`
- Candidate revision: `011d96f77c24b6afa621f202b3114de5dd40069d`
- Candidate tree: `0d29e76fbe18211e27af18c54bf24d6a2820aa25`
- Akita revision: `a56b933cd097deebd3a1d937d1308e7d02ea6e0a`
- Rust: `rustc 1.95.0 (59807616e 2026-04-14)`
- Physical one-hot `K = 256`
- `PERF_LOG_K_CHUNK=8`
- `PERF_LOOKUPS_RA_VIRTUAL_LOG_K_CHUNK=32`
- Release profile, proof verification required

Benchmark command:

```bash
/usr/bin/time -l env \
  PERF_LOG_T=28 \
  PERF_LOG_K_CHUNK=8 \
  PERF_LOOKUPS_RA_VIRTUAL_LOG_K_CHUNK=32 \
  PERF_TRACE=1 \
  cargo nextest run --release \
    -p jolt-prover-legacy --features akita \
    -E 'test(sha2_chain_akita_perf)' \
    --run-ignored all --no-capture --cargo-quiet
```

The same command with `PERF_LOG_T=22` and `PERF_LOG_T=26` supplies the small
screen and intermediate replication.

## Metrics

The localized aggregate is:

```text
fused_inc_one_hot_columns
+ fused_inc_deltas
+ prove_stage6a_lattice
+ prove_stage6b_lattice
+ prove_stage7_lattice
```

The candidate has no `fused_inc_one_hot_columns` span because it removes that
work. Inclusive Perfetto durations are used throughout. Whole-prover time and
commitment time are reported separately so machine drift outside the changed
path is visible.

The analytical memory metric is the removed live owner:

```text
9 one-hot columns * T rows * 1 byte = 9T bytes.
```

This allocation begins after Stage 5. The experiment therefore should reduce
Stage 6/7 live memory but is not expected to lower the current Stage 5 process
peak.

## Gates

1. Strided and contiguous RA polynomials must agree through every binding
   state in both binding orders.
2. Natural, forced-K256, and committed-program Akita proofs must verify.
3. The localized aggregate must not regress in the repeated `T = 2^22` and
   `T = 2^26` screens.
4. One exact `T = 2^28` proof must verify with no swaps and no affected-stage
   regression.
5. The candidate must delete exactly `9T` bytes of live data without adding
   another trace-scaled owner.
6. Standard and ZK Dory regression tests, scoped and workspace Clippy, and
   formatting must pass.

